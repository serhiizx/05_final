"""MAS HR-скринінгу в LangGraph: supervisor + чотири спеціалізовані агенти.

Supervisor — LLM-маршрутизатор зі structured output; він класифікує ЗАПИТ і
через conditional edge передає його одному агенту. Кожен агент має власний
системний промпт, власний allowlist інструментів і власний патерн:

    screening     — Plan-and-Execute підграф (з ДЗ2): planner → executor → replanner
    researcher    — Agentic RAG на ChromaDB (з ДЗ2): search_hr_kb + MCP resource
    communicator  — ReAct із шаблоном MCP Prompt, далі HITL
    general       — fallback без інструментів

Домен живе в MCP-сервері (окремий процес), цей модуль відповідає лише за
оркестрацію, захист і трасування.
"""

import asyncio
import contextvars
import json
import operator
import sys
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from config import (
    AGENT_TIMEOUT_S,
    EXECUTOR_MAX_STEPS,
    MAX_REPLANS,
    MAX_STEPS,
    ROOT,
    load_prompt,
    make_llm,
)
from guardrails import (
    ToolDenied,
    check_input_length,
    check_tool_call,
    detect_injection,
    rate_limiter,
    redact_pii,
)
from schemas import (
    EmailDraft,
    Plan,
    PlanStep,
    ReplanDecision,
    RouteDecision,
    ScreeningVerdict,
)
from tools_legacy import search_hr_kb
from trajectory_logger import log_step

# MCP-сервер піднімається як підпроцес того самого інтерпретатора.
MCP_SERVER_CONFIG = {
    "hr": {
        "command": sys.executable,
        "args": [str(ROOT / "mcp_server.py")],
        "transport": "stdio",
    }
}

# Текст резюме недовірений, тому саме його результат проганяється крізь
# input guardrail на межі інструмента — до того, як його побачить модель.
UNTRUSTED_TOOLS = {"fetch_resume"}

# Куди обгортки інструментів складають події траєкторії поточного прогону.
# ContextVar, а не глобальний список: вузли графа асинхронні, і два одночасні
# прогони інакше писали б в один буфер.
_trace_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "trace_sink", default=None
)


def _trace(event: dict) -> None:
    """Додати подію в буфер поточного прогону, якщо він відкритий."""
    sink = _trace_sink.get()
    if sink is not None:
        sink.append(event)


# ── Стан ────────────────────────────────────────────────────────────────────


class MASState(TypedDict, total=False):
    """Стан MAS, спільний для всіх вузлів графа.

    Поля походять з попередніх робіт: step_count і trajectory — з ДЗ1,
    plan/current_step/results — з Plan-and-Execute у ДЗ2.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    current_agent: str
    candidate_id: str
    job_id: str
    plan: list[dict]
    current_step: int
    results: list[str]
    step_count: int
    trajectory: Annotated[list, operator.add]
    verdict: dict | None
    email_draft: dict | None
    hitl_action: str
    completed: bool
    pending_approval: bool
    report: str


# ── Інструменти під захистом ────────────────────────────────────────────────


def _tool_result_text(content: Any) -> str:
    """Дістати текст із результату MCP-інструмента.

    langchain_mcp_adapters повертає не рядок, а список content-блоків
    [{"type": "text", "text": "..."}] — і в прямому ainvoke, і в ToolMessage.
    """
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def guarded_tool(agent: str, tool: BaseTool) -> BaseTool:
    """Обгорнути інструмент трьома перевірками, які агент не може обійти.

    1. allowlist і валідація аргументів (check_tool_call) — до виклику;
    2. input guardrail на недовіреному результаті (fetch_resume) — після;
    3. запис події в траєкторію з іменем агента.

    Обгортка живе на межі інструмента, а не всередині вузла. Тому вона
    працює однаково і в ReAct-циклі, і в Plan-and-Execute executor'і: модель
    не бачить сирого тексту резюме в жодному з них.
    """

    async def run(**kwargs) -> str:
        started = time.perf_counter()
        try:
            args = check_tool_call(agent, tool.name, kwargs)
        except ToolDenied as exc:
            _trace(
                log_step(
                    agent, tool.name, f"виклик {tool.name}", str(exc),
                    kind="guard", duration_ms=(time.perf_counter() - started) * 1000,
                    blocked=True,
                )
            )
            return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

        raw = await tool.ainvoke(args)
        text = _tool_result_text(raw)

        if tool.name in UNTRUSTED_TOOLS:
            payload = json.loads(text)
            if payload.get("status") == "ok":
                verdict = detect_injection(payload["data"]["resume_text"])
                payload["data"]["resume_text"] = verdict.safe_text
                payload["injection_detected"] = verdict.detected
                payload["injection_patterns"] = verdict.patterns
                text = json.dumps(payload, ensure_ascii=False)
                if verdict.detected:
                    _trace(
                        log_step(
                            agent, tool.name, "виявлено спробу маніпуляції в резюме",
                            ", ".join(verdict.patterns), kind="guard",
                        )
                    )

        # output у log_step обрізається до 300 символів заради читабельності
        # трейсу, тому структурований результат кладемо окремим полем: з нього
        # потім береться авторитетний бал. Недовірені інструменти сюди не
        # потрапляють — текст резюме не має дублюватись у трейсі.
        extra = {}
        if tool.name not in UNTRUSTED_TOOLS:
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                payload = None
            if isinstance(payload, dict) and payload.get("status") == "ok":
                extra["data"] = payload["data"]

        _trace(
            log_step(
                agent, tool.name, f"{tool.name}({json.dumps(args, ensure_ascii=False)[:120]})",
                text, kind="tool", tools=[tool.name],
                duration_ms=(time.perf_counter() - started) * 1000, **extra,
            )
        )
        return text

    return StructuredTool.from_function(
        coroutine=run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def tools_for(agent: str, tools: list[BaseTool]) -> list[BaseTool]:
    """Інструменти агента: спершу фільтр за allowlist, потім обгортка.

    Два рубежі замість одного. Розводка графа не дає агенту інструмент
    взагалі; check_tool_call усередині обгортки ловить випадок, коли
    інструмент усе ж дійшов до виклику — наприклад, через помилку в коді.
    """
    from guardrails import AGENT_TOOL_ALLOWLIST

    allowed = AGENT_TOOL_ALLOWLIST.get(agent, set())
    return [guarded_tool(agent, t) for t in tools if t.name in allowed]


async def load_mcp_tools() -> tuple[MultiServerMCPClient, list[BaseTool]]:
    """Підключитися до власного MCP-сервера і забрати його інструменти.

    Локальний RAG-інструмент з ДЗ2 додається до набору тут же: для агентів
    різниці між MCP-інструментом і LangChain-інструментом немає.
    """
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)
    tools = await client.get_tools()
    return client, [*tools, search_hr_kb]


async def read_mcp_resource(client: MultiServerMCPClient, uri: str) -> str:
    """Прочитати MCP Resource. Використовує researcher як довідник політики."""
    blobs = await client.get_resources("hr", uris=[uri])
    return "\n".join(getattr(b, "data", "") or getattr(b, "text", "") for b in blobs)


# ── Вузол supervisor ────────────────────────────────────────────────────────


def _supervisor_node(llm):
    """Маршрутизатор. Класифікує запит і записує current_agent.

    Structured output, а не вільний текст: інакше назву агента довелося б
    вигрібати регуляркою з відповіді моделі, і кожне перефразування ламало б
    маршрутизацію. Rate-limit перевіряється тут, бо це вхід у систему.
    """
    router = llm.with_structured_output(RouteDecision)
    system = load_prompt("supervisor")

    async def node(state: MASState) -> dict:
        started = time.perf_counter()
        user_msg = state["messages"][-1].content if state.get("messages") else ""
        session_id = state.get("session_id", "default")

        allowed, reason = rate_limiter.check(session_id)
        if not allowed:
            return {
                "current_agent": "general",
                "completed": True,
                "report": f"Запит відхилено: {reason}",
                "trajectory": [
                    log_step("supervisor", "rate_limit", user_msg, reason, kind="guard", blocked=True)
                ],
            }

        ok, message = check_input_length(user_msg)
        if not ok:
            return {
                "current_agent": "general",
                "completed": True,
                "report": f"Запит відхилено: {message}",
                "trajectory": [
                    log_step("supervisor", "input_guard", str(user_msg)[:100], message,
                             kind="guard", blocked=True)
                ],
            }

        decision = await router.ainvoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
        )
        update = {
            "current_agent": decision.next_agent,
            "step_count": state.get("step_count", 0) + 1,
            "trajectory": [
                log_step(
                    "supervisor", "route", user_msg,
                    f"→ {decision.next_agent}: {decision.reason}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            ],
        }
        # ID з запиту мають пріоритет над тими, що вже є у стані: користувач
        # міг назвати іншого кандидата в наступному повідомленні тієї ж сесії.
        if decision.candidate_id:
            update["candidate_id"] = decision.candidate_id
        if decision.job_id:
            update["job_id"] = decision.job_id
        return update

    return node


# ── Агент screening: Plan-and-Execute підграф (з ДЗ2) ───────────────────────


class PlanExecuteState(TypedDict, total=False):
    """Стан вкладеного Plan-and-Execute графа."""

    question: str
    plan: list[dict]
    past_steps: Annotated[list, operator.add]
    final_answer: str
    replan_count: int
    events: Annotated[list, operator.add]


def _build_plan_execute(llm, tools: list[BaseTool]):
    """Зібрати підграф planner → executor → replanner.

    Навіщо переплановування: план складають до того, як агент побачив дані.
    Якщо резюме не знайдено або навичок витягти не вдалося, початковий план
    втрачає сенс, і replanner має право його переписати.
    """
    planner = llm.with_structured_output(Plan)
    replanner = llm.with_structured_output(ReplanDecision)
    planner_prompt = load_prompt("screening_planner")
    replanner_prompt = load_prompt("screening_replanner")
    budget_note = load_prompt("replan_budget_exhausted")

    # Крок плану виконує вкладений ReAct-агент з нижчим лімітом кроків:
    # людини поруч немає, тому бюджет має бути кінцевим.
    executor_agent = create_react_agent(
        llm, tools_for("screening", tools), prompt=load_prompt("screening_executor")
    )

    async def planner_node(state: PlanExecuteState) -> dict:
        started = time.perf_counter()
        plan: Plan = await planner.ainvoke(planner_prompt.format(question=state["question"]))
        steps = [step.model_dump() for step in plan.steps]
        return {
            "plan": steps,
            "events": [
                log_step("screening", "planner", state["question"],
                         f"складено кроків: {len(steps)}",
                         duration_ms=(time.perf_counter() - started) * 1000)
            ],
        }

    async def executor_node(state: PlanExecuteState) -> dict:
        step = state["plan"][0]
        started = time.perf_counter()
        # Результати попередніх кроків обов'язкові в контексті: без них
        # виконавець кроку «порахувати бал» не знає навичок, які витягнув
        # попередній крок, і або вигадає їх, або викличе інструмент порожнім.
        done = "\n".join(
            f"- {s['step']}\n  результат: {s['result']}" for s in state.get("past_steps", [])
        ) or "кроків ще не виконано"
        task = (
            f"Загальний запит: {state['question']}\n\n"
            f"Уже виконано:\n{done}\n\n"
            f"Виконай саме цей крок: {step['description']}\n"
            f"Очікуваний результат: {step['expected_outcome']}"
        )
        sink: list = []
        token = _trace_sink.set(sink)
        try:
            result = await executor_agent.ainvoke(
                {"messages": [HumanMessage(content=task)]},
                config={"recursion_limit": EXECUTOR_MAX_STEPS * 2},
            )
        finally:
            _trace_sink.reset(token)

        # kind="span", а не "llm": події самого прогону вже в sink, інакше
        # кроки й час подвоювалися б.
        span = log_step(
            "screening", "executor", step["description"],
            result["messages"][-1].content, kind="span",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return {
            "plan": state["plan"][1:],
            "past_steps": [
                {"step": step["description"], "result": result["messages"][-1].content}
            ],
            "events": sink + [span],
        }

    def _summarize_steps(past_steps: list[dict]) -> str:
        """Підсумок із виконаних кроків, якщо replanner висновку не дав.

        Текст прямо каже, що підсумок автоматичний, — щоб його не прийняли
        за вердикт скринінгу."""
        if not past_steps:
            return "Автоматичний підсумок: жодного кроку виконати не встигли."
        lines = ["Автоматичний підсумок (replanner не дав власного висновку):"]
        lines += [f"- {s['step']}: {s['result']}" for s in past_steps]
        return "\n".join(lines)

    async def replanner_node(state: PlanExecuteState) -> dict:
        started = time.perf_counter()
        done = "\n".join(f"- {s['step']}: {s['result']}" for s in state["past_steps"]) or "нічого"
        remaining = "\n".join(f"- {s['description']}" for s in state["plan"]) or "порожньо"
        exhausted = state.get("replan_count", 0) >= MAX_REPLANS

        prompt = replanner_prompt.format(
            question=state["question"], done=done, remaining=remaining
        )
        if exhausted:
            prompt += budget_note

        decision: ReplanDecision = await replanner.ainvoke(prompt)

        # Структурний захист, а не прохання в промпті: модель схильна
        # «дійти висновку» після другого кроку, не викликавши score_candidate.
        # Поки бюджет є, такий finish не приймається — замість нього в план
        # ставиться детермінований крок підрахунку.
        has_score = _score_from_events(state.get("events", [])) is not None
        forced_replan = decision.action == "finish" and not has_score and not exhausted
        if forced_replan:
            decision.action = "replan"
            decision.reasoning = (
                "Примусове переплановування: вердикт неможливий без результату "
                "інструмента score_candidate."
            )
            decision.updated_steps = [
                PlanStep(
                    description=(
                        "Виклич інструмент score_candidate, передавши навички та роки "
                        "досвіду, витягнуті з резюме на попередньому кроці, і job_id вакансії."
                    ),
                    expected_outcome="Інструмент повернув score та decision.",
                )
            ]

        update: dict = {
            "events": [
                log_step("screening", "replanner", decision.action, decision.reasoning,
                         duration_ms=(time.perf_counter() - started) * 1000,
                         replan_budget_exhausted=exhausted, forced=forced_replan)
            ]
        }
        if decision.action == "finish":
            update["final_answer"] = decision.final_answer
        elif not exhausted:
            update["plan"] = [s.model_dump() for s in decision.updated_steps]
            update["replan_count"] = state.get("replan_count", 0) + 1

        # Граф не повинен завершуватись мовчки: план вичерпано, висновку немає.
        if not update.get("final_answer") and not update.get("plan", state.get("plan")):
            update["final_answer"] = _summarize_steps(state["past_steps"])
        return update

    def route_after_replan(state: PlanExecuteState) -> str:
        return END if state.get("final_answer") else "executor"

    builder = StateGraph(PlanExecuteState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replanner", replanner_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "replanner")
    builder.add_conditional_edges(
        "replanner", route_after_replan, {"executor": "executor", END: END}
    )
    return builder.compile()


def _screening_node(llm, tools: list[BaseTool]):
    """Агент оцінки кандидата. Всередині — Plan-and-Execute підграф з ДЗ2."""
    subgraph = _build_plan_execute(llm, tools)
    verdict_llm = llm.with_structured_output(ScreeningVerdict)

    async def node(state: MASState) -> dict:
        question = state["messages"][-1].content
        candidate_id = state.get("candidate_id") or "невідомий"
        job_id = state.get("job_id") or "JOB-BACKEND"
        task = (
            f"{question}\nКандидат: {candidate_id}. Вакансія: {job_id}."
        )

        try:
            result = await asyncio.wait_for(
                subgraph.ainvoke(
                    {"question": task, "plan": [], "past_steps": [], "final_answer": "",
                     "replan_count": 0, "events": []},
                    config={"recursion_limit": MAX_STEPS * 3},
                ),
                timeout=AGENT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return {
                "completed": True,
                "report": f"Скринінг перервано: перевищено ліміт {AGENT_TIMEOUT_S} с.",
                "trajectory": [
                    log_step("screening", "timeout", task, "перевищено ліміт часу", kind="guard")
                ],
            }

        events = result.get("events", [])
        summary = result.get("final_answer", "")
        # Авторитетний бал беремо з ToolMessage інструмента, а не з тексту
        # моделі: текст — переказ, який ін'єкція могла спотворити.
        score_data = _score_from_events(events)

        verdict = await verdict_llm.ainvoke(
            [
                {"role": "system", "content": load_prompt("screening_executor")},
                {"role": "user", "content":
                    f"Сформуй вердикт для {candidate_id} на {job_id}.\n"
                    f"Підсумок скринінгу:\n{summary}\n"
                    f"Результат інструмента score_candidate: "
                    f"{json.dumps(score_data, ensure_ascii=False)}"},
            ]
        )
        verdict.candidate_id = candidate_id
        verdict.job_id = job_id
        if score_data:
            verdict.score = score_data["score"]
            verdict.decision = score_data["decision"]
        else:
            # Немає авторитетного балу — це має бути ВИДНО, а не тихо
            # замінене числом, яке вигадала модель.
            verdict.score = 0
            verdict.decision = "reject"
            verdict.rationale = (
                "Вердикт неповний: інструмент score_candidate не повернув балу. "
                + verdict.rationale
            )
        verdict.injection_detected = any(
            e.get("kind") == "guard" and "маніпуляції" in e.get("action", "") for e in events
        )

        return {
            "verdict": verdict.model_dump(),
            "completed": True,
            "step_count": state.get("step_count", 0) + len(events),
            "messages": [HumanMessage(content=f"screening: {summary}")],
            "trajectory": events,
        }

    return node


def _score_from_events(events: list[dict]) -> dict | None:
    """Дістати результат score_candidate з подій траєкторії.

    Береться ОСТАННІЙ успішний виклик: агент міг схибити з аргументами і
    повторити виклик уже правильно.
    """
    found = None
    for event in events:
        if event.get("kind") != "tool" or "score_candidate" not in event.get("tools", []):
            continue
        data = event.get("data")
        if isinstance(data, dict) and "score" in data:
            found = data
    return found


# ── Агент researcher: Agentic RAG на ChromaDB (з ДЗ2) ───────────────────────


def _researcher_node(llm, tools: list[BaseTool], client=None):
    """Агент політик. ReAct-цикл поверх search_hr_kb, плюс MCP Resource.

    Ресурс hrpolicy://screening підвантажується в контекст безумовно: це
    короткий довідник порогів і строків, який потрібен майже в кожній
    відповіді. База знань лишається інструментом — чи йти в неї, вирішує
    модель, читаючи докстрінг search_hr_kb.
    """
    agent = create_react_agent(llm, tools_for("researcher", tools), prompt=load_prompt("researcher"))

    async def node(state: MASState) -> dict:
        started = time.perf_counter()
        question = state["messages"][-1].content

        policy = ""
        if client is not None:
            try:
                policy = await read_mcp_resource(client, "hrpolicy://screening")
            except Exception as exc:  # ресурс недоступний — працюємо без нього
                _trace(log_step("researcher", "mcp_resource", "hrpolicy://screening",
                                f"недоступний: {exc}", kind="guard"))

        prefix = f"Довідник політики (MCP resource hrpolicy://screening):\n{policy}\n\n" if policy else ""
        sink: list = []
        token = _trace_sink.set(sink)
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=f"{prefix}Питання: {question}")]},
                    config={"recursion_limit": MAX_STEPS * 2},
                ),
                timeout=AGENT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return {
                "completed": True,
                "report": f"Пошук у політиках перервано: ліміт {AGENT_TIMEOUT_S} с.",
                "trajectory": sink + [log_step("researcher", "timeout", question,
                                               "перевищено ліміт часу", kind="guard")],
            }
        finally:
            _trace_sink.reset(token)

        answer = result["messages"][-1].content
        return {
            "completed": True,
            "report": answer,
            "step_count": state.get("step_count", 0) + 1,
            "messages": [HumanMessage(content=f"researcher: {answer}")],
            "trajectory": sink + [
                log_step("researcher", "answer", question, answer,
                         duration_ms=(time.perf_counter() - started) * 1000,
                         used_mcp_resource=bool(policy))
            ],
        }

    return node


# ── Агент communicator: ReAct + MCP Prompt, далі HITL ───────────────────────


def _communicator_node(llm, client=None):
    """Готує чернетку листа. Інструмент надсилання НЕ викликає.

    Шаблон листа береться з MCP Prompt candidate_reply — тобто формулювання
    політики живе на сервері поруч із доменом, а не зашите в код агента.
    """
    email_llm = llm.with_structured_output(EmailDraft)

    async def node(state: MASState) -> dict:
        started = time.perf_counter()
        verdict = state.get("verdict") or {}
        candidate_id = verdict.get("candidate_id") or state.get("candidate_id") or "CAND-001"
        decision = verdict.get("decision") or "maybe"
        gaps = ", ".join(verdict.get("gaps") or [])

        template = ""
        if client is not None:
            try:
                prompts = await client.get_prompt(
                    "hr", "candidate_reply",
                    arguments={"candidate_name": candidate_id, "decision": decision, "gaps": gaps},
                )
                template = "\n".join(getattr(m, "content", "") if isinstance(getattr(m, "content", ""), str)
                                     else getattr(m.content, "text", "") for m in prompts)
            except Exception as exc:  # шаблон недоступний — пишемо без нього
                _trace(log_step("communicator", "mcp_prompt", "candidate_reply",
                                f"недоступний: {exc}", kind="guard"))

        context = {
            "candidate_id": candidate_id,
            "job_id": verdict.get("job_id") or state.get("job_id") or "JOB-BACKEND",
            "decision": decision,
            "gaps": verdict.get("gaps") or [],
            "rationale": verdict.get("rationale", ""),
        }
        draft = await email_llm.ainvoke(
            [
                {"role": "system", "content": load_prompt("communicator")},
                {"role": "user", "content":
                    (f"Шаблон з MCP:\n{template}\n\n" if template else "")
                    + f"Дані вердикту:\n{json.dumps(context, ensure_ascii=False, indent=2)}"},
            ]
        )
        draft.candidate_id = candidate_id
        # Output guardrail: навіть якщо промпт не вберіг лист від PII,
        # останній рубіж — маскування перед тим, як лист побачить людина.
        draft.body, pii_found = redact_pii(draft.body)

        events = [
            log_step("communicator", "draft", f"лист для {candidate_id}", draft.subject,
                     duration_ms=(time.perf_counter() - started) * 1000,
                     used_mcp_prompt=bool(template))
        ]
        if pii_found:
            events.append(
                log_step("communicator", "output_guard", "маскування PII у листі",
                         ", ".join(pii_found), kind="guard")
            )

        return {
            "email_draft": draft.model_dump(),
            "pending_approval": True,
            "step_count": state.get("step_count", 0) + 1,
            "trajectory": events,
        }

    return node


# ── Агент general ───────────────────────────────────────────────────────────


def _general_node(llm):
    """Fallback: вітання й нерозпізнані запити. Інструментів не має."""

    async def node(state: MASState) -> dict:
        started = time.perf_counter()
        question = state["messages"][-1].content
        answer = await llm.ainvoke(
            [
                {"role": "system", "content": load_prompt("general")},
                {"role": "user", "content": question},
            ]
        )
        text = answer.content
        return {
            "completed": True,
            "report": text,
            "step_count": state.get("step_count", 0) + 1,
            "trajectory": [
                log_step("general", "answer", question, text,
                         duration_ms=(time.perf_counter() - started) * 1000)
            ],
        }

    return node


# ── HITL і надсилання ───────────────────────────────────────────────────────


def _human_approval_node(state: MASState) -> dict:
    """HITL: граф зупиняється і чекає рішення людини щодо листа.

    Ризиковий send_candidate_email викликає граф ПІСЛЯ цієї зупинки, а не
    агент за власним рішенням — точка зупинки детермінована незалежно від
    того, чи «захоче» модель викликати інструмент. Людина бачить адресата,
    рішення, тему й повний текст. Формат відповіді:
    {"action": "approve" | "reject" | "edit", "body": "..."}.
    """
    draft = state.get("email_draft")
    if draft is None:
        return {"hitl_action": "reject", "pending_approval": False}

    response = interrupt(
        {
            "type": "email_approval",
            "candidate_id": draft["candidate_id"],
            "decision": draft["decision"],
            "subject": draft["subject"],
            "body": draft["body"],
        }
    )

    action = response.get("action", "reject")
    if action == "edit":
        draft = {**draft, "body": response["body"]}

    return {
        "email_draft": draft,
        "hitl_action": action,
        "pending_approval": False,
        "trajectory": [
            log_step("graph", "human_approval", f"рішення людини: {action}",
                     draft["subject"], kind="interrupt")
        ],
    }


def _send_email_node(tools: list[BaseTool]):
    """Викликає ризиковий MCP-інструмент — лише після схвалення людиною.

    Виклик іде під іменем агента "graph": в allowlist send_candidate_email
    належить саме графу, а не комусь із агентів.
    """
    send_tool = next((t for t in tools if t.name == "send_candidate_email"), None)
    if send_tool is None:
        # Падаємо одразу при збірці, а не посеред прогону, коли лист уже
        # схвалила людина. Голий StopIteration тут нічого не пояснював би.
        raise RuntimeError(
            "у наборі інструментів немає send_candidate_email — MCP-сервер не "
            "піднявся або не віддав інструмент; перевірте mcp_server.py"
        )
    guarded = guarded_tool("graph", send_tool)

    async def node(state: MASState) -> dict:
        draft = state.get("email_draft")
        if draft is None:
            return {"trajectory": [
                log_step("graph", "send_email", "пропущено", "у стані немає чернетки",
                         kind="guard")
            ]}

        sink: list = []
        token = _trace_sink.set(sink)
        try:
            result = await guarded.ainvoke(
                {
                    "candidate_id": draft["candidate_id"],
                    "decision": draft["decision"],
                    "subject": draft["subject"],
                    "body": draft["body"],
                }
            )
        finally:
            _trace_sink.reset(token)

        return {
            "messages": [HumanMessage(content=f"send_candidate_email: {result}")],
            "trajectory": sink,
        }

    return node


# ── Звіт ────────────────────────────────────────────────────────────────────


def _build_report(state: MASState) -> str:
    """Фінальний звіт користувачу. Проходить output guardrail — це останній
    рубіж перед тим, як звіт побачить людина."""
    if state.get("report"):
        lines = [state["report"]]
    else:
        lines = []

    verdict = state.get("verdict")
    if verdict:
        lines = [
            f"Кандидат: {verdict.get('candidate_id')}",
            f"Вакансія: {verdict.get('job_id')}",
            f"Бал: {verdict.get('score')}  →  {verdict.get('decision')}",
            f"Обґрунтування: {verdict.get('rationale')}",
            f"Чого бракує: {', '.join(verdict.get('gaps') or []) or 'нічого'}",
            f"Спроба маніпуляції в резюме: "
            f"{'виявлена' if verdict.get('injection_detected') else 'не виявлена'}",
        ] + lines

    draft = state.get("email_draft")
    if draft:
        action = state.get("hitl_action", "")
        status = {
            "approve": "надіслано після підтвердження людиною",
            "edit": "надіслано у відредагованій людиною версії",
            "reject": "НЕ надіслано: людина відхилила",
        }.get(action, "чернетка, рішення людини не отримано")
        lines += ["", f"Лист ({draft['subject']}) — {status}:", draft["body"]]

    report, found_pii = redact_pii("\n".join(lines) or "Порожній результат.")
    if found_pii:
        report += f"\n\n[Замасковано PII: {', '.join(found_pii)}]"
    return report


def _report_node(state: MASState) -> dict:
    return {"report": _build_report(state), "completed": True}


# ── Збірка графа ────────────────────────────────────────────────────────────


def _route(state: MASState) -> str:
    """Conditional edge після supervisor'а."""
    if state.get("completed"):
        return "report"
    return state.get("current_agent", "general")


def _route_after_approval(state: MASState) -> str:
    """approve та edit ведуть до надсилання, reject — одразу до звіту."""
    return "send_email" if state.get("hitl_action") in ("approve", "edit") else "report"


def build_graph(tools: list[BaseTool], checkpointer=None, llm=None, client=None):
    """Зібрати MAS: supervisor маршрутизує, агент відпрацьовує, граф звітує.

    Шлях communicator'а довший за інші: він веде не в звіт, а на зупинку
    human_approval, і лише схвалення людини відкриває send_email.

    `llm` можна передати готовим (тести зі стаб-моделлю, без мережі й без
    ключів) — інакше створюється справжня модель через make_llm().
    """
    if llm is None:
        llm = make_llm()

    builder = StateGraph(MASState)
    builder.add_node("supervisor", _supervisor_node(llm))
    builder.add_node("screening", _screening_node(llm, tools))
    builder.add_node("researcher", _researcher_node(llm, tools, client))
    builder.add_node("communicator", _communicator_node(llm, client))
    builder.add_node("general", _general_node(llm))
    builder.add_node("human_approval", _human_approval_node)
    builder.add_node("send_email", _send_email_node(tools))
    builder.add_node("report", _report_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route,
        {
            "screening": "screening",
            "researcher": "researcher",
            "communicator": "communicator",
            "general": "general",
            "report": "report",
        },
    )
    builder.add_edge("screening", "report")
    builder.add_edge("researcher", "report")
    builder.add_edge("general", "report")
    builder.add_edge("communicator", "human_approval")
    builder.add_conditional_edges(
        "human_approval", _route_after_approval,
        {"send_email": "send_email", "report": "report"},
    )
    builder.add_edge("send_email", "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=checkpointer)


def initial_state(query: str, session_id: str = "default", **overrides) -> dict:
    """Початковий стан прогону. Усі поля задані явно, щоб вузли не гадали."""
    state = {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "current_agent": "",
        "candidate_id": "",
        "job_id": "",
        "plan": [],
        "current_step": 0,
        "results": [],
        "step_count": 0,
        "trajectory": [],
        "verdict": None,
        "email_draft": None,
        "hitl_action": "",
        "completed": False,
        "pending_approval": False,
        "report": "",
    }
    state.update(overrides)
    return state
