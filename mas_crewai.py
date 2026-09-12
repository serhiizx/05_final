"""MAS HR-скринінгу в CrewAI: той самий кейс, що й у mas_langgraph.py.

Домен — той самий MCP-сервер, guardrails — ті самі функції, база знань — та
сама ChromaDB-колекція. Відрізняється лише оркестрація, і саме ця різниця
вимірюється в порівняльній таблиці README.

Головна архітектурна різниця. У LangGraph маршрут задано явно: supervisor
повертає structured output, а conditional edge веде до одного вузла. У CrewAI
прямого аналога conditional edge немає, тому роль супервізора виконує
Process.hierarchical із manager_llm: менеджер сам вирішує, кому делегувати
задачу. Маршрут стає рішенням моделі всередині фреймворку, а не ребром, яке
можна прочитати в коді.

Запуск: uv run python mas_crewai.py "Оціни кандидата CAND-001 на JOB-BACKEND"
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool as crew_tool
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from config import ROOT, load_prompt
from guardrails import AGENT_TOOL_ALLOWLIST, redact_pii
from observability import langfuse_enabled
from tools_legacy import search_hr_kb

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(ROOT / "mcp_server.py")],
    env={**os.environ},
)


@crew_tool("search_hr_kb")
def crew_search_hr_kb(query: str) -> str:
    """Знайти правило у внутрішній базі HR-політик компанії.

    База містить політику скринінгу, рівні seniority, строки відповіді (SLA),
    правила відмови, роботу з персональними даними, антидискримінаційні
    вимоги, формулу підрахунку балу та етапи процесу після скринінгу.

    Args:
        query: Питання до бази політик.
    """
    # Та сама ChromaDB-колекція, що й у LangGraph-реалізації: різниця між
    # фреймворками не має тягнути за собою другу копію бази знань.
    return search_hr_kb.invoke({"query": query, "top_k": 3})


def _instrument_crewai() -> None:
    """CrewAI віддає трейси в Langfuse через OpenTelemetry, а не через
    CallbackHandler — тому міст інший, а бекенд той самий."""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client
        from openinference.instrumentation.crewai import CrewAIInstrumentor

        # get_client() піднімає TracerProvider Langfuse. Без цього виклику
        # OpenInference створить спани, яким нікуди їхати.
        get_client()
        CrewAIInstrumentor().instrument(skip_dep_check=True)
    except Exception:  # інструментація не критична для роботи команди
        pass


def _tools_for(agent: str, tools: list) -> list:
    """Той самий allowlist, що й у LangGraph (mas_langgraph.tools_for).

    Різниця в тому, що тут він працює лише як фільтр списку. Рантаймової
    перевірки check_tool_call на межі виклику немає: CrewAI-агент сам
    вирішує, коли викликати інструмент, і перехопити цей момент у коді
    оркестрації ніде.
    """
    allowed = AGENT_TOOL_ALLOWLIST[agent]
    named = {t.name: t for t in tools}
    return [named[name] for name in allowed if name in named]


def make_llm(temperature: float = 0.0) -> LLM:
    """Та сама модель і провайдер, що й config.make_llm.

    Без явного llm= CrewAI/LiteLLM читає MODEL/OPENAI_MODEL_NAME, а не
    LLM_MODEL, тож агенти мовчки отримали б іншу модель, ніж LangGraph, і
    порівняння перестало б бути порівнянням. provider="openai" передано
    ЯВНО: LiteLLM інакше трактує "/" у назві моделі як префікс провайдера
    (наприклад google/gemma-4-e4b з LM Studio — це власна назва моделі, а не
    вказівка йти в Gemini API).
    """
    return LLM(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        provider="openai",
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=temperature,
    )


def build_crew(tools: list, query: str, auto_approve: bool = True) -> Crew:
    """Скласти команду з трьох агентів під керуванням менеджера.

    Винесено окремою функцією, щоб тести могли перевірити фільтрацію
    інструментів і побудову агентів на стабах, без підняття MCP-підпроцесу.
    """
    llm = make_llm()

    screener = Agent(
        role="Screener",
        goal="Оцінити кандидата на вакансію: витягти факти з резюме, звірити з "
             "вимогами і порахувати бал інструментом",
        backstory=load_prompt("screening_executor"),
        tools=_tools_for("screening", tools),
        llm=llm,
        verbose=True,
    )
    researcher = Agent(
        role="Researcher",
        goal="Відповідати на питання про HR-політики, спираючись лише на базу знань",
        backstory=load_prompt("researcher"),
        tools=[crew_search_hr_kb, *_tools_for("researcher", tools)],
        llm=llm,
        verbose=True,
    )
    communicator = Agent(
        role="Communicator",
        goal="Підготувати лист кандидату за результатом скринінгу",
        backstory=load_prompt("communicator"),
        tools=_tools_for("communicator", tools),
        llm=llm,
        verbose=True,
    )

    task = Task(
        description=(
            f"Запит користувача: {query}\n\n"
            "Визнач, хто з команди має його виконати: Screener для оцінки "
            "конкретного кандидата, Researcher для питань про політики й строки, "
            "Communicator для підготовки листа. Передай задачу відповідному "
            "виконавцю і поверни його результат.\n"
            "Бал бери рівно той, що повернув інструмент score_candidate — не "
            "переписуй його. Лист без підтвердження людини не надсилай."
        ),
        expected_output="Результат роботи відповідного агента: вердикт із балом, "
                        "відповідь із посиланням на джерела або чернетка листа.",
        # HITL у CrewAI: блокуючий input() у консолі, без збереження стану.
        # Це не те саме, що interrupt() у LangGraph, і різниця в README §7.
        human_input=not auto_approve,
    )

    return Crew(
        agents=[screener, researcher, communicator],
        tasks=[task],
        # hierarchical — найближчий аналог supervisor-патерну. Менеджер
        # створюється фреймворком, його промпт нам не належить.
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )


# Ідентифікатори в запиті. Потрібні пост-перевірці нижче: вона має знати,
# кого саме звіряти, а розібраного стану, як у LangGraph, тут немає.
_CANDIDATE_RE = re.compile(r"CAND-\d{3}")
_JOB_RE = re.compile(r"JOB-[A-Z]+")


def authoritative_verdict(query: str) -> dict | None:
    """Порахувати бал детерміновано, обійшовши переказ між агентами.

    Навіщо це знадобилося саме тут. У LangGraph авторитетний бал береться зі
    структурованої події виклику інструмента (_score_from_events), і текст
    агента на нього не впливає. У CrewAI результат інструмента повертається
    менеджеру РЯДКОМ і далі переказується через delegate_work_to_coworker.
    У прогоні на CAND-001 інструмент повернув 93/strong_match, а у фінальному
    звіті після трьох переказів рішення стало "in_progress". Тобто губиться не
    форматування, а сам вердикт.

    Тому в CrewAI-реалізації потрібен окремий рубіж, якого в LangGraph немає:
    після прогону бал перераховується тим самим інструментом від фактів
    кандидата. Це не прикраса результату, а визнання, що переказ тексту між
    агентами не є надійним каналом для авторитетних даних (OWASP ASI07).
    """
    candidate = _CANDIDATE_RE.search(query)
    job = _JOB_RE.search(query)
    if not candidate:
        return None

    from mcp_server import fetch_job_requirements, fetch_resume, score_candidate

    resume = fetch_resume(candidate.group())
    if resume["status"] != "ok":
        return None

    job_id = job.group() if job else "JOB-BACKEND"
    if fetch_job_requirements(job_id)["status"] != "ok":
        return None

    # Навички шукаємо ВИКЛЮЧНО в тексті резюме, а не у звіті агентів.
    # Звіт — це переказ, і саме він тут під підозрою: якби ми брали навички
    # звідти, вигадана моделлю навичка підняла б бал, і пост-перевірка
    # перестала б бути незалежною від того, що вона перевіряє.
    known = fetch_job_requirements(job_id)["data"]
    haystack = resume["data"]["resume_text"].lower()
    skills = [s for s in known["must_have"] + known["nice_to_have"] if s.lower() in haystack]
    # Порожній збіг — це не «не вдалося порахувати», а найгірший можливий
    # результат, і саме там пост-перевірка найпотрібніша: модель могла назвати
    # strong_match кандидату без жодної потрібної навички. Плейсхолдер не
    # збігається ні з чим, тому обидві частини за навички дорівнюють нулю, і
    # лишається тільки частка за стаж — це коректний бал, а не заглушка.
    # (Порожній список не приймає схема ScoreArgs, і це правильно: агент не
    # має права викликати інструмент порожнім.)
    skills = skills or ["(жодної з вимог)"]

    years = 0.0
    # «6 років», «1 рік», «4 роки» — усі три форми, інакше junior із «1 рік
    # досвіду» отримав би нуль за стаж і занижений бал.
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:рок|рік|роки)", resume["data"]["resume_text"])
    if match:
        years = float(match.group(1).replace(",", "."))

    scored = score_candidate(skills=skills, years_experience=years, job_id=job_id)
    return scored["data"] if scored["status"] == "ok" else None


def run_crew(query: str, auto_approve: bool = True) -> str:
    """Повний прогін на CrewAI. Повертає фінальний звіт після маскування PII."""
    _instrument_crewai()

    with MCPServerAdapter(SERVER_PARAMS) as tools:
        crew = build_crew(list(tools), query, auto_approve)
        # crew.kickoff() кидає RuntimeError, якщо його викликати з коду, що вже
        # працює під активним event loop (наприклад, з комірки Jupyter). Щоб
        # лишити run_crew синхронною, виконуємо kickoff в окремому потоці.
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(crew.kickoff).result()

    text = str(result)

    # Рубіж, якого в LangGraph не треба: перевіряємо, чи переказ між агентами
    # не втратив авторитетний вердикт. Див. докстрінг authoritative_verdict.
    verdict = authoritative_verdict(query)
    if verdict:
        text += (
            f"\n\n[Авторитетний вердикт від score_candidate: "
            f"{verdict['score']} → {verdict['decision']}. "
            f"Якщо в тексті вище названо інше рішення, правильним є це: "
            f"бал рахує інструмент, а не переказ між агентами.]"
        )

    # Той самий output guardrail, що й у LangGraph: маскування — останній
    # рубіж незалежно від того, який фреймворк готував текст.
    report, found_pii = redact_pii(text)
    if found_pii:
        report += f"\n\n[Замасковано PII: {', '.join(found_pii)}]"
    return report


if __name__ == "__main__":
    user_query = (
        sys.argv[1] if len(sys.argv) > 1 else "Оціни кандидата CAND-001 на вакансію JOB-BACKEND"
    )
    print(run_crew(user_query))
