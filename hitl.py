"""Human-in-the-loop: підтвердження людиною перед незворотною дією.

Сам approval gate живе у графі (mas_langgraph._human_approval_node) — тут
демонстрація трьох сценаріїв і хелпери, якими користуються main.py та тести.

Чому зупинка стоїть у графі, а не в агенті: точка зупинки має бути
детермінованою. Якби рішення «спитати людину» ухвалював агент, ін'єкція в
тексті резюме могла б переконати його не питати. Агент `communicator` навіть
не має send_candidate_email у своєму allowlist — інструмент належить графу.

Запуск: uv run python hitl.py
"""

import asyncio
import json
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from config import CHECKPOINT_DB
from mas_langgraph import build_graph, initial_state, load_mcp_tools

# Ризикові інструменти: незворотні дії, які не можна виконувати без людини.
# send_candidate_email — комунікація з живою людиною; решта зарезервовані
# під майбутні дії того ж класу.
RISKY_TOOLS = {"send_candidate_email", "delete_candidate", "send_mass_email"}


@asynccontextmanager
async def screening_app(db_path=None, llm=None):
    """Граф із підключеним checkpointer'ом і піднятим MCP-сервером.

    Checkpointer обов'язковий для HITL: interrupt() зберігає стан і віддає
    керування назовні, а Command(resume=...) продовжує рівно з того місця.
    Без checkpointer'а продовжувати не було б з чого.
    """
    client, tools = await load_mcp_tools()
    async with AsyncSqliteSaver.from_conn_string(str(db_path or CHECKPOINT_DB)) as saver:
        yield build_graph(tools, checkpointer=saver, llm=llm, client=client)


def approval_payload(result: dict) -> dict | None:
    """Дістати з результату прогону те, що система показує людині.

    Повертає None, якщо граф не зупинявся — наприклад, запит пішов не до
    communicator'а і незворотної дії в ньому немає.
    """
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


async def run_until_approval(graph, query: str, thread_id: str, **overrides) -> dict:
    """Прогнати граф до зупинки на підтвердженні (або до кінця, якщо її немає)."""
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(initial_state(query, session_id=thread_id, **overrides), config=config)


async def resume(graph, thread_id: str, decision: dict) -> dict:
    """Продовжити зупинений прогін рішенням людини.

    decision: {"action": "approve"} | {"action": "reject"} |
              {"action": "edit", "body": "виправлений текст"}
    """
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(Command(resume=decision), config=config)


async def demo() -> list[dict]:
    """Три сценарії HITL на одному ризиковому інструменті.

    Кожен сценарій — окремий thread_id, тому прогони не заважають один одному
    і кожен можна відновити незалежно.
    """
    from config import OUTBOX_PATH

    verdict = {
        "candidate_id": "CAND-002",
        "job_id": "JOB-BACKEND",
        "score": 28,
        "decision": "reject",
        "rationale": "Бракує всіх трьох обов'язкових навичок і стажу.",
        "gaps": ["Python", "PostgreSQL", "Docker"],
        "injection_detected": False,
    }
    query = "Напиши лист кандидату CAND-002 за результатом скринінгу"

    scenarios = [
        ("approve", {"action": "approve"}, "людина підтверджує чернетку без змін"),
        ("reject", {"action": "reject", "reason": "надішлемо пізніше"}, "людина відхиляє"),
        (
            "edit",
            {"action": "edit", "body": "Доброго дня! Дякуємо за інтерес до вакансії. "
                                       "Цього разу ми обрали іншого кандидата, бо бракує "
                                       "досвіду з Python, PostgreSQL і Docker. Будемо раді "
                                       "розглянути вашу заявку за пів року."},
            "людина переписує текст листа",
        ),
    ]

    results = []
    letters_before = _outbox_size(OUTBOX_PATH)

    async with screening_app() as graph:
        for action, response, note in scenarios:
            thread_id = f"hitl-{action}"
            paused = await run_until_approval(graph, query, thread_id, verdict=verdict)
            payload = approval_payload(paused)

            record = {
                "scenario": action,
                "note": note,
                "paused": payload is not None,
                "shown_to_human": payload,
            }

            if payload is None:
                record["error"] = "граф не зупинився на підтвердженні"
                results.append(record)
                continue

            finished = await resume(graph, thread_id, response)
            letters_after = _outbox_size(OUTBOX_PATH)
            record["letters_added"] = letters_after - letters_before
            letters_before = letters_after
            record["final_action"] = finished.get("hitl_action")
            record["report_tail"] = (finished.get("report") or "")[-300:]
            results.append(record)

    return results


def _outbox_size(path) -> int:
    """Скільки листів уже лежить у вихідних. Порожній файл — нуль."""
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    outcome = asyncio.run(demo())
    for record in outcome:
        print("=" * 72)
        print(f"СЦЕНАРІЙ: {record['scenario']} — {record['note']}")
        print(f"  граф зупинився: {'так' if record['paused'] else 'НІ'}")
        shown = record.get("shown_to_human") or {}
        if shown:
            print(f"  людина бачила: {shown.get('candidate_id')} / {shown.get('decision')} / "
                  f"«{shown.get('subject')}»")
        print(f"  рішення людини: {record.get('final_action')}")
        print(f"  листів надіслано: {record.get('letters_added')}")
    print("=" * 72)
    approved = sum(1 for r in outcome if r.get("letters_added"))
    print(f"Надіслано листів усього: {approved} з {len(outcome)} сценаріїв "
          f"(очікується 2: approve та edit).")
