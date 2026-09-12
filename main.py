"""Демонстрація MAS: маршрутизація, робота агентів, persistence.

Запуск:
    uv run python main.py                  — три запити різного типу
    uv run python main.py --persistence     — обрив прогону і відновлення
    uv run python main.py "довільний запит" — один власний запит
"""

import argparse
import asyncio
import sys

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import CHECKPOINT_DB, TRAJECTORY_PATH
from hitl import screening_app
from mas_langgraph import build_graph, initial_state, load_mcp_tools
from observability import tracing_config, tracing_status
from trajectory_logger import dump_trajectory, summarize

# Три запити різного типу — по одному на кожного змістовного агента.
DEMO_QUERIES = [
    ("demo-screening", "Оціни кандидата CAND-001 на вакансію JOB-BACKEND"),
    ("demo-researcher", "За скільки робочих днів ми зобов'язані відповісти кандидату "
                        "після скринінгу і що означає вердикт maybe?"),
    ("demo-injection", "Оціни кандидата CAND-003 на вакансію JOB-BACKEND"),
]


async def run_demo() -> list[dict]:
    """Прогнати демо-запити й зібрати траєкторію всього MAS в один файл."""
    all_events: list[dict] = []
    runs: list[dict] = []

    print(f"Трасування: {tracing_status()}\n")

    async with screening_app() as graph:
        for thread_id, query in DEMO_QUERIES:
            config = {"configurable": {"thread_id": thread_id}, **tracing_config()}
            result = await graph.ainvoke(
                initial_state(query, session_id=thread_id), config=config
            )

            events = result.get("trajectory", [])
            all_events += events
            stats = summarize(events)
            runs.append({"thread_id": thread_id, "query": query, **stats})

            print("=" * 78)
            print(f"[{thread_id}] {query}")
            print(f"  агент: {result.get('current_agent')} | кроків: {stats['steps']} | "
                  f"інструментів: {stats['tool_calls']} | час: {stats['seconds']} с")
            print(f"  інструменти: {', '.join(stats['tools_called']) or 'жодного'}")
            print("-" * 78)
            print(result.get("report", "").strip()[:900])
            print()

    payload = dump_trajectory(
        all_events,
        TRAJECTORY_PATH,
        meta={"runs": runs, "queries": len(DEMO_QUERIES)},
    )
    print("=" * 78)
    print(f"Траєкторію збережено: {TRAJECTORY_PATH}")
    print(f"  подій: {len(all_events)} | агентів задіяно: {payload['summary']['agents_used']}")
    return runs


async def run_persistence_demo() -> dict:
    """Показати, що прогін переживає «крах» процесу.

    Сценарій: запускаємо скринінг із зупинкою на підтвердженні, «вмираємо»
    (закриваємо граф і з'єднання з базою), відкриваємо все заново і
    продовжуємо той самий thread_id. Стан береться з agent_state.db.
    """
    thread_id = "persistence-demo"
    query = "Напиши лист кандидату CAND-002 за результатом скринінгу"
    verdict = {
        "candidate_id": "CAND-002", "job_id": "JOB-BACKEND", "score": 28,
        "decision": "reject", "rationale": "Бракує обов'язкових навичок.",
        "gaps": ["Python", "PostgreSQL", "Docker"], "injection_detected": False,
    }
    config = {"configurable": {"thread_id": thread_id}}

    print("КРОК 1. Запускаємо прогін до зупинки на підтвердженні людиною.")
    async with screening_app() as graph:
        paused = await graph.ainvoke(
            initial_state(query, session_id=thread_id, verdict=verdict), config=config
        )
        interrupts = paused.get("__interrupt__") or []
        subject = interrupts[0].value.get("subject") if interrupts else None
        print(f"   граф зупинився: {'так' if interrupts else 'НІ'}")
        print(f"   людина бачить лист: «{subject}»")

    print("\nКРОК 2. «Крах»: граф і з'єднання з базою закриті, об'єкти знищені.")
    del graph
    print(f"   у пам'яті нічого не лишилось, стан живе тільки у {CHECKPOINT_DB.name}")

    print("\nКРОК 3. Новий процес-еквівалент: відкриваємо базу заново.")
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        client, tools = await load_mcp_tools()
        restored_graph = build_graph(tools, checkpointer=saver, client=client)
        snapshot = await restored_graph.aget_state(config)
        print(f"   стан відновлено: наступний вузол = {snapshot.next}")
        print(f"   кандидат у стані: {snapshot.values.get('email_draft', {}).get('candidate_id')}")

        print("\nКРОК 4. Продовжуємо той самий thread_id рішенням людини.")
        from langgraph.types import Command

        finished = await restored_graph.ainvoke(Command(resume={"action": "reject"}), config=config)
        print(f"   прогін завершено, рішення: {finished.get('hitl_action')}")

    return {
        "thread_id": thread_id,
        "paused": bool(interrupts),
        "resumed_after_restart": True,
        "final_action": finished.get("hitl_action"),
        "next_node_after_restore": list(snapshot.next),
    }


async def run_single(query: str) -> None:
    """Один довільний запит — для ручних перевірок."""
    async with screening_app() as graph:
        result = await graph.ainvoke(
            initial_state(query, session_id="manual"),
            config={"configurable": {"thread_id": "manual"}, **tracing_config()},
        )
        print(f"агент: {result.get('current_agent')}")
        print(result.get("report", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо MAS HR-скринінгу")
    parser.add_argument("query", nargs="?", help="довільний запит")
    parser.add_argument("--persistence", action="store_true",
                        help="демонстрація обриву і відновлення стану")
    args = parser.parse_args()

    if args.persistence:
        outcome = asyncio.run(run_persistence_demo())
        print("\n" + "=" * 78)
        print(f"Persistence підтверджено: {outcome}")
    elif args.query:
        asyncio.run(run_single(args.query))
    else:
        asyncio.run(run_demo())


if __name__ == "__main__":
    sys.exit(main())
