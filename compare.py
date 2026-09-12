"""Порівняння LangGraph і CrewAI на однаковому наборі прогонів.

Умова чесності: ті самі запити, та сама модель, той самий MCP-сервер, та сама
база знань. Без цього порівнювати токени й час не можна.

Що вимірюється офлайн: рядки коду оркестрації. Що вимірюється лише з
ключами: час, токени, вартість. Невиміряне позначається явним текстом, а не
нулем — щоб його неможливо було сплутати з результатом.

Запуск: uv run python compare.py
"""

import asyncio
import json
import time
from pathlib import Path

from config import ROOT
from observability import langfuse_enabled

COMPARISON_PATH = ROOT / "comparison.json"

# Три запити різного типу — той самий набір для обох фреймворків.
BENCHMARK_QUERIES = [
    "Оціни кандидата CAND-001 на вакансію JOB-BACKEND",
    "Оціни кандидата CAND-002 на вакансію JOB-BACKEND",
    "За скільки робочих днів ми маємо відповісти кандидату після скринінгу?",
]

NOT_MEASURED = "потребує прогону з ключами API"


def count_loc(path: Path) -> int:
    """Рядки коду без порожніх і без рядків-коментарів.

    Докстрінги рахуються як код: це рядкові літерали, а не коментарі.
    Показник чесний лише як розмір коду оркестрації для одного сценарію. Він
    не враховує, скільки логіки CrewAI ховає в декларативних структурах
    (Task.description, backstory), і не є оцінкою якості.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


async def run_langgraph_benchmark() -> dict:
    """Прогнати набір запитів через LangGraph і зібрати заміри."""
    from hitl import screening_app
    from mas_langgraph import initial_state
    from trajectory_logger import summarize

    runs = []
    async with screening_app() as graph:
        for i, query in enumerate(BENCHMARK_QUERIES):
            started = time.perf_counter()
            result = await graph.ainvoke(
                initial_state(query, session_id=f"bench-lg-{i}"),
                config={"configurable": {"thread_id": f"bench-lg-{i}"}},
            )
            elapsed = time.perf_counter() - started
            stats = summarize(result.get("trajectory", []))
            runs.append({
                "query": query,
                "seconds": round(elapsed, 2),
                "routed_to": result.get("current_agent"),
                "agents_used": stats["agents_used"],
                "tools_called": stats["tools_called"],
                "llm_steps": stats["steps"],
                "verdict": (result.get("verdict") or {}).get("decision"),
                "score": (result.get("verdict") or {}).get("score"),
            })
    return {"runs": runs, "total_seconds": round(sum(r["seconds"] for r in runs), 2)}


def run_crewai_benchmark() -> dict:
    """Той самий набір через CrewAI."""
    from mas_crewai import run_crew

    runs = []
    for query in BENCHMARK_QUERIES:
        started = time.perf_counter()
        report = run_crew(query, auto_approve=True)
        elapsed = time.perf_counter() - started
        runs.append({
            "query": query,
            "seconds": round(elapsed, 2),
            "report_length": len(report),
            "authoritative_block_needed": "Авторитетний вердикт" in report,
        })
    return {"runs": runs, "total_seconds": round(sum(r["seconds"] for r in runs), 2)}


def fetch_langfuse_usage() -> dict:
    """Зібрати з Langfuse кількість трейсів і вартість по фреймворках.

    Запит іде в публічне API напряму, бо SDK не дає агрегації. Трейси
    LangGraph мають ім'я "LangGraph", CrewAI — "Crew_<uuid>.kickoff", тому
    групуємо за префіксом імені.

    Обмеження, яке варто назвати прямо: поле usage на рівні трейсу Langfuse
    не заповнює, тому сумарні токени звідси взяти не можна — лише вартість,
    і то лише там, де її порахував провайдер. Для CrewAI вартість виходить
    нульовою, бо OTEL-міст OpenInference її не передає.
    """
    if not langfuse_enabled():
        return {"status": "unavailable", "reason": f"ключі Langfuse відсутні — {NOT_MEASURED}"}

    import base64
    import collections
    import os
    import urllib.request

    try:
        from langfuse import get_client

        get_client().flush()  # дочекатись, поки трейси поточного прогону доїдуть

        host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
        token = base64.b64encode(
            f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
        ).decode()
        request = urllib.request.Request(
            f"{host}/api/public/traces?limit=100", headers={"Authorization": f"Basic {token}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)

        buckets = collections.defaultdict(lambda: {"traces": 0, "cost_usd": 0.0, "links": []})
        for trace in data.get("data", []):
            name = trace.get("name") or "?"
            framework = "CrewAI" if name.startswith("Crew_") else (
                "LangGraph" if name == "LangGraph" else "інше"
            )
            bucket = buckets[framework]
            bucket["traces"] += 1
            bucket["cost_usd"] += trace.get("totalCost") or 0.0
            if len(bucket["links"]) < 2:
                bucket["links"].append(
                    f"{host}/project/{trace.get('projectId')}/traces/{trace.get('id')}"
                )

        return {
            "status": "ok",
            "total_traces": data.get("meta", {}).get("totalItems"),
            "by_framework": {k: {**v, "cost_usd": round(v["cost_usd"], 4)}
                             for k, v in buckets.items()},
            "note": "поле usage на рівні трейсу Langfuse не заповнює, тому сумарні "
                    "токени звідси недоступні; для CrewAI вартість нульова, бо "
                    "OTEL-міст OpenInference її не передає",
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


# Якісні оцінки за шкалою 1–5. Це судження з ЦЬОГО проєкту, а не з лекцій:
# кожен рядок спирається на конкретну подію, названу в полі "підстава".
QUALITATIVE = {
    "Контроль над маршрутом": {
        "LangGraph": 5,
        "CrewAI": 2,
        "підстава": "У LangGraph маршрут — conditional edge зі словником переходів, "
                    "його видно в коді. У CrewAI Process.hierarchical делегує через "
                    "менеджера, чий промпт нам не належить; у прогоні на CAND-001 "
                    "менеджер викликав delegate_work_to_coworker тричі й залучив усіх "
                    "трьох агентів замість одного.",
    },
    "Надійність передачі даних між агентами": {
        "LangGraph": 5,
        "CrewAI": 2,
        "підстава": "LangGraph бере бал зі структурованої події виклику інструмента "
                    "(_score_from_events), тому текст агента на нього не впливає. "
                    "У CrewAI результат переказується рядком: інструмент повернув "
                    "93/strong_match, а у фінальному звіті рішення стало in_progress. "
                    "Довелося додати окремий рубіж authoritative_verdict.",
    },
    "Зручність пошуку помилок": {
        "LangGraph": 5,
        "CrewAI": 3,
        "підстава": "graph.aget_state(config) дає знімок стану на будь-якому кроці, "
                    "кожен вузол — окремий запис у checkpointer. У CrewAI доступний "
                    "лише verbose-вивід у консоль, і стан між задачами передається "
                    "неявно через context.",
    },
    "Швидкість першого прототипу": {
        "LangGraph": 3,
        "CrewAI": 5,
        "підстава": "Три агенти й одна задача в CrewAI описуються декларативно. "
                    "У LangGraph довелося заздалегідь спроєктувати MASState на 16 "
                    "полів і розписати всі ребра.",
    },
    "Підтвердження людиною": {
        "LangGraph": 5,
        "CrewAI": 2,
        "підстава": "interrupt() зберігає стан у SQLite, і прогін відновлюється за тим "
                    "самим thread_id навіть після перезапуску процесу (main.py "
                    "--persistence). human_input=True в CrewAI дає блокуючий input() "
                    "у консолі без збереження стану.",
    },
    "Контроль над правами агентів": {
        "LangGraph": 5,
        "CrewAI": 3,
        "підстава": "В обох allowlist фільтрує список інструментів, але в LangGraph "
                    "є другий рубіж: guarded_tool викликає check_tool_call перед "
                    "кожним виконанням. У CrewAI агент сам вирішує, коли викликати "
                    "інструмент, і перехопити цей момент у коді оркестрації ніде.",
    },
}


def build_comparison(live: bool = False) -> dict:
    """Зібрати порівняння. live=True додає справжні прогони обох реалізацій."""
    loc = {
        "LangGraph": {
            "mas_langgraph.py": count_loc(ROOT / "mas_langgraph.py"),
            "hitl.py": count_loc(ROOT / "hitl.py"),
        },
        "CrewAI": {"mas_crewai.py": count_loc(ROOT / "mas_crewai.py")},
    }
    loc["LangGraph"]["всього"] = sum(loc["LangGraph"].values())
    loc["CrewAI"]["всього"] = sum(loc["CrewAI"].values())

    payload = {
        "benchmark_queries": BENCHMARK_QUERIES,
        "lines_of_code": loc,
        "qualitative": QUALITATIVE,
        "langfuse": fetch_langfuse_usage() if live else {"status": "skipped"},
    }

    if live:
        print("Прогін LangGraph…", flush=True)
        payload["langgraph"] = asyncio.run(run_langgraph_benchmark())
        print("Прогін CrewAI…", flush=True)
        payload["crewai"] = run_crewai_benchmark()
        payload["wall_clock"] = {
            "LangGraph_seconds": payload["langgraph"]["total_seconds"],
            "CrewAI_seconds": payload["crewai"]["total_seconds"],
        }
    else:
        payload["langgraph"] = {"status": NOT_MEASURED}
        payload["crewai"] = {"status": NOT_MEASURED}

    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Порівняння LangGraph і CrewAI")
    parser.add_argument("--offline", action="store_true",
                        help="лише те, що вимірюється без ключів (рядки коду)")
    args = parser.parse_args()

    outcome = build_comparison(live=not args.offline)
    Path(COMPARISON_PATH).write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("РЯДКИ КОДУ ОРКЕСТРАЦІЇ")
    for framework, files in outcome["lines_of_code"].items():
        print(f"  {framework}: {files['всього']}")
    if "wall_clock" in outcome:
        print("\nЧАС НА ТРЬОХ ЗАПИТАХ")
        for key, value in outcome["wall_clock"].items():
            print(f"  {key}: {value} с")
    print("\nЯКІСНІ ОЦІНКИ (1–5)")
    for criterion, scores in outcome["qualitative"].items():
        print(f"  {criterion:42s} LangGraph {scores['LangGraph']}  CrewAI {scores['CrewAI']}")
    print(f"\nЗбережено: {COMPARISON_PATH}")
