"""JSON-логування траєкторії MAS.

Перевикористано з ДЗ1 (logger.py) і розширено ключовим для MAS полем
`agent_name`: у мультиагентній системі при post-mortem треба бачити не лише
який вузол помилився, а й який агент ним керував.

Події накопичуються у стані графа, а не в глобальному списку: інакше при
відновленні з checkpointer лог не збігався б зі станом.
"""

import json
import time
from pathlib import Path

# kind-и, чия тривалість входить в активний час. "span" лише обгортає
# вкладений прогін, чиї власні події вже в тому самому списку, — інакше
# вкладений прогін рахувався б двічі.
TIMED_KINDS = {"llm", "tool", "interrupt", "guard"}


def log_step(
    agent_name: str,
    node: str,
    action: str,
    output: str = "",
    kind: str = "llm",
    duration_ms: float = 0.0,
    tools: list[str] | None = None,
    **fields,
) -> dict:
    """Одна подія траєкторії MAS.

    agent_name — хто виконував крок (supervisor, screening, researcher,
    communicator, general, graph). Саме це поле відрізняє лог MAS від лога
    одиночного агента з ДЗ1.

    kind: llm — виклик моделі; tool — виконання інструмента; interrupt —
    зупинка на підтвердженні людиною; guard — спрацювання захисту; span —
    обгортка над вкладеним прогоном (крок Plan-and-Execute executor'а).

    Тексти обрізаються: трейс має лишатись читабельним, а повні відповіді
    моделі вже є в messages стану.
    """
    event = {
        "ts": round(time.time(), 3),
        "agent_name": agent_name,
        "node": node,
        "kind": kind,
        "action": action[:200],
        "output": output[:300],
        "tools": tools or [],
        "duration_ms": round(duration_ms, 2),
    }
    event.update(fields)
    return event


def active_seconds(events: list[dict]) -> float:
    """Сумарний активний час, секунди.

    Не різниця настінного часу: між зупинкою на interrupt і відповіддю людини
    можуть минути хвилини, і таймаут спрацював би на повільній людині.
    """
    return sum(e.get("duration_ms", 0.0) for e in events if e.get("kind") in TIMED_KINDS) / 1000.0


def agents_used(events: list[dict]) -> list[str]:
    """Перелік агентів, які брали участь у прогоні, у порядку появи.

    Службовий агент "graph" не рахується: це сам граф (HITL, звіт), а не агент.
    """
    seen: list[str] = []
    for event in events:
        name = event.get("agent_name")
        if name and name not in seen and name != "graph":
            seen.append(name)
    return seen


def tools_called(events: list[dict]) -> list[str]:
    """Перелік інструментів, викликаних за прогін, без повторів."""
    seen: list[str] = []
    for event in events:
        for tool in event.get("tools", []):
            if tool not in seen:
                seen.append(tool)
    return seen


def summarize(events: list[dict]) -> dict:
    """Зведення по траєкторії: кроки, виклики, активний час, склад учасників."""
    return {
        "steps": sum(1 for e in events if e["kind"] == "llm"),
        "tool_calls": sum(1 for e in events if e["kind"] == "tool"),
        "interrupts": sum(1 for e in events if e["kind"] == "interrupt"),
        "guard_hits": sum(1 for e in events if e["kind"] == "guard"),
        "seconds": round(active_seconds(events), 3),
        "agents_used": agents_used(events),
        "tools_called": tools_called(events),
    }


def dump_trajectory(events: list[dict], path: str | Path, meta: dict | None = None) -> dict:
    """Записати траєкторію у JSON-файл і повернути записану структуру."""
    payload = {"meta": meta or {}, "summary": summarize(events), "events": events}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
