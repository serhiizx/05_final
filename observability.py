"""Observability: трасування прогонів MAS у Langfuse або LangSmith.

Обидва провайдери підтримані, бо вмикаються по-різному. LangSmith працює
через змінні оточення і не потребує коду взагалі. Langfuse потребує
callback-хендлера в config кожного виклику графа — саме його віддає
tracing_config().

Без ключів трасування просто вимкнене: жодного звернення до мережі, жодного
винятку. Це важливо для тестів, які мають працювати офлайн.
"""

import os

from dotenv import load_dotenv

# Читаємо .env тут, а не покладаємось на config: модуль має працювати
# самостійно, зокрема коли його імпортує тест без решти системи.
load_dotenv()


def langfuse_enabled() -> bool:
    """Чи налаштований Langfuse. Потрібні обидва ключі.

    Значення обрізається від пробілів: ключ із самих пробілів — це порожня
    конфігурація, а не валідний ключ.
    """
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    )


def langsmith_enabled() -> bool:
    """Чи увімкнений LangSmith. Він трасує сам, без коду в графі."""
    return (
        os.environ.get("LANGSMITH_TRACING", "").strip().lower() == "true"
        and bool(os.environ.get("LANGSMITH_API_KEY", "").strip())
    )


def make_langfuse_handler():
    """CallbackHandler Langfuse або None, якщо трасування вимкнено.

    Імпорт навмисно всередині функції: без ключів модуль імпортується без
    побічних ефектів і без мережі. У Langfuse 3.x шлях імпорту —
    langfuse.langchain (у 2.x був langfuse.callback, той більше не працює).
    """
    if not langfuse_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:  # бібліотека не встановлена або несумісна версія
        return None


def tracing_config() -> dict:
    """Частина config для graph.ainvoke, яка вмикає трасування.

    Порожній словник, якщо трасування вимкнене — виклик графа виглядає
    однаково в обох випадках.
    """
    handler = make_langfuse_handler()
    return {"callbacks": [handler]} if handler else {}


def tracing_status() -> str:
    """Людський опис стану трасування — друкується на початку демо."""
    active = []
    if langfuse_enabled():
        active.append(f"Langfuse ({os.environ.get('LANGFUSE_BASE_URL', 'cloud.langfuse.com')})")
    if langsmith_enabled():
        active.append(f"LangSmith (проєкт {os.environ.get('LANGSMITH_PROJECT', 'default')})")
    if not active:
        return "вимкнено (немає ключів — це нормально, система працює без них)"
    return " + ".join(active)


if __name__ == "__main__":
    print("Стан трасування:", tracing_status())
    print("Langfuse:", "увімкнено" if langfuse_enabled() else "вимкнено")
    print("LangSmith:", "увімкнено" if langsmith_enabled() else "вимкнено")
    print("config для ainvoke:", "з callbacks" if tracing_config() else "порожній")
