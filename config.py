"""Конфігурація проєкту: шляхи, ліміти, завантаження даних і промптів, фабрика LLM."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
PROMPTS_DIR = ROOT / "prompts"
CHECKPOINT_DB = ROOT / "agent_state.db"
OUTBOX_PATH = DATA_DIR / "outbox.json"
TRAJECTORY_PATH = ROOT / "trajectory.json"
CHROMA_PATH = str(ROOT / "chroma_db")
COLLECTION_NAME = "hr_policy_kb"

# ── Захисні механізми з ДЗ1, застосовані per-agent ──────────────────────────
# Ліміти навмисно різні: researcher робить один пошук і відповідає, а
# screening планує кілька кроків, тому йому потрібен більший бюджет.
MAX_STEPS = 8                # максимум ітерацій ReAct-циклу одного агента
AGENT_TIMEOUT_S = 90.0       # ліміт активного часу одного агент-вузла
EXECUTOR_MAX_STEPS = 5       # бюджет вкладеного виконавця кроку плану
MAX_REPLANS = 2              # переплановувань за один прогін screening-агента
RATE_LIMIT_CALLS = 30        # запитів на сесію...
RATE_LIMIT_WINDOW_S = 60     # ...за це вікно, секунд


def load_candidates() -> dict[str, dict]:
    """Мокові резюме кандидатів. У production тут був би ATS чи CRM."""
    return json.loads((DATA_DIR / "candidates.json").read_text(encoding="utf-8"))


def load_jobs() -> dict[str, dict]:
    """Мокові вимоги вакансій."""
    return json.loads((DATA_DIR / "jobs.json").read_text(encoding="utf-8"))


def load_prompt(name: str) -> str:
    """Системний промпт із prompts/<name>.md."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Фабрика LLM. Провайдер повністю визначається змінними .env —
    код однаково працює і з OpenAI, і з локальним LM Studio.

    temperature=0 за замовчуванням, щоб демо-прогони були відтворюваними."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY відсутній. Скопіюйте .env.example у .env "
            "(cp .env.example .env) і підставте ключ — для справжнього OpenAI "
            "реальний ключ, для локального LM Studio будь-який непорожній рядок."
        )
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=api_key,
        temperature=temperature,
    )
