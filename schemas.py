"""Pydantic v2 схеми: аргументи інструментів і доменні моделі.

Перевикористано з ДЗ1/ДЗ2 і розширено під MAS: додано RouteDecision для
supervisor'а та Plan/ReplanDecision для Plan-and-Execute підграфа.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CANDIDATE_ID_PATTERN = r"^CAND-\d{3}$"
JOB_ID_PATTERN = r"^JOB-[A-Z]+$"

Decision = Literal["strong_match", "maybe", "reject"]
# Агенти MAS. "general" — fallback для вітань і нерозпізнаних запитів.
AgentName = Literal["screening", "researcher", "communicator", "general"]


# --- Аргументи MCP-інструментів -------------------------------------------


class FetchResumeArgs(BaseModel):
    """Аргументи fetch_resume."""

    candidate_id: str = Field(
        pattern=CANDIDATE_ID_PATTERN, description="Ідентифікатор кандидата, напр. CAND-001"
    )


class FetchJobArgs(BaseModel):
    """Аргументи fetch_job_requirements."""

    job_id: str = Field(
        pattern=JOB_ID_PATTERN, description="Ідентифікатор вакансії, напр. JOB-BACKEND"
    )


class ScoreArgs(BaseModel):
    """Аргументи score_candidate."""

    skills: list[str] = Field(min_length=1, description="Навички, витягнуті з резюме")
    years_experience: float = Field(ge=0, le=60, description="Роки релевантного досвіду")
    job_id: str = Field(pattern=JOB_ID_PATTERN)

    @field_validator("skills")
    @classmethod
    def navychky_ne_porozhni(cls, value: list[str]) -> list[str]:
        cleaned = [s.strip() for s in value if s and s.strip()]
        if not cleaned:
            raise ValueError("список навичок не може складатися з порожніх рядків")
        return cleaned


class SendEmailArgs(BaseModel):
    """Аргументи ризикового інструмента send_candidate_email."""

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN)
    decision: Decision
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)

    @field_validator("subject", "body")
    @classmethod
    def tekst_ne_porozhnii(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("текст не може складатися лише з пробілів")
        return value.strip()


class SearchCandidatesArgs(BaseModel):
    """Аргументи search_candidates. Усі фільтри необов'язкові —
    без жодного інструмент повертає всю базу."""

    job_id: str | None = Field(default=None, pattern=JOB_ID_PATTERN)
    status: Literal["new", "screened", "contacted"] | None = None
    min_years: float | None = Field(default=None, ge=0, le=60)


class KbSearchArgs(BaseModel):
    """Аргументи search_hr_kb — інструмента Agentic RAG."""

    query: str = Field(min_length=3, max_length=500, description="Питання до бази HR-політик")
    top_k: int = Field(default=3, ge=1, le=10)


# --- Доменні моделі --------------------------------------------------------


class ResumeFacts(BaseModel):
    """Структуровані факти, витягнуті з сирого тексту резюме."""

    skills: list[str] = Field(default_factory=list)
    years_experience: float = Field(default=0, ge=0, le=60)
    education: str = ""
    location: str = ""


class ScreeningVerdict(BaseModel):
    """Фінальний вердикт скринінгу."""

    candidate_id: str
    job_id: str
    score: int = Field(ge=0, le=100)
    decision: Decision
    rationale: str
    gaps: list[str] = Field(default_factory=list)
    injection_detected: bool = False


class EmailDraft(BaseModel):
    """Чернетка листа кандидату — те, що людина бачить під час HITL."""

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN)
    decision: Decision
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


# --- Оркестрація -----------------------------------------------------------


class RouteDecision(BaseModel):
    """Рішення supervisor'а, якому агенту віддати запит.

    Structured output, а не вільний текст: інакше маршрут довелося б
    вигрібати регуляркою з відповіді моделі."""

    next_agent: AgentName = Field(description="Цільовий агент або general для нерозпізнаних запитів")
    reason: str = Field(min_length=1, description="Коротке пояснення вибору")
    candidate_id: str | None = Field(
        default=None, description="ID кандидата, якщо він згаданий у запиті"
    )
    job_id: str | None = Field(default=None, description="ID вакансії, якщо він згаданий у запиті")


class PlanStep(BaseModel):
    """Один крок плану Plan-and-Execute агента."""

    description: str = Field(min_length=1, description="Що саме зробити на цьому кроці")
    expected_outcome: str = Field(min_length=1, description="Який результат вважати успішним")


class Plan(BaseModel):
    """План скринінгу, складений до того, як агент побачив дані."""

    steps: list[PlanStep] = Field(min_length=1, max_length=6)


class ReplanDecision(BaseModel):
    """Рішення replanner'а після чергового кроку: завершити чи перепланувати."""

    action: Literal["finish", "replan"]
    reasoning: str = ""
    final_answer: str = ""
    updated_steps: list[PlanStep] = Field(default_factory=list)
