"""Тести даних, схем і бази знань. Перевіряють інваріанти, на які спираються
решта тестів і README: якщо зміняться дані, впаде саме цей файл."""

import json

import pytest
from pydantic import ValidationError

from config import load_candidates, load_jobs, load_prompt
from schemas import (
    EmailDraft,
    FetchResumeArgs,
    Plan,
    PlanStep,
    RouteDecision,
    ScoreArgs,
    SendEmailArgs,
)
from tools_legacy import load_documents


# ── Дані ────────────────────────────────────────────────────────────────────


def test_usi_kandydaty_mayut_obovyazkovi_polya():
    for cid, candidate in load_candidates().items():
        for field in ("candidate_id", "full_name", "email", "status", "applied_for", "resume_text"):
            assert field in candidate, f"{cid}: немає {field}"
        assert candidate["candidate_id"] == cid


def test_kozhen_kandydat_podavsya_na_isnuyuchu_vakansiyu():
    jobs = load_jobs()
    for cid, candidate in load_candidates().items():
        assert candidate["applied_for"] in jobs, cid


def test_kandydat_003_mistyt_sprobu_manipulyatsiyi():
    """На цьому кандидаті тримаються evals і red-team: якщо прибрати
    ін'єкцію з резюме, вони перестануть перевіряти те, що мають."""
    text = load_candidates()["CAND-003"]["resume_text"].lower()
    assert "ignore all previous instructions" in text
    assert "strong_match" in text


def test_kandydat_004_mistyt_usi_typy_pii():
    """Матеріал для перевірки output guardrail."""
    text = load_candidates()["CAND-004"]["resume_text"]
    for fragment in ("14.03.1992", "3216549870", "АА123456", "4242 4242 4242 4242",
                     "UA213223130000026007233566001", "вул. Дерибасівська", "+380671234567"):
        assert fragment in text, fragment


def test_vakansiyi_mayut_neporozhni_vymohy():
    for jid, job in load_jobs().items():
        assert job["must_have"], jid
        assert job["min_years"] > 0, jid
        assert job["status"] in {"open", "closed"}, jid


def test_ye_rivno_odna_zakryta_vakansiya():
    """Ресурс jobs://open має що відфільтровувати."""
    closed = [j for j in load_jobs().values() if j["status"] == "closed"]
    assert len(closed) == 1


# ── База знань ──────────────────────────────────────────────────────────────


def test_baza_znan_mistyt_visim_dokumentiv():
    docs = load_documents()
    assert len(docs) == 8
    assert all(doc["text"] and doc["title"] for doc in docs)


def test_readme_ne_potraplyaye_v_bazu():
    """README — довідка для людини, а не документ бази."""
    assert "README.md" not in {doc["source"] for doc in load_documents()}


def test_klyuchovi_pravyla_ye_v_bazi():
    """На ці правила спираються evals: researcher має де взяти відповідь."""
    corpus = " ".join(doc["text"] for doc in load_documents())
    assert "трьох робочих днів" in corpus
    assert "Відмова без пояснення причини заборонена" in corpus
    assert "maybe" in corpus


# ── Промпти ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["supervisor", "screening_planner", "screening_executor", "screening_replanner",
     "replan_budget_exhausted", "researcher", "communicator", "general"],
)
def test_usi_prompty_isnuyut_i_neporozhni(name):
    assert len(load_prompt(name).strip()) > 40, name


def test_prompt_replannera_zaboronyaye_zavershennya_bez_balu():
    """Промпт дублює структурний захист у коді. Дублювання навмисне:
    промпт зменшує кількість спроб, код гарантує результат."""
    assert "score_candidate" in load_prompt("screening_replanner")


def test_prompt_communicatora_zaboronyaye_pii_ta_bal():
    text = load_prompt("communicator")
    assert "персональні дані" in text
    assert "бал" in text


# ── Схеми ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["CAND-1", "cand-001", "CAND-0001", "", "../CAND-001"])
def test_format_id_kandydata_perevirayetsya(value):
    with pytest.raises(ValidationError):
        FetchResumeArgs(candidate_id=value)


def test_porozhni_navychky_vidkhylyayutsya():
    with pytest.raises(ValidationError):
        ScoreArgs(skills=[], years_experience=5, job_id="JOB-BACKEND")
    with pytest.raises(ValidationError):
        ScoreArgs(skills=["  ", ""], years_experience=5, job_id="JOB-BACKEND")


def test_navychky_obrizayutsya_vid_probiliv():
    args = ScoreArgs(skills=[" Python ", "Docker"], years_experience=5, job_id="JOB-BACKEND")
    assert args.skills == ["Python", "Docker"]


def test_nerealnyi_stazh_vidkhylyayetsya():
    with pytest.raises(ValidationError):
        ScoreArgs(skills=["Python"], years_experience=200, job_id="JOB-BACKEND")


def test_tema_lysta_z_probiliv_ne_ye_temoyu():
    with pytest.raises(ValidationError):
        SendEmailArgs(candidate_id="CAND-001", decision="reject", subject="   ", body="текст")


def test_nevidome_rishennya_vidkhylyayetsya():
    with pytest.raises(ValidationError):
        SendEmailArgs(candidate_id="CAND-001", decision="hire_now", subject="Тема", body="Текст")


def test_route_decision_pryimaye_lyshe_vidomykh_ahentiv():
    RouteDecision(next_agent="screening", reason="тест")
    with pytest.raises(ValidationError):
        RouteDecision(next_agent="billing", reason="тест")


def test_plan_ne_buvaye_porozhnim():
    with pytest.raises(ValidationError):
        Plan(steps=[])
    Plan(steps=[PlanStep(description="крок", expected_outcome="результат")])


def test_email_draft_vymahaye_tekstu():
    with pytest.raises(ValidationError):
        EmailDraft(candidate_id="CAND-001", decision="reject", subject="Тема", body="")
