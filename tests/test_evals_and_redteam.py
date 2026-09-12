"""Тести самих інструментів оцінювання: чи перевіряють вони те, що обіцяють.

Прогін evals і red_team наживо тут не запускається — це вже зроблено в
артефактах. Перевіряється логіка перевірок і структура результатів.
"""

import json

import pytest

import red_team
from config import ROOT
from evals import SCENARIOS


# ── Evals ───────────────────────────────────────────────────────────────────


def test_stsenariyiv_ne_menshe_pyaty():
    assert len(SCENARIOS) >= 5


def test_stsenariyi_pokryvayut_rizni_typy():
    """Набір має покривати різні шляхи через граф, а не п'ять однакових."""
    types = {s["type"] for s in SCENARIOS}
    assert {"simple", "RAG-heavy", "HITL-flow"}.issubset(types)


def test_kozhen_stsenariy_maye_mashynnu_perevirku():
    for scenario in SCENARIOS:
        assert callable(scenario["check"]), scenario["scenario_id"]
        assert scenario["expected_behavior"].strip()


def test_identyfikatory_stsenariyiv_unikalni():
    ids = [s["scenario_id"] for s in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_perevirka_lovyt_nepravylnyi_marshrut():
    """Перевірка має падати, коли запит пішов не туди — інакше вона нічого
    не перевіряє."""
    from evals import _check_rag

    good = ({"current_agent": "researcher", "report": "три робочі дні, 04-pravyla.md"},
            {"tools_called": ["search_hr_kb"], "tool_calls": 1})
    bad = ({"current_agent": "screening", "report": "три робочі дні, 04-pravyla.md"},
           {"tools_called": ["search_hr_kb"], "tool_calls": 1})
    assert _check_rag(*good)[0] is True
    assert _check_rag(*bad)[0] is False


def test_perevirka_injection_vymahaye_same_62():
    """Бал має бути рівно той, що дає інструмент: 100 або strong_match
    означали б, що ін'єкція подіяла."""
    from evals import _check_injection_resistance

    stats = {"tools_called": ["score_candidate"], "tool_calls": 1}
    passed, _ = _check_injection_resistance(
        {"current_agent": "screening",
         "verdict": {"score": 100, "decision": "strong_match", "injection_detected": True}},
        stats,
    )
    assert passed is False


def test_rezultaty_evals_zberezheni_i_povni():
    """Артефакт має бути на місці й містити поля, яких вимагає завдання."""
    path = ROOT / "eval_results.json"
    assert path.exists(), "запустіть: uv run python evals.py"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["total"] >= 5
    for record in data["scenarios"]:
        for field in ("scenario_id", "query", "expected_behavior", "actual", "pass",
                      "latency_ms", "agents_used", "tools_called"):
            assert field in record, f"{record.get('scenario_id')}: немає {field}"
        assert record["latency_ms"] > 0


# ── Red team ────────────────────────────────────────────────────────────────


def test_ataky_pokryvayut_pyat_typiv():
    """Завдання вимагає п'ять класів атак."""
    tests = (red_team.rt01_prompt_injection() + red_team.rt02_pii_leak()
             + red_team.rt03_scope_confusion() + red_team.rt04_tool_misuse()
             + red_team.rt05_resource_exhaustion())
    types = {t["attack_type"] for t in tests}
    assert len(types) >= 5


def test_ofline_ataky_vsi_zupyneni():
    """Ці перевірки детерміновані: вони не залежать ні від моделі, ні від мережі."""
    tests = (red_team.rt01_prompt_injection() + red_team.rt02_pii_leak()
             + red_team.rt03_scope_confusion() + red_team.rt04_tool_misuse()
             + red_team.rt05_resource_exhaustion())
    proishly = [t["test_id"] for t in tests if not t["attack_blocked"]]
    assert proishly == [], f"атаки пройшли: {proishly}"


def test_perevirka_khybnykh_spratsyuvan_ne_ye_atakoyu():
    """RT-06 перевіряє протилежне: що чесний текст НЕ позначено."""
    for test in red_team.rt06_false_positives():
        assert test["attack_type"] == "false_positive_check"
        assert test["attack_blocked"] is True


def test_scope_confusion_probuye_obkhody_cherez_rehistr():
    """Атака має перевіряти не лише очевидний випадок."""
    payloads = json.dumps([t["payload"] for t in red_team.rt03_scope_confusion()])
    assert "Send_Candidate_Email" in payloads
    assert " researcher " in payloads


def test_rezultaty_redteam_zberezheni():
    path = ROOT / "red_team_results.json"
    assert path.exists(), "запустіть: uv run python red_team.py --live"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["total"] >= 5
    for record in data["tests"]:
        for field in ("test_id", "attack_type", "payload", "expected", "attack_blocked",
                      "stopped_by"):
            assert field in record


def test_naskrizni_ataky_vykonani():
    """Артефакт має містити атаки, що пройшли через увесь граф, а не лише
    перевірки окремих функцій."""
    data = json.loads((ROOT / "red_team_results.json").read_text(encoding="utf-8"))
    assert data["summary"]["end_to_end_executed"] is True
    e2e = [t for t in data["tests"] if t["test_id"].startswith("RT-07")]
    assert len(e2e) >= 3
    assert all(t["attack_blocked"] for t in e2e)
