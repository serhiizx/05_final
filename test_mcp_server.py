"""Unit tests для MCP-сервера HR-скринінгу.

Тести ходять через справжній протокольний шар FastMCP — list_tools,
call_tool, list_resources, read_resource, list_prompts, render_prompt, — а не
викликають доменні функції напряму. Так перевіряється й сама реєстрація
примітивів, і серіалізація результату.

Запуск: uv run python test_mcp_server.py    (друкує PASS по кожному тесту)
        uv run pytest test_mcp_server.py -v (той самий набір під pytest)
"""

import asyncio
import json

import pytest

from mcp_server import mcp


async def call_tool(name: str, args: dict) -> dict:
    """Хелпер: викликати MCP-tool і розібрати JSON з текстового блоку."""
    result = await mcp.call_tool(name, args)
    blocks = result[0] if isinstance(result, tuple) else result.content
    return json.loads(blocks[0].text)


async def read_resource(uri: str) -> dict:
    """Хелпер: прочитати MCP-resource за URI."""
    result = await mcp.read_resource(uri)
    return json.loads(result.contents[0].content)


async def render_prompt(name: str, args: dict) -> str:
    """Хелпер: відрендерити MCP-prompt і дістати текст першого повідомлення."""
    result = await mcp.render_prompt(name, args)
    return result.messages[0].content.text


# ── 1. Реєстрація примітивів ────────────────────────────────────────────────


async def test_01_zareyestrovani_vsi_pyat_instrumentiv():
    """list_tools повертає всі п'ять інструментів сервера."""
    names = {t.name for t in await mcp.list_tools()}
    assert names == {
        "fetch_resume",
        "fetch_job_requirements",
        "score_candidate",
        "search_candidates",
        "send_candidate_email",
    }


async def test_02_instrumenty_mayut_docstring_dlya_modeli():
    """У кожного інструмента є непорожній опис — його читає модель,
    вирішуючи, чи викликати інструмент."""
    for tool in await mcp.list_tools():
        assert tool.description and len(tool.description) > 50, tool.name


# ── 2. fetch_resume ─────────────────────────────────────────────────────────


async def test_03_fetch_resume_znaydenyi_kandydat():
    """Наявний кандидат повертається з текстом резюме."""
    data = await call_tool("fetch_resume", {"candidate_id": "CAND-001"})
    assert data["status"] == "ok"
    assert data["data"]["candidate_id"] == "CAND-001"
    assert "Python" in data["data"]["resume_text"]


async def test_04_fetch_resume_nevidomyi_kandydat_daye_pomylku():
    """Відсутній кандидат дає error, а не виняток."""
    data = await call_tool("fetch_resume", {"candidate_id": "CAND-999"})
    assert data["status"] == "error"
    assert "немає в базі" in data["error"]


async def test_05_fetch_resume_nevalidnyi_format_id():
    """Порушення формату CAND-XXX ловить Pydantic-схема, а не база."""
    data = await call_tool("fetch_resume", {"candidate_id": "не-ід"})
    assert data["status"] == "error"
    assert "валідації" in data["error"]


# ── 3. score_candidate: детермінована арифметика ────────────────────────────


async def test_06_score_candidate_rakhuye_ochikuvanyi_bal():
    """Чотири з п'яти навичок і подвійний стаж дають 93 і strong_match.

    Число зафіксоване навмисно: 70 + 7.5 + 15 = 92.5, і звичайне округлення
    дає 93. Стандартний round() дав би 92 — саме тому в коді int(x + 0.5)."""
    data = await call_tool(
        "score_candidate",
        {
            "skills": ["Python", "PostgreSQL", "Docker", "Kubernetes"],
            "years_experience": 6,
            "job_id": "JOB-BACKEND",
        },
    )
    assert data["status"] == "ok"
    assert data["data"]["score"] == 93
    assert data["data"]["decision"] == "strong_match"
    assert data["data"]["missing_must_have"] == []


async def test_07_score_candidate_slabkyi_kandydat_dictaye_reject():
    """Жодної обов'язкової навички і стаж нижче мінімуму — reject."""
    data = await call_tool(
        "score_candidate",
        {"skills": ["HTML", "CSS"], "years_experience": 1, "job_id": "JOB-BACKEND"},
    )
    assert data["data"]["decision"] == "reject"
    assert set(data["data"]["missing_must_have"]) == {"Python", "PostgreSQL", "Docker"}


async def test_08_score_candidate_porozhni_navychky_vidkhylyayutsya():
    """Порожній список навичок відсікає field_validator схеми ScoreArgs."""
    data = await call_tool(
        "score_candidate", {"skills": [], "years_experience": 5, "job_id": "JOB-BACKEND"}
    )
    assert data["status"] == "error"


# ── 4. fetch_job_requirements і search_candidates ───────────────────────────


async def test_09_fetch_job_requirements_povertaye_vymohy():
    """Вакансія повертається з обома списками навичок і мінімумом років."""
    data = await call_tool("fetch_job_requirements", {"job_id": "JOB-BACKEND"})
    assert data["data"]["must_have"] == ["Python", "PostgreSQL", "Docker"]
    assert data["data"]["min_years"] == 3


async def test_10_search_candidates_filtruye_za_vakansiyeyu():
    """Фільтр за вакансією звужує вибірку, резюме в неї не потрапляє."""
    data = await call_tool("search_candidates", {"job_id": "JOB-ML"})
    assert data["data"]["count"] == 1
    assert data["data"]["candidates"][0]["candidate_id"] == "CAND-005"
    assert "resume_text" not in data["data"]["candidates"][0]


async def test_11_search_candidates_bez_filtriv_daye_vsikh():
    """Без фільтрів повертається вся база."""
    data = await call_tool("search_candidates", {})
    assert data["data"]["count"] == 5


# ── 5. send_candidate_email: ризиковий інструмент ───────────────────────────


async def test_12_send_candidate_email_validuye_rishennya():
    """Невідоме рішення відхиляється до будь-якого запису у файл."""
    data = await call_tool(
        "send_candidate_email",
        {
            "candidate_id": "CAND-001",
            "decision": "hire_immediately",
            "subject": "Тема",
            "body": "Текст",
        },
    )
    assert data["status"] == "error"


async def test_13_send_candidate_email_vidkhylyaye_porozhniu_temu():
    """Тема з самих пробілів не є темою — ловить field_validator."""
    data = await call_tool(
        "send_candidate_email",
        {"candidate_id": "CAND-001", "decision": "reject", "subject": "   ", "body": "Текст"},
    )
    assert data["status"] == "error"


# ── 6. Resources ────────────────────────────────────────────────────────────


async def test_14_zareyestrovani_obydva_resursy():
    """list_resources повертає обидва довідники."""
    uris = {str(r.uri) for r in await mcp.list_resources()}
    assert uris == {"hrpolicy://screening", "jobs://open"}


async def test_15_resurs_polityky_mistyt_porohy_i_vahy():
    """Ресурс політики віддає ті самі пороги, за якими рахує score_candidate."""
    policy = await read_resource("hrpolicy://screening")
    assert policy["thresholds"]["strong_match"] == ">= 75"
    assert policy["weights"]["must_have"] == 70
    assert any("підтвердження людиною" in rule for rule in policy["rules"])


async def test_16_resurs_vakansiy_ne_pokazuye_zakryti():
    """Закрита вакансія JOB-QA у довідник відкритих не потрапляє."""
    jobs = await read_resource("jobs://open")
    ids = {job["job_id"] for job in jobs["jobs"]}
    assert ids == {"JOB-BACKEND", "JOB-ML"}


# ── 7. Prompts ──────────────────────────────────────────────────────────────


async def test_17_zareyestrovani_obydva_shablony():
    """list_prompts повертає обидва шаблони."""
    names = {p.name for p in await mcp.list_prompts()}
    assert names == {"candidate_reply", "screening_summary"}


async def test_18_shablon_vidmovy_vymahaye_prychyny_i_zaboronyaye_pii():
    """Шаблон відмови вимагає назвати причину і прямо забороняє PII в листі."""
    text = await render_prompt(
        "candidate_reply",
        {"candidate_name": "Олена", "decision": "reject", "gaps": "PostgreSQL, Docker"},
    )
    assert "назви конкретну причину" in text
    assert "PostgreSQL, Docker" in text
    assert "персональних даних" in text


async def test_19_shablon_pidsumku_zadaye_strukturu():
    """Шаблон підсумку перелічує обов'язкові розділи."""
    text = await render_prompt(
        "screening_summary", {"candidate_id": "CAND-001", "job_id": "JOB-BACKEND"}
    )
    assert "CAND-001" in text and "JOB-BACKEND" in text
    assert "вердикт" in text and "маніпуляції" in text


# ── Запуск як скрипта ───────────────────────────────────────────────────────


async def _run_all() -> None:
    """Прогнати всі тести послідовно і надрукувати результат кожного."""
    tests = sorted(
        ((name, fn) for name, fn in globals().items() if name.startswith("test_")),
        key=lambda pair: pair[0],
    )
    for name, fn in tests:
        await fn()
        print(f"PASS — {name}")
    print(f"\nУсі {len(tests)} MCP unit tests пройдено.")


if __name__ == "__main__":
    asyncio.run(_run_all())
