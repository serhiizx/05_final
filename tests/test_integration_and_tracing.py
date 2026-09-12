"""Тести інтеграції MCP, CrewAI-реалізації, трейсингу й логера траєкторії.

Мережа не потрібна: MCP-сервер піднімається як локальний підпроцес, CrewAI
перевіряється на стабах, а трейсинг — через змінні оточення.
"""

import json

import pytest

import mas_langgraph
from conftest import StubTool
from observability import langfuse_enabled, langsmith_enabled, tracing_config, tracing_status
from schemas import FetchResumeArgs, KbSearchArgs, SendEmailArgs
from trajectory_logger import (
    agents_used,
    dump_trajectory,
    log_step,
    summarize,
    tools_called,
)


# ── MCP-інтеграція (піднімає справжній підпроцес) ───────────────────────────


@pytest.mark.asyncio
async def test_mcp_klient_viddaye_usi_instrumenty():
    """MultiServerMCPClient справді бачить сервер і його інструменти."""
    client, tools = await mas_langgraph.load_mcp_tools()
    names = {t.name for t in tools}
    assert {"fetch_resume", "fetch_job_requirements", "score_candidate",
            "search_candidates", "send_candidate_email"}.issubset(names)
    # search_hr_kb додається локально: для агентів різниці немає.
    assert "search_hr_kb" in names


@pytest.mark.asyncio
async def test_mcp_resurs_chytayetsya_cherez_kliyenta():
    client, _ = await mas_langgraph.load_mcp_tools()
    policy = await mas_langgraph.read_mcp_resource(client, "hrpolicy://screening")
    assert '"strong_match": ">= 75"' in policy


@pytest.mark.asyncio
async def test_mcp_prompt_dostupnyi_cherez_kliyenta():
    client, _ = await mas_langgraph.load_mcp_tools()
    messages = await client.get_prompt(
        "hr", "candidate_reply",
        arguments={"candidate_name": "Олена", "decision": "reject", "gaps": "Python"},
    )
    text = " ".join(
        m.content if isinstance(m.content, str) else getattr(m.content, "text", "")
        for m in messages
    )
    assert "назви конкретну причину" in text


# ── Розподіл інструментів по агентах ───────────────────────────────────────


def test_bez_ryzykovoho_instrumenta_hraf_ne_zbyrayetsya(monkeypatch):
    """Fail fast при збірці, а не посеред прогону після схвалення людини."""
    from conftest import StubLLM, stub_react_agent_with_text

    monkeypatch.setattr(mas_langgraph, "create_react_agent",
                        lambda *a, **kw: stub_react_agent_with_text("готово"))
    with pytest.raises(RuntimeError, match="send_candidate_email"):
        mas_langgraph.build_graph([], llm=StubLLM())


def test_kozhen_ahent_otrymuye_lyshe_svoye():
    tools = [
        StubTool("fetch_resume", args_schema=FetchResumeArgs),
        StubTool("send_candidate_email", args_schema=SendEmailArgs),
        StubTool("search_hr_kb", args_schema=KbSearchArgs),
    ]
    assert [t.name for t in mas_langgraph.tools_for("screening", tools)] == ["fetch_resume"]
    assert [t.name for t in mas_langgraph.tools_for("researcher", tools)] == ["search_hr_kb"]
    assert mas_langgraph.tools_for("supervisor", tools) == []
    assert mas_langgraph.tools_for("general", tools) == []


def test_hraf_maye_vsi_ochikuvani_vuzly(monkeypatch):
    from conftest import StubLLM, stub_react_agent_with_text

    # create_react_agent вимагає справжній Runnable; тут перевіряється лише
    # розводка графа.
    monkeypatch.setattr(mas_langgraph, "create_react_agent",
                        lambda *a, **kw: stub_react_agent_with_text("готово"))
    graph = mas_langgraph.build_graph(
        [StubTool("send_candidate_email", args_schema=SendEmailArgs)], llm=StubLLM()
    )
    nodes = set(graph.get_graph().nodes)
    assert {"supervisor", "screening", "researcher", "communicator", "general",
            "human_approval", "send_email", "report"}.issubset(nodes)


# ── Логер траєкторії ───────────────────────────────────────────────────────


def test_podiya_mistyt_imya_ahenta():
    """Ключова відмінність лога MAS від лога одиночного агента з ДЗ1."""
    event = log_step("screening", "planner", "склав план", "3 кроки")
    assert event["agent_name"] == "screening"
    assert set(event) >= {"ts", "agent_name", "node", "kind", "action", "output",
                          "tools", "duration_ms"}


def test_dovhi_teksty_obrizayutsya():
    """Трейс має лишатись читабельним; повні відповіді є в messages."""
    event = log_step("a", "n", "x" * 500, "y" * 500)
    assert len(event["action"]) == 200 and len(event["output"]) == 300


def test_hraf_ne_rakhuyetsya_yak_ahent():
    """«graph» — це сам граф (HITL, звіт), а не учасник MAS."""
    events = [log_step("supervisor", "route", "a"), log_step("graph", "hitl", "b")]
    assert agents_used(events) == ["supervisor"]


def test_zvedennya_rakhuye_kroky_i_instrumenty():
    events = [
        log_step("supervisor", "route", "a", duration_ms=100),
        log_step("screening", "fetch", "b", kind="tool", tools=["fetch_resume"], duration_ms=50),
        log_step("screening", "fetch", "c", kind="tool", tools=["fetch_resume"], duration_ms=50),
        log_step("graph", "hitl", "d", kind="interrupt", duration_ms=10),
        log_step("screening", "guard", "e", kind="guard", duration_ms=5),
    ]
    stats = summarize(events)
    assert stats == {
        "steps": 1, "tool_calls": 2, "interrupts": 1, "guard_hits": 1,
        "seconds": 0.215, "agents_used": ["supervisor", "screening"],
        "tools_called": ["fetch_resume"],
    }


def test_span_ne_dodaye_chasu_dvichi():
    """span лише обгортає вкладений прогін, чиї події вже враховані."""
    events = [
        log_step("screening", "executor", "крок", kind="span", duration_ms=1000),
        log_step("screening", "tool", "виклик", kind="tool", duration_ms=200),
    ]
    assert summarize(events)["seconds"] == 0.2


def test_trayektoriya_zapysuyetsya_u_fayl(tmp_path):
    path = tmp_path / "trajectory.json"
    payload = dump_trajectory([log_step("screening", "n", "a")], path, meta={"queries": 1})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == payload
    assert saved["meta"]["queries"] == 1
    assert saved["events"][0]["agent_name"] == "screening"


# ── Observability ──────────────────────────────────────────────────────────


def test_bez_klyuchiv_trasuvannya_vymkneno(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGSMITH_TRACING",
                "LANGSMITH_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert langfuse_enabled() is False
    assert langsmith_enabled() is False
    assert tracing_config() == {}
    assert "вимкнено" in tracing_status()


def test_klyuch_z_probiliv_ne_vvazhayetsya_klyuchem(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "   ")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "   ")
    assert langfuse_enabled() is False


def test_langsmith_potrebuye_oboh_zminnykh(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert langsmith_enabled() is False
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    assert langsmith_enabled() is True


# ── CrewAI-реалізація ──────────────────────────────────────────────────────


def test_crewai_vykorystovuye_toi_samyi_allowlist():
    from mas_crewai import _tools_for

    tools = [StubTool("fetch_resume"), StubTool("send_candidate_email"),
             StubTool("score_candidate"), StubTool("fetch_job_requirements")]
    assert set(t.name for t in _tools_for("screening", tools)) == {
        "fetch_resume", "fetch_job_requirements", "score_candidate"}
    assert _tools_for("communicator", tools) == []


@pytest.mark.parametrize(
    "query,ochikuvanyi_bal",
    [("Оціни CAND-001 на JOB-BACKEND", 93),
     ("Оціни CAND-002 на JOB-BACKEND", 5),
     ("Оціни CAND-005 на JOB-ML", 93)],
)
def test_avtorytetnyi_verdykt_rakhuyetsya_detektovano(query, ochikuvanyi_bal):
    """Рубіж, якого LangGraph не потребує: у CrewAI вердикт губиться при
    переказі між агентами, тому бал перераховується інструментом."""
    from mas_crewai import authoritative_verdict

    verdict = authoritative_verdict(query)
    assert verdict["score"] == ochikuvanyi_bal


def test_bez_kandydata_verdyktu_nemaye():
    from mas_crewai import authoritative_verdict

    assert authoritative_verdict("Які правила відмови?") is None
