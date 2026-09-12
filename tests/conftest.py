"""Спільні стаби для офлайн-тестів: без мережі, без ключів, без LLM.

StubLLM підтримує рівно той інтерфейс, який справді використовує
mas_langgraph.py: with_structured_output(Schema).ainvoke(...) та ainvoke(...)
для general-агента. StubTool відтворює контракт MCP-інструмента —
langchain_mcp_adapters повертає список content-блоків, а не рядок (див.
_tool_result_text). StubReactAgent підмінює create_react_agent там, де вузол
лишається ReAct-агентом.
"""

import json

import pytest
from langchain_core.messages import AIMessage


class StubTool:
    """Мінімальний двійник BaseTool.

    .name потрібен allowlist'у, .args_schema — обгортці guarded_tool,
    ainvoke повертає результат у форматі реального MCP-адаптера:
    [{"type": "text", "text": "<json-рядок>"}].
    """

    def __init__(self, name: str, result: dict | None = None, args_schema=None):
        self.name = name
        self.description = f"стаб інструмента {name}"
        self.args_schema = args_schema
        self._result = result
        self.calls: list[dict] = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return [{"type": "text", "text": json.dumps(self._result, ensure_ascii=False)}]


class _StubStructured:
    """Віддає з наперед заданої черги об'єкт, що відповідає схемі."""

    def __init__(self, parent: "StubLLM", schema):
        self._parent = parent
        self._schema = schema

    async def ainvoke(self, messages):
        self._parent.calls.append((self._schema, messages))
        queue = self._parent.responses[self._schema]
        # Останню відповідь повторюємо, якщо викликів більше, ніж заготовок.
        return queue.pop(0) if len(queue) > 1 else queue[0]


class StubLLM:
    """Стаб LLM.

    responses: {Схема: об'єкт | список об'єктів (черга за порядком викликів)}.
    text_response — те, що повертає прямий ainvoke (його використовує
    general-агент, який structured output не застосовує).
    """

    def __init__(self, responses: dict[type, object] | None = None, text_response: str = "ок"):
        self.responses = {
            schema: (value if isinstance(value, list) else [value])
            for schema, value in (responses or {}).items()
        }
        self.text_response = text_response
        self.calls: list[tuple] = []

    def with_structured_output(self, schema):
        return _StubStructured(self, schema)

    def bind_tools(self, tools, **kwargs):
        """create_react_agent прив'язує інструменти до моделі при побудові
        графа. Стаб інструментів не викликає, тому віддає себе ж — цього
        достатньо, щоб граф зібрався."""
        self.bound_tools = list(tools)
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append(("text", messages))
        return AIMessage(content=self.text_response)


class StubReactAgent:
    """Двійник того, що повертає create_react_agent: лише ainvoke."""

    def __init__(self, messages: list):
        self.messages = messages
        self.invocations = 0

    async def ainvoke(self, _input, config=None):
        self.invocations += 1
        return {"messages": self.messages}


def stub_react_agent_with_text(final_text: str) -> StubReactAgent:
    """Конструктор для випадків, коли важливий лише фінальний текст."""
    return StubReactAgent([AIMessage(content=final_text)])


@pytest.fixture
def tymchasovyi_outbox(tmp_path, monkeypatch):
    """Підміняє файл вихідних листів тимчасовим.

    Потрібно тестам HITL: вони перевіряють, що лист справді записався, і не
    мають при цьому забруднювати data/outbox.json проєкту.
    """
    import config
    import mcp_server

    path = tmp_path / "outbox.json"
    monkeypatch.setattr(config, "OUTBOX_PATH", path)
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", path)
    return path


@pytest.fixture(autouse=True)
def chystyi_rate_limiter():
    """Скидає спільний лічильник запитів між тестами.

    Без цього тести, що проганяють граф, поступово вичерпують вікно, і
    останній із них падає не через свою логіку, а через сусідів.
    """
    from guardrails import rate_limiter

    rate_limiter.reset()
    yield
    rate_limiter.reset()
