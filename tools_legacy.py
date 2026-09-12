"""Reused-компоненти з ДЗ1 і ДЗ2: Pydantic-tools і Agentic RAG на ChromaDB.

Перевикористано з ДЗ2 (knowledge.py) і переорієнтовано на HR-домен. База
знань — не обов'язковий етап кожного запиту, а звичайний інструмент: чи йти в
неї, вирішує сама модель, читаючи докстрінг search_hr_kb. Тому той докстрінг
написаний як інструкція для моделі, а не для людини.
"""

import json
import os

from langchain_core.tools import tool

from config import CHROMA_PATH, COLLECTION_NAME, KB_DIR
from schemas import KbSearchArgs


def _ok(data: dict) -> str:
    return json.dumps({"status": "ok", "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "error": message}, ensure_ascii=False)


def load_documents() -> list[dict]:
    """Прочитати документи з data/kb/: один файл — один документ.

    На чанки не ділимо: документи короткі й цілісні, а сплітер розрізав би
    правило посередині. README.md — довідка для людини, не документ бази.
    """
    documents = []
    for path in sorted(KB_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        first_line = text.splitlines()[0] if text else ""
        title = first_line[2:].strip() if first_line.startswith("# ") else path.name
        documents.append({"source": path.name, "title": title, "text": text})
    return documents


def build_collection(path: str, embedding_function, collection_name: str):
    """Створити або відкрити персистентну колекцію ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )


def index_knowledge_base(collection) -> int:
    """Проіндексувати документи. Повертає кількість документів у колекції.

    upsert за іменем файлу як id: повторний запуск не дублює документи, а
    зміна тексту потрапляє в колекцію.
    """
    documents = load_documents()
    collection.upsert(
        ids=[doc["source"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[{"source": doc["source"], "title": doc["title"]} for doc in documents],
    )
    return collection.count()


def search_documents(collection, query: str, top_k: int) -> list[dict]:
    """Знайти найрелевантніші документи. Повертає список {source, title, text}."""
    response = collection.query(query_texts=[query], n_results=top_k)
    return [
        {"source": meta["source"], "title": meta.get("title", meta["source"]), "text": text}
        for text, meta in zip(response["documents"][0], response["metadatas"][0])
    ]


_default_collection = None


def default_collection():
    """Колекція для роботи агента: персистентна, зі справжніми ембедінгами.

    Модель ембедінгів беремо з того самого провайдера, що й LLM: дефолтна
    англомовна модель ChromaDB на українських документах працює помітно гірше.
    """
    global _default_collection
    if _default_collection is None:
        from chromadb.utils import embedding_functions

        _default_collection = build_collection(
            path=CHROMA_PATH,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                api_base=os.environ.get("OPENAI_BASE_URL") or None,
            ),
            collection_name=COLLECTION_NAME,
        )
        index_knowledge_base(_default_collection)
    return _default_collection


@tool("search_hr_kb", args_schema=KbSearchArgs)
def search_hr_kb(query: str, top_k: int = 3) -> str:
    """Знайти правило у внутрішній базі HR-політик компанії.

    База містить: політику скринінгу з порогами вердиктів, рівні seniority та
    очікування від них, строки відповіді кандидату (SLA), правила відмови й
    перелік допустимих і недопустимих причин, роботу з персональними даними,
    антидискримінаційні вимоги, формулу підрахунку балу та етапи процесу після
    скринінгу.

    Звертайся сюди щоразу, коли питання стосується того, як МАЄ бути за
    політикою: чи можна відмовити без пояснення, за скільки днів відповісти,
    що означає вердикт maybe, які дані не можна писати в листі. Інструменти
    fetch_* цих відповідей не дають — вони показують лише факти про конкретного
    кандидата чи вакансію.

    Приклад: search_hr_kb(query="чи можна відмовити без причини", top_k=3)

    Args:
        query: Питання до бази, від 3 до 500 символів.
        top_k: Скільки документів повернути, від 1 до 10. За замовчуванням 3.

    Returns:
        JSON {"status": "ok", "data": {"query": ..., "results": [{source,
        title, text}]}} або {"status": "error", "error": "..."}.
    """
    try:
        args = KbSearchArgs(query=query, top_k=top_k)
    except Exception as exc:  # ValidationError від Pydantic
        return _error(f"помилка валідації аргументів: {exc}")

    try:
        results = search_documents(default_collection(), args.query, args.top_k)
    except Exception as exc:  # мережа, ключ, пошкоджене сховище
        return _error(f"пошук у базі знань не вдався: {exc}")

    if not results:
        return _error(f"за запитом «{args.query}» у базі знань нічого не знайдено")
    return _ok({"query": args.query, "results": results})
