"""STEP 6 - GENERATE: give the LLM the retrieved chunks and ask it to answer.

This is the "Augmented Generation" part. We use LangChain here for prompting:
  * ChatPromptTemplate  - a reusable prompt with {context} and {question} slots
  * ChatOpenAI          - the chat model
  * StrOutputParser     - pulls the plain string out of the model's reply
They are joined with `|` into a "chain" (LangChain Expression Language).

The system prompt is what keeps the model *grounded*: answer only from the
context, say "I don't know" otherwise, and cite sections.
"""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
# Low temperature = less random, more faithful to the context. Some newer models
# only accept their default temperature, so we only send it when it's set in .env.
CHAT_TEMPERATURE = os.getenv("CHAT_TEMPERATURE", "").strip()

SYSTEM = """You are an assistant that answers questions about a person's resume.
Use ONLY the resume excerpts in the context below.
- If the answer is not in the context, say: "I don't know based on the resume."
- Do not invent employers, dates, skills, or numbers.
- Cite the section(s) you used in square brackets, e.g. [Skills].
- Be concise.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{question}"),
])

_chain = None


def chain():
    global _chain
    if _chain is None:
        kwargs = {"model": CHAT_MODEL}
        if CHAT_TEMPERATURE:
            kwargs["temperature"] = float(CHAT_TEMPERATURE)
        _chain = prompt | ChatOpenAI(**kwargs) | StrOutputParser()
    return _chain


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant excerpts found)"
    return "\n\n---\n\n".join(c["text"] for c in chunks)


def answer(question: str, chunks: list[dict]) -> str:
    return chain().invoke({"context": format_context(chunks), "question": question})


if __name__ == "__main__":
    # Show the fully rendered prompt so you can see exactly what the LLM receives.
    demo = [{"text": "[Skills]\nPython, SQL, Docker"}]
    for m in prompt.format_messages(context=format_context(demo), question="What languages do I know?"):
        print(f"=== {m.type} ===\n{m.content}\n")
