"""Chat with your resume: retrieve -> generate, in a loop.

Usage:
  python ask.py                       # interactive
  python ask.py "Where did I study?"  # one question
  python ask.py --hide-context        # don't print retrieved chunks
  python ask.py --no-rag              # ask the LLM WITHOUT context (compare!)
"""
import argparse

from rag.generate import answer
from rag.retrieve import search


def ask(question: str, show_context: bool, use_rag: bool) -> None:
    chunks = search(question) if use_rag else []
    if show_context and use_rag:
        print("\n  retrieved chunks:")
        for c in chunks or []:
            body = c["text"].split("\n", 1)[-1]  # drop the "[Section]" prefix line
            preview = body.replace("\n", " ")[:90]
            print(f"   {c['score']:.3f}  [{c['section']}] {preview}...")
        if not chunks:
            print("   (none above the minimum score)")
    print(f"\n{answer(question, chunks)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--hide-context", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()
    show, use_rag = not args.hide_context, not args.no_rag

    if args.question:
        ask(" ".join(args.question), show, use_rag)
        return

    print("Ask about your resume (empty line or 'exit' to quit).")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        ask(q, show, use_rag)


if __name__ == "__main__":
    main()
