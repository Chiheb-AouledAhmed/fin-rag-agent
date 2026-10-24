from __future__ import annotations

import sys

from finrag.factory import build_agent


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Question: ")
    r = build_agent().ask(question)
    print(r.answer)
    for c in r.sources:
        print(f"  - [{c.id}] {c.title} / {c.section}")
    print(f"({r.latency_ms:.0f} ms)")


if __name__ == "__main__":
    main()
