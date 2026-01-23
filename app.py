from __future__ import annotations

import argparse

from data import load_dataframe
from agent import build_df_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a CSV dataframe using OpenAI only.")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (overrides CSV_PATH in .env).")
    args = parser.parse_args()

    if args.csv:
        import os
        os.environ["CSV_PATH"] = args.csv

    df = load_dataframe()
    agent = build_df_agent(df)

    print("\n✅ Data loaded.")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            return 0

        try:
            # New LC agents return dicts from invoke
            out = agent.invoke({"input": q})
            if isinstance(out, dict):
                ans = out.get("output") or out.get("final") or str(out)
            else:
                ans = str(out)

            print("\nAssistant:\n" + ans + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
