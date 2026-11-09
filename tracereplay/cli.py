"""tracereplay CLI.

  tracereplay demo   [--db traces.db]              record the 10 demo tasks
  tracereplay list   [--db traces.db]
  tracereplay show   RUN_ID                        step-by-step replay in the terminal
  tracereplay step   RUN_ID N                      exact recorded context at step N
  tracereplay verify RUN_ID                        re-run fully from the cassette; must reproduce
  tracereplay rerun  RUN_ID --from N [--llm v2]    reuse steps < N, run live from N
  tracereplay serve  [--port 8765]                 web viewer
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import TASKS, TOOLS, run_agent, simulated_llm
from .recorder import Recorder, context_at, record, rerun


def _task_of(rec, run_id):
    return next(r["task"] for r in rec.store.runs() if r["run_id"] == run_id)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tracereplay")
    ap.add_argument("--db", default="traces.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("list")
    s = sub.add_parser("show"); s.add_argument("run_id")
    s = sub.add_parser("step"); s.add_argument("run_id"); s.add_argument("n", type=int)
    s = sub.add_parser("verify"); s.add_argument("run_id")
    s = sub.add_parser("rerun"); s.add_argument("run_id"); s.add_argument("--from", dest="frm", type=int, required=True)
    s.add_argument("--llm", default="v1")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    rec = Recorder(a.db)
    if a.cmd == "demo":
        for t in TASKS:
            sess, ans = record(rec, run_agent, t, simulated_llm("v1"), TOOLS)
            print(sess.run_id, ans, "|", t)
    elif a.cmd == "list":
        for r in rec.store.runs():
            print(json.dumps(r))
    elif a.cmd == "show":
        for st in rec.store.steps(a.run_id):
            req = st["request"]
            what = req["messages"][-1]["content"][:60] if st["kind"] == "llm.call" else f"{req['tool']}({req['args']})"
            out = st["response"].get("text") if st["kind"] == "llm.call" else st["response"]["result"]
            tok = st["attributes"].get("gen_ai.usage.input_tokens")
            print(f"[{st['step']}] {st['kind']:<9} {st['ms']:>7}ms  in={what!r}  out={out!r}" +
                  (f"  tokens_in={tok}" if tok is not None else ""))
    elif a.cmd == "step":
        print(json.dumps(context_at(rec, a.run_id, a.n), indent=2))
    elif a.cmd == "verify":
        orig = rec.store.steps(a.run_id)
        sess, ans = rerun(rec, a.run_id, run_agent, _task_of(rec, a.run_id), simulated_llm("v1"), TOOLS,
                          from_step=len(orig))
        same = ans == next(r["final"] for r in rec.store.runs() if r["run_id"] == a.run_id)
        print(json.dumps({"replay_run": sess.run_id, "served_from_cassette": sess.served_from_cassette,
                          "steps": len(orig), "final_answer_identical": same}))
        return 0 if same else 1
    elif a.cmd == "rerun":
        sess, ans = rerun(rec, a.run_id, run_agent, _task_of(rec, a.run_id), simulated_llm(a.llm), TOOLS,
                          from_step=a.frm)
        print(json.dumps({"new_run": sess.run_id, "from_step": a.frm, "served_from_cassette": sess.served_from_cassette,
                          "final_answer": ans}))
    elif a.cmd == "serve":
        from .web import serve
        serve(rec, a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
