"""Replay fidelity + overhead on the 10 demo tasks (SIMULATED LLM policy, real OTel/SQLite path).

1. record each task live
2. verify: full re-run from the cassette -> identical final answer, every step served from cassette
3. rerun from every step k with the SAME policy -> identical answer (exact context reuse)
4. rerun from the calc-request step with policy v2 (rounds ratios) -> answers change only on ratio tasks
5. divergence: a cassette whose tool response is tampered is still served, but a changed REQUEST raises
6. overhead: recorded vs bare agent loop wall time
"""
import json
import statistics
import tempfile
import time
from pathlib import Path

from tracereplay.agent import TASKS, TOOLS, run_agent, simulated_llm
from tracereplay.recorder import Divergence, Recorder, record, rerun


class Bare:
    def __init__(self, llm):
        self.l = llm

    def llm(self, m, **p):
        return self.l(m, **p)["text"]

    def tool(self, n, a):
        return TOOLS[n](a)


def main():
    db = Path(tempfile.mkdtemp()) / "eval.db"
    rec = Recorder(str(db))
    out = {"tasks": len(TASKS), "llm": "simulated deterministic policy (no model weights)"}
    runs = []
    for t in TASKS:
        s, a = record(rec, run_agent, t, simulated_llm("v1"), TOOLS)
        runs.append((s.run_id, t, a, len(rec.cassette(s.run_id))))
    out["steps_recorded"] = sum(r[3] for r in runs)
    v_ok = 0
    for rid, t, a, n in runs:
        s, a2 = rerun(rec, rid, run_agent, t, simulated_llm("v1"), TOOLS, from_step=n)
        v_ok += (a2 == a and s.served_from_cassette == n)
    out["full_replay_identical"] = f"{v_ok}/{len(runs)}"
    k_total = k_ok = 0
    for rid, t, a, n in runs:
        for k in range(n):
            s, a2 = rerun(rec, rid, run_agent, t, simulated_llm("v1"), TOOLS, from_step=k)
            k_total += 1
            k_ok += (a2 == a and s.served_from_cassette == k)
    out["rerun_from_every_step_identical"] = f"{k_ok}/{k_total}"
    changed = []
    for rid, t, a, n in runs:
        steps = rec.cassette(rid)
        k = next(i for i, st in enumerate(steps) if st["kind"] == "llm.call" and "calc" in st["response"]["text"])
        s, a2 = rerun(rec, rid, run_agent, t, simulated_llm("v2"), TOOLS, from_step=k)
        if a2 != a:
            changed.append({"task": t, "before": a, "after": a2, "reused_steps": s.served_from_cassette})
    out["v2_rerun_changed_answers"] = changed
    div = edited = 0
    for rid, t, a, n in runs:
        if "sum" not in t:
            continue
        edited += 1
        try:
            rerun(rec, rid, run_agent, t.replace("sum", "product"), simulated_llm("v1"), TOOLS, from_step=n)
        except Divergence:
            div += 1
    out["divergence_detected_on_edited_tasks"] = f"{div}/{edited} (task text sum->product, replayed against original cassette)"
    rec_t, bare_t = [], []
    for _ in range(20):
        for t in TASKS:
            t0 = time.perf_counter(); run_agent(Bare(simulated_llm("v1")), t); bare_t.append(time.perf_counter() - t0)
            s, _ = record(rec, run_agent, t, simulated_llm("v1"), TOOLS); rec_t.append(s.seconds)
    out["mean_ms_per_run_bare"] = round(statistics.mean(bare_t) * 1e3, 3)
    out["mean_ms_per_run_recorded"] = round(statistics.mean(rec_t) * 1e3, 3)
    out["mean_steps_per_run"] = round(out["steps_recorded"] / len(runs), 2)
    out["db_bytes_after_eval"] = db.stat().st_size
    out["db_runs_after_eval"] = len(rec.store.runs())
    Path(__file__).resolve().parents[1].joinpath("results", "replay_eval.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
