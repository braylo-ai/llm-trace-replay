import json
import threading
import urllib.request

import pytest

from tracereplay import cli
from tracereplay.agent import TASKS, TOOLS, run_agent, simulated_llm
from tracereplay.recorder import Divergence, Recorder, context_at, record, req_hash, rerun


@pytest.fixture
def rec(tmp_path):
    return Recorder(str(tmp_path / "t.db"))


@pytest.fixture
def run1(rec):
    s, a = record(rec, run_agent, TASKS[0], simulated_llm(), TOOLS)
    return s.run_id, a


def test_record_creates_run_and_steps(rec, run1):
    rid, ans = run1
    runs = rec.store.runs()
    assert runs[0]["run_id"] == rid and runs[0]["final"] == ans == "4766650"
    steps = rec.store.steps(rid)
    assert [s["kind"] for s in steps] == ["llm.call", "tool.call"] * 3 + ["llm.call"]
    assert [s["step"] for s in steps] == list(range(7))


def test_genai_semconv_attributes(rec, run1):
    st = rec.store.steps(run1[0])
    a = st[0]["attributes"]
    assert a["gen_ai.operation.name"] == "chat" and a["gen_ai.request.model"] == "sim-policy-v1"
    assert a["gen_ai.usage.input_tokens"] > 0 and a["gen_ai.usage.output_tokens"] > 0
    assert st[1]["attributes"]["gen_ai.tool.name"] == "lookup"


def test_spans_share_trace_and_parent(rec, run1):
    rows = rec.store.conn.execute("SELECT trace_id, parent_id, name FROM spans WHERE run_id=?", (run1[0],)).fetchall()
    assert len({r[0] for r in rows}) == 1
    root = [r for r in rows if r[2] == "agent.run"]
    assert len(root) == 1 and root[0][1] is None and all(r[1] for r in rows if r[2] != "agent.run")


def test_context_at_has_full_messages(rec, run1):
    c = context_at(rec, run1[0], 4)
    assert c["kind"] == "llm.call" and len(c["request"]["messages"]) == 6
    assert c["response"]["text"].startswith("CALL calc")


def test_full_replay_served_from_cassette(rec, run1):
    calls = []

    def spy(m, **p):
        calls.append(1)
        return simulated_llm()(m, **p)

    s, a = rerun(rec, run1[0], run_agent, TASKS[0], spy, TOOLS, from_step=7)
    assert a == run1[1] and s.served_from_cassette == 7 and calls == []


@pytest.mark.parametrize("k", [0, 3, 6])
def test_rerun_from_step_k(rec, run1, k):
    s, a = rerun(rec, run1[0], run_agent, TASKS[0], simulated_llm(), TOOLS, from_step=k)
    assert a == run1[1] and s.served_from_cassette == k
    sources = [st["attributes"]["replay.source"] for st in rec.store.steps(s.run_id)]
    assert sources == ["cassette"] * k + ["live"] * (7 - k)


def test_rerun_links_parent(rec, run1):
    s, _ = rerun(rec, run1[0], run_agent, TASKS[0], simulated_llm(), TOOLS, from_step=2)
    r = next(r for r in rec.store.runs() if r["run_id"] == s.run_id)
    assert r["parent_run"] == run1[0] and r["from_step"] == 2


def test_rerun_with_new_model_changes_downstream(rec):
    s, a = record(rec, run_agent, TASKS[2], simulated_llm("v1"), TOOLS)
    s2, a2 = rerun(rec, s.run_id, run_agent, TASKS[2], simulated_llm("v2"), TOOLS, from_step=4)
    assert a == "1.0390625" and a2 == "1.039" and s2.served_from_cassette == 4


def test_divergence_raises(rec, run1):
    with pytest.raises(Divergence):
        rerun(rec, run1[0], run_agent, TASKS[0].replace("paris", "tokyo"), simulated_llm(), TOOLS, from_step=7)


def test_req_hash_order_independent():
    assert req_hash({"a": 1, "b": 2}) == req_hash({"b": 2, "a": 1}) != req_hash({"a": 2, "b": 2})


def test_cli_demo_list_verify_rerun(tmp_path, capsys):
    db = str(tmp_path / "c.db")
    assert cli.main(["--db", db, "demo"]) == 0
    capsys.readouterr()
    cli.main(["--db", db, "list"])
    rid = json.loads(capsys.readouterr().out.splitlines()[0])["run_id"]
    assert cli.main(["--db", db, "verify", rid]) == 0
    assert json.loads(capsys.readouterr().out)["final_answer_identical"] is True
    cli.main(["--db", db, "rerun", rid, "--from", "3"])
    assert json.loads(capsys.readouterr().out)["served_from_cassette"] == 3
    cli.main(["--db", db, "show", rid])
    assert "[6] llm.call" in capsys.readouterr().out


def test_web_viewer_api(rec, run1):
    from http.server import ThreadingHTTPServer

    from tracereplay.web import make_handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(rec))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    runs = json.loads(urllib.request.urlopen(base + "/api/runs").read())
    steps = json.loads(urllib.request.urlopen(base + f"/api/steps?run={run1[0]}").read())
    html = urllib.request.urlopen(base + "/").read().decode()
    srv.shutdown()
    assert runs[0]["run_id"] == run1[0] and len(steps) == 7 and "Trace Replay" in html


def test_store_persists_across_processes(tmp_path):
    db = str(tmp_path / "p.db")
    s, a = record(Recorder(db), run_agent, TASKS[1], simulated_llm(), TOOLS)
    again = Recorder(db)
    assert again.store.runs()[0]["final"] == a == "238" and len(again.cassette(s.run_id)) == 7


def test_cassette_response_is_source_of_truth(rec, run1):
    rec.store.conn.execute(
        "UPDATE spans SET attributes=json_set(attributes, '$.\"replay.response\"', '{\"result\": 1}') "
        "WHERE run_id=? AND step=5", (run1[0],))
    rec.store.conn.commit()
    s, a = rerun(rec, run1[0], run_agent, TASKS[0], simulated_llm(), TOOLS, from_step=6)
    assert a == "1"  # replays the recorded (edited) tool output, not a live recomputation
