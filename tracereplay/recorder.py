"""Recorder + replayer.

An agent talks to the world only through `Session.llm(messages, **params)` and
`Session.tool(name, args)`. Every call becomes an OpenTelemetry span (GenAI
semantic-convention attributes + the full request/response JSON), exported to
SQLite. The same Session can be built from a recorded run ("cassette"): steps
before `from_step` are served from the recording - with a request-hash check
that raises `Divergence` if the agent asks for something different - and steps
from `from_step` on go to the live LLM/tools and are recorded as a new run.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Callable

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .store import SQLiteSpanExporter, Store

LLMFn = Callable[
    ..., dict
]  # (messages, **params) -> {"text", "usage": {"input_tokens","output_tokens"}, "model"}
ToolFn = Callable[[dict], Any]


class Divergence(RuntimeError):
    pass


def req_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


class Recorder:
    def __init__(self, db_path: str = "traces.db"):
        self.store = Store(db_path)
        self.provider = TracerProvider(
            resource=Resource.create({"service.name": "llm-trace-replay"})
        )
        self.provider.add_span_processor(
            SimpleSpanProcessor(SQLiteSpanExporter(self.store))
        )
        self.tracer = self.provider.get_tracer("tracereplay")

    def session(
        self,
        task: str,
        llm: LLMFn,
        tools: dict[str, ToolFn],
        cassette: list[dict] | None = None,
        from_step: int | None = None,
        parent_run: str | None = None,
    ) -> "Session":
        return Session(self, task, llm, tools, cassette, from_step, parent_run)

    def cassette(self, run_id: str) -> list[dict]:
        return self.store.steps(run_id)


class Session:
    def __init__(
        self, rec: Recorder, task, llm, tools, cassette, from_step, parent_run
    ):
        self.rec, self.task, self.live_llm, self.live_tools = rec, task, llm, tools
        self.cassette = cassette or []
        self.from_step = (
            from_step
            if from_step is not None
            else (len(self.cassette) if cassette else 0)
        )
        self.parent_run = parent_run
        self.run_id = uuid.uuid4().hex[:12]
        self.step = 0
        self.served_from_cassette = 0

    # ---- run lifecycle ----
    def __enter__(self):
        self._cm = self.rec.tracer.start_as_current_span("agent.run")
        self._root = self._cm.__enter__()
        self._root.set_attribute("agent.run_id", self.run_id)
        self._root.set_attribute("agent.task", self.task)
        if self.parent_run:
            self._root.set_attribute("replay.parent_run", self.parent_run)
            self._root.set_attribute("replay.from_step", self.from_step)
        return self

    def finish(self, answer: str):
        self._root.set_attribute("agent.final_answer", answer)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    # ---- recorded / replayed calls ----
    def _recorded(self, kind: str, request: dict):
        if self.step >= self.from_step or self.step >= len(self.cassette):
            return None
        rec = self.cassette[self.step]
        if rec["kind"] != kind or req_hash(rec["request"]) != req_hash(request):
            raise Divergence(
                f"step {self.step}: recorded {rec['kind']} request differs from replayed {kind} request"
            )
        self.served_from_cassette += 1
        return rec["response"]

    def _span(self, kind: str, request: dict, attrs: dict, live: Callable[[], Any], post=None):
        with self.rec.tracer.start_as_current_span(kind) as sp:
            sp.set_attribute("agent.run_id", self.run_id)
            sp.set_attribute("replay.step", self.step)
            sp.set_attribute("replay.request", json.dumps(request))
            sp.set_attribute("replay.request_hash", req_hash(request))
            for k, v in attrs.items():
                sp.set_attribute(k, v)
            resp = self._recorded(kind, request)
            sp.set_attribute(
                "replay.source", "cassette" if resp is not None else "live"
            )
            if resp is None:
                resp = live()
            sp.set_attribute("replay.response", json.dumps(resp))
            if post:
                post(sp, resp)
            self.step += 1
            return sp, resp

    def llm(self, messages: list[dict], **params) -> str:
        request = {"messages": messages, "params": params}

        def post(sp, resp):
            sp.set_attribute("gen_ai.request.model", resp.get("model", "unknown"))
            sp.set_attribute("gen_ai.usage.input_tokens", resp["usage"]["input_tokens"])
            sp.set_attribute("gen_ai.usage.output_tokens", resp["usage"]["output_tokens"])

        _, resp = self._span("llm.call", request, {"gen_ai.operation.name": "chat"},
                             lambda: self.live_llm(messages, **params), post)
        return resp["text"]

    def tool(self, name: str, args: dict):
        request = {"tool": name, "args": args}
        _, resp = self._span(
            "tool.call",
            request,
            {"gen_ai.tool.name": name},
            lambda: {"result": self.live_tools[name](args)},
        )
        return resp["result"]


def context_at(rec: Recorder, run_id: str, step: int) -> dict:
    """Exact recorded request (full message list for LLM steps) at a step."""
    s = rec.cassette(run_id)[step]
    return {"kind": s["kind"], "request": s["request"], "response": s["response"]}


def rerun(
    rec: Recorder,
    run_id: str,
    agent: Callable,
    task: str,
    llm: LLMFn,
    tools: dict,
    from_step: int,
    **kw,
):
    """Re-execute `agent` reusing the recording up to `from_step`, live afterwards. Returns (session, answer)."""
    sess = rec.session(
        task,
        llm,
        tools,
        cassette=rec.cassette(run_id),
        from_step=from_step,
        parent_run=run_id,
    )
    with sess:
        ans = agent(sess, task, **kw)
        sess.finish(ans)
    return sess, ans


def record(rec: Recorder, agent: Callable, task: str, llm: LLMFn, tools: dict, **kw):
    sess = rec.session(task, llm, tools)
    t0 = time.perf_counter()
    with sess:
        ans = agent(sess, task, **kw)
        sess.finish(ans)
    sess.seconds = time.perf_counter() - t0
    return sess, ans
