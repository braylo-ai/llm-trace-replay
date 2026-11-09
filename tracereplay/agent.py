"""Demo multi-step tool agent + a SIMULATED LLM (deterministic policy, no model weights).

Protocol: the LLM replies `CALL <tool> <json-args>` or `FINAL: <answer>`.
The simulated LLM parses the conversation and picks the next action; its
token usage is a whitespace word count (labelled approximate). Swap in any
real chat model with the same `(messages, **params) -> dict` signature.
"""
from __future__ import annotations

import json
import re

FACTS = {
    "paris_population": 2102650, "tokyo_population": 13960000, "lagos_population": 15388000,
    "chicago_population": 2664000, "everest_height_m": 8849, "k2_height_m": 8611,
    "nile_length_km": 6650, "amazon_length_km": 6400,
}
TOOLS = {
    "lookup": lambda a: FACTS.get(a["key"], "unknown"),
    "calc": lambda a: eval(a["expr"], {"__builtins__": {}}, {"round": round}),  # demo only: numeric expressions
}
SYSTEM = "Answer with tools. Reply `CALL <tool> <json>` or `FINAL: <answer>`. Tools: lookup(key), calc(expr)."


def _words(msgs):
    return sum(len(m["content"].split()) for m in msgs)


def simulated_llm(version: str = "v1"):
    def llm(messages, temperature=0.0, model=f"sim-policy-{version}"):
        task = messages[1]["content"]
        keys = re.findall(r"\b([a-z0-9]+_[a-z0-9_]+)\b", task)
        obs = [m["content"] for m in messages[2:] if m["role"] == "tool"]
        if len(obs) < len(keys):
            text = f'CALL lookup {json.dumps({"key": keys[len(obs)]})}'
        elif len(obs) == len(keys):
            vals = obs[: len(keys)]
            op = "-" if "difference" in task else "+" if "sum" in task else "/" if "ratio" in task else "*"
            expr = f" {op} ".join(vals)
            if version == "v2" and op == "/":
                expr = f"round({expr}, 3)"
            text = f'CALL calc {json.dumps({"expr": expr})}'
        else:
            text = f"FINAL: {obs[-1]}"
        return {"text": text, "model": model,
                "usage": {"input_tokens": _words(messages), "output_tokens": len(text.split())}}
    return llm


def run_agent(sess, task: str, max_steps: int = 8) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for _ in range(max_steps):
        reply = sess.llm(messages, temperature=0.0)
        messages.append({"role": "assistant", "content": reply})
        if reply.startswith("FINAL:"):
            return reply[len("FINAL:"):].strip()
        m = re.match(r"CALL (\w+) (.*)", reply, re.S)
        if not m:
            return "error: unparseable reply"
        out = sess.tool(m.group(1), json.loads(m.group(2)))
        messages.append({"role": "tool", "content": str(out)})
    return "error: step budget exhausted"


TASKS = [
    "What is the sum of paris_population and chicago_population?",
    "What is the difference of everest_height_m and k2_height_m?",
    "What is the ratio of nile_length_km and amazon_length_km?",
    "What is the sum of tokyo_population and lagos_population?",
    "What is the ratio of tokyo_population and paris_population?",
    "What is the product of everest_height_m and nile_length_km?",
    "What is the difference of lagos_population and chicago_population?",
    "What is the sum of everest_height_m and k2_height_m and nile_length_km?",
    "What is the ratio of lagos_population and chicago_population?",
    "What is the difference of nile_length_km and amazon_length_km?",
]
