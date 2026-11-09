"""Minimal web viewer (stdlib http.server): run list -> step timeline -> full context."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html><meta charset=utf-8><title>Trace Replay</title>
<style>body{font:14px system-ui;margin:16px;max-width:1100px}pre{background:#f4f4f4;padding:8px;white-space:pre-wrap}
li{cursor:pointer;margin:4px 0}.s{border-left:3px solid #888;padding-left:8px;margin:8px 0}
button{margin:0 4px}</style>
<h1>Trace Replay</h1><div style=display:flex;gap:24px><ul id=runs style=width:40%></ul><div id=view style=flex:1></div></div>
<script>
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let steps=[],i=0;
fetch('api/runs').then(r=>r.json()).then(rs=>{runs.innerHTML=rs.map(r=>`<li onclick="load('${r.run_id}')"><b>${r.run_id}</b> ${r.parent_run?'(rerun of '+r.parent_run+' @'+r.from_step+') ':''}${esc(r.task)} &rarr; ${esc(r.final)}</li>`).join('')});
function load(id){fetch('api/steps?run='+id).then(r=>r.json()).then(s=>{steps=s;i=0;draw()})}
function draw(){const s=steps[i];if(!s)return;view.innerHTML=`<div><button onclick="i=Math.max(0,i-1);draw()">&larr; prev</button>step ${i+1}/${steps.length}<button onclick="i=Math.min(steps.length-1,i+1);draw()">next &rarr;</button></div>
<div class=s><b>${s.kind}</b> ${s.ms} ms, source=${s.attributes['replay.source']}, tokens in/out=${s.attributes['gen_ai.usage.input_tokens']??'-'}/${s.attributes['gen_ai.usage.output_tokens']??'-'}
<h4>Request (exact recorded context)</h4><pre>${esc(JSON.stringify(s.request,null,2))}</pre><h4>Response</h4><pre>${esc(JSON.stringify(s.response,null,2))}</pre></div>`}
</script>"""


def make_handler(rec):
    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype="application/json"):
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/runs":
                self._send(json.dumps(rec.store.runs()))
            elif u.path == "/api/steps":
                self._send(json.dumps(rec.store.steps(parse_qs(u.query)["run"][0])))
            else:
                self._send(PAGE, "text/html")

        def log_message(self, *a):
            pass
    return H


def serve(rec, port=8765):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(rec))
    print(f"viewer on http://127.0.0.1:{port}")
    srv.serve_forever()
