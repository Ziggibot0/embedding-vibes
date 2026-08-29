"""Renderer — canonical step text -> N synthetic harness renderings.

The Barlow Twins pairs for exp9. Same action, different format -> same 8-dim
point. Formats are deliberately varied along the axes real harnesses differ:
wrapper syntax, role markers, metadata placement, field ordering, verbosity.

IMPORTANT: renderers are pure text transforms of the canonical step. They add
no outcome information (no success/fail leakage is possible from a renderer).
"""
import re
import json as _json


def _strip_canonical(t: str) -> str:
    """Remove OUR canonical markers ([tool_call] ...) so renderers re-add
    their own; keeps the pair factory honest (renderers differ, content same)."""
    m = re.match(r"^\[tool_call\]\s*([A-Za-z0-9_.\-]+)\s*:\s*(.*)$", t, re.S)
    if m:
        return ("TOOL", m.group(1), m.group(2))
    return ("TEXT", None, t)


def render_chat(t: str) -> str:
    kind, name, body = _strip_canonical(t)
    if kind == "TOOL":
        return f"user: run {name}\nassistant: <tool>{body}</tool>"
    return f"assistant: {body}"


def render_xml(t: str) -> str:
    kind, name, body = _strip_canonical(t)
    if kind == "TOOL":
        return f'<action type="tool" name="{name}">{body}</action>'
    return f"<thought>{body}</thought>"


def render_json(t: str) -> str:
    kind, name, body = _strip_canonical(t)
    if kind == "TOOL":
        return _json.dumps({"role": "tool", "name": name, "args": body})
    return _json.dumps({"role": "assistant", "text": body})


def render_otel(t: str) -> str:
    kind, name, body = _strip_canonical(t)
    if kind == "TOOL":
        return (f"gen_ai.tool.name={name} gen_ai.output.type=tool "
                f"payload<<{body}>>")
    return f"gen_ai.output.messages=text payload<<{body}>>"


def render_terse(t: str) -> str:
    kind, name, body = _strip_canonical(t)
    if kind == "TOOL":
        # aggressive truncation + no name: formatting-blur stress test
        return f"-> {body[:200]}"
    return body[:200]


RENDERERS = {
    "chat": render_chat,
    "xml": render_xml,
    "json": render_json,
    "otel": render_otel,
    "terse": render_terse,
}

if __name__ == "__main__":
    demo = ['[tool_call] mcp__env__read_file: {"path": "/etc/hosts"}',
            "I will inspect the file first, then edit it."]
    for name, fn in RENDERERS.items():
        print(f"--- {name}")
        for d in demo:
            print("  ", fn(d)[:110])