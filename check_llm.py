"""Is the LLM connection working, and is it fast enough for a live demo?

    python check_llm.py

Reads the same .env / environment the app does, builds the configured backend (Anthropic or Ollama), then runs
the three calls the agent makes - parse, plan (tool call), explain - and prints what came back and how long each
took. Exit code 0 only if all three worked, so it is safe to use in a script.
"""
import json
import os
import sys
import time

from agent import agent as agent_core
from agent import config
from agent import llm as agent_llm
from agent.prompts import EXPLAINER_PROMPT_V1, PLANNER_PROMPT_V1
from agent.tools import TOOL_SCHEMAS


def main():
    client = agent_llm.get_client()
    provider = os.environ.get("AVATARBLEND_LLM_PROVIDER", "anthropic")
    where = os.environ.get("OLLAMA_HOST", config.OLLAMA_HOST) if provider == "ollama" else "api.anthropic.com"
    print(f"provider: {provider}   target: {where}")
    if not client.available:
        print(f"NOT CONNECTED: {client.reason}")
        return 1
    print(f"connected: {client.name} / {getattr(client, 'model', '?')}")
    ok = True

    prompt = "romba curly red hair venum, blue eyes, no glasses"
    t0 = time.time()
    parse = agent_core.parse_request(client, prompt)
    dt = time.time() - t0
    good = parse.source in ("llm", "llm_retry")
    ok &= good
    print(f"\n1. parse   {dt:5.1f}s  source={parse.source}  {'OK' if good else 'FELL BACK TO RULES'}")
    print(f"   {prompt!r} -> {parse.constraints}")
    if parse.error:
        print(f"   problem: {parse.error[:300]}")

    t0 = time.time()
    resp = client.chat(system=PLANNER_PROMPT_V1, tools=TOOL_SCHEMAS, messages=[{"role": "user", "content": (
        "User request: make an avatar with curly red hair\n\nParsed trait constraints: "
        f"{json.dumps(parse.constraints)}\n\nAvatars already in this session: (none yet)")}])
    dt = time.time() - t0
    call = resp.tool_calls[0] if resp.tool_calls else None
    bad = agent_core._validate_tool_call(call["name"], call["input"]) if call else "no tool call was made"
    good = call is not None and bad is None
    ok &= good
    print(f"\n2. plan    {dt:5.1f}s  {'OK' if good else 'PROBLEM'}  tool={call['name'] if call else None}")
    if resp.error or bad:
        print(f"   problem: {resp.error or bad}")

    t0 = time.time()
    resp = client.chat(system=EXPLAINER_PROMPT_V1, messages=[{"role": "user", "content": json.dumps({
        "request": prompt, "requested_traits": parse.constraints, "understood": parse.understood,
        "unknown_words": parse.unknown, "tool_used": "generate_from_constraints", "verification": None})}])
    dt = time.time() - t0
    good = bool(resp.text.strip()) and not resp.error
    ok &= good
    print(f"\n3. explain {dt:5.1f}s  {'OK' if good else 'PROBLEM'}")
    print(f"   {resp.text.strip()[:300] if good else resp.error}")

    print("\nRESULT:", "all three stages work - ready for the demo" if ok else
          "something failed - the app will still run, using the rule-based fallback for the failed stage(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
