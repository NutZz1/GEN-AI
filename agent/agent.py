"""The avatar-design agent: Parse -> Plan+Act (tool calls) -> Verify -> Self-correct -> Explain.

Every stage degrades gracefully with no LLM configured: parsing falls back to text2avatar.parse(),
planning falls back to a small keyword heuristic, and explaining falls back to a templated sentence.
This is what lets `agent/agent.py` be imported and exercised by tests and by evaluate_agent.py without
any API key - see agent/llm.py's `LLMClient.available`.
"""
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

import text2avatar as t2a
from traits import TRAITS as TRAIT_NAMES

from . import config
from .prompts import EXPLAINER_PROMPT_V1, EXPLAINER_PROMPT_VERSION, PARSER_PROMPT_V1, PARSER_PROMPT_VERSION, \
    PLANNER_PROMPT_V1, PLANNER_PROMPT_VERSION
from .schemas import validate_parser_json
from .tools import TOOL_IMPLS, TOOL_SCHEMAS, classify_avatar


@dataclass
class ParseResult:
    constraints: dict
    understood: list
    unknown: list
    source: str            # "llm" | "llm_retry" | "rule_fallback"
    retries: int = 0
    error: str = ""


@dataclass
class ToolTrace:
    tool: str
    args: dict
    result_summary: dict   # no images - JSON-loggable
    verify: dict | None    # per-trait {requested, achieved, hit}, or None if not verified


@dataclass
class AgentRunResult:
    prompt: str
    parse: ParseResult
    plan_source: str       # "llm" | "heuristic"
    trace: list             # list[ToolTrace]
    final_avatar: str | None
    explanation: str
    latency_ms: float
    llm_backend: str
    retries: int = 0


def parse_request(llm, user_text):
    if not llm.available:
        cons, understood, unknown = t2a.parse(user_text)
        return ParseResult(cons, understood, unknown, source="rule_fallback")

    resp = llm.chat(system=PARSER_PROMPT_V1, messages=[{"role": "user", "content": user_text}])
    parsed, err = (None, resp.error) if resp.error else validate_parser_json(resp.text)
    source = "llm"
    if parsed is None:
        retry_msg = (f"Your previous response was invalid: {err}. Respond again with ONLY a valid JSON "
                     f"object matching the schema - no prose, no markdown fences.")
        resp2 = llm.chat(system=PARSER_PROMPT_V1, messages=[
            {"role": "user", "content": user_text}, {"role": "assistant", "content": resp.text or "(no output)"},
            {"role": "user", "content": retry_msg}])
        parsed, err2 = (None, resp2.error) if resp2.error else validate_parser_json(resp2.text)
        source = "llm_retry"
        if parsed is None:
            cons, understood, unknown = t2a.parse(user_text)
            return ParseResult(cons, understood, unknown, source="rule_fallback", retries=1,
                               error=f"first: {err}; retry: {err2}")
    return ParseResult(parsed.constraints(), parsed.understood, parsed.unknown, source=source,
                       retries=(1 if source == "llm_retry" else 0))


def _heuristic_plan(ctx, user_text, constraints):
    """No LLM available: a small deterministic router, good enough to keep the app usable offline."""
    lower = user_text.lower()
    names = list(ctx.avatars)
    if len(names) >= 2 and any(w in lower for w in ("blend", "mix", "combine")):
        return {"name": "blend", "input": {"avatar_a": names[-2], "avatar_b": names[-1], "alpha": 0.5}}
    if len(names) >= 2 and any(w in lower for w in ("breed", "evolve", "kids", "children")):
        return {"name": "breed", "input": {"avatar_a": names[-2], "avatar_b": names[-1], "n": 6}}
    if names and any(w in lower for w in ("change", "edit", "make the", "make it", "give it", "give this")):
        trait = next((t for t in constraints if t in TRAIT_NAMES), None)
        if trait:
            return {"name": "edit_trait", "input": {"avatar": names[-1], "trait": trait,
                                                     "variant": int(constraints[trait][0]), "strength": 1.0}}
    return {"name": "generate_from_constraints", "input": {"constraints": constraints}}


def _verify(ctx, avatar, constraints):
    predicted = classify_avatar(ctx, avatar)
    if predicted is None:
        return None
    return {t: {"requested": ids, "achieved": predicted[t], "hit": predicted[t] in ids}
            for t, ids in constraints.items() if t in predicted}


def _run_tool(ctx, call):
    result = TOOL_IMPLS[call["name"]](ctx, **call["input"])
    if call["name"] == "breed":
        primary = result["children"][0]["avatar"]
        summary = {"children": [{"avatar": c["avatar"], "labels": c["labels"]} for c in result["children"]]}
    else:
        primary = result["avatar"]
        summary = {"avatar": result["avatar"], "labels": result["labels"]}
    return primary, summary


def plan_act_verify(ctx, llm, user_text, parse: ParseResult):
    trace, retries = [], 0
    plan_source = "llm" if llm.available else "heuristic"
    session_note = f"Avatars already in this session: {list(ctx.avatars) or '(none yet)'}"
    messages = [{"role": "user", "content": f"User request: {user_text}\n\nParsed trait constraints: "
                                            f"{json.dumps(parse.constraints)}\n\n{session_note}"}]
    final_avatar = None
    while True:
        if llm.available:
            resp = llm.chat(system=PLANNER_PROMPT_V1, messages=messages, tools=TOOL_SCHEMAS)
            if resp.tool_calls:
                call = resp.tool_calls[0]
                tool_call = {"name": call["name"], "input": call["input"]}
            else:
                tool_call = None
        else:
            tool_call = _heuristic_plan(ctx, user_text, parse.constraints) if retries == 0 else None

        if tool_call is None or tool_call["name"] not in TOOL_IMPLS:
            break
        try:
            avatar, summary = _run_tool(ctx, tool_call)
        except Exception as e:
            trace.append(ToolTrace(tool_call["name"], tool_call["input"], {"error": str(e)}, None))
            break
        final_avatar = avatar
        verify = _verify(ctx, avatar, parse.constraints)
        trace.append(ToolTrace(tool_call["name"], tool_call["input"], summary, verify))

        missed = {t: v for t, v in (verify or {}).items() if not v["hit"]}
        if not missed or retries >= config.MAX_RETRIES - 1:
            break
        retries += 1
        if llm.available:
            messages.append({"role": "assistant", "content": resp.raw_content})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(summary)},
                {"type": "text", "text": f"verification_failed: {json.dumps(missed)}. Try again with "
                                          f"different arguments to fix the trait(s) that missed."}]})
        else:
            # heuristic self-correction: re-roll generation with a fresh random draw
            tool_call = {"name": "generate_from_constraints", "input": {"constraints": parse.constraints}}
            try:
                avatar, summary = _run_tool(ctx, tool_call)
            except Exception as e:
                trace.append(ToolTrace(tool_call["name"], tool_call["input"], {"error": str(e)}, None))
                break
            final_avatar = avatar
            verify = _verify(ctx, avatar, parse.constraints)
            trace.append(ToolTrace(tool_call["name"], tool_call["input"], summary, verify))
            if not any(not v["hit"] for v in (verify or {}).values()):
                break
    return final_avatar, trace, retries, plan_source


def explain(llm, user_text, parse: ParseResult, trace):
    last = trace[-1] if trace else None
    verify = last.verify if last else None
    facts = {"request": user_text, "requested_traits": parse.constraints, "understood": parse.understood,
             "unknown_words": parse.unknown, "tool_used": last.tool if last else None, "verification": verify}
    if llm.available:
        resp = llm.chat(system=EXPLAINER_PROMPT_V1, messages=[{"role": "user", "content": json.dumps(facts)}])
        if not resp.error and resp.text.strip():
            return resp.text.strip()
    # template fallback
    bits = [f"Generated an avatar from your request \"{user_text}\"."]
    if verify:
        hits = [t for t, v in verify.items() if v["hit"]]
        misses = [t for t, v in verify.items() if not v["hit"]]
        if hits:
            bits.append(f"Confirmed: {', '.join(hits)}.")
        if misses:
            bits.append(f"Didn't quite land: {', '.join(misses)}.")
    if parse.unknown:
        bits.append(f"Couldn't map: {', '.join(parse.unknown)}.")
    return " ".join(bits)


def run_once(ctx, llm, user_text, log=True):
    t0 = time.time()
    parse = parse_request(llm, user_text)
    final_avatar, trace, retries, plan_source = plan_act_verify(ctx, llm, user_text, parse)
    explanation = explain(llm, user_text, parse, trace)
    result = AgentRunResult(prompt=user_text, parse=parse, plan_source=plan_source, trace=trace,
                            final_avatar=final_avatar, explanation=explanation,
                            latency_ms=(time.time() - t0) * 1000, llm_backend=llm.name, retries=retries)
    if log:
        _log_run(result)
    return result


def _log_run(result: AgentRunResult):
    record = {"timestamp": time.time(), "parser_prompt_version": PARSER_PROMPT_VERSION,
              "planner_prompt_version": PLANNER_PROMPT_VERSION, "explainer_prompt_version": EXPLAINER_PROMPT_VERSION,
              **asdict(result)}
    config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.LOG_PATH.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
