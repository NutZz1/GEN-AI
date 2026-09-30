"""The avatar-design agent: Parse -> Plan+Act (tool calls) -> Verify -> Self-correct -> Explain.

Every stage degrades gracefully with no LLM configured: parsing falls back to text2avatar.parse(),
planning falls back to a small keyword heuristic, and explaining falls back to a templated sentence.
This is what lets `agent/agent.py` be imported and exercised by tests and by evaluate_agent.py without
any API key - see agent/llm.py's `LLMClient.available`.
"""
import json
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

import text2avatar as t2a
from traits import TRAITS as TRAIT_NAMES

from . import config
from .prompts import EXPLAINER_PROMPT_V1, EXPLAINER_PROMPT_VERSION, PARSER_PROMPT_V1, PARSER_PROMPT_VERSION, \
    PLANNER_PROMPT_V1, PLANNER_PROMPT_VERSION
from .schemas import validate_parser_json
from .tools import TOOL_IMPLS, TOOL_SCHEMAS, classify_avatar, generate_from_constraints


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

    resp = llm.chat(system=PARSER_PROMPT_V1, messages=[{"role": "user", "content": user_text}], json_mode=True)
    parsed, err = (None, resp.error) if resp.error else validate_parser_json(resp.text)
    source = "llm"
    if parsed is None:
        retry_msg = (f"Your previous response was invalid: {err}. Respond again with ONLY a valid JSON "
                     f"object matching the schema - no prose, no markdown fences.")
        resp2 = llm.chat(system=PARSER_PROMPT_V1, messages=[
            {"role": "user", "content": user_text}, {"role": "assistant", "content": resp.text or "(no output)"},
            {"role": "user", "content": retry_msg}], json_mode=True)
        parsed, err2 = (None, resp2.error) if resp2.error else validate_parser_json(resp2.text)
        source = "llm_retry"
        if parsed is None:
            cons, understood, unknown = t2a.parse(user_text)
            return ParseResult(cons, understood, unknown, source="rule_fallback", retries=1,
                               error=f"first: {err}; retry: {err2}")
    return ParseResult(parsed.constraints(), parsed.understood, parsed.unknown, source=source,
                       retries=(1 if source == "llm_retry" else 0))


EDIT_TOOLS = ("edit_trait", "edit_traits")
# phrasings that modify the avatar the user is looking at ("make it blonde", "add glasses", "remove the beard",
# "take off the glasses", "now a mustache")
_EDIT_PATTERN = re.compile(r"\b(change|edit|make|give|add|remove|take off|take away|put|turn|switch|swap|set|"
                           r"get rid|lose|drop|delete|shave|instead|now|also|without|no|more|less|dye|paint)\b")
# phrasings that ask for a different avatar, even when one is active
_NEW_PATTERN = re.compile(r"\b(new (avatar|one|person|character)|from scratch|start over|start again|brand new|"
                          r"another (one|avatar)|generate|create)\b")


def _heuristic_plan(ctx, user_text, constraints, active=None):
    """No LLM available: a small deterministic router, good enough to keep the app usable offline.
    `active` is the avatar the user is currently looking at; follow-up edits apply to it (falling back to the
    most recently created avatar), not to whichever candidate happened to be generated last."""
    lower = user_text.lower()
    names = list(ctx.avatars)
    current = active if active in ctx.avatars else (names[-1] if names else None)
    if len(names) >= 2 and any(w in lower for w in ("blend", "mix", "combine")):
        other = next((n for n in reversed(names) if n != current), names[-2])
        return {"name": "blend", "input": {"avatar_a": current, "avatar_b": other, "alpha": 0.5}}
    if len(names) >= 2 and any(w in lower for w in ("breed", "evolve", "kids", "children")):
        other = next((n for n in reversed(names) if n != current), names[-2])
        return {"name": "breed", "input": {"avatar_a": current, "avatar_b": other, "n": 6}}
    if current and _EDIT_PATTERN.search(lower) and not _NEW_PATTERN.search(lower):
        edits = {t: ids for t, ids in constraints.items() if t in TRAIT_NAMES and ids}
        if not edits:
            return None  # an edit we can't map to any trait: keep the avatar rather than replace it with a random one
        return {"name": "edit_traits", "input": {"avatar": current, "edits": edits, "strength": 1.0}}
    return {"name": "generate_from_constraints", "input": {"constraints": constraints}}


def _edit_key(call):
    """(avatar, traits) identifying one edit, so a retry of the same edit can restore what it started from."""
    inp = call["input"]
    if call["name"] == "edit_trait":
        return inp.get("avatar"), (inp.get("trait"),)
    if call["name"] == "edit_traits":
        return inp.get("avatar"), tuple(sorted(inp.get("edits", {})))
    return None


def _prepare_edit(ctx, call, snapshots):
    """Edits overwrite the avatar in place and relabel it, so re-applying the same edit (a stronger retry after a
    failed verification) on the already-edited avatar is a no-op. Remember the state an edit started from and put
    it back before that same edit is tried again."""
    key = _edit_key(call)
    if key is None or key[0] not in ctx.avatars:
        return
    if key in snapshots:
        a = snapshots[key]
        ctx.avatars[key[0]] = {"z": a["z"].copy(), "labels": dict(a["labels"]), "img": a["img"].copy()}
    else:
        a = ctx.avatars[key[0]]
        snapshots[key] = {"z": a["z"].copy(), "labels": dict(a["labels"]), "img": a["img"].copy()}


def verify_avatar(ctx, avatar, constraints):
    predicted = classify_avatar(ctx, avatar)
    if predicted is None:
        return None
    return {t: {"requested": ids, "achieved": predicted[t], "hit": predicted[t] in ids}
            for t, ids in constraints.items() if t in predicted}


def _validate_tool_call(name, args):
    """None if `args` matches the tool's schema, else a human-readable reason. Anthropic's API mostly
    enforces this itself, but smaller local models (Ollama) readily invent argument names outside the
    schema - this catches that before it reaches the Python function and turns into a bare TypeError."""
    schema = next((s for s in TOOL_SCHEMAS if s["name"] == name), None)
    if schema is None:
        return f"there is no tool named {name!r}"
    props = schema["input_schema"]["properties"]
    required = schema["input_schema"].get("required", [])
    missing = [r for r in required if r not in args]
    unknown = [k for k in args if k not in props]
    if missing or unknown:
        return (f"call to {name!r} is missing required argument(s) {missing} and/or uses unrecognised "
                f"argument(s) {unknown}; valid arguments are {list(props)}")
    return None


def _coerce_tool_args(name, args):
    """Some local models serialise object/array/number arguments as JSON strings instead of native
    JSON (observed with llama3.2 via Ollama: {"constraints": "{\"hair\": [5]}"} instead of a nested
    object) - decode those against the tool's declared type before they reach the Python function."""
    schema = next((s for s in TOOL_SCHEMAS if s["name"] == name), None)
    if schema is None:
        return args
    props = schema["input_schema"]["properties"]
    out = dict(args)
    for k, v in list(out.items()):
        want = props.get(k, {}).get("type")
        if not isinstance(v, str):
            continue
        try:
            if want in ("object", "array"):
                out[k] = json.loads(v)
            elif want == "number":
                out[k] = float(v)
            elif want == "integer":
                out[k] = int(v)
        except (json.JSONDecodeError, ValueError):
            pass  # leave it as-is; _run_tool's caller will surface whatever error results
    return out


def _run_tool(ctx, call):
    result = TOOL_IMPLS[call["name"]](ctx, **_coerce_tool_args(call["name"], call["input"]))
    if call["name"] == "breed":
        primary = result["children"][0]["avatar"]
        summary = {"children": [{"avatar": c["avatar"], "labels": c["labels"]} for c in result["children"]]}
    else:
        primary = result["avatar"]
        summary = {"avatar": result["avatar"], "labels": result["labels"]}
    return primary, summary


def plan_act_verify(ctx, llm, user_text, parse: ParseResult, active=None):
    trace, retries, snapshots = [], 0, {}
    plan_source = "llm" if llm.available else "heuristic"
    session_note = f"Avatars already in this session: {list(ctx.avatars) or '(none yet)'}"
    if active in ctx.avatars:
        session_note += (f"\nActive avatar (the one the user is looking at - \"it\", \"this\" and \"the hair\" "
                         f"refer to it): {active}")
    messages = [{"role": "user", "content": f"User request: {user_text}\n\nParsed trait constraints: "
                                            f"{json.dumps(parse.constraints)}\n\n{session_note}"}]
    final_avatar = None
    while True:
        if llm.available:
            resp = llm.chat(system=PLANNER_PROMPT_V1, messages=messages, tools=TOOL_SCHEMAS)
            call = resp.tool_calls[0] if resp.tool_calls else None
            bad_args = _validate_tool_call(call["name"], call["input"]) if call else None
            if bad_args:  # one re-ask, same pattern as the parser's validate-then-retry
                messages.append({"role": "assistant", "content": resp.raw_content})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps({"error": bad_args}),
                     "is_error": True},
                    {"type": "text", "text": "Call the tool again with corrected arguments."}]})
                resp = llm.chat(system=PLANNER_PROMPT_V1, messages=messages, tools=TOOL_SCHEMAS)
                call = resp.tool_calls[0] if resp.tool_calls else None
            tool_call = {"name": call["name"], "input": call["input"]} if call else None
        else:
            tool_call = _heuristic_plan(ctx, user_text, parse.constraints, active) if retries == 0 else None

        if tool_call is None or tool_call["name"] not in TOOL_IMPLS:
            break
        try:
            _prepare_edit(ctx, tool_call, snapshots)
            avatar, summary = _run_tool(ctx, tool_call)
        except Exception as e:
            trace.append(ToolTrace(tool_call["name"], tool_call["input"], {"error": str(e)}, None))
            break
        final_avatar = avatar
        verify = verify_avatar(ctx, avatar, parse.constraints)
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
            if tool_call["name"] in EDIT_TOOLS:
                # heuristic self-correction for an edit: redo it from the avatar's original state, pushed harder
                tool_call = {"name": tool_call["name"],
                             "input": {**tool_call["input"], "strength": 1.0 + 0.5 * retries}}
            else:
                # heuristic self-correction for a generation: re-roll with a fresh random draw
                tool_call = {"name": "generate_from_constraints", "input": {"constraints": parse.constraints}}
            try:
                _prepare_edit(ctx, tool_call, snapshots)
                avatar, summary = _run_tool(ctx, tool_call)
            except Exception as e:
                trace.append(ToolTrace(tool_call["name"], tool_call["input"], {"error": str(e)}, None))
                break
            final_avatar = avatar
            verify = verify_avatar(ctx, avatar, parse.constraints)
            trace.append(ToolTrace(tool_call["name"], tool_call["input"], summary, verify))
            if not any(not v["hit"] for v in (verify or {}).values()):
                break
    return final_avatar, trace, retries, plan_source


def generate_verified_candidate(ctx, parse: ParseResult, residual_strength=0.7, name=None):
    """Generate one avatar from already-parsed constraints and self-correct by re-rolling (fresh
    random residual/unconstrained traits) up to MAX_RETRIES times if a requested trait doesn't verify.
    No planner call needed here - the intent (generate) is already known, unlike plan_act_verify()."""
    name = name or f"cand_{1}"
    retries = 0
    result = generate_from_constraints(ctx, parse.constraints, residual_strength, name=name)
    verify = verify_avatar(ctx, result["avatar"], parse.constraints)
    missed = {t: v for t, v in (verify or {}).items() if not v["hit"]}
    while missed and retries < config.MAX_RETRIES - 1:
        retries += 1
        result = generate_from_constraints(ctx, parse.constraints, residual_strength, name=result["avatar"])
        verify = verify_avatar(ctx, result["avatar"], parse.constraints)
        missed = {t: v for t, v in (verify or {}).items() if not v["hit"]}
    return {"name": result["avatar"], "img": result["img"], "labels": result["labels"],
            "verify": verify, "retries": retries}


def generate_candidates_from_parse(ctx, parse: ParseResult, n=4, residual_strength=0.7):
    """`n` independently-sampled, independently-verified candidates from already-parsed constraints -
    shared by generate_candidates() (text) and the photo-upload path in app.py (photo2avatar.py)."""
    return [generate_verified_candidate(ctx, parse, residual_strength, name=f"cand_{len(ctx.avatars) + 1}")
            for _ in range(n)]


def generate_candidates(ctx, llm, user_text, n=4, residual_strength=0.7):
    """Parse once, then generate `n` candidates - used by the "Describe it" tab's first turn (before
    the user has picked one to keep editing)."""
    parse = parse_request(llm, user_text)
    return parse, generate_candidates_from_parse(ctx, parse, n, residual_strength)


def explain(llm, user_text, parse: ParseResult, trace):
    last = trace[-1] if trace else None
    if last is None or "error" in last.result_summary:
        # A failed/empty trace is not something to hand to an LLM to narrate - asked to explain a
        # failure with no real result to describe, a model will readily fabricate a success story
        # (observed with a local model: it claimed "the classifier verified all traits matched" when
        # no tool had actually run). Be direct instead of risking that.
        if last is None and not llm.available:
            # rule-based planner found nothing to change: say what it does understand instead of a generic error
            return ("I couldn't tell which trait to change, so I left your avatar as it is. Try naming a trait, "
                    "e.g. \"make the hair red\", \"add sunglasses\", \"remove the beard\" or \"make the eyes blue\".")
        reason = last.result_summary.get("error") if last else "couldn't decide what to do with that request"
        return (f"Sorry, I couldn't generate that ({reason}). Try rephrasing, or turn off \"Use LLM "
                f"agent\" in the sidebar to fall back to the rule-based parser.")
    verify = last.verify
    facts = {"request": user_text, "requested_traits": parse.constraints, "understood": parse.understood,
             "unknown_words": parse.unknown, "tool_used": last.tool, "verification": verify}
    if llm.available:
        resp = llm.chat(system=EXPLAINER_PROMPT_V1, messages=[{"role": "user", "content": json.dumps(facts)}])
        if not resp.error and resp.text.strip():
            return resp.text.strip()
    # template fallback
    done = {"blend": "Blended two avatars", "breed": "Bred a litter", "trait_mix": "Mixed traits between two avatars"}
    if last.tool in EDIT_TOOLS:
        bits = [f"Updated your avatar for \"{user_text}\"."]
    else:
        bits = [f"{done.get(last.tool, 'Generated an avatar')} from your request \"{user_text}\"."]
    if verify:
        hits = [t.replace("_", " ") for t, v in verify.items() if v["hit"]]
        misses = [t.replace("_", " ") for t, v in verify.items() if not v["hit"]]
        if hits:
            bits.append(f"Confirmed: {', '.join(hits)}.")
        if misses:
            bits.append(f"Didn't quite land: {', '.join(misses)}.")
    if parse.unknown:
        bits.append(f"Couldn't map: {', '.join(parse.unknown)}.")
    return " ".join(bits)


def run_once(ctx, llm, user_text, log=True, active=None):
    """`active`: name of the avatar the user is currently looking at, so "make it red" edits that one."""
    t0 = time.time()
    parse = parse_request(llm, user_text)
    final_avatar, trace, retries, plan_source = plan_act_verify(ctx, llm, user_text, parse, active)
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
