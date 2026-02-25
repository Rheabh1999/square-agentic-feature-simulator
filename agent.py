from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI

from simulator import (
    SimulationAssumptions,
    compute_baseline_metrics,
    detect_fragility_signals,
    simulate_feature_impact,
)
from prompts import SYSTEM_INSTRUCTIONS, build_user_request


# ----------------------------
# Tools schema (Chat Completions format)
# ----------------------------
def _tool_schema() -> List[Dict[str, Any]]:
    """
    Chat Completions tool schema uses:
      tools=[{"type":"function","function":{"name":...,"description":...,"parameters":...}}, ...]
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_baseline",
                "description": "Compute baseline seller metrics from normalized transactions.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_fragility",
                "description": "Detect fragility signals (volatility, concentration, margin pressure) from normalized transactions.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "simulate_feature",
                "description": "Run a feature simulation using default assumptions for the selected feature.",
                "parameters": {
                    "type": "object",
                    "properties": {"feature": {"type": "string"}},
                    "required": ["feature"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_scenarios",
                "description": "Compute a concise comparison view from before/after simulation outputs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "simulation_result": {
                            "type": "object",
                            "description": "The object returned by simulate_feature.",
                        }
                    },
                    "required": ["simulation_result"],
                },
            },
        },
    ]


# ----------------------------
# JSON-safe conversion
# ----------------------------
def _make_json_safe(obj: Any) -> Any:
    """
    Convert common non-JSON-serializable objects (pandas, datetime, numpy) into JSON-safe forms.
    """
    # Python date/time
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    # Pandas Timestamp/Timedelta
    try:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return str(obj)
    except Exception:
        pass

    # Pandas containers
    if isinstance(obj, pd.DataFrame):
        return [_make_json_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return _make_json_safe(obj.to_dict())

    # Nested containers
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_make_json_safe(v) for v in obj]

    # Numpy scalars / datetime64 (often appear via pandas)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.datetime64):
            return str(obj)
    except Exception:
        pass

    return obj


# ----------------------------
# Public entrypoint used by app.py
# ----------------------------
def run_strategy_agent(
    df: pd.DataFrame,
    feature: str,
    persona: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_output_tokens: int = 1100,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Stable tool-calling agent using Chat Completions (works across SDK versions).

    Returns: (memo_text, trace)
    """
    client = OpenAI(api_key=api_key)
    tools = _tool_schema()

    state: Dict[str, Any] = {"feature": feature, "persona": persona}
    trace: List[Dict[str, Any]] = []
    step = 1

    user_msg = build_user_request(feature=feature, persona=persona)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_msg},
    ]

    # Safety valve so we don't loop forever
    for _ in range(12):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        msg = resp.choices[0].message

        # If the model returned normal text and no tool calls, we're done
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            final_text = msg.content or ""
            return final_text, trace

        # Add the assistant message that contains the tool calls
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Execute each tool call and append tool outputs as role="tool"
        for tc in tool_calls:
            fn_name = tc.function.name
            args_raw = tc.function.arguments

            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except Exception:
                args = {}

            trace.append({"step": step, "type": "tool_call", "name": fn_name, "input": _make_json_safe(args)})
            step += 1

            tool_output = _execute_tool(fn_name, args, df, state)
            safe_output = _make_json_safe(tool_output)

            trace.append({"step": step, "type": "tool_output", "name": fn_name, "output": safe_output})
            step += 1

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(safe_output),
                }
            )

    # If we hit the max loops, return whatever we have
    return "Agent stopped early (too many tool-call rounds). Try again with a smaller dataset or fewer steps.", trace


# ----------------------------
# Tool execution (deterministic)
# ----------------------------
def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    df: pd.DataFrame,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    if name == "analyze_baseline":
        baseline = compute_baseline_metrics(df)
        state["baseline"] = baseline
        return baseline

    if name == "detect_fragility":
        fragility = detect_fragility_signals(df)
        state["fragility"] = fragility
        return fragility

    if name == "simulate_feature":
        feat = arguments.get("feature") or state.get("feature", "")
        assumptions = SimulationAssumptions.default_for_feature(feat)
        sim = simulate_feature_impact(df, feat, assumptions)
        state["simulation"] = sim
        return sim

    if name == "compare_scenarios":
        sim = arguments.get("simulation_result") or state.get("simulation") or {}

        before = sim.get("before", {}) if isinstance(sim, dict) else {}
        after = sim.get("after", {}) if isinstance(sim, dict) else {}
        deltas = sim.get("deltas", {}) if isinstance(sim, dict) else {}

        compact = {
            "before": before,
            "after": after,
            "deltas": deltas,
            "key_takeaways": _key_takeaways(before, after, deltas),
        }
        state["comparison"] = compact
        return compact

    return {"error": f"Unknown tool: {name}"}


def _key_takeaways(before: Dict[str, Any], after: Dict[str, Any], deltas: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    def _fmt_pct(x: Any) -> str:
        try:
            return f"{float(x):+.1f}%"
        except Exception:
            return str(x)

    def _fmt_pp(x: Any) -> str:
        try:
            return f"{float(x):+.2f} pp"
        except Exception:
            return str(x)

    if "gross_revenue_pct" in deltas:
        out.append(f"Gross revenue change: {_fmt_pct(deltas['gross_revenue_pct'])}")
    if "net_revenue_pct" in deltas:
        out.append(f"Net revenue change (after fees): {_fmt_pct(deltas['net_revenue_pct'])}")
    if "gross_margin_pp" in deltas:
        out.append(f"Gross margin change: {_fmt_pp(deltas['gross_margin_pp'])}")
    if "orders_pct" in deltas:
        out.append(f"Orders change: {_fmt_pct(deltas['orders_pct'])}")
    if "aov_pct" in deltas:
        out.append(f"Average order value change: {_fmt_pct(deltas['aov_pct'])}")
    if "daily_revenue_cv_delta" in deltas:
        try:
            out.append(f"Volatility change (daily revenue CV): {float(deltas['daily_revenue_cv_delta']):+.2f}")
        except Exception:
            out.append(f"Volatility change (daily revenue CV): {deltas['daily_revenue_cv_delta']}")

    return out