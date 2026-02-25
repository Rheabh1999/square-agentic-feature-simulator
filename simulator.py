from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "date",
    "order_id",
    "item",
    "category",
    "unit_price",
    "quantity",
    "cogs_per_unit",
    "payment_type",
]


@dataclass
class SimulationAssumptions:
    conversion_uplift_pct: float = 0.0
    aov_uplift_pct: float = 0.0
    processing_fee_delta_pct: float = 0.0
    bnpl_fee_pct: float = 0.0
    instant_payout_adoption_pct: float = 0.0
    instant_payout_fee_pct: float = 0.0

    baseline_processing_fee_pct: float = 2.9
    baseline_processing_fee_fixed: float = 0.30

    @staticmethod
    def default_for_feature(feature: str) -> "SimulationAssumptions":
        a = SimulationAssumptions()
        if feature.startswith("BNPL enabled"):
            a.conversion_uplift_pct = 6.0
            a.aov_uplift_pct = 4.0
            a.bnpl_fee_pct = 3.5
        elif feature.startswith("Processing fee increase"):
            a.processing_fee_delta_pct = 0.5
        elif feature.startswith("AI upsell prompts"):
            a.aov_uplift_pct = 5.0
        elif feature.startswith("Instant payout pricing change"):
            a.instant_payout_adoption_pct = 35.0
            a.instant_payout_fee_pct = 1.75
        return a


def validate_and_normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}\nExpected: {REQUIRED_COLS}")

    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    if out["date"].isna().any():
        raise ValueError("Some `date` values could not be parsed. Use ISO like 2026-02-01.")

    for col in ["unit_price", "quantity", "cogs_per_unit"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"Some `{col}` values are not numeric.")

    out["quantity"] = out["quantity"].astype(int)
    if (out["quantity"] <= 0).any():
        raise ValueError("Quantity must be positive integers.")

    out["line_revenue"] = out["unit_price"] * out["quantity"]
    out["line_cogs"] = out["cogs_per_unit"] * out["quantity"]
    return out


def compute_baseline_metrics(df: pd.DataFrame) -> Dict[str, float]:
    orders = int(df["order_id"].nunique())
    gross_revenue = float(df["line_revenue"].sum())
    gross_cogs = float(df["line_cogs"].sum())
    gross_profit = gross_revenue - gross_cogs
    gross_margin_pct = (gross_profit / gross_revenue * 100.0) if gross_revenue > 0 else 0.0
    aov = gross_revenue / orders if orders > 0 else 0.0

    daily = df.groupby("date")["line_revenue"].sum()
    daily_mean = float(daily.mean()) if len(daily) else 0.0
    daily_std = float(daily.std(ddof=0)) if len(daily) else 0.0
    daily_cv = (daily_std / daily_mean) if daily_mean > 0 else 0.0

    baseline_fee_pct = 2.9 / 100.0
    baseline_fixed = 0.30
    processing_fees = (gross_revenue * baseline_fee_pct) + (orders * baseline_fixed)
    net_revenue = gross_revenue - processing_fees

    return {
        "orders": orders,
        "gross_revenue": gross_revenue,
        "gross_cogs": gross_cogs,
        "gross_profit": gross_profit,
        "gross_margin_pct": float(gross_margin_pct),
        "aov": float(aov),
        "processing_fees": float(processing_fees),
        "net_revenue": float(net_revenue),
        "daily_revenue_cv": float(daily_cv),
    }


def detect_fragility_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple but PM-relevant fragility signals:
    - revenue concentration by item/category
    - day-of-week dependence
    - volatility proxy
    - margin pressure
    """
    base = compute_baseline_metrics(df)

    # Concentration
    item_rev = df.groupby("item")["line_revenue"].sum().sort_values(ascending=False)
    cat_rev = df.groupby("category")["line_revenue"].sum().sort_values(ascending=False)

    total = base["gross_revenue"] if base["gross_revenue"] > 0 else 1.0
    top_item_share = float(item_rev.iloc[0] / total) if len(item_rev) else 0.0
    top_cat_share = float(cat_rev.iloc[0] / total) if len(cat_rev) else 0.0

    # Day-of-week dependence
    d = df.copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    d["dow"] = d["date_dt"].dt.day_name()
    dow_rev = d.groupby("dow")["line_revenue"].sum().sort_values(ascending=False)
    top_dow_share = float(dow_rev.iloc[0] / total) if len(dow_rev) else 0.0

    # Flags
    flags = []
    if base["gross_margin_pct"] < 35:
        flags.append("Low gross margin (proxy) — seller may be sensitive to fee increases.")
    if base["daily_revenue_cv"] > 0.65:
        flags.append("High daily revenue volatility — cashflow risk and operational instability.")
    if top_item_share > 0.45:
        flags.append("Revenue concentrated in a single item — supply/competition risk.")
    if top_dow_share > 0.40:
        flags.append("Revenue concentrated on one day-of-week — schedule/seasonality fragility.")

    return {
        "baseline": base,
        "top_item_share": top_item_share,
        "top_category_share": top_cat_share,
        "top_day_of_week_share": top_dow_share,
        "top_items": item_rev.head(5).to_dict(),
        "top_categories": cat_rev.head(5).to_dict(),
        "flags": flags,
    }


def _apply_orders_uplift(df: pd.DataFrame, uplift_pct: float) -> Tuple[pd.DataFrame, str]:
    if uplift_pct <= 0:
        return df.copy(), "No order uplift applied."

    orders = df["order_id"].unique()
    n_orders = len(orders)
    add_orders = int(round(n_orders * (uplift_pct / 100.0)))
    if add_orders <= 0:
        return df.copy(), "Order uplift rounded to 0 additional orders."

    rng = np.random.default_rng(7)
    chosen = rng.choice(orders, size=min(add_orders, n_orders), replace=(add_orders > n_orders))

    dup = df[df["order_id"].isin(chosen)].copy()
    dup["order_id"] = dup["order_id"].astype(str) + "_SIM" + pd.Series(range(len(dup))).astype(str)

    out = pd.concat([df, dup], ignore_index=True)
    return out, f"Simulated +{uplift_pct:.1f}% orders by duplicating {len(chosen)} orders."


def _apply_aov_uplift(df: pd.DataFrame, uplift_pct: float) -> Tuple[pd.DataFrame, str]:
    if uplift_pct <= 0:
        return df.copy(), "No AOV uplift applied."

    out = df.copy()
    factor = 1.0 + (uplift_pct / 100.0)
    out["unit_price"] = out["unit_price"] * factor
    out["line_revenue"] = out["unit_price"] * out["quantity"]
    return out, f"Applied +{uplift_pct:.1f}% AOV uplift via price factor {factor:.3f}."


def _compute_processing_fees(gross_revenue: float, orders: int, pct: float, fixed: float) -> float:
    return (gross_revenue * (pct / 100.0)) + (orders * fixed)


def simulate_feature_impact(df: pd.DataFrame, feature: str, assumptions: SimulationAssumptions) -> Dict[str, object]:
    before = compute_baseline_metrics(df)
    notes = []
    sim_df = df.copy()

    sim_df, note = _apply_orders_uplift(sim_df, assumptions.conversion_uplift_pct)
    notes.append(note)

    sim_df, note = _apply_aov_uplift(sim_df, assumptions.aov_uplift_pct)
    notes.append(note)

    after = compute_baseline_metrics(sim_df)

    fee_pct = assumptions.baseline_processing_fee_pct + assumptions.processing_fee_delta_pct
    processing_fees = _compute_processing_fees(
        gross_revenue=after["gross_revenue"],
        orders=after["orders"],
        pct=fee_pct,
        fixed=assumptions.baseline_processing_fee_fixed,
    )

    bnpl_fees = 0.0
    if feature.startswith("BNPL enabled") and assumptions.bnpl_fee_pct > 0:
        bnpl_share = 0.35
        bnpl_fees = after["gross_revenue"] * bnpl_share * (assumptions.bnpl_fee_pct / 100.0)
        notes.append(f"BNPL fees modeled as {bnpl_share:.0%} of revenue * {assumptions.bnpl_fee_pct:.2f}%.")

    instant_payout_fees = 0.0
    if feature.startswith("Instant payout pricing change"):
        adoption = max(0.0, min(assumptions.instant_payout_adoption_pct, 100.0)) / 100.0
        instant_payout_fees = after["gross_revenue"] * adoption * (assumptions.instant_payout_fee_pct / 100.0)
        notes.append(f"Instant payout fees modeled as adoption {adoption:.0%} of revenue * {assumptions.instant_payout_fee_pct:.2f}%.")

    total_fees = processing_fees + bnpl_fees + instant_payout_fees
    net_revenue = after["gross_revenue"] - total_fees

    after_adj = dict(after)
    after_adj["processing_fees"] = float(processing_fees)
    after_adj["bnpl_fees"] = float(bnpl_fees)
    after_adj["instant_payout_fees"] = float(instant_payout_fees)
    after_adj["total_fees"] = float(total_fees)
    after_adj["net_revenue"] = float(net_revenue)

    deltas = {
        "gross_revenue_pct": _pct_change(before["gross_revenue"], after_adj["gross_revenue"]),
        "orders_pct": _pct_change(before["orders"], after_adj["orders"]),
        "aov_pct": _pct_change(before["aov"], after_adj["aov"]),
        "gross_margin_pp": after_adj["gross_margin_pct"] - before["gross_margin_pct"],
        "processing_fees_pct": _pct_change(before["processing_fees"], after_adj["processing_fees"]),
        "net_revenue_pct": _pct_change(before["net_revenue"], after_adj["net_revenue"]),
        "daily_revenue_cv_delta": after_adj["daily_revenue_cv"] - before["daily_revenue_cv"],
    }

    ts_before = df.groupby("date")["line_revenue"].sum().rename("gross_revenue_before")
    ts_after = sim_df.groupby("date")["line_revenue"].sum().rename("gross_revenue_after")
    ts = pd.concat([ts_before, ts_after], axis=1).fillna(0.0).reset_index()

    return {"before": before, "after": after_adj, "deltas": deltas, "timeseries": ts, "notes": notes}


def _pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0 if after == 0 else 100.0
    return (after - before) / before * 100.0
