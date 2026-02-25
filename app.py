import os
import pandas as pd
import streamlit as st

from simulator import validate_and_normalize_transactions
from agent import run_strategy_agent


# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Feature Impact Simulator",
    page_icon="🧠",
    layout="wide",
)


# ------------------------------------------------
# Light, stable styling (no fragile HTML wrappers)
# ------------------------------------------------
st.markdown(
    """
<style>
/* Overall width & padding */
.block-container { max-width: 1100px; padding-top: 1.8rem; padding-bottom: 2rem; }

/* Make headings feel more premium */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Make Streamlit expanders + inputs slightly rounder */
div[data-testid="stExpander"] { border-radius: 14px; }
section[data-testid="stFileUploaderDropzone"] { border-radius: 14px; }

/* Primary button */
.stButton > button[kind="primary"] {
  border-radius: 12px;
  padding: 0.7rem 1.1rem;
  font-weight: 650;
}

/* Sidebar readability */
.sidebar-note {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 12px;
  padding: 12px 12px;
  background: rgba(49, 51, 63, 0.03);
  font-size: 0.92rem;
  line-height: 1.35;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------
# Sidebar: Clear, descriptive settings
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scenario & AI Settings")

    st.markdown(
        """
<div class="sidebar-note">
<b>What this panel controls</b><br/>
1) Which product change (“feature”) you are evaluating<br/>
2) How the AI writes the recommendation memo<br/><br/>
If you’re unsure, leave defaults as-is.
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("📦 Feature to evaluate")
    feature = st.selectbox(
        "Select a Square product change",
        [
            "BNPL enabled (Buy Now, Pay Later)",
            "Processing fee increase",
            "AI upsell prompts",
            "Instant payout pricing change",
        ],
    )

    st.caption("What each option means:")
    st.markdown(
        """
- **BNPL (Buy Now, Pay Later)**: Customers split payments → can increase conversion/order size, but adds fees and risk.
- **Processing fee increase**: Raises take-rate → improves platform revenue, may reduce seller margins and increase churn risk.
- **AI upsell prompts**: Suggests add-ons at checkout → may increase average order value, could impact customer experience.
- **Instant payout pricing change**: Changes cost of immediate payouts → affects seller cashflow and working capital behavior.
"""
    )

    st.divider()

    st.subheader("🏪 Seller context (optional)")
    persona = st.text_area(
        "Describe the seller",
        value="A 1-location coffee shop with 2 employees in an urban neighborhood.",
        height=90,
    )
    st.caption(
        "Why this matters: transaction data shows numbers, but not context. "
        "This helps the AI reason about operational constraints and behavioral effects."
    )

    st.divider()

    with st.expander("🧠 Advanced AI controls (optional)", expanded=False):
        model = st.text_input("Model", value="gpt-4o-mini")
        st.caption(
            "The model is the AI system that writes the memo. Different models trade off speed, cost, and reasoning depth."
        )

        temperature = st.slider("Creativity level", 0.0, 1.0, 0.2, 0.1)
        st.caption(
            "Creativity level (temperature) controls how predictable vs exploratory the writing is.\n"
            "• Low (0.0–0.3): more consistent/structured (recommended for decision support)\n"
            "• High (0.7+): more varied phrasing and more speculative reasoning"
        )

        max_output_tokens = st.slider("Maximum memo length", 300, 2000, 1100, 50)
        st.caption(
            "Limits how long the memo can be. Higher values allow deeper analysis but increase time/cost slightly."
        )


# ------------------------------------------------
# Main: Purpose (clear, non-jargony)
# ------------------------------------------------
st.title("🧠 Agentic AI Feature Impact Simulator")

st.markdown(
    """
### What is this?

This application helps evaluate how a new Square product change (“feature”) might impact a small business **before launch**.

A **feature** here means a change to how Square works or charges sellers, such as BNPL, fee changes, AI upsells, or instant payout pricing.

### What am I supposed to do?

1) Upload a seller’s historical transaction CSV  
2) Choose a feature to evaluate (left sidebar)  
3) Click **Run Agent** to generate a recommendation memo  

### What does the tool output?

A structured PM-style memo that covers:
- baseline seller health (revenue, orders, margin proxy, volatility)
- the simulated financial impact of the selected feature
- second-order effects (incentives, retention/churn risk, unintended consequences)
- mitigations / guardrails
- a suggested experiment design
"""
)

with st.expander("What makes this “agentic” (not just a single ChatGPT response)?", expanded=False):
    st.markdown(
        """
The AI is constrained to **use tools** before it writes conclusions.

It must:
1) compute baseline metrics from the uploaded CSV  
2) detect fragility signals  
3) run a deterministic simulation of the selected feature  
4) compare before vs after  
5) then generate the memo based on computed results

You can view the full tool-call trace after running.
"""
    )

st.divider()


# ------------------------------------------------
# Step 1: Upload
# ------------------------------------------------
st.header("1) Upload seller transaction CSV")
st.markdown(
    """
Upload historical transaction data for **a single seller**.

**Why this matters:**  
The tool uses this data to compute a baseline financial profile before simulating any product change.
"""
)

st.markdown("**Expected columns (exact):**")
st.code("date, order_id, item, category, unit_price, quantity, cogs_per_unit, payment_type", language="text")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to continue. (You can use the sample file: `sample_data/seller_transactions_sample.csv`.)")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

try:
    df = validate_and_normalize_transactions(raw_df)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.success("CSV loaded successfully.")
st.caption("Preview (first 20 rows):")
st.dataframe(df.head(20), use_container_width=True)

st.divider()


# ------------------------------------------------
# Step 2: Confirm scenario
# ------------------------------------------------
st.header("2) Confirm scenario")
st.markdown(
    f"""
**Selected feature:** {feature}

When you run the agent, it will:
- analyze baseline seller health
- detect fragility signals
- simulate the feature’s first-order economic impact
- generate a PM-style recommendation memo
"""
)

with st.expander("Assumptions & limitations (important)", expanded=False):
    st.markdown(
        """
- Simulations use reasonable placeholders (e.g., “BNPL may lift conversion but adds fees”)
- Results are directional and meant for decision support (not financial advice)
- In a production tool, assumptions would be calibrated using real experiment data
"""
    )

st.divider()


# ------------------------------------------------
# API key handling
# ------------------------------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning(
        "No OPENAI_API_KEY found.\n\n"
        "Local: add it to `.streamlit/secrets.toml`\n"
        "Streamlit Cloud: add it in App Settings → Secrets."
    )
    st.stop()


# ------------------------------------------------
# Step 3: Run agent
# ------------------------------------------------
st.header("3) Generate recommendation memo")
st.markdown(
    """
Click **Run Agent** to generate a structured recommendation memo based on:
- computed seller metrics (from your CSV)
- a deterministic simulation of the selected feature
- AI reasoning about risks, incentives, and guardrails
"""
)

run = st.button("🚀 Run Agent", type="primary")

if run:
    with st.spinner("Analyzing data, simulating impact, and writing memo..."):
        try:
            memo, trace = run_strategy_agent(
                df=df,
                feature=feature,
                persona=persona,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            st.subheader("✅ Recommendation memo")
            st.markdown(memo)

            with st.expander("🔎 Show agent trace (tool calls + outputs)"):
                st.markdown("This trace shows the tool outputs the model used to avoid guessing numbers.")
                for step in trace:
                    st.markdown(f"**Step {step['step']} — {step['type']}**")
                    if "name" in step:
                        st.code(step["name"], language="text")
                    if "input" in step:
                        st.json(step["input"])
                    if "output" in step:
                        st.json(step["output"])

        except Exception as e:
            st.error(f"Agent run failed: {e}")


st.divider()