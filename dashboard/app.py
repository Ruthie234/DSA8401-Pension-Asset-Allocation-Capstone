"""Single-screen interactive pension portfolio monitoring and early-warning tool.

The user selects an anonymised scheme reference and current reporting year,
updates current investment amounts for the asset classes held, and receives:
- current-versus-previous-year portfolio comparison;
- concentration and diversification measures; and
- the existing calibrated Gradient Boosting model's next-year probability
  and classification.

The authorised scheme-level CSV is mounted read-only at runtime. It is not
copied into the Docker image and is not exported by this application.
"""
from __future__ import annotations

import altair as alt  # pyright: ignore[reportMissingImports]
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pension_capstone.config import ASSET_COLUMNS  # noqa: E402
from pension_capstone.data_pipeline import run_data_preparation  # noqa: E402
from pension_capstone.feature_engineering import build_portfolio_features  # noqa: E402
from dashboard.assessment import build_assessment_features  # noqa: E402

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
RBA_DATA_PATH = Path(
    os.getenv("RBA_DATA_PATH", str(PROJECT_ROOT / "COPY 1 FINAL.csv"))
).expanduser()

ASSET_LABELS = {
    "Cash, Demand & Fixed Deposits": "Cash, demand and fixed deposits",
    "Commercial Paper and Corporate Bonds": "Corporate bonds",
    "Guaranteed Funds": "Guaranteed funds",
    "Immovable Property": "Immovable property",
    "Kenya Government Securities": "Kenya government securities",
    "Offshore Investments": "Offshore investments",
    "Property Unit Trusts": "Property unit trusts",
    "Quoted Equity": "Quoted equity",
    "Unquoted Equity": "Unquoted equity",
}

st.set_page_config(
    page_title="Pension Portfolio Monitor & Early Warning",
    page_icon="📊",
    layout="wide",
)


def clear_assessment() -> None:
    """Prevent an old assessment being displayed after selection changes."""
    st.session_state.pop("assessment_result", None)


@st.cache_data(show_spinner="Loading and validating the authorised portfolio history…")
def load_historical_portfolio(data_path: str, modified_ns: int) -> pd.DataFrame:
    """Rebuild the project's cleaned annual panel and feature history in memory."""
    del modified_ns  # Included in the cache key so a changed source file is reloaded.
    prepared = run_data_preparation(data_path)
    portfolio, _, _ = build_portfolio_features(prepared.df_modelling)
    if portfolio.empty:
        raise ValueError("The authorised source file contains no valid positive-investment portfolios.")
    if portfolio.duplicated(["Ref", "Year"]).any():
        raise ValueError("The prepared history contains duplicate scheme-year records.")
    return portfolio


def api_is_available() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=4)
        response.raise_for_status()
        payload = response.json()
        if payload.get("model_loaded"):
            return True, str(payload.get("selected_model", "Prediction model online"))
        return False, "Prediction API is responding, but the model is not loaded."
    except requests.RequestException as exc:
        return False, f"Prediction API unavailable ({type(exc).__name__})."


def money(value: float) -> str:
    return f"KSh {value:,.2f}"


def render_assessment(result: dict[str, Any]) -> None:
    """Render all outputs together on this one dashboard page."""
    comparison = result["comparison"]
    prediction = result["prediction"]
    current_year = int(comparison["current_year"])
    next_year = current_year + 1

    st.divider()
    st.header("Portfolio assessment results")
    st.caption(
        f"Scheme reference {result['scheme_ref']} · Current reporting year {current_year} · "
        f"Historical comparison year {comparison['previous_year']}"
    )

    # First row: portfolio values and change in size.
    st.subheader("1. Portfolio at a glance")
    growth = float(comparison["total_investment_growth_pct"])
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Previous portfolio value · {comparison['previous_year']}",
        money(float(comparison["previous_total"])),
    )
    c2.metric(
        f"Current portfolio value · {current_year}",
        money(float(comparison["current_total"])),
    )
    c3.metric("Change in total investments", f"{growth:+.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Active asset classes · previous year", f"{comparison['previous_active_classes']} / 9")
    c5.metric("Active asset classes · current year", f"{comparison['current_active_classes']} / 9")
    c6.metric("Current dominant asset class", str(comparison["current_dominant_asset"]))

    # Detailed historical-to-current comparison.
    st.subheader("2. How the portfolio has changed")
    table_rows: list[dict[str, Any]] = []
    for asset in ASSET_COLUMNS:
        previous_amount = float(comparison["previous_amounts"][asset])
        current_amount = float(comparison["current_amounts"][asset])
        previous_share = float(comparison["previous_shares"][f"{_share_name(asset)}"])
        current_share = float(comparison["current_shares"][f"{_share_name(asset)}"])
        table_rows.append({
            "Asset class": ASSET_LABELS[asset],
            f"Previous amount ({comparison['previous_year']})": previous_amount,
            f"Current amount ({current_year})": current_amount,
            "Previous allocation (%)": previous_share * 100.0,
            "Current allocation (%)": current_share * 100.0,
            "Change (percentage points)": (current_share - previous_share) * 100.0,
        })
    comparison_table = pd.DataFrame(table_rows)

    chart_data = comparison_table.melt(
        id_vars="Asset class",
        value_vars=["Previous allocation (%)", "Current allocation (%)"],
        var_name="Reporting period",
        value_name="Allocation (%)",
    )

    chart_data["Reporting period"] = chart_data["Reporting period"].map({
        "Previous allocation (%)": f"Previous year ({comparison['previous_year']})",
        "Current allocation (%)": f"Current year ({current_year})",
    })

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Asset class:N",
                title=None,
                sort=list(ASSET_LABELS.values()),
                axis=alt.Axis(labelAngle=-40),
            ),
            xOffset="Reporting period:N",
            y=alt.Y(
                "Allocation (%):Q",
                title="Portfolio allocation (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color("Reporting period:N", title="Reporting period"),
            tooltip=[
                "Asset class:N",
                "Reporting period:N",
                alt.Tooltip("Allocation (%):Q", format=".2f"),
            ],
        )
        .properties(height=360)
    )

    st.markdown("**Previous-year versus current-year asset allocation**")
    st.altair_chart(chart, use_container_width=True)
    st.caption("Bars compare allocation shares, not raw investment values.")

    formatted_table = comparison_table.copy()
    for column in [f"Previous amount ({comparison['previous_year']})", f"Current amount ({current_year})"]:
        formatted_table[column] = formatted_table[column].map(lambda value: f"KSh {value:,.2f}")
    for column in ["Previous allocation (%)", "Current allocation (%)", "Change (percentage points)"]:
        formatted_table[column] = formatted_table[column].map(lambda value: f"{value:+.2f}" if column == "Change (percentage points)" else f"{value:.2f}")
    st.dataframe(formatted_table, use_container_width=True, hide_index=True)

    changes = comparison_table["Change (percentage points)"].astype(float)

    if changes.abs().max() < 1e-9:
        st.info(
            "No allocation movement observed between the previous "
            "and current reporting years."
        )

    else:
        increase_row = comparison_table.loc[changes.idxmax()]
        decrease_row = comparison_table.loc[changes.idxmin()]

        inc_col, dec_col = st.columns(2)

        inc_col.metric(
            "Largest allocation increase",
            str(increase_row["Asset class"]),
            f"{float(increase_row['Change (percentage points)']):+.2f} pp",
        )

        dec_col.metric(
            "Largest allocation decrease",
            str(decrease_row["Asset class"]),
            f"{float(decrease_row['Change (percentage points)']):+.2f} pp",
        )

    # Concentration/diversification for both years.
    st.subheader("3. Concentration and diversification")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Previous HHI", f"{float(comparison['previous_hhi']):.3f}")
    h2.metric("Current HHI", f"{float(comparison['current_hhi']):.3f}", f"{float(comparison['current_hhi'] - comparison['previous_hhi']):+.3f}")
    h3.metric("Previous diversification", f"{float(comparison['previous_diversification']):.3f}")
    h4.metric("Current diversification", f"{float(comparison['current_diversification']):.3f}", f"{float(comparison['current_diversification'] - comparison['previous_diversification']):+.3f}")
    st.caption("Higher HHI indicates greater concentration. The diversification index is 1 − HHI.")

    # Single headline prediction output. SHAP/API factor details are intentionally not displayed.
    st.subheader("4. Next-year early-warning assessment")
    probability = float(prediction["predicted_probability_material_change"])
    predicted_class = int(prediction["predicted_material_change"])
    threshold = float(prediction["decision_threshold"])
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted probability", f"{probability:.1%}")
    p2.metric("Prediction for reporting year", str(next_year))
    p3.metric("Decision threshold", f"{threshold:.0%}")

    if predicted_class == 1:
        st.warning(
            f"**Material allocation change predicted for {next_year}.** "
            "The estimated probability is at or above the configured decision threshold. "
            "Closer monitoring may be warranted."
        )
    else:
        st.info(
            f"**No material allocation change predicted for {next_year} at the configured threshold.** "
            "This does not mean that no allocation movement will occur."
        )

    st.progress(min(max(probability, 0.0), 1.0), text=f"Estimated probability: {probability:.1%}")
    st.caption(
        "A material change is an absolute change of at least five percentage points in the share "
        "of at least one asset class between consecutive reporting years. The model does not "
        "identify which asset class will change or whether a change is harmful or non-compliant."
    )
    if current_year == 2025:
        st.info(
            "Evaluation note: the available held-out test period covers outcomes through 2025. "
            "This assessment predicts 2026, which is beyond that historical test window; "
            "interpret it as a prospective research-prototype signal, not as independently "
            "validated 2026 performance."
        )


def _share_name(asset: str) -> str:
    from pension_capstone.config import SHARE_MAPPING
    return SHARE_MAPPING[asset]


# ---------------------------------------------------------------------------
# Page and private historical source
# ---------------------------------------------------------------------------
st.title("Pension Portfolio Monitor & Early-Warning System")
st.caption(
    "Enter the latest portfolio amounts for a selected anonymised scheme. "
    "The application retrieves the previous reporting year, compares the portfolio "
    "and requests a next-year prediction from the trained model."
)

if not RBA_DATA_PATH.is_file():
    st.error(
        "The private historical source file is not available to the dashboard. "
        "Keep `COPY 1 FINAL.csv` in the project root and confirm that Docker Compose "
        "mounts it read-only at `/private_data/rba_source.csv`. The raw dataset is not "
        "copied into the Docker image or public repository."
    )
    st.stop()

try:
    source_mtime = RBA_DATA_PATH.stat().st_mtime_ns
    history = load_historical_portfolio(str(RBA_DATA_PATH), source_mtime)
except Exception as exc:
    st.error(f"Could not prepare the authorised historical portfolio data: {type(exc).__name__}: {exc}")
    st.stop()

api_ok, model_status = api_is_available()
status_col, model_col = st.columns([2, 3])
if api_ok:
    status_col.success("Prediction service online")
else:
    status_col.error(model_status)
model_col.caption(f"Model service: {model_status} · API: {API_BASE_URL}")

available_years = sorted(int(value) for value in history["Year"].dropna().unique())
year_set = set(available_years)
assessment_years = [year for year in available_years if year - 1 in year_set]
if not assessment_years:
    st.error("No consecutive reporting-year records are available for portfolio assessment.")
    st.stop()

st.subheader("Portfolio input")
st.write(
    "Choose the scheme and current reporting year. Previous-year figures will be retrieved "
    "automatically. Asset classes held last year are preselected, and their amounts are "
    "shown as starting values for convenience—update them to the current reporting amounts."
)

year_col, scheme_col = st.columns(2)
with year_col:
    current_year = st.selectbox(
        "Current reporting year",
        options=assessment_years,
        index=len(assessment_years) - 1,
        key="assessment_year",
        on_change=clear_assessment,
    )

previous_year_rows = history.loc[history["Year"].eq(int(current_year) - 1)].copy()
reference_options = sorted(int(value) for value in previous_year_rows["Ref"].unique())
if not reference_options:
    st.warning(f"No valid historical records are available for reporting year {current_year - 1}.")
    st.stop()

with scheme_col:
    scheme_ref = st.selectbox(
        "Anonymised scheme reference",
        options=reference_options,
        format_func=lambda value: str(value),
        key=f"assessment_scheme_{current_year}",
        on_change=clear_assessment,
        help="References are anonymised and are only used to retrieve the preceding year's record from the private local source.",
    )

previous_matches = history.loc[
    history["Ref"].eq(int(scheme_ref)) & history["Year"].eq(int(current_year) - 1)
]
if len(previous_matches) != 1:
    st.error("A unique previous-year record could not be retrieved for this selection.")
    st.stop()
previous_record = previous_matches.iloc[0]
previous_active_assets = [asset for asset in ASSET_COLUMNS if float(previous_record[asset]) > 0]

st.caption(
    f"Historical record retrieved: {int(current_year) - 1} · "
    f"Previous total investments: {money(float(previous_record['Total_Investments']))} · "
    f"Previous active asset classes: {len(previous_active_assets)} of 9"
)

selected_assets = st.multiselect(
    "Asset classes held in the current portfolio",
    options=ASSET_COLUMNS,
    default=previous_active_assets,
    format_func=lambda asset: ASSET_LABELS[asset],
    key=f"current_assets_{scheme_ref}_{current_year}",
    on_change=clear_assessment,
    help="The previous year's active classes are preselected. Remove classes no longer held or add newly held classes.",
)

with st.form(key=f"portfolio_assessment_form_{scheme_ref}_{current_year}"):
    st.markdown("**Current-year investment amounts (KSh)**")
    st.caption(
        "Starting values are copied from the previous year to reduce typing. "
        "Replace each value with its current-year amount. Classes not selected above are treated as zero."
    )
    entered_amounts: dict[str, float] = {asset: 0.0 for asset in ASSET_COLUMNS}
    if selected_assets:
        amount_cols = st.columns(2)
        for index, asset in enumerate(selected_assets):
            default_amount = max(0.0, float(previous_record[asset]))
            with amount_cols[index % 2]:
                entered_amounts[asset] = st.number_input(
                    ASSET_LABELS[asset],
                    min_value=0.0,
                    value=default_amount,
                    step=10000.0,
                    format="%.2f",
                    key=f"amount_{scheme_ref}_{current_year}_{asset}",
                )
    else:
        st.info("Select at least one asset class above to enter its current investment amount.")
    submitted = st.form_submit_button("Assess portfolio", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("assessment_result", None)
    try:
        features, comparison = build_assessment_features(
            previous_record=previous_record,
            current_amounts=entered_amounts,
            reporting_year=int(current_year),
        )
        if not api_ok:
            raise RuntimeError("The prediction API is not currently available. Check that the API container is healthy.")
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=features,
            timeout=45,
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"Prediction service returned HTTP {response.status_code}: {detail}")
        prediction = response.json()
        st.session_state["assessment_result"] = {
            "scheme_ref": int(scheme_ref),
            "comparison": comparison,
            "prediction": prediction,
        }
    except Exception as exc:
        st.error(f"Assessment could not be completed: {type(exc).__name__}: {exc}")

saved_result = st.session_state.get("assessment_result")
if saved_result and saved_result.get("scheme_ref") == int(scheme_ref):
    render_assessment(saved_result)

st.divider()
st.caption(
    "Research prototype for authorised users. Scheme-level historical records are read from "
    "a private, read-only local data mount. Individual assessment results are held only in "
    "the active Streamlit session and are not written to the aggregate dashboard exports. "
    "Predictions are decision-support signals, not determinations of harm or regulatory non-compliance."
)
