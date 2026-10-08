from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_DIR, RAW_DATA, OUTPUT_DIR
from src.data_loader import load_supply_chain_csv
from src.feature_engineering import prepare_for_modeling
from src.predict import SupplyGuardPredictor

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="SupplyGuard | Supply Chain Intelligence",
    page_icon="SG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# VISUAL SYSTEM
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --sg-yellow: #F4C542;
        --sg-yellow-dark: #D4A900;
        --sg-yellow-soft: #FFF6D8;
        --sg-yellow-pale: #FFFBEF;
        --sg-ink: #202124;
        --sg-muted: #74716A;
        --sg-border: #E7E3D9;
        --sg-bg: #FBFAF7;
        --sg-card: #FFFFFF;
        --sg-green: #2E7D5B;
        --sg-orange: #BA7110;
        --sg-red: #C94B42;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--sg-bg);
        color: var(--sg-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(251,250,247,0.96);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--sg-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .main .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .brand {
        padding: .3rem .45rem 1.2rem;
    }

    .brand-mark {
        width: 44px;
        height: 44px;
        border-radius: 13px;
        background: var(--sg-yellow);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: 'Manrope';
        font-weight: 800;
        color: #191919;
        margin-bottom: .7rem;
        box-shadow: 0 5px 16px rgba(244,197,66,.18);
    }

    .brand-title {
        font-family: 'Manrope';
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -.04em;
        color: var(--sg-ink);
    }

    .brand-sub {
        color: var(--sg-muted);
        font-size: .76rem;
        margin-top: .18rem;
        line-height: 1.45;
    }

    .section-label {
        color: #9B968A;
        font-size: .66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .14em;
        margin: 1.0rem .45rem .45rem;
    }

    .page-kicker {
        color: #9A8A58;
        font-size: .7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .14em;
        margin-bottom: .2rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -.04em !important;
        color: var(--sg-ink) !important;
    }

    h1 {
        font-size: 2.05rem !important;
        margin-bottom: .15rem !important;
    }

    h2 {
        font-size: 1.28rem !important;
    }

    h3 {
        font-size: 1rem !important;
    }

    .hero {
        background: linear-gradient(135deg, #FFFDF6 0%, #FFF8DF 100%);
        border: 1px solid #F0E4B7;
        border-radius: 22px;
        padding: 1.45rem 1.65rem;
        margin: 1rem 0 1.15rem;
    }

    .hero-title {
        font-family: 'Manrope';
        font-size: 1.62rem;
        font-weight: 800;
        letter-spacing: -.04em;
        color: #29261D;
    }

    .hero-copy {
        color: #696459;
        max-width: 850px;
        line-height: 1.6;
        margin-top: .35rem;
        font-size: .88rem;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid var(--sg-border);
        border-radius: 17px;
        padding: 1rem 1.05rem;
        min-height: 112px;
        box-shadow: 0 4px 18px rgba(35,30,15,.025);
    }

    .metric-label {
        color: var(--sg-muted);
        font-size: .74rem;
        font-weight: 600;
    }

    .metric-value {
        font-family: 'Manrope';
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -.035em;
        margin-top: .3rem;
        color: #202124;
    }

    .metric-note {
        color: #A09B90;
        font-size: .68rem;
        margin-top: .15rem;
    }

    .panel-title {
        font-family: 'Manrope';
        font-size: .98rem;
        font-weight: 700;
        color: #202124;
        margin-bottom: .08rem;
    }

    .panel-sub {
        color: var(--sg-muted);
        font-size: .74rem;
        line-height: 1.45;
        margin-bottom: .55rem;
    }

    .insight {
        border-left: 3px solid var(--sg-yellow);
        background: #FFFDF6;
        padding: .78rem .95rem;
        border-radius: 0 12px 12px 0;
        color: #5E594C;
        font-size: .8rem;
        line-height: 1.55;
        margin: .35rem 0 .9rem;
    }

    .risk-pill {
        display: block;
        width: 100%;
        padding: .42rem .62rem;
        border-radius: 9px;
        font-size: .7rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }

    .risk-low { background:#EAF5EF; color:#2E7D5B; }
    .risk-moderate { background:#FFF5DC; color:#9A6A12; }
    .risk-high { background:#FFF0D9; color:#B56708; }
    .risk-critical { background:#FBE9E7; color:#B33F36; }

    .recommendation {
        border: 1px solid #EFE3B8;
        background: #FFFDF6;
        border-radius: 12px;
        padding: .72rem .8rem;
        margin-bottom: .5rem;
        color: #514D43;
        font-size: .79rem;
        line-height: 1.5;
    }

    .footer {
        color:#AAA59A;
        font-size:.68rem;
        padding:1.6rem 0 .5rem;
        text-align:center;
    }

    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--sg-border);
        padding: .85rem 1rem;
        border-radius: 16px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--sg-border);
        border-radius: 13px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--sg-border);
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: var(--sg-yellow-dark);
        background: var(--sg-yellow-pale);
        color: #66520A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HELPERS
# =============================================================================


def plot_style(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the common SupplyGuard Plotly visual system."""
    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=10, r=14, t=52, b=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans", color="#30302D", size=12),
        title=dict(
            font=dict(family="Manrope", size=15, color="#202124"),
            x=0.01,
            xanchor="left",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(bgcolor="#202124", font_color="#FFFFFF"),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#EAE6DD",
        tickfont=dict(color="#77736B"),
    )
    fig.update_yaxes(
        gridcolor="#F0EEE8",
        zeroline=False,
        tickfont=dict(color="#77736B"),
    )
    return fig


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def panel_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)


def classify_risk(probability: float) -> str:
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Moderate"
    return "Low"


def risk_class(band: str) -> str:
    return {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Critical": "risk-critical",
    }.get(str(band), "risk-moderate")


def score_frame(df: pd.DataFrame, predictor: SupplyGuardPredictor, sample_size: int = 15000) -> pd.DataFrame:
    """Prepare and score a dataframe for portfolio-level visualisation."""
    prepared = prepare_for_modeling(df.copy())
    if len(prepared) > sample_size:
        prepared = prepared.sample(sample_size, random_state=42)
    probabilities = predictor.pipeline.predict_proba(prepared[predictor.features])[:, 1]
    prepared["Risk Probability"] = probabilities
    prepared["Risk Band"] = [classify_risk(float(x)) for x in probabilities]
    return prepared


@st.cache_resource
def get_predictor() -> SupplyGuardPredictor:
    return SupplyGuardPredictor()


@st.cache_data
def get_raw_data() -> pd.DataFrame:
    return load_supply_chain_csv(RAW_DATA)


# =============================================================================
# INITIALISE
# =============================================================================

if not (MODEL_DIR / "model.joblib").exists():
    st.markdown(
        '<div class="hero">'
        '<div class="page-kicker">SupplyGuard</div>'
        '<div class="hero-title">Your model is not trained yet</div>'
        '<div class="hero-copy">'
        'Run <b>python -m src.train</b> from the SupplyGuard project root, '
        'then refresh this dashboard.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    predictor = get_predictor()
    raw = get_raw_data()
except Exception as exc:
    st.error(f"Unable to initialise SupplyGuard: {exc}")
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-mark">SG</div>'
        '<div class="brand-title">SupplyGuard</div>'
        '<div class="brand-sub">Supply chain risk intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Shipment Analyzer",
            "Risk Explorer",
            "What-if Simulator",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Risk bands</div>', unsafe_allow_html=True)
    st.markdown('<span class="risk-pill risk-low">LOW · 0–30%</span>', unsafe_allow_html=True)
    st.markdown('<span class="risk-pill risk-moderate">MODERATE · 30–60%</span>', unsafe_allow_html=True)
    st.markdown('<span class="risk-pill risk-high">HIGH · 60–80%</span>', unsafe_allow_html=True)
    st.markdown('<span class="risk-pill risk-critical">CRITICAL · 80–100%</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    st.caption(f"{len(raw):,} records loaded")

# =============================================================================
# OVERVIEW
# =============================================================================

if page == "Overview":
    st.markdown('<div class="page-kicker">Executive dashboard</div>', unsafe_allow_html=True)
    st.title("Supply chain at a glance")

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">Find risk before it becomes a delivery problem.</div>'
        '<div class="hero-copy">'
        'SupplyGuard scores shipments, surfaces operational risk factors, and '
        'turns model output into clear actions for operations teams.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    scored = score_frame(raw, predictor, 12000)
    probs = scored["Risk Probability"]

    total = len(raw)
    high = int((probs >= 0.60).sum())
    critical = int((probs >= 0.80).sum())

    if "Late_delivery_risk" in raw.columns:
        observed = pd.to_numeric(raw["Late_delivery_risk"], errors="coerce")
        observed_delay_rate = observed.mean()
    else:
        observed_delay_rate = None

    cols = st.columns(4)
    with cols[0]:
        metric_card("Total orders", f"{total:,}", "Dataset records")
    with cols[1]:
        metric_card("High / critical", f"{high:,}", f"of {len(scored):,} scored orders")
    with cols[2]:
        metric_card("Critical risk", f"{critical:,}", "Predicted probability ≥ 80%")
    with cols[3]:
        metric_card(
            "Observed delay rate",
            f"{observed_delay_rate:.1%}" if observed_delay_rate is not None else "N/A",
            "Dataset outcome field",
        )

    st.markdown("### Risk landscape")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            panel_header(
                "Risk distribution",
                "Portfolio share across the four transparent risk bands.",
            )

            band_counts = (
                scored["Risk Band"]
                .value_counts()
                .reindex(["Low", "Moderate", "High", "Critical"], fill_value=0)
                .reset_index()
            )
            band_counts.columns = ["Risk Band", "Shipments"]

            fig = px.pie(
                band_counts,
                names="Risk Band",
                values="Shipments",
                hole=0.68,
            )
            fig.update_traces(
                textinfo="percent+label",
                textposition="outside",
                marker=dict(line=dict(color="#FFFFFF", width=2)),
            )
            plot_style(fig, 335)
            fig.update_layout(
                title=None,
                showlegend=False,
                margin=dict(l=30, r=30, t=8, b=8),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            panel_header(
                "Risk by shipping mode",
                "Mean model-estimated risk by shipping method.",
            )

            if "Shipping Mode" in scored.columns:
                agg = (
                    scored.groupby("Shipping Mode", as_index=False)["Risk Probability"]
                    .mean()
                    .sort_values("Risk Probability", ascending=True)
                )
                fig = px.bar(
                    agg,
                    x="Risk Probability",
                    y="Shipping Mode",
                    orientation="h",
                    text="Risk Probability",
                )
                fig.update_traces(
                    marker_color="#F4C542",
                    texttemplate="%{text:.0%}",
                    textposition="outside",
                    cliponaxis=False,
                )
                fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Mean predicted risk")
                fig.update_yaxes(title=None)
                plot_style(fig, 335)
                fig.update_layout(title=None)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Shipping Mode is not available in the loaded dataset.")

    if "Order Region" in scored.columns:
        with st.container(border=True):
            panel_header(
                "Highest risk-associated regions",
                "Top regions ranked by mean predicted risk. This is association, not causation.",
            )

            region = (
                scored.groupby("Order Region", as_index=False)
                .agg(
                    risk=("Risk Probability", "mean"),
                    shipments=("Risk Probability", "size"),
                )
                .sort_values("risk", ascending=False)
                .head(10)
                .sort_values("risk")
            )

            fig = px.bar(
                region,
                x="risk",
                y="Order Region",
                orientation="h",
                text="risk",
            )
            fig.update_traces(
                marker_color="#E5B52C",
                texttemplate="%{text:.0%}",
                textposition="outside",
                cliponaxis=False,
            )
            fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Mean predicted risk")
            fig.update_yaxes(title=None)
            plot_style(fig, 425)
            fig.update_layout(title=None)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        '<div class="insight"><b>How to read this dashboard:</b> '
        'risk is model probability, not a guarantee of delay. Use the Shipment '
        'Analyzer for order-level explanations and the What-if Simulator for '
        'controlled scenario comparisons.</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# SHIPMENT ANALYZER
# =============================================================================

elif page == "Shipment Analyzer":
    st.markdown('<div class="page-kicker">Order-level intelligence</div>', unsafe_allow_html=True)
    st.title("Shipment risk analyzer")
    st.caption("Select an order to inspect its risk score, contributing factors, and recommended actions.")

    df = prepare_for_modeling(raw.copy())

    if "Order Id" not in df.columns:
        st.error("Order Id is not available in the dataset.")
        st.stop()

    order_ids = df["Order Id"].dropna().drop_duplicates().tolist()
    selected_id = st.selectbox("Search order", order_ids)
    row = df[df["Order Id"] == selected_id].iloc[[0]].copy()
    result = predictor.predict(row)

    top1, top2, top3 = st.columns(3)
    with top1:
        metric_card("Risk probability", f"{result['probability']:.1%}", "Model-estimated probability")
    with top2:
        metric_card("Risk band", result["risk_band"], "Operational severity")
    with top3:
        metric_card("Prediction", "At risk" if result["prediction"] else "On track", "Binary model decision")

    left, right = st.columns([1.25, 0.9])

    with left:
        with st.container(border=True):
            panel_header(
                "Why is this shipment risky?",
                "SHAP contribution shows which model features push the prediction up or down.",
            )

            exp = result.get("explanations", pd.DataFrame()).copy()

            if not exp.empty:
                exp = exp.sort_values("shap_value")
                if len(exp) > 12:
                    exp = exp.tail(12)
                exp["Direction"] = exp["shap_value"].apply(
                    lambda x: "Increases risk" if x > 0 else "Reduces risk"
                )

                fig = px.bar(
                    exp,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={
                        "Increases risk": "#D95B50",
                        "Reduces risk": "#5D9C7A",
                    },
                    text="shap_value",
                )
                fig.add_vline(x=0, line_color="#BEB9AE", line_width=1)
                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                    cliponaxis=False,
                )
                fig.update_xaxes(title="SHAP contribution")
                fig.update_yaxes(title=None)
                plot_style(fig, max(400, 32 * len(exp) + 120))
                fig.update_layout(title=None, legend_title=None)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No explanation data available for this model.")

    with right:
        with st.container(border=True):
            panel_header(
                "Recommended actions",
                "Transparent rules convert the risk signal into an operational next step.",
            )
            actions = result.get("actions", [])
            if actions:
                for action in actions:
                    st.markdown(f'<div class="recommendation">{action}</div>', unsafe_allow_html=True)
            else:
                st.info("No additional action is recommended.")

        with st.container(border=True):
            panel_header("Shipment details", "Key fields for the selected order.")
            display_cols = [
                c for c in [
                    "Order Id",
                    "Shipping Mode",
                    "Order Priority",
                    "Market",
                    "Order Region",
                    "Category Name",
                    "Order Item Quantity",
                    "Order Item Product Price",
                    "Sales",
                ]
                if c in row.columns
            ]
            detail = row[display_cols].T.reset_index()
            detail.columns = ["Field", "Value"]
            st.dataframe(detail, use_container_width=True, hide_index=True, height=370)

# =============================================================================
# RISK EXPLORER
# =============================================================================

elif page == "Risk Explorer":
    st.markdown('<div class="page-kicker">Portfolio analysis</div>', unsafe_allow_html=True)
    st.title("Explore where risk is concentrated")
    st.caption("Filter the scored shipment sample and compare risk across operational dimensions.")

    scored = score_frame(raw, predictor, 15000)

    with st.container(border=True):
        panel_header("Portfolio filters", "Use these filters to narrow the current analytical scope.")

        f1, f2, f3, f4 = st.columns(4)
        filter_pairs = [
            (f1, "Market"),
            (f2, "Order Region"),
            (f3, "Category Name"),
            (f4, "Shipping Mode"),
        ]

        for container, column in filter_pairs:
            if column not in scored.columns:
                continue
            values = sorted(scored[column].dropna().astype(str).unique().tolist())
            selected = container.multiselect(
                column,
                values,
                key=f"filter_{column}",
            )
            if selected:
                scored = scored[scored[column].astype(str).isin(selected)]

    if scored.empty:
        st.warning("No shipments match the current filters.")
        st.stop()

    high_count = int((scored["Risk Probability"] >= 0.60).sum())
    critical_count = int((scored["Risk Probability"] >= 0.80).sum())
    average_risk = scored["Risk Probability"].mean()

    s1, s2, s3 = st.columns(3)
    with s1:
        metric_card("Filtered shipments", f"{len(scored):,}", "Current filter scope")
    with s2:
        metric_card("Average risk", f"{average_risk:.1%}", "Mean predicted probability")
    with s3:
        metric_card("High + critical", f"{high_count:,}", f"{critical_count:,} critical")

    dimensions = [
        c for c in [
            "Market",
            "Order Region",
            "Category Name",
            "Shipping Mode",
            "Customer Segment",
        ]
        if c in scored.columns
    ]

    if not dimensions:
        st.warning("No supported categorical dimensions are available.")
        st.stop()

    group = st.selectbox("Compare by", dimensions)

    agg = (
        scored.groupby(group, as_index=False)
        .agg(
            mean_risk=("Risk Probability", "mean"),
            shipments=("Risk Probability", "size"),
        )
        .sort_values("mean_risk", ascending=False)
    )

    with st.container(border=True):
        panel_header(
            f"Risk by {group}",
            "Higher bars indicate higher average predicted risk in the selected scope.",
        )

        display_agg = agg.head(15).sort_values("mean_risk")
        fig = px.bar(
            display_agg,
            x="mean_risk",
            y=group,
            orientation="h",
            text="mean_risk",
        )
        fig.update_traces(
            marker_color="#F4C542",
            texttemplate="%{text:.0%}",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Mean predicted risk")
        fig.update_yaxes(title=None)
        plot_style(fig, max(420, 29 * len(display_agg) + 120))
        fig.update_layout(title=None)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        panel_header("Detailed ranking", "Use the table for exact shipment counts and risk values.")
        display = agg.rename(
            columns={
                group: "Segment",
                "mean_risk": "Mean risk",
                "shipments": "Shipments",
            }
        )[["Segment", "Mean risk", "Shipments"]]
        display["Mean risk"] = display["Mean risk"].map(lambda x: f"{x:.1%}")
        st.dataframe(display, use_container_width=True, hide_index=True, height=360)

# =============================================================================
# WHAT-IF SIMULATOR
# =============================================================================

elif page == "What-if Simulator":
    st.markdown('<div class="page-kicker">Scenario planning</div>', unsafe_allow_html=True)
    st.title("What-if simulator")
    st.caption("Change controllable shipment attributes and compare model-estimated risk.")

    df = prepare_for_modeling(raw.copy())
    if "Order Id" not in df.columns:
        st.error("Order Id is not available in the dataset.")
        st.stop()

    order_ids = df["Order Id"].dropna().drop_duplicates().tolist()
    selected_id = st.selectbox("Base order", order_ids)
    base = df[df["Order Id"] == selected_id].iloc[[0]].copy()
    scenario = base.copy()

    with st.container(border=True):
        panel_header(
            "Scenario controls",
            "Modify selected attributes available to the model. Changes are simulations, not guarantees.",
        )

        a, b = st.columns(2)

        with a:
            if "Shipping Mode" in scenario.columns:
                options = sorted(df["Shipping Mode"].dropna().astype(str).unique().tolist())
                current = str(base.iloc[0]["Shipping Mode"])
                selected = st.selectbox(
                    "Shipping mode",
                    options,
                    index=options.index(current) if current in options else 0,
                )
                scenario.at[scenario.index[0], "Shipping Mode"] = selected

            if "Order Priority" in scenario.columns:
                options = sorted(df["Order Priority"].dropna().astype(str).unique().tolist())
                current = str(base.iloc[0]["Order Priority"])
                selected = st.selectbox(
                    "Order priority",
                    options,
                    index=options.index(current) if current in options else 0,
                )
                scenario.at[scenario.index[0], "Order Priority"] = selected

        with b:
            if "Order Item Quantity" in scenario.columns:
                try:
                    current = float(base.iloc[0]["Order Item Quantity"])
                except Exception:
                    current = 1.0
                current = max(1.0, current)
                quantity = st.number_input(
                    "Item quantity",
                    min_value=1.0,
                    value=current,
                    step=1.0,
                )
                scenario.at[scenario.index[0], "Order Item Quantity"] = quantity

            if "Order Item Discount Rate" in scenario.columns:
                try:
                    current = float(base.iloc[0]["Order Item Discount Rate"])
                except Exception:
                    current = 0.0
                current = min(1.0, max(0.0, current))
                discount = st.number_input(
                    "Discount rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=current,
                    step=0.01,
                    format="%.2f",
                )
                scenario.at[scenario.index[0], "Order Item Discount Rate"] = discount

    base_result = predictor.predict(base)
    scenario_result = predictor.predict(scenario)
    delta = scenario_result["probability"] - base_result["probability"]

    st.markdown("### Scenario result")

    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("Base risk", f"{base_result['probability']:.1%}", base_result["risk_band"])
    with r2:
        metric_card("Scenario risk", f"{scenario_result['probability']:.1%}", scenario_result["risk_band"])
    with r3:
        metric_card("Risk change", f"{delta:+.1%}", "Negative is better")

    with st.container(border=True):
        panel_header(
            "Before vs simulated risk",
            "Direct comparison of model probability for the base order and scenario.",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Base", "Scenario"],
                y=[base_result["probability"], scenario_result["probability"]],
                text=[
                    f"{base_result['probability']:.0%}",
                    f"{scenario_result['probability']:.0%}",
                ],
                textposition="outside",
                marker_color="#F4C542",
                width=0.55,
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Predicted risk")
        fig.update_xaxes(title=None)
        plot_style(fig, 350)
        fig.update_layout(title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if delta < 0:
        st.success(f"This scenario lowers the model's predicted risk by {abs(delta):.1%}.")
    elif delta > 0:
        st.warning(f"This scenario increases the model's predicted risk by {delta:.1%}.")
    else:
        st.info("This scenario does not change the predicted risk.")

    st.markdown(
        '<div class="insight"><b>Important:</b> What-if results are model '
        'simulations used for operational comparison. They are not guarantees '
        'of real-world delivery outcomes.</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

else:
    st.markdown('<div class="page-kicker">Machine learning quality</div>', unsafe_allow_html=True)
    st.title("Model performance")
    st.caption("Compare candidate models and inspect the final model's evaluation artifacts.")

    metrics_path = MODEL_DIR / "model_metrics.json"
    metadata_path = MODEL_DIR / "metadata.json"

    if not metrics_path.exists() or not metadata_path.exists():
        st.warning("Training metrics are not available yet. Run the training pipeline first.")
        st.stop()

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Unable to read training metadata: {exc}")
        st.stop()

    rows = []
    for name, m in metrics.get("metrics", {}).items():
        rows.append(
            {
                "Model": name,
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1": m.get("f1"),
                "ROC-AUC": m.get("roc_auc"),
                "PR-AUC": m.get("pr_auc"),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty and "PR-AUC" in table.columns:
        table = table.sort_values("PR-AUC", ascending=False)

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card(
            "Selected model",
            metadata.get("selected_model", "N/A"),
            "Chosen by training pipeline",
        )
    with m2:
        metric_card(
            "Training rows",
            f"{metadata.get('train_rows', 0):,}",
            "Used for model fitting",
        )
    with m3:
        metric_card(
            "Test rows",
            f"{metadata.get('test_rows', 0):,}",
            "Held-out evaluation",
        )

    if not table.empty:
        with st.container(border=True):
            panel_header(
                "Model comparison",
                "Precision, recall, F1, ROC-AUC and PR-AUC are prioritised over accuracy alone.",
            )
            chart_metrics = [
                c for c in ["F1", "ROC-AUC", "PR-AUC"]
                if c in table.columns
            ]
            fig = px.bar(
                table,
                x="Model",
                y=chart_metrics,
                barmode="group",
            )
            fig.update_yaxes(range=[0, 1], title="Score")
            fig.update_xaxes(title=None)
            plot_style(fig, 400)
            fig.update_layout(title=None, legend_title=None)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.container(border=True):
            panel_header("Metric table", "Exact evaluation values from the training pipeline.")
            numeric_cols = [c for c in table.columns if c != "Model"]
            st.dataframe(
                table.style.format({c: "{:.3f}" for c in numeric_cols}),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No model metrics are available yet.")

    with st.container(border=True):
        panel_header("Evaluation artifacts", "Plots generated during model evaluation.")
        artifact_cols = st.columns(3)
        artifacts = [
            ("roc_curve.png", "ROC curve"),
            ("precision_recall_curve.png", "Precision–Recall curve"),
            ("confusion_matrix.png", "Confusion matrix"),
        ]
        for container, (filename, title) in zip(artifact_cols, artifacts):
            with container:
                path = OUTPUT_DIR / "plots" / filename
                if path.exists():
                    st.image(str(path), caption=title, use_container_width=True)
                else:
                    st.info(f"{title} not generated yet.")

    with st.container(border=True):
        panel_header(
            "Leakage audit",
            "Columns excluded from predictors because they are not valid inputs at prediction time.",
        )
        excluded = metadata.get("excluded_columns_present", [])
        if excluded:
            st.markdown(
                '<div class="insight"><b>Excluded from predictors:</b> '
                + ", ".join(map(str, excluded))
                + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No excluded leakage-prone columns were recorded in metadata.")

    feature_columns = metadata.get("feature_columns", [])
    if feature_columns:
        with st.container(border=True):
            panel_header("Features used by final model", "Predictor columns persisted with the selected model.")
            feature_df = pd.DataFrame({"Feature": feature_columns})
            st.dataframe(feature_df, use_container_width=True, hide_index=True, height=300)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    '<div class="footer">SupplyGuard · AI-powered supply chain risk intelligence · '
    'Model outputs are decision-support signals, not delivery guarantees.</div>',
    unsafe_allow_html=True,
)
