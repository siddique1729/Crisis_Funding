import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, ast, warnings, json, re
from collections import Counter
from typing import List
warnings.filterwarnings("ignore")

os.environ["GRPC_ARG_KEEPALIVE_TIME_MS"]                            = "60000"
os.environ["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"]                         = "20000"
os.environ["GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS"] = "60000"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Allocaid — Humanitarian Funding Fairness",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ISO3 → Country Name Mapping (main app)
# ============================================================
ISO_NAMES = {
    "AFG": "Afghanistan", "BFA": "Burkina Faso", "CAF": "Central African Republic",
    "COD": "DR Congo", "COL": "Colombia", "ETH": "Ethiopia", "HTI": "Haiti",
    "IRQ": "Iraq", "JOR": "Jordan", "LBN": "Lebanon", "MLI": "Mali",
    "MMR": "Myanmar", "MOZ": "Mozambique", "NER": "Niger", "NGA": "Nigeria",
    "PAK": "Pakistan", "PSE": "Palestine", "SDN": "Sudan", "SOM": "Somalia",
    "SSD": "South Sudan", "SYR": "Syria", "TCD": "Chad", "UKR": "Ukraine",
    "VEN": "Venezuela", "YEM": "Yemen"
}

# ============================================================
# CUSTOM CSS (main app)
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: transparent;
        padding: 1rem 0 0.5rem 0;
        margin-bottom: 1.5rem;
        color: #111827;
    }
    .main-header h1 { font-size: 2.2rem; margin-bottom: 0.3rem; color: #111827; }
    .main-header p { font-size: 1.05rem; opacity: 0.75; margin-bottom: 0; color: #374151; }

    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #e74c3c;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { font-size: 1.8rem; margin: 0; color: #0f172a; }
    .metric-card p { font-size: 0.85rem; color: #666; margin: 0; }

    .insight-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .insight-blue {
        background: #e0f2fe;
        border: 1px solid #0ea5e9;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #888;
        font-size: 0.85rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }

    /* ── Allocaid RAG Chat CSS ─────────────────────────────────────────── */
    :root{
        --accent:#e07b39;--accent2:#2563eb;
        --danger:#dc2626;--success:#16a34a;
        --muted:#6b7280;--text:#111827;--border:#e5e7eb;
        --surface:#ffffff;--surface2:#f9fafb;
    }
    .panel{background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,0.06);}
    .panel-title{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#6b7280;
        margin-bottom:0.8rem;display:flex;align-items:center;gap:0.4rem;}
    .panel-title::before{content:'';width:5px;height:5px;border-radius:50%;
        background:#e07b39;display:inline-block;}

    .chat-scroll{max-height:480px;overflow-y:auto;padding-right:4px;margin-bottom:0.6rem;}
    .msg{padding:0.65rem 0.9rem;border-radius:8px;margin-bottom:0.5rem;font-size:0.84rem;line-height:1.65;color:#111827;}
    .msg-user{background:#f3f4f6;border:1px solid #e5e7eb;}
    .msg-ai{background:#fff7f0;border:1px solid #fbd5b5;}
    .msg-role{font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;
        color:#9ca3af;margin-bottom:0.2rem;}
    .tag{display:inline-block;padding:1px 8px;border-radius:100px;font-size:0.67rem;margin:1px;font-weight:500;}
    .tag-red  {background:#fee2e2;color:#dc2626;border:1px solid #fca5a5;}
    .tag-green{background:#dcfce7;color:#16a34a;border:1px solid #86efac;}
    .tag-blue {background:#dbeafe;color:#2563eb;border:1px solid #93c5fd;}
    .tag-grey {background:#f3f4f6;color:#6b7280;border:1px solid #d1d5db;}
    .tag-orange{background:#fff7ed;color:#e07b39;border:1px solid #fed7aa;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA (main app)
# ============================================================
@st.cache_data
def load_data():
    scored = pd.read_csv("data/scored_funding.csv")
    bench = pd.read_csv("data/benchmarking.csv")
    scored["Country"] = scored["ISO3"].map(ISO_NAMES).fillna(scored["ISO3"])
    bench["Country"] = bench["ISO3"].map(ISO_NAMES).fillna(bench["ISO3"])
    return scored, bench

scored_df, bench_df = load_data()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔎 Allocaid</h1>
    <p>Databricks × United Nations — Where Does Humanitarian Funding Fall Short of Need?</p>
    <p style="font-size: 0.8rem; color: #6b7280; margin-top: 0.5rem;">
        Hacklytics 2026 | ML pipeline in Databricks (XGBoost + MLflow) • Visualized with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    available_years = sorted(scored_df["Year"].unique(), reverse=True)
    selected_year = st.selectbox("Year", available_years, index=0)
    continents = ["All"] + sorted(scored_df["Continent"].dropna().unique().tolist())
    selected_continent = st.selectbox("Region", continents)
    show_predicted_only = st.checkbox("Only show rows with model predictions", value=True)
    st.markdown("---")
    st.markdown("### 📊 How It Works")
    st.markdown(
        "Allocaid uses an **XGBoost model** trained on humanitarian indicators "
        "(INFORM Risk, Vulnerability, Population, Conflict Probability, etc.) to predict "
        "what CBPF funding **should** look like given a country's need level. "
        "The gap between actual and expected funding reveals which crises are **overlooked**."
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline:** HumData → Databricks (Bronze/Silver/Gold) → MLflow → Streamlit"
    )
    st.markdown(
        "**Data:** [UN OCHA](https://data.humdata.org/) • "
        "[CBPF](https://cbpf.data.unocha.org/) • "
        "[INFORM](https://drmkc.jrc.ec.europa.eu/inform-index/)"
    )

# ============================================================
# APPLY FILTERS
# ============================================================
filtered = scored_df.copy()
if selected_year:
    filtered = filtered[filtered["Year"] == selected_year]
if selected_continent != "All":
    filtered = filtered[filtered["Continent"] == selected_continent]
if show_predicted_only:
    filtered = filtered[filtered["pred_funding"].notna()]

bench_filtered = bench_df.copy()
if selected_year:
    bench_filtered = bench_filtered[bench_filtered["Year"] == selected_year]
if show_predicted_only:
    bench_filtered = bench_filtered[bench_filtered["pred_funding"].notna()]

# ============================================================
# KEY METRICS
# ============================================================
if len(filtered) > 0:
    total_actual = filtered["actual_funding"].sum()
    total_predicted = filtered["pred_funding"].sum() if filtered["pred_funding"].notna().any() else 0
    n_overlooked = filtered["flag_overlooked"].sum()
    n_countries = filtered["ISO3"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${total_actual / 1e6:,.0f}M</h3>
            <p>Total CBPF Funding ({selected_year})</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #f39c12;">
            <h3>${total_predicted / 1e6:,.0f}M</h3>
            <p>Model-Expected Funding</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #e74c3c;">
            <h3>{int(n_overlooked)}</h3>
            <p>Overlooked Crises (≥35% below expected)</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #3498db;">
            <h3>{n_countries}</h3>
            <p>Countries Tracked</p>
        </div>
        """, unsafe_allow_html=True)

    if filtered["funding_ratio_gap"].notna().any():
        worst_idx = filtered["funding_ratio_gap"].idxmin()
        worst = filtered.loc[worst_idx]
        st.markdown(f"""
        <div class="insight-box">
            ⚠️ <strong>Key Finding ({selected_year}):</strong> {worst['Country']} received
            <strong>${worst['actual_funding']/1e6:.1f}M</strong> in CBPF funding —
            but our model expected <strong>${worst['pred_funding']/1e6:.1f}M</strong> based on
            humanitarian need. That's <strong>{abs(worst['funding_ratio_gap']):.0%} below</strong>
            what comparable crises receive.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Global Map",
    "📊 Funding Gaps",
    "🔍 Peer Benchmarking",
    "⚡ Efficiency",
    "📋 Data Explorer",
    "🤖 Allocaid Intelligence"
])

# ============================================================
# TAB 1: GLOBAL MAP
# ============================================================
with tab1:
    st.markdown("### Funding Fairness Map")
    st.markdown(
        "Each country is colored by its **funding ratio gap**: how much actual CBPF funding "
        "deviates from what our model predicts based on humanitarian need. "
        "**Red = underfunded**, **Blue = overfunded** relative to expected."
    )

    map_data = filtered[filtered["funding_ratio_gap"].notna()].copy()

    if len(map_data) > 0:
        map_data["gap_display"] = map_data["funding_ratio_gap"].clip(-1, 2)
        map_data["gap_pct"] = (map_data["funding_ratio_gap"] * 100).round(1)
        map_data["actual_M"] = (map_data["actual_funding"] / 1e6).round(1)
        map_data["expected_M"] = (map_data["pred_funding"] / 1e6).round(1)

        fig_map = px.choropleth(
            map_data,
            locations="ISO3",
            color="gap_display",
            hover_name="Country",
            hover_data={
                "gap_display": False,
                "gap_pct": ":.1f",
                "actual_M": ":.1f",
                "expected_M": ":.1f",
                "INFORM_Risk": ":.1f",
                "ISO3": False,
            },
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            labels={
                "gap_pct": "Funding Gap %",
                "actual_M": "Actual ($M)",
                "expected_M": "Expected ($M)",
                "INFORM_Risk": "INFORM Risk",
            },
            title=f"CBPF Funding Fairness — {selected_year}"
        )
        fig_map.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)"
            ),
            coloraxis_colorbar=dict(
                title="Funding Gap",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-100%", "-50%", "Fair", "+50%", "+100%+"],
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### Bubble Map — Need vs. Funding Gap")

        fig_bubble = px.scatter_geo(
            map_data,
            lat="Latitude",
            lon="Longitude",
            size=np.sqrt(map_data["Need_Proxy"].clip(lower=1).abs()),
            color="funding_ratio_gap",
            hover_name="Country",
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            size_max=25,
            hover_data={
                "actual_M": ":.1f",
                "expected_M": ":.1f",
                "gap_pct": ":.1f",
                "Latitude": False,
                "Longitude": False,
            },
            labels={
                "actual_M": "Actual ($M)",
                "expected_M": "Expected ($M)",
                "gap_pct": "Gap %",
            }
        )
        fig_bubble.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth")
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("No data with model predictions for the selected filters.")

# ============================================================
# TAB 2: FUNDING GAPS
# ============================================================
with tab2:
    st.markdown("### Actual vs. Expected Funding")
    st.markdown(
        "Our XGBoost model predicts what each country **should** receive based on "
        "INFORM Risk, Vulnerability, Population, Conflict Probability, and other indicators. "
        "The difference reveals where funding allocation doesn't match humanitarian need."
    )

    gap_data = filtered[filtered["pred_funding"].notna()].copy()
    gap_data = gap_data.sort_values("funding_ratio_gap")

    if len(gap_data) > 0:
        fig_dumbbell = go.Figure()
        for _, row in gap_data.iterrows():
            color = "#e74c3c" if row["funding_ratio_gap"] < -0.35 else (
                "#3b82f6" if row["funding_ratio_gap"] > 0.35 else "#94a3b8"
            )
            fig_dumbbell.add_trace(go.Scatter(
                x=[row["actual_funding"] / 1e6, row["pred_funding"] / 1e6],
                y=[row["Country"], row["Country"]],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=10),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Country']}</b><br>"
                    f"Actual: ${row['actual_funding']/1e6:.1f}M<br>"
                    f"Expected: ${row['pred_funding']/1e6:.1f}M<br>"
                    f"Gap: {row['funding_ratio_gap']:.0%}<extra></extra>"
                )
            ))

        fig_dumbbell.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#e74c3c"), name="Overlooked (≥35% below)"))
        fig_dumbbell.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#3b82f6"), name="Overfunded (≥35% above)"))
        fig_dumbbell.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#94a3b8"), name="Within range"))

        fig_dumbbell.update_layout(
            height=max(400, len(gap_data) * 35),
            margin=dict(l=0, t=10, b=40),
            xaxis_title="CBPF Funding ($ Millions)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_dumbbell, use_container_width=True)
        st.markdown("*Left dot = actual funding, Right dot = model-expected funding. "
                     "Red lines = crises receiving ≥35% less than expected.*")

        st.markdown("### Funding Ratio Gap by Country")
        fig_gap_bar = px.bar(
            gap_data,
            x="funding_ratio_gap",
            y="Country",
            orientation="h",
            color="funding_ratio_gap",
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            labels={"funding_ratio_gap": "Funding Ratio Gap", "Country": ""},
            hover_data={"actual_funding": ":.0f", "pred_funding": ":.0f"}
        )
        fig_gap_bar.update_layout(
            height=max(400, len(gap_data) * 30),
            margin=dict(l=0, t=10),
            xaxis_tickformat=".0%",
            xaxis_title="← Underfunded | Overfunded →",
            coloraxis_colorbar=dict(tickformat=".0%")
        )
        fig_gap_bar.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig_gap_bar, use_container_width=True)

        st.markdown("### Funding Gap Over Time")
        countries_list = sorted(filtered["Country"].unique())
        selected_ts_country = st.selectbox("Select country", countries_list, key="ts_country")

        ts_data = scored_df[
            (scored_df["Country"] == selected_ts_country) &
            (scored_df["pred_funding"].notna())
        ].sort_values("Year")

        if len(ts_data) > 0:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts_data["Year"], y=ts_data["actual_funding"] / 1e6,
                name="Actual CBPF", mode="lines+markers",
                line=dict(color="#e74c3c", width=3),
                marker=dict(size=8)
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts_data["Year"], y=ts_data["pred_funding"] / 1e6,
                name="Model Expected", mode="lines+markers",
                line=dict(color="#3b82f6", width=3, dash="dash"),
                marker=dict(size=8)
            ))
            fig_ts.update_layout(
                height=400,
                yaxis_title="Funding ($ Millions)",
                xaxis_title="Year",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=30)
            )
            st.plotly_chart(fig_ts, use_container_width=True)

# ============================================================
# TAB 3: PEER BENCHMARKING
# ============================================================
with tab3:
    st.markdown("### Peer Comparison — Is This Crisis an Outlier?")
    st.markdown(
        "For each country, our model finds the **5 most similar crises** (by humanitarian "
        "indicators like INFORM Risk, Vulnerability, Population, GDP, etc.) using nearest-neighbor "
        "matching. We then compare: is this country's funding gap unusual relative to its peers?"
    )

    bench_valid = bench_filtered[bench_filtered["pred_funding"].notna()].copy()

    if len(bench_valid) > 0:
        bench_valid["Country"] = bench_valid["ISO3"].map(ISO_NAMES).fillna(bench_valid["ISO3"])
        bench_valid = bench_valid.sort_values("relative_to_peers")

        fig_peers = px.bar(
            bench_valid,
            x="relative_to_peers",
            y="Country",
            orientation="h",
            color="relative_to_peers",
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            labels={"relative_to_peers": "vs Peers", "Country": ""},
            hover_data={
                "funding_ratio_gap": ":.1%",
                "neighbor_avg_ratio_gap": ":.1%",
                "relative_to_peers": ":.1%",
            }
        )
        fig_peers.update_layout(
            height=max(400, len(bench_valid) * 30),
            margin=dict(l=0, t=10),
            xaxis_title="← Worse than peers | Better than peers →",
            xaxis_tickformat=".0%",
            coloraxis_colorbar=dict(title="vs Peers", tickformat=".0%"),
        )
        fig_peers.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig_peers, use_container_width=True)

        st.markdown("#### Detailed Peer Comparison")
        display_bench = bench_valid[["Country", "actual_funding", "pred_funding",
                                      "funding_ratio_gap", "neighbor_avg_ratio_gap",
                                      "neighbor_avg_funding", "relative_to_peers"]].copy()
        display_bench.columns = ["Country", "Actual ($)", "Expected ($)",
                                  "Own Gap", "Peers Avg Gap", "Peers Avg Funding ($)", "vs Peers"]
        display_bench["Actual ($)"] = display_bench["Actual ($)"].apply(lambda x: f"${x/1e6:.1f}M")
        display_bench["Expected ($)"] = display_bench["Expected ($)"].apply(lambda x: f"${x/1e6:.1f}M")
        display_bench["Own Gap"] = display_bench["Own Gap"].apply(lambda x: f"{x:.0%}")
        display_bench["Peers Avg Gap"] = display_bench["Peers Avg Gap"].apply(lambda x: f"{x:.0%}")
        display_bench["Peers Avg Funding ($)"] = display_bench["Peers Avg Funding ($)"].apply(lambda x: f"${x/1e6:.1f}M")
        display_bench["vs Peers"] = display_bench["vs Peers"].apply(lambda x: f"{x:+.0%}")
        st.dataframe(display_bench, use_container_width=True, hide_index=True)
    else:
        st.info("No benchmarking data available for the selected filters.")

# ============================================================
# TAB 4: EFFICIENCY
# ============================================================
with tab4:
    st.markdown("### CBPF Delivery Efficiency")
    st.markdown(
        "How many beneficiaries does each dollar of CBPF funding reach? "
        "This metric helps identify which country operations are unusually efficient "
        "or inefficient at converting funding into impact."
    )

    eff_data = filtered[filtered["beneficiaries_per_million"].notna()].copy()
    eff_data = eff_data[eff_data["beneficiaries_per_million"] > 0]

    if len(eff_data) > 0:
        eff_data = eff_data.sort_values("beneficiaries_per_million", ascending=False)

        fig_eff = px.bar(
            eff_data,
            x="beneficiaries_per_million",
            y="Country",
            orientation="h",
            color="efficiency_robust_z",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={
                "beneficiaries_per_million": "Beneficiaries per $1M",
                "efficiency_robust_z": "Efficiency Z-Score"
            },
            hover_data={
                "CBPF_Reached": ":,.0f",
                "actual_funding": ":,.0f",
            }
        )
        fig_eff.update_layout(
            height=max(400, len(eff_data) * 30),
            margin=dict(l=0, t=10),
            xaxis_title="Beneficiaries Reached per $1M CBPF",
        )
        st.plotly_chart(fig_eff, use_container_width=True)

        st.markdown("### Efficiency vs. Funding Gap")
        st.markdown("Are the most efficient operations also the most underfunded?")

        eff_scatter = filtered[
            (filtered["beneficiaries_per_million"].notna()) &
            (filtered["funding_ratio_gap"].notna()) &
            (filtered["beneficiaries_per_million"] > 0)
        ].copy()

        if len(eff_scatter) > 0:
            fig_eff_scatter = px.scatter(
                eff_scatter,
                x="funding_ratio_gap",
                y="beneficiaries_per_million",
                text="Country",
                color="Continent",
                size="actual_funding",
                size_max=40,
                labels={
                    "funding_ratio_gap": "Funding Gap (← underfunded | overfunded →)",
                    "beneficiaries_per_million": "Beneficiaries per $1M",
                    "Continent": "Region",
                },
                hover_data={"actual_funding": ":,.0f"}
            )
            fig_eff_scatter.update_traces(textposition="top center", textfont_size=9)
            fig_eff_scatter.update_layout(
                height=500,
                margin=dict(t=10),
                xaxis_tickformat=".0%",
            )
            fig_eff_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_eff_scatter, use_container_width=True)
            st.markdown(
                "*Top-left quadrant = high efficiency but underfunded — these are the crises "
                "that deliver the most impact per dollar yet receive less than expected.*"
            )
    else:
        st.info("No efficiency data available for the selected filters.")

# ============================================================
# TAB 5: DATA EXPLORER
# ============================================================
with tab5:
    st.markdown("### Explore the Data")

    display_cols = ["Country", "ISO3", "Year", "actual_funding", "pred_funding",
                    "funding_ratio_gap", "flag_overlooked", "flag_overfunded",
                    "INFORM_Risk", "Vulnerability", "Need_Proxy", "Pop_Used",
                    "Continent", "CBPF_Reached", "CBPF_Targeted",
                    "beneficiaries_per_million"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    explorer = filtered[display_cols].copy()
    money_cols = ["actual_funding", "pred_funding"]
    for c in money_cols:
        if c in explorer.columns:
            explorer[c] = explorer[c].apply(lambda x: f"${x/1e6:.1f}M" if pd.notna(x) else "—")
    if "funding_ratio_gap" in explorer.columns:
        explorer["funding_ratio_gap"] = explorer["funding_ratio_gap"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "—"
        )
    if "Need_Proxy" in explorer.columns:
        explorer["Need_Proxy"] = explorer["Need_Proxy"].apply(
            lambda x: f"{x/1e6:.1f}M" if pd.notna(x) else "—"
        )
    if "Pop_Used" in explorer.columns:
        explorer["Pop_Used"] = explorer["Pop_Used"].apply(
            lambda x: f"{x/1e6:.1f}M" if pd.notna(x) else "—"
        )

    st.dataframe(explorer, use_container_width=True, hide_index=True)
    csv_data = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_data,
        file_name=f"allocaid_data_{selected_year}.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 6: ALLOCAID INTELLIGENCE (RAG CHATBOT)
# ============================================================

# ── RAG config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
RAG_CSV_PATH = "datanew.csv"
TOP_K        = 6

COUNTRY_ISO = {
    "AF":"Afghanistan","BD":"Bangladesh","ET":"Ethiopia","IQ":"Iraq",
    "ML":"Mali","NE":"Niger","PK":"Pakistan","PS":"Palestine","SD":"Sudan",
    "SY":"Syria","TD":"Chad","YE":"Yemen","SO":"Somalia","SS":"South Sudan",
    "CD":"DR Congo","CF":"Central African Republic","NG":"Nigeria",
    "MM":"Myanmar","UA":"Ukraine","LB":"Lebanon","HT":"Haiti",
    "VE":"Venezuela","MZ":"Mozambique","BF":"Burkina Faso","BI":"Burundi",
    "CM":"Cameroon","CO":"Colombia","ER":"Eritrea","LY":"Libya","ZW":"Zimbabwe",
    "KE":"Kenya","UG":"Uganda","TZ":"Tanzania","RW":"Rwanda","ZA":"South Africa",
}
CRISIS_SEVERITY = {
    "YE":10,"SY":10,"SS":9,"SD":9,"SO":9,"AF":9,"CD":8,"CF":8,"ET":8,
    "MM":8,"NG":7,"ML":7,"NE":7,"IQ":7,"LB":7,"HT":7,"TD":7,"UA":8,
    "BF":8,"MZ":6,"CM":6,"BD":5,"PK":6,"PS":8,"LY":7,"BI":6,"ER":6,
    "VE":6,"CO":5,"ZW":5,"KE":5,"UG":4,"TZ":4,"RW":4,"ZA":4,
}
TEXT_COLUMNS_RAG = [
    "title_narrative","description_narrative","sector_narrative",
    "recipient_country_narrative","participating_org_narrative",
    "result_title_narrative","policy_marker_narrative",
]

RAG_PROMPTS = [
    {"id":"bbr_outliers","icon":"⚠️","title":"Flag Budget Outliers",
     "desc":"Projects with unusually high or low budget-to-activity ratios",
     "query":"Flag projects with unusually high or low budget ratios compared to similar projects. Identify outliers.",
     "viz":"bbr"},
    {"id":"neglected","icon":"🔴","title":"Most Neglected Crises",
     "desc":"Countries where severity far exceeds funding coverage",
     "query":"Which crises are most neglected? Compare crisis severity scores against total funding.",
     "viz":"neglect"},
    {"id":"benchmarks","icon":"📐","title":"Benchmark Projects",
     "desc":"Find well-funded comparable projects for benchmarking",
     "query":"Identify the best benchmark projects — well-funded, high-impact, replicable across similar crisis contexts.",
     "viz":"benchmark"},
    {"id":"sector_gaps","icon":"📊","title":"Sector Funding Gaps",
     "desc":"Which humanitarian clusters are chronically underfunded?",
     "query":"Which sectors and humanitarian clusters receive the least funding relative to need?",
     "viz":"treemap"},
    {"id":"attention_fade","icon":"📉","title":"Fading Crisis Attention",
     "desc":"Crises that lost donor attention over time",
     "query":"Which crises experienced the sharpest decline in project funding after 2018? Show funding fatigue.",
     "viz":"heatmap"},
    {"id":"food_security","icon":"🌾","title":"Food Security Gaps",
     "desc":"Underfunded food and nutrition projects by country",
     "query":"Analyze food security and nutrition project budgets. Which countries are most underfunded in this sector?",
     "viz":"sector_bar","sector_kw":"food"},
    {"id":"health","icon":"🏥","title":"Health Cluster Analysis",
     "desc":"Health project coverage vs crisis severity",
     "query":"Analyze health cluster projects. Flag countries where health funding is lowest relative to crisis severity.",
     "viz":"sector_bar","sector_kw":"health"},
    {"id":"protection","icon":"🛡️","title":"Protection Funding",
     "desc":"Child protection and GBV project coverage",
     "query":"Analyze protection cluster funding including child protection and GBV. Which contexts are most underserved?",
     "viz":"sector_bar","sector_kw":"protection"},
    {"id":"org_efficiency","icon":"🏛️","title":"Organization Efficiency",
     "desc":"Which organizations deliver the most projects per dollar?",
     "query":"Which organizations have the best budget efficiency — most projects relative to total spend? Flag high and low performers.",
     "viz":"org_bar"},
    {"id":"emergency_response","icon":"🚨","title":"Emergency Response Coverage",
     "desc":"Emergency response project gaps by severity tier",
     "query":"Analyze emergency response project coverage. Which high-severity countries have the fewest emergency projects?",
     "viz":"scatter"},
    {"id":"policy_recs","icon":"📋","title":"Policy Recommendations",
     "desc":"AI-generated actionable recommendations from data",
     "query":"Based on funding gaps, neglect scores, and project data, generate 3 specific policy recommendations for donors and humanitarian coordinators.",
     "viz":"none"},
    {"id":"compare_sudan_yemen","icon":"⚖️","title":"Sudan vs Yemen",
     "desc":"Side-by-side funding comparison for two top crises",
     "query":"Compare Sudan and Yemen across total budget, project count, sector coverage, and funding efficiency. Which is more overlooked?",
     "viz":"compare","countries":["SD","YE"]},
]

# ── RAG data helpers ──────────────────────────────────────────────────────────
def safe_parse(val) -> str:
    if pd.isna(val) or val == "": return ""
    try:
        p = ast.literal_eval(str(val))
        if isinstance(p, list):
            return " | ".join(str(v) for v in p if str(v) not in ["nan","None",""])
        return str(p)
    except: return str(val)

def safe_list(val) -> list:
    if pd.isna(val): return []
    try:
        p = ast.literal_eval(str(val))
        return p if isinstance(p, list) else [p]
    except: return [str(val)]

def extract_max_budget(val) -> float:
    nums = [x for x in safe_list(val) if isinstance(x,(int,float)) and x>0]
    return max(nums) if nums else 0.0

def extract_years(val) -> List[int]:
    years = []
    for d in safe_list(val):
        try: years.append(int(str(d)[:4]))
        except: pass
    return [y for y in years if 1990<y<2035]

@st.cache_data(show_spinner=False)
def load_rag_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_budget"]     = df["budget_value"].apply(extract_max_budget)
    df["_years"]      = df["activity_date_iso_date"].apply(extract_years)
    df["_start_year"] = df["_years"].apply(lambda y: min(y) if y else None)
    df["_end_year"]   = df["_years"].apply(lambda y: max(y) if y else None)
    df["_countries"]  = df["recipient_country_code"].apply(safe_list)
    df["_sectors"]    = df["sector_narrative"].apply(safe_list)
    df["_title"]      = df["title_narrative"].apply(safe_parse)
    df["_orgs"]       = df["participating_org_narrative"].apply(safe_parse)
    return df

@st.cache_data(show_spinner=False)
def get_rag_country_stats(_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, name in COUNTRY_ISO.items():
        sub = _df[_df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        if not len(sub): continue
        tb  = sub["_budget"].sum()
        pc  = len(sub)
        sev = CRISIS_SEVERITY.get(iso, 3)
        gap = round(max(0, sev - min(tb/1e8,10)), 2)
        rows.append({"iso":iso,"country":name,"projects":pc,"total_budget":tb,
                     "severity":sev,"gap_score":gap})
    return pd.DataFrame(rows).sort_values("gap_score", ascending=False)

@st.cache_resource
def get_groq():
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except: return None

def ask_groq(gc, question: str, context: str) -> str:
    if not gc: return "⚠️ Groq unavailable — check API key."
    system = """You are Allocaid, a humanitarian policy intelligence assistant.
You analyze real IATI aid project data to surface funding gaps and policy opportunities.
Respond with concise bullet points. Be specific — cite countries, budgets, organizations, sector names.
Focus on: budget ratios, neglected crises, benchmark projects, actionable policy implications.
Use bold for key figures. Keep response under 250 words."""
    try:
        resp = gc.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":f"Context from dataset:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.35, max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error: {e}"

def build_rag_context(df: pd.DataFrame, stats: pd.DataFrame, prompt: dict) -> str:
    lines = [f"Dataset: {len(df):,} projects across {len(stats)} countries."]
    viz = prompt.get("viz","")
    if viz == "bbr":
        budgets = df[df["_budget"]>0]["_budget"]
        median  = budgets.median()
        q25, q75 = budgets.quantile(0.25), budgets.quantile(0.75)
        high = df[df["_budget"] > q75*4].nlargest(5,"_budget")
        low  = df[(df["_budget"]>0) & (df["_budget"]<q25*0.2)].nsmallest(5,"_budget")
        lines.append(f"Median project budget: ${median:,.0f}. Q25=${q25:,.0f}, Q75=${q75:,.0f}.")
        lines.append("High-budget outliers:")
        for _,r in high.iterrows():
            lines.append(f"  - {r['_title'][:60]} | ${r['_budget']:,.0f} | {safe_parse(r.get('recipient_country_narrative',''))[:40]}")
        lines.append("Low-budget projects:")
        for _,r in low.iterrows():
            lines.append(f"  - {r['_title'][:60]} | ${r['_budget']:,.0f} | {safe_parse(r.get('recipient_country_narrative',''))[:40]}")
    elif viz == "neglect":
        for _,r in stats.head(8).iterrows():
            lines.append(f"  {r['country']}: severity={r['severity']}/10, budget=${r['total_budget']/1e6:.1f}M, neglect={r['gap_score']}/10")
    elif viz == "benchmark":
        top = df.nlargest(8,"_budget")
        for _,r in top.iterrows():
            lines.append(f"  - {r['_title'][:60]} | ${r['_budget']:,.0f} | {r['_orgs'][:40]}")
    elif viz in ("treemap","sector_bar"):
        kw = prompt.get("sector_kw","")
        sub = df if not kw else df[df["_sectors"].apply(
            lambda s: any(kw.lower() in x.lower() for x in s if isinstance(x,str)))]
        all_secs = [s for row in sub["_sectors"] for s in row
                    if isinstance(s,str) and len(s)>3 and not s.replace(".","").isdigit()]
        top_secs = Counter(all_secs).most_common(10)
        lines.append(f"Top sectors{' ('+kw+')' if kw else ''}:")
        for sec,cnt in top_secs:
            lines.append(f"  - {sec}: {cnt} projects")
        lines.append(f"Total budget in sector: ${sub['_budget'].sum():,.0f}")
    elif viz == "org_bar":
        orgs = []
        for _,r in df.iterrows():
            for o in r["_orgs"].split(" | ") if r["_orgs"] else []:
                if o.strip(): orgs.append({"org":o.strip()[:50],"budget":r["_budget"]})
        if orgs:
            odf = pd.DataFrame(orgs).groupby("org").agg(projects=("budget","count"),
                                                          total=("budget","sum")).reset_index()
            odf["per_project"] = odf["total"]/odf["projects"]
            for _,r in odf.nlargest(6,"projects").iterrows():
                lines.append(f"  {r['org']}: {r['projects']} projects, ${r['total']/1e6:.1f}M total, ${r['per_project']:,.0f}/project")
    elif viz == "scatter":
        for _,r in stats.iterrows():
            lines.append(f"  {r['country']}: {r['projects']} projects, severity={r['severity']}, budget=${r['total_budget']/1e6:.1f}M")
    elif viz == "compare":
        for iso in prompt.get("countries",["SD","YE"]):
            name = COUNTRY_ISO.get(iso,iso)
            sub  = df[df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
            secs = Counter([s for r in sub["_sectors"] for s in r
                            if isinstance(s,str) and len(s)>3]).most_common(3)
            lines.append(f"{name}: {len(sub)} projects, ${sub['_budget'].sum()/1e6:.1f}M, "
                         f"severity={CRISIS_SEVERITY.get(iso,0)}/10, "
                         f"top sectors: {', '.join([s for s,_ in secs])}")
    elif viz == "heatmap":
        for _,r in stats.head(10).iterrows():
            sub = df[df["_countries"].apply(lambda c: r['iso'] in [x.upper() for x in c])]
            early = sub[sub["_years"].apply(lambda y: any(yr<2018 for yr in y))].shape[0]
            late  = sub[sub["_years"].apply(lambda y: any(yr>=2018 for yr in y))].shape[0]
            lines.append(f"  {r['country']}: pre-2018={early} projects, post-2018={late} projects")
    return "\n".join(lines)

# ── RAG visualizations ────────────────────────────────────────────────────────
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(249,250,251,0.6)",
            font=dict(color="#111827", family="sans-serif", size=11),
            margin=dict(l=0,r=0,t=20,b=0))
GRID = dict(gridcolor="#e5e7eb", linecolor="#e5e7eb")

def rag_fig_bbr(df):
    b = df[df["_budget"]>0].copy()
    median = b["_budget"].median()
    b["ratio"] = b["_budget"] / median
    b["status"] = b["ratio"].apply(lambda r: "High outlier" if r>4 else ("Low outlier" if r<0.15 else "Normal"))
    b["country"] = b["_countries"].apply(lambda c: COUNTRY_ISO.get(c[0].upper() if c else "","Unknown") if c else "Unknown")
    b["label"]   = b["_title"].str[:40]
    cmap = {"High outlier":"#f85149","Low outlier":"#3fb950","Normal":"#9ca3af"}
    fig = px.scatter(b.head(200), x="ratio", y="label", color="status",
        color_discrete_map=cmap, size_max=12,
        hover_data={"ratio":":.2f","status":True},
        labels={"ratio":"Budget / Median","label":"Project","status":""},
        title="Budget Ratio vs Median — Outlier Detection")
    fig.add_vline(x=1, line_dash="dash", line_color="#9ca3af", opacity=0.5)
    fig.add_vline(x=4, line_dash="dot",  line_color="#f85149", opacity=0.4)
    fig.add_vline(x=0.15, line_dash="dot", line_color="#3fb950", opacity=0.4)
    fig.update_layout(**DARK, height=360, xaxis=dict(**GRID,type="log"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=9)),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def rag_fig_neglect(stats):
    df = stats.head(12).sort_values("gap_score", ascending=True)
    colors = ["#f85149" if g>=8 else "#d29922" if g>=5 else "#3fb950" for g in df["gap_score"]]
    fig = go.Figure(go.Bar(
        x=df["gap_score"], y=df["country"], orientation="h",
        marker_color=colors,
        customdata=np.stack([df["total_budget"]/1e6, df["projects"], df["severity"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Neglect: %{x:.1f}<br>Budget: $%{customdata[0]:.1f}M<br>"
                      "Projects: %{customdata[1]}<br>Severity: %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(**DARK, title="Crisis Neglect Score (Severity − Funding Coverage)",
        xaxis=dict(**GRID, title="Neglect Score", range=[0,11]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"), height=360, showlegend=False)
    return fig

def rag_fig_benchmark(df):
    top = df.nlargest(15,"_budget").copy()
    top["budget_M"] = top["_budget"]/1e6
    top["label"]    = top["_title"].str[:45]
    top["org"]      = top["_orgs"].str[:35]
    top["country"]  = top["_countries"].apply(lambda c: COUNTRY_ISO.get(c[0].upper() if c else "","?") if c else "?")
    fig = px.bar(top, x="budget_M", y="label", orientation="h", color="country",
        hover_data={"org":True,"budget_M":":.1f"},
        labels={"budget_M":"Budget (M USD equiv.)","label":"Project","country":"Country"},
        title="Top Benchmark Projects by Budget",
        color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(**DARK, height=400,
        xaxis=dict(**GRID), yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=9)),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9)))
    return fig

def rag_fig_treemap(df, kw=""):
    sub = df if not kw else df[df["_sectors"].apply(
        lambda s: any(kw.lower() in x.lower() for x in s if isinstance(x,str)))]
    rows = []
    for _,row in sub.iterrows():
        for s in row["_sectors"]:
            if isinstance(s,str) and len(s)>3 and not s.replace(".","").isdigit():
                rows.append({"sector":s[:45],"budget":row["_budget"]})
    if not rows: return None
    sec = pd.DataFrame(rows).groupby("sector")["budget"].agg(["sum","count"]).reset_index()
    sec.columns = ["sector","total_budget","count"]
    sec = sec[sec["total_budget"]>0].nlargest(14,"total_budget")
    fig = px.treemap(sec, path=["sector"], values="total_budget", color="count",
        color_continuous_scale=["#fff7ed","#e07b39"],
        hover_data={"count":True,"total_budget":":,.0f"},
        title=f"Sector Funding Distribution{' — '+kw.title() if kw else ''}")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827",family="sans-serif",size=11),
        margin=dict(l=0,r=0,t=30,b=0), height=360)
    fig.update_traces(marker=dict(line=dict(color="#ffffff",width=1.5)))
    return fig

def rag_fig_sector_bar(df, stats, kw):
    sub = df[df["_sectors"].apply(
        lambda s: any(kw.lower() in x.lower() for x in s if isinstance(x,str)))]
    rows = []
    for iso,name in COUNTRY_ISO.items():
        s2 = sub[sub["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        if not len(s2): continue
        sev = CRISIS_SEVERITY.get(iso,3)
        tb  = s2["_budget"].sum()
        rows.append({"country":name,"projects":len(s2),"budget_M":tb/1e6,"severity":sev,
                     "gap":max(0, sev - min(tb/5e7,10))})
    if not rows: return None
    sdf = pd.DataFrame(rows).sort_values("gap", ascending=False).head(14)
    fig = px.bar(sdf, x="country", y="budget_M", color="gap",
        color_continuous_scale=["#3fb950","#d29922","#f85149"],
        hover_data={"projects":True,"severity":True,"gap":":.1f"},
        labels={"budget_M":f"{kw.title()} Budget (M)","country":"","gap":"Gap Score"},
        title=f"{kw.title()} Sector Funding by Country — Sorted by Gap")
    fig.update_layout(**DARK, xaxis=dict(**GRID,tickangle=-35),
        yaxis=dict(**GRID), height=360,
        coloraxis_colorbar=dict(title="Gap",thickness=10,tickfont=dict(color="#6b7280")))
    return fig

def rag_fig_org_bar(df):
    orgs = []
    for _,r in df.iterrows():
        for o in (r["_orgs"] or "").split(" | "):
            if o.strip() and len(o.strip())>3:
                orgs.append({"org":o.strip()[:45],"budget":r["_budget"]})
    if not orgs: return None
    odf = pd.DataFrame(orgs).groupby("org").agg(
        projects=("budget","count"), total=("budget","sum")).reset_index()
    odf["per_project"] = (odf["total"]/odf["projects"]/1e6).round(2)
    top = odf.nlargest(12,"projects")
    fig = px.scatter(top, x="projects", y="total", size="per_project", color="per_project",
        hover_name="org", hover_data={"per_project":":.2f","projects":True},
        color_continuous_scale=["#dbeafe","#2563eb","#e07b39"],
        labels={"projects":"Number of Projects","total":"Total Budget","per_project":"M$/Project"},
        title="Organization Efficiency — Projects vs Total Budget")
    fig.update_layout(**DARK, xaxis=dict(**GRID), yaxis=dict(**GRID), height=360,
        coloraxis_colorbar=dict(title="M$/proj",thickness=10,tickfont=dict(color="#6b7280")))
    return fig

def rag_fig_scatter(stats):
    fig = px.scatter(stats, x="projects", y="severity", size="gap_score",
        color="gap_score", hover_name="country",
        color_continuous_scale=["#3fb950","#d29922","#f85149"],
        hover_data={"total_budget":":,.0f","gap_score":":.1f"},
        labels={"projects":"Projects","severity":"Crisis Severity","gap_score":"Neglect"},
        title="Emergency Response Coverage — Projects vs Crisis Severity",
        size_max=30)
    fig.update_layout(**DARK, xaxis=dict(**GRID), yaxis=dict(**GRID), height=360,
        coloraxis_colorbar=dict(title="Neglect",thickness=10,tickfont=dict(color="#6b7280")))
    return fig

def rag_fig_heatmap(df):
    years = list(range(2012,2025))
    rows  = []
    for iso,name in COUNTRY_ISO.items():
        sub = df[df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        if not len(sub): continue
        rd = {"country":name}
        for yr in years:
            rd[str(yr)] = sub[sub["_years"].apply(lambda y: yr in y)].shape[0]
        rd["total"] = len(sub)
        rows.append(rd)
    if not rows: return None
    heat = pd.DataFrame(rows).sort_values("total",ascending=False).head(16)
    mat  = heat[["country"]+[str(y) for y in years]].set_index("country")
    fig  = go.Figure(data=go.Heatmap(
        z=mat.values, x=[str(y) for y in years], y=mat.index.tolist(),
        colorscale=[[0,"#fff7ed"],[0.4,"#e07b39"],[1,"#dc2626"]],
        colorbar=dict(title="Projects",tickfont=dict(color="#6b7280"),thickness=10),
    ))
    fig.update_layout(**DARK, title="Crisis Attention Over Time — Funding Fatigue",
        xaxis=dict(**GRID), yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=10)),
        height=380)
    return fig

def rag_fig_compare(df, stats, countries):
    rows = []
    for iso in countries:
        name = COUNTRY_ISO.get(iso,iso)
        sub  = df[df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        secs = Counter([s for r in sub["_sectors"] for s in r
                        if isinstance(s,str) and len(s)>3]).most_common(5)
        for sec,cnt in secs:
            rows.append({"country":name,"sector":sec[:40],"projects":cnt,
                         "budget":sub[sub["_sectors"].apply(lambda s: sec in s)]["_budget"].sum()/1e6})
    if not rows: return None
    cdf = pd.DataFrame(rows)
    fig = px.bar(cdf, x="sector", y="budget", color="country", barmode="group",
        labels={"budget":"Budget (M)","sector":"Sector","country":""},
        title=f"Sector Funding Comparison: {' vs '.join([COUNTRY_ISO.get(c,c) for c in countries])}",
        color_discrete_sequence=["#58a6ff","#e07b39"])
    fig.update_layout(**DARK, xaxis=dict(**GRID,tickangle=-30),
        yaxis=dict(**GRID), height=360,
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def get_rag_viz(prompt, df, stats):
    viz = prompt.get("viz","none")
    kw  = prompt.get("sector_kw","")
    if viz=="bbr":         return rag_fig_bbr(df)
    if viz=="neglect":     return rag_fig_neglect(stats)
    if viz=="benchmark":   return rag_fig_benchmark(df)
    if viz=="treemap":     return rag_fig_treemap(df, kw)
    if viz=="sector_bar":  return rag_fig_sector_bar(df, stats, kw)
    if viz=="org_bar":     return rag_fig_org_bar(df)
    if viz=="scatter":     return rag_fig_scatter(stats)
    if viz=="heatmap":     return rag_fig_heatmap(df)
    if viz=="compare":     return rag_fig_compare(df, stats, prompt.get("countries",["SD","YE"]))
    return None

def format_chat_content(text: str, role: str) -> str:
    """Convert markdown-style AI responses to clean styled HTML."""
    if role == "user":
        return f'<span style="font-size:0.84rem;">{text}</span>'

    import html as html_lib
    lines = text.split("\n")
    result = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            result.append("</ul>")
            in_ul = False
        if in_ol:
            result.append("</ol>")
            in_ol = False

    def inline_fmt(s):
        # bold
        import re
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#111827;">\1</strong>', s)
        # italic
        s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
        # backtick code
        s = re.sub(r'`(.+?)`', r'<code style="background:#f3f4f6;color:#1d4ed8;padding:1px 5px;border-radius:3px;font-size:0.82rem;">\1</code>', s)
        return s

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.lstrip()

        # blank line
        if not stripped:
            close_lists()
            continue

        # headings
        if stripped.startswith("### "):
            close_lists()
            result.append(f'<div style="font-size:0.82rem;font-weight:700;color:#e07b39;margin:0.8rem 0 0.3rem;">{inline_fmt(stripped[4:])}</div>')
        elif stripped.startswith("## "):
            close_lists()
            result.append(f'<div style="font-size:0.88rem;font-weight:700;color:#e07b39;margin:0.8rem 0 0.3rem;">{inline_fmt(stripped[3:])}</div>')
        elif stripped.startswith("# "):
            close_lists()
            result.append(f'<div style="font-size:0.95rem;font-weight:700;color:#e07b39;margin:0.8rem 0 0.4rem;">{inline_fmt(stripped[2:])}</div>')

        # unordered bullets: *, -, +
        elif stripped.startswith(("* ", "- ", "+ ")):
            if not in_ul:
                close_lists()
                result.append('<ul style="margin:0.4rem 0 0.4rem 1.1rem;padding:0;list-style:none;">')
                in_ul = True
            bullet_text = inline_fmt(stripped[2:])
            result.append(f'<li style="position:relative;padding-left:1rem;margin-bottom:0.25rem;font-size:0.83rem;line-height:1.6;color:#374151;">'
                           f'<span style="position:absolute;left:0;color:#e07b39;">›</span>{bullet_text}</li>')

        # numbered list
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)" :
            if not in_ol:
                close_lists()
                result.append('<ol style="margin:0.4rem 0 0.4rem 1.1rem;padding:0;list-style:none;counter-reset:item;">')
                in_ol = True
            num_end = stripped.index(stripped[1]) + 1
            ol_text = inline_fmt(stripped[num_end:].lstrip())
            num = stripped[:num_end-1]
            result.append(f'<li style="padding-left:1.5rem;margin-bottom:0.25rem;font-size:0.83rem;line-height:1.6;position:relative;color:#374151;">'
                           f'<span style="position:absolute;left:0;color:#e07b39;font-weight:600;">{num}.</span>{ol_text}</li>')

        # plain paragraph
        else:
            close_lists()
            result.append(f'<p style="margin:0.3rem 0;font-size:0.83rem;line-height:1.65;color:#374151;">{inline_fmt(stripped)}</p>')

    close_lists()
    return "\n".join(result)


# ── Tab 6 render ──────────────────────────────────────────────────────────────
with tab6:
    gc = get_groq()

    if not os.path.exists(RAG_CSV_PATH):
        st.warning(f"RAG dataset not found: `{RAG_CSV_PATH}`. Place `datanew.csv` in the app directory to enable Allocaid Intelligence.")
    else:
        with st.spinner("Loading intelligence dataset..."):
            rag_df    = load_rag_data(RAG_CSV_PATH)
            rag_stats = get_rag_country_stats(rag_df)

        # Init session state
        if "rag_messages"      not in st.session_state: st.session_state.rag_messages = []
        if "rag_active_prompt" not in st.session_state: st.session_state.rag_active_prompt = None

        # Header
        st.markdown("""
        <div style="padding:0.7rem 0 1rem 0; border-bottom:1px solid #e5e7eb; margin-bottom:1rem;">
            <span style="font-size:1.4rem; font-weight:700;">🤖 Allocaid Intelligence</span>
            <span style="font-size:0.75rem; color:#9ca3af; margin-left:1rem; text-transform:uppercase; letter-spacing:0.08em;">
                Humanitarian Policy RAG Assistant
            </span>
        </div>
        """, unsafe_allow_html=True)

        left_rag = st.container()

        # ── LEFT: chat ────────────────────────────────────────────────────────
        with left_rag:
            # Chat history
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Intelligence Assistant</div>', unsafe_allow_html=True)

            chat_html = '<div class="chat-scroll">'
            if not st.session_state.rag_messages:
                chat_html += """<div style="color:#9ca3af;font-size:0.83rem;padding:0.8rem 0;line-height:1.8;">
                    Select a prompt above to generate a policy analysis,<br>
                    or type your own question below.
                </div>"""
            else:
                for msg in st.session_state.rag_messages:
                    role_lbl = "You" if msg["role"]=="user" else "Allocaid"
                    cls = "msg-user" if msg["role"]=="user" else "msg-ai"
                    rendered = format_chat_content(msg["content"], msg["role"])
                    chat_html += f'<div class="msg {cls}"><div class="msg-role">{role_lbl}</div>{rendered}</div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            inp_c, btn_c, clr_c = st.columns([6,1,1])
            with inp_c:
                user_q = st.text_input("q", placeholder="Ask a custom question...",
                                       label_visibility="collapsed", key="rag_chat_input")
            with btn_c:
                send = st.button("↑", use_container_width=True, key="rag_send")
            with clr_c:
                if st.button("✕", use_container_width=True, key="rag_clear"):
                    st.session_state.rag_messages      = []
                    st.session_state.rag_active_prompt = None
                    st.rerun()

            if send and user_q.strip():
                kw = user_q.lower()

                # ── Smart viz detection from free-text ──────────────────────
                iso2_map = {
                    "afghanistan":"AF","bangladesh":"BD","ethiopia":"ET","iraq":"IQ",
                    "mali":"ML","niger":"NE","pakistan":"PK","palestine":"PS","sudan":"SD",
                    "syria":"SY","chad":"TD","yemen":"YE","somalia":"SO","south sudan":"SS",
                    "dr congo":"CD","congo":"CD","central african republic":"CF","nigeria":"NG",
                    "myanmar":"MM","ukraine":"UA","lebanon":"LB","haiti":"HT",
                    "venezuela":"VE","mozambique":"MZ","burkina faso":"BF","burundi":"BI",
                    "cameroon":"CM","colombia":"CO","eritrea":"ER","libya":"LY","zimbabwe":"ZW",
                    "kenya":"KE","uganda":"UG","tanzania":"TZ","rwanda":"RW","south africa":"ZA",
                }
                mentioned_isos = []
                for country_name, iso in iso2_map.items():
                    if country_name in kw and iso not in mentioned_isos:
                        mentioned_isos.append(iso)
                # also catch bare ISO codes
                for iso in COUNTRY_ISO:
                    if iso.lower() in kw.split() and iso not in mentioned_isos:
                        mentioned_isos.append(iso)

                if len(mentioned_isos) >= 2:
                    p = {"viz": "compare", "countries": mentioned_isos[:2]}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif len(mentioned_isos) == 1:
                    p = {"viz": "scatter"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["budget", "outlier", "ratio"]):
                    p = {"viz": "bbr"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["neglect", "ignored", "gap", "severity", "worst"]):
                    p = {"viz": "neglect"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["food", "nutrition", "hunger"]):
                    p = {"viz": "sector_bar", "sector_kw": "food"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["health", "medical", "hospital"]):
                    p = {"viz": "sector_bar", "sector_kw": "health"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["protection", "gbv", "child"]):
                    p = {"viz": "sector_bar", "sector_kw": "protection"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["sector", "cluster"]):
                    p = {"viz": "treemap"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["org", "organization", "ngo", "efficient", "efficiency"]):
                    p = {"viz": "org_bar"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["time", "trend", "year", "decline", "fade", "attention"]):
                    p = {"viz": "heatmap"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["emergency", "response", "coverage"]):
                    p = {"viz": "scatter"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["benchmark", "best", "top", "replicable"]):
                    p = {"viz": "benchmark"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                else:
                    mask = rag_df.apply(lambda r: any(kw in str(r.get(c,"")).lower() for c in TEXT_COLUMNS_RAG), axis=1)
                    hits = rag_df[mask].head(6)
                    ctx_lines = [f"Dataset: {len(rag_df):,} projects."]
                    for _,r in hits.iterrows():
                        ctx_lines.append(f"- {r['_title'][:60]} | ${r['_budget']:,.0f} | "
                                          f"{safe_parse(r.get('recipient_country_narrative',''))[:30]}")
                    ctx = "\n".join(ctx_lines)

                st.session_state.rag_messages.append({"role":"user","content":user_q})
                reply = ask_groq(gc, user_q, ctx)
                st.session_state.rag_messages.append({"role":"assistant","content":reply})
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)



# ============================================================
# METHODOLOGY
# ============================================================
with st.expander("📐 Methodology — How Allocaid Works"):
    st.markdown("""
    **The Core Question:** Given a country's humanitarian indicators, how much CBPF funding
    *should* it receive? Where does actual funding deviate from what the data suggests is fair?

    **Model:** XGBoost regression trained with walk-forward time-series validation (2018-2019
    as initial training window, predictions from 2020 onward). Target variable is log-transformed
    Total CBPF allocation.

    **Features (19 indicators):** INFORM Risk, Vulnerability, Conflict Probability, Food Security,
    Governance, Healthcare Access, Uprooted People, Hazard Exposure, GDP per capita, Population,
    Population Density, Urban %, Latitude/Longitude, and year-over-year changes in risk and vulnerability.

    **Key Design Choice:** We intentionally exclude prior-year CBPF funding as a feature. Including
    it would let the model learn "countries that got funded before get funded again" — which bakes
    in historical bias rather than surfacing it.

    **Fairness Gap:** `(Actual / Expected) - 1`. A value of -0.5 means the country received 50%
    less than the model predicts for crises with similar humanitarian indicators.

    **Peer Benchmarking:** For each country-year, we find the 5 most similar observations using
    standardized Euclidean distance across all numeric features, then compare funding gaps.

    **Pipeline:**
    - 🥉 **Bronze:** Raw data ingested from HumData into Delta Lake
    - 🥈 **Silver:** Cleaned, typed, Need_Proxy engineered, data quality validated
    - 🥇 **Gold:** XGBoost model trained with MLflow experiment tracking (3 hyperparameter configs),
      SHAP explainability, nearest-neighbor benchmarking
    - 📦 **Export:** Gold tables → CSV → Streamlit Cloud

    All model experiments tracked in **MLflow** on Databricks. Best model selected by walk-forward R².
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <strong>Allocaid</strong> — Built with Databricks + Streamlit for Hacklytics 2026<br>
    Databricks × United Nations Geo-Insight Challenge<br>
    Data: UN OCHA HumData • CBPF Data Hub • INFORM Risk Index
</div>
""", unsafe_allow_html=True)
