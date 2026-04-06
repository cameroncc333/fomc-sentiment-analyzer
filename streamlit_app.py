"""
FOMC Sentiment Analyzer — Live Dashboard
=========================================
Interactive Streamlit dashboard displaying pre-computed FinBERT sentiment
analysis results. The heavy computation (transformer model scoring) runs
locally via fomc_sentiment_analyzer.py; this dashboard consumes the
exported CSVs and renders them interactively.

Architecture note: The FinBERT model (110M parameters, 438MB) requires
~2GB RAM — beyond free-tier hosting limits. Separating compute from
visualization mirrors how institutional quant teams deploy ML research:
heavy inference offline, lightweight display online.

Author: Cameron Camarotti | GitHub: cameroncc333
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="FOMC Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================



# ============================================================================
# DATA LOADING
# ============================================================================

DATA_DIR = Path("output")

@st.cache_data
def load_data():
    """Load all pre-computed CSV files."""
    data = {}
    files = {
        "pmsi": "pmsi.csv",
        "correlations": "correlations.csv",
        "regime": "regime_analysis.csv",
        "sensitivity": "sensitivity_ranking.csv",
        "scored_headlines": "scored_headlines.csv",
        "sector_returns": "sector_returns.csv",
    }
    for key, filename in files.items():
        path = DATA_DIR / filename
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            st.error(f"Missing data file: {path}. Run fomc_sentiment_analyzer.py first.")
            return None
    return data

data = load_data()
if data is None:
    st.stop()

pmsi = data["pmsi"]
corr_df = data["correlations"]
regime_df = data["regime"]
sensitivity_df = data["sensitivity"]
scored_df = data["scored_headlines"]
returns_df = data["sector_returns"]

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🧠 FOMC Sentiment Analyzer")
    st.markdown("---")
    st.markdown(
        "**Research Question:**  \n"
        "Does pre-meeting financial news sentiment predict which S&P 500 "
        "sectors outperform after Federal Reserve decisions?"
    )
    st.markdown("---")

    st.markdown("### Dataset")
    st.metric("FOMC Meetings", len(pmsi))
    st.metric("Headlines Scored", len(scored_df))
    st.metric("Sector ETFs", scored_df.get("fomc_date", pd.Series()).nunique() and 11)
    st.metric("Correlation Tests", len(corr_df))

    st.markdown("---")
    st.markdown("### Model")
    st.markdown(
        "**FinBERT** (ProsusAI/finbert)  \n"
        "110M params · Fine-tuned on 48K  \n"
        "financial text samples  \n"
        "Classes: Positive / Negative / Neutral"
    )

    st.markdown("---")
    st.markdown(
        "### Architecture\n"
        "FinBERT (438MB) runs locally.  \n"
        "This dashboard displays  \n"
        "pre-computed results.  \n"
        "Heavy compute offline,  \n"
        "lightweight viz online."
    )

    st.markdown("---")
    st.markdown(
        "Built by [Cameron Camarotti](https://github.com/cameroncc333)  \n"
        "Founder, [All Around Services](https://allaroundservice.com)"
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown("# FOMC Sentiment Analyzer")
st.markdown(
    "FinBERT transformer NLP × S&P 500 sector return analysis across 90+ Federal Reserve meetings"
)

# Top-level metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Mean PMSI", f"{pmsi['pmsi'].mean():.3f}")
with col2:
    n_bull = int((pmsi['pmsi'] > 0.15).sum())
    st.metric("Bullish Meetings", n_bull)
with col3:
    n_bear = int((pmsi['pmsi'] < -0.15).sum())
    st.metric("Bearish Meetings", n_bear)
with col4:
    n_sig = int(corr_df["significant_bh_5pct"].sum()) if "significant_bh_5pct" in corr_df.columns else 0
    st.metric("BH-Significant", f"{n_sig}/33")
with col5:
    if "significant_raw_5pct" in corr_df.columns:
        n_raw = int(corr_df["significant_raw_5pct"].sum())
    else:
        n_raw = 0
    st.metric("Raw Significant", f"{n_raw}/33")

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Sentiment Timeline",
    "🔥 Correlation Heatmap",
    "📈 Regime Analysis",
    "🎯 Top Sectors",
    "📋 Sensitivity Ranking",
    "🔬 Raw Data",
])

# Color scheme
GREEN = "#00e676"
RED = "#ff1744"
GRAY = "#607d8b"
GOLD = "#ffd740"
BLUE = "#42a5f5"
BG = "#0a0a0a"
CARD_BG = "#111111"

# --------------------------------------------------------------------------
# TAB 1: SENTIMENT TIMELINE
# --------------------------------------------------------------------------

with tab1:
    st.markdown("### Pre-Meeting Sentiment Index (PMSI) · 2015–2026")
    st.markdown(
        "Each bar represents the average FinBERT sentiment score of financial "
        "news headlines in the 5 trading days before an FOMC meeting. "
        "Green = bullish (>0.15), red = bearish (<-0.15), gray = neutral."
    )

    dates = pd.to_datetime(pmsi["fomc_date"])
    scores = pmsi["pmsi"].values
    colors = [GREEN if s > 0.15 else RED if s < -0.15 else GRAY for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=scores, marker_color=colors, marker_opacity=0.85,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>PMSI: %{y:.3f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_color="#444444", line_width=1)
    fig.add_hline(y=0.15, line_color=GREEN, line_width=0.5, line_dash="dash", opacity=0.4)
    fig.add_hline(y=-0.15, line_color=RED, line_width=0.5, line_dash="dash", opacity=0.4)
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="#e0e0e0", family="DM Sans"),
        xaxis=dict(gridcolor="#1a1a1a"),
        yaxis=dict(gridcolor="#1a1a1a", title="PMSI Score"),
        height=450, margin=dict(l=60, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution below
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=scored_df["sentiment_score"], nbinsx=50,
            marker_color=BLUE, opacity=0.8,
            hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>"
        ))
        fig_hist.add_vline(x=0, line_color=RED, line_dash="dash", opacity=0.5)
        fig_hist.add_vline(x=scored_df["sentiment_score"].mean(), line_color=GREEN, line_width=2)
        fig_hist.update_layout(
            title="FinBERT Score Distribution",
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color="#e0e0e0", family="DM Sans"),
            xaxis=dict(title="Sentiment Score (-1 to +1)", gridcolor="#1a1a1a"),
            yaxis=dict(title="Count", gridcolor="#1a1a1a"),
            height=350, margin=dict(l=60, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        label_counts = scored_df["label"].value_counts()
        pie_colors = {"positive": GREEN, "negative": RED, "neutral": GRAY}
        fig_pie = go.Figure(data=[go.Pie(
            labels=[l.capitalize() for l in label_counts.index],
            values=label_counts.values,
            marker=dict(colors=[pie_colors.get(l, GRAY) for l in label_counts.index],
                       line=dict(color=BG, width=2)),
            textfont=dict(color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        )])
        fig_pie.update_layout(
            title="Classification Breakdown",
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color="#e0e0e0", family="DM Sans"),
            height=350, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 2: CORRELATION HEATMAP
# --------------------------------------------------------------------------

with tab2:
    st.markdown("### Sentiment–Return Correlation Matrix")
    st.markdown(
        "Pearson correlation between PMSI and post-meeting sector returns. "
        "✦ marks correlations significant at α=0.05 after Benjamini-Hochberg "
        "false discovery rate correction."
    )

    return_type = st.radio(
        "Return Type", ["Total Returns", "Excess Returns (vs SPY)"],
        horizontal=True, key="heatmap_return_type"
    )

    col_name = "pearson_r" if return_type == "Total Returns" else "excess_pearson_r"
    p_col = "pearson_p" if return_type == "Total Returns" else "excess_pearson_p"

    pivot = corr_df.pivot_table(index="sector", columns="window_days", values=col_name)
    pivot.columns = [f"{w}d" for w in pivot.columns]
    pivot = pivot.reindex(pivot.abs().mean(axis=1).sort_values(ascending=False).index)

    # Build annotation text with significance markers
    annot_text = []
    for sector in pivot.index:
        row_text = []
        for w, col_label in zip([30, 60, 90], pivot.columns):
            val = pivot.loc[sector, col_label]
            r = corr_df[(corr_df["sector"] == sector) & (corr_df["window_days"] == w)]
            marker = ""
            if len(r) > 0 and "significant_bh_5pct" in r.columns:
                if r["significant_bh_5pct"].values[0]:
                    marker = " ✦"
                elif r.get("significant_raw_5pct", pd.Series([False])).values[0]:
                    marker = " *"
            row_text.append(f"{val:.3f}{marker}")
        annot_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        text=annot_text, texttemplate="%{text}", textfont=dict(size=13, color="white"),
        colorscale="RdYlGn", zmid=0, zmin=-0.4, zmax=0.4,
        colorbar=dict(title=dict(text="Pearson r", font=dict(color="#e0e0e0")), tickfont=dict(color="#e0e0e0")),
        hovertemplate="<b>%{y}</b><br>Window: %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="#e0e0e0", family="DM Sans"),
        xaxis=dict(title="Post-Meeting Window", side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=550, margin=dict(l=150, r=20, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "*✦ = significant after BH correction (α=0.05) · "
        "\\* = nominally significant (raw p<0.05)*"
    )

# --------------------------------------------------------------------------
# TAB 3: REGIME ANALYSIS
# --------------------------------------------------------------------------

with tab3:
    st.markdown("### Sector Returns by Pre-Meeting Sentiment Regime")
    st.markdown(
        "How do sectors perform when pre-meeting sentiment is bullish vs bearish? "
        "Grouped by PMSI regime classification."
    )

    window_choice = st.selectbox(
        "Post-Meeting Window", [30, 60, 90],
        format_func=lambda x: f"{x}-Day Window", key="regime_window"
    )

    regime_sub = regime_df[regime_df["window_days"] == window_choice].copy()

    if len(regime_sub) > 0:
        # Pivot for plotting
        pivot_regime = regime_sub.pivot_table(
            index="sector", columns="sentiment_regime", values="mean_return"
        )

        # Sort by bullish-bearish spread
        if "Bullish" in pivot_regime.columns and "Bearish" in pivot_regime.columns:
            pivot_regime["spread"] = pivot_regime.get("Bullish", 0) - pivot_regime.get("Bearish", 0)
            pivot_regime = pivot_regime.sort_values("spread", ascending=True)
            pivot_regime = pivot_regime.drop("spread", axis=1)

        fig = go.Figure()
        regime_colors = {"Bearish": RED, "Neutral": GRAY, "Bullish": GREEN}

        for regime in ["Bearish", "Neutral", "Bullish"]:
            if regime in pivot_regime.columns:
                vals = pivot_regime[regime].values * 100
                fig.add_trace(go.Bar(
                    y=pivot_regime.index, x=vals, name=regime,
                    orientation="h", marker_color=regime_colors[regime],
                    opacity=0.85,
                    hovertemplate=f"<b>%{{y}}</b><br>{regime}: %{{x:.2f}}%<extra></extra>"
                ))

        fig.update_layout(
            barmode="group",
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color="#e0e0e0", family="DM Sans"),
            xaxis=dict(title=f"Mean {window_choice}-Day Return (%)", gridcolor="#1a1a1a",
                      zeroline=True, zerolinecolor="#444444"),
            yaxis=dict(gridcolor="#1a1a1a"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0")),
            height=500, margin=dict(l=150, r=20, t=20, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show Cohen's d where available
        cohens = regime_sub[regime_sub["cohens_d_vs_bearish"].notna()][
            ["sector", "cohens_d_vs_bearish"]
        ].sort_values("cohens_d_vs_bearish", ascending=False) if "cohens_d_vs_bearish" in regime_sub.columns else pd.DataFrame()

        if len(cohens) > 0:
            with st.expander("Effect Size: Cohen's d (Bullish vs Bearish)"):
                st.markdown(
                    "Cohen's d measures the standardized difference between bullish and "
                    "bearish regime means. |d| > 0.2 = small, > 0.5 = medium, > 0.8 = large."
                )
                for _, row in cohens.iterrows():
                    d = row["cohens_d_vs_bearish"]
                    size = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
                    st.markdown(f"**{row['sector']}**: d = {d:.3f} ({size})")

# --------------------------------------------------------------------------
# TAB 4: TOP SECTORS SCATTER
# --------------------------------------------------------------------------

with tab4:
    st.markdown("### Most Sentiment-Sensitive Sectors")
    st.markdown(
        "Scatter plots showing PMSI vs post-meeting returns for the sectors "
        "with the strongest correlation. Regression line shows the linear fit."
    )

    scatter_window = st.selectbox(
        "Post-Meeting Window", [30, 60, 90],
        index=1, format_func=lambda x: f"{x}-Day Window", key="scatter_window"
    )

    # Merge PMSI with sector returns for scatter
    merged = returns_df[returns_df["ticker"] != "SPY"].merge(
        pmsi[["fomc_date", "pmsi", "sentiment_regime"]], on="fomc_date", how="inner"
    )

    # Find top 4 by abs correlation
    if len(corr_df) > 0:
        top_sectors = (
            corr_df.groupby("ticker")["pearson_r"]
            .apply(lambda x: np.abs(x).mean())
            .sort_values(ascending=False)
            .head(4).index.tolist()
        )

        sector_map = dict(zip(corr_df["ticker"], corr_df["sector"]))

        cols = st.columns(2)
        for idx, ticker in enumerate(top_sectors):
            with cols[idx % 2]:
                subset = merged[
                    (merged["ticker"] == ticker) &
                    (merged["window_days"] == scatter_window)
                ].dropna(subset=["pmsi", "return"])

                if len(subset) < 5:
                    continue

                x = subset["pmsi"].values
                y = subset["return"].values * 100
                point_colors = [GREEN if s > 0.15 else RED if s < -0.15 else GRAY for s in x]

                slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
                x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
                y_line = slope * x_line + intercept

                sig = "★★" if p_val < 0.01 else "★" if p_val < 0.05 else ""
                sector_name = sector_map.get(ticker, ticker)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="markers",
                    marker=dict(color=point_colors, size=8, opacity=0.7,
                               line=dict(color="#333333", width=0.5)),
                    hovertemplate="PMSI: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    line=dict(color=GOLD, width=2, dash="dash"),
                    showlegend=False,
                ))
                fig.update_layout(
                    title=dict(
                        text=f"{sector_name} ({ticker})<br>"
                             f"<span style='font-size:12px;color:{GOLD}'>r={r_val:.3f} {sig} · p={p_val:.4f} · n={len(subset)}</span>",
                        font=dict(size=14),
                    ),
                    plot_bgcolor=BG, paper_bgcolor=BG,
                    font=dict(color="#e0e0e0", family="DM Sans"),
                    xaxis=dict(title="PMSI", gridcolor="#1a1a1a",
                              zeroline=True, zerolinecolor="#333333"),
                    yaxis=dict(title=f"{scatter_window}d Return (%)", gridcolor="#1a1a1a",
                              zeroline=True, zerolinecolor="#333333"),
                    height=380, margin=dict(l=60, r=20, t=60, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 5: SENSITIVITY RANKING
# --------------------------------------------------------------------------

with tab5:
    st.markdown("### Sector Sensitivity to Pre-Meeting Sentiment")
    st.markdown(
        "Sectors ranked by mean |Pearson r| between PMSI and post-meeting returns "
        "across all measurement windows. Higher = more predictable from sentiment."
    )

    sens_sorted = sensitivity_df.sort_values("avg_abs_correlation", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sens_sorted["sector"],
        x=sens_sorted["avg_abs_correlation"],
        orientation="h",
        marker=dict(
            color=sens_sorted["avg_abs_correlation"],
            colorscale="YlGn",
            line=dict(width=0),
        ),
        text=[f"{v:.3f}" for v in sens_sorted["avg_abs_correlation"]],
        textposition="outside",
        textfont=dict(color="#e0e0e0", size=12),
        hovertemplate="<b>%{y}</b><br>|r̄| = %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="#e0e0e0", family="DM Sans"),
        xaxis=dict(title="Mean |Pearson r|", gridcolor="#1a1a1a"),
        yaxis=dict(gridcolor="#1a1a1a"),
        height=450, margin=dict(l=150, r=60, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # PMSI vs SPY validation
    st.markdown("---")
    st.markdown("### Validation: PMSI vs S&P 500 Returns")
    st.markdown(
        "Does the sentiment index track actual broad market outcomes? "
        "If PMSI doesn't correlate with SPY, sector-level analysis is suspect."
    )

    spy_30d = returns_df[
        (returns_df["ticker"] == "SPY") & (returns_df["window_days"] == 30)
    ][["fomc_date", "return"]].rename(columns={"return": "spy_30d"})
    val_merged = pmsi.merge(spy_30d, on="fomc_date", how="inner")

    if len(val_merged) >= 10:
        x_v = val_merged["pmsi"].values
        y_v = val_merged["spy_30d"].values * 100
        v_colors = [GREEN if s > 0.15 else RED if s < -0.15 else GRAY for s in x_v]

        sl, ic, rv, pv, _ = stats.linregress(x_v, y_v)
        xl = np.linspace(x_v.min() - 0.05, x_v.max() + 0.05, 100)

        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=x_v, y=y_v, mode="markers",
            marker=dict(color=v_colors, size=9, opacity=0.7,
                       line=dict(color="#333333", width=0.5)),
            hovertemplate="PMSI: %{x:.3f}<br>SPY 30d: %{y:.2f}%<extra></extra>"
        ))
        fig_val.add_trace(go.Scatter(
            x=xl, y=sl * xl + ic, mode="lines",
            line=dict(color=GOLD, width=2, dash="dash"), showlegend=False,
        ))
        v_sig = "★★" if pv < 0.01 else "★" if pv < 0.05 else ""
        fig_val.update_layout(
            title=f"r = {rv:.3f} {v_sig} · p = {pv:.4f} · n = {len(val_merged)}",
            title_font=dict(color=GOLD, size=14),
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color="#e0e0e0", family="DM Sans"),
            xaxis=dict(title="Pre-Meeting Sentiment Index", gridcolor="#1a1a1a",
                      zeroline=True, zerolinecolor="#333333"),
            yaxis=dict(title="30-Day SPY Return (%)", gridcolor="#1a1a1a",
                      zeroline=True, zerolinecolor="#333333"),
            height=420, margin=dict(l=60, r=20, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_val, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 6: RAW DATA
# --------------------------------------------------------------------------

with tab6:
    st.markdown("### Explore Raw Data")

    data_choice = st.selectbox("Dataset", [
        "Pre-Meeting Sentiment Index (PMSI)",
        "Correlation Results",
        "Regime Analysis",
        "Sensitivity Ranking",
        "Scored Headlines (sample)",
        "Sector Returns (sample)",
    ])

    if data_choice == "Pre-Meeting Sentiment Index (PMSI)":
        st.dataframe(pmsi.style.format({
            "pmsi": "{:.4f}", "sentiment_std": "{:.4f}",
            "bullish_ratio": "{:.2%}", "bearish_ratio": "{:.2%}",
            "neutral_ratio": "{:.2%}", "avg_confidence": "{:.3f}",
        }), use_container_width=True, height=500)

    elif data_choice == "Correlation Results":
        display_corr = corr_df.copy()
        st.dataframe(display_corr.style.format({
            "pearson_r": "{:.4f}", "pearson_p": "{:.4f}",
            "spearman_rho": "{:.4f}", "spearman_p": "{:.4f}",
            "excess_pearson_r": "{:.4f}", "excess_pearson_p": "{:.4f}",
        }), use_container_width=True, height=500)

    elif data_choice == "Regime Analysis":
        st.dataframe(regime_df.style.format({
            "mean_return": "{:.4f}", "median_return": "{:.4f}",
            "std_return": "{:.4f}", "mean_excess_return": "{:.4f}",
        }), use_container_width=True, height=500)

    elif data_choice == "Sensitivity Ranking":
        st.dataframe(sensitivity_df.style.format({
            "avg_abs_correlation": "{:.4f}", "avg_correlation": "{:.4f}",
            "max_abs_correlation": "{:.4f}",
        }), use_container_width=True, height=400)

    elif data_choice == "Scored Headlines (sample)":
        st.dataframe(scored_df.head(100), use_container_width=True, height=500)

    elif data_choice == "Sector Returns (sample)":
        st.dataframe(returns_df.head(100), use_container_width=True, height=500)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555555; font-size:0.85rem; padding:20px;'>"
    "FOMC Sentiment Analyzer · Built by Cameron Camarotti · "
    "<a href='https://github.com/cameroncc333/fomc-sentiment-analyzer' style='color:#42a5f5;'>GitHub</a> · "
    "FinBERT model by ProsusAI · Data from Yahoo Finance<br>"
    "Not financial advice. Built for analytical and educational purposes."
    "</div>",
    unsafe_allow_html=True,
)
