"""
trading_dashboard.py - VIEW layer of the trading dashboard (Dash + Plotly + dash-bootstrap-components).

Split in two files:
    trading_data.py       reads the Excel sheet and computes every number  (no UI code)
    trading_dashboard.py  this file: theme, charts, tables, layout, callbacks (no sheet / finance maths)

Dependencies:
    pip install dash dash-bootstrap-components pandas plotly openpyxl numpy

Run:   python trading_dashboard.py            (uses the path in DEFAULT_EXCEL_FILE)
       python trading_dashboard.py "C:\\path\\to\\statements.xlsx"

Tabs: Overview | Performance | Attribution | Income | Costs & cash | Risk | Behaviour | Projection
      | Net worth (NEW) | Data

What changed for the new sheet layout (v3):
  1. All sheet knowledge moved to ROW_PATTERNS in trading_data.py: rows are found by regex on the
     normalised label, so colons, double spaces and long note-style labels no longer matter.
  2. Fees now come from two rows ("Equity charges and fees" + "Commissions and fees"); both are counted
     and shown separately in the cost chart.
  3. '?' placeholders (Opening/Closing balance) and blank cells are read as "no data" instead of crashing.
  4. NEW Net worth tab: all accounts stacked (Trading212 + Polivios + cash held + Revolut), cash-flow
     breakdown, spending, wallet cash and the Polivios account - each cross-checked against the totals
     you calculate in the sheet.
  5. NEW KPI card and insight cards for net worth and the Polivios account; Data tab has the new columns.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

from trading_data import (build_portfolio, fnum, mdiff,
                          INCOME_LABELS, FEE_LABELS, NW_LABELS, CF_LABELS, MONTHLY_NAMES,
                          DEFAULT_CONTRIBUTION)

# =============================================================== CONFIG  (the only block you normally edit)
BASE_DIR = Path(__file__).resolve().parent

def _resolve_excel_file() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    preferred = BASE_DIR / "Trading121 monthly statements december 2023-december 2024.xlsx"
    if preferred.exists():
        return preferred
    candidates = [p for p in BASE_DIR.glob("*.xlsx") if not p.name.startswith("~$") and not p.name.startswith("test_")]
    if candidates:
        return candidates[0]
    return preferred

DEFAULT_EXCEL_FILE = _resolve_excel_file()
EXCEL_FILE = DEFAULT_EXCEL_FILE
START_DATE = "2023-10-01"   # statement months before this date are ignored
GRAPH_HEIGHT = 520
RISK_FREE_ANNUAL = 0.02     # used for Sharpe / Sortino / cash drag. Set to your cash/T-bill rate.
N_SIMS = 3000               # Monte-Carlo paths in the Projection tab
MIN_PROJECTION_MONTHS = 6   # the projection needs at least this many monthly returns
DEFAULT_CONTRIBUTION = 300  # suggested monthly deposit in the Projection tab
# Is the "unrealised" row a month-end BALANCE of open-position P&L (True), a monthly FLOW (False),
# or should the script decide by reconciling against the equity curve (None)?
UNREALISED_IS_BALANCE = None

# =============================================================== LOAD DATA (all maths happens in trading_data.py)
P = build_portfolio(EXCEL_FILE, start_date=START_DATE, risk_free_annual=RISK_FREE_ANNUAL,
                    unrealised_is_balance=UNREALISED_IS_BALANCE,
                    default_contribution=DEFAULT_CONTRIBUTION)
PF = P.perf          # shorthand: flow-adjusted performance of the Trading212 account
idx = P.idx          # month-start timestamps shared by every chart

# =============================================================== THEME
BG, GRID, FONT = "#2e2e2e", "#444444", "JetBrains Mono, monospace"
C = dict(blue="#4C9BE8", green="#2ECC71", red="#E74C3C", gold="#F1C40F",
         orange="#FFA726", teal="#1ABC9C", purple="#A678DE", grey="#9AA0A6",
         brown="#C47A3A")
EUR = "%{y:,.2f} €<extra></extra>"
PCT = "%{y:.2f}%<extra></extra>"
XRANGE = P.x_range

# One colour per series, keyed by the internal field names from trading_data.py.
INCOME_COLORS = {"dividends": C["gold"], "interest": C["blue"], "cashback": C["green"], "share_lending": C["purple"]}
FEE_COLORS = {"equity_fees": C["red"], "commission_fees": C["orange"]}
ATTR_COLORS = {"Realised P&L": C["blue"], "Income": C["gold"], "Fees": C["red"], "Open positions / other": C["brown"]}
NW_COLORS = {"t212": C["blue"], "polivios": C["purple"], "wallet": C["teal"], "revolut": C["orange"]}
CF_COLORS = {"t212_flow": C["blue"], "spending": C["red"], "polivios": C["purple"],
             "wallet_new": C["teal"], "revolut_in": C["green"], "revolut_out": C["orange"]}

RANGESELECTOR = dict(
    buttons=[
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ],
    bgcolor="rgba(46, 46, 46, 0.7)",
    activecolor=C["blue"],
    font=dict(family=FONT, size=11, color="#e0e0e0"),
    bordercolor="#555555",
    borderwidth=1,
)


def style(fig, title, ytitle=None, height=GRAPH_HEIGHT, time_axis=True, hovermode="x unified"):
    """Apply the shared dark look to every chart so individual figure functions only describe their data."""
    t_margin = 60 if height <= 350 else 100
    b_margin = 40 if height <= 350 else 70
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.01, xanchor="left", y=0.96, yanchor="top", font=dict(size=18)),
        font=dict(family=FONT, color="#d0d0d0", size=12),
        paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        margin=dict(l=70, r=40, t=t_margin, b=b_margin), hovermode=hovermode,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.04, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#888888", zerolinewidth=1.5)
    if time_axis:
        fig.update_xaxes(tickformat="%b-%y", dtick="M2", tickangle=-45, range=XRANGE,
                         showspikes=True, spikemode="across", spikethickness=1, spikecolor="#888888",
                         rangeslider=dict(visible=False))
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    return fig


def empty_fig(msg):
    """Placeholder chart for data that is missing, so a missing sheet row never breaks the page."""
    return style(go.Figure(), msg, height=300, time_axis=False)


def bar_colors(series):
    return [C["green"] if v >= 0 else C["red"] for v in series.fillna(0)]


# =============================================================== FIGURES - OVERVIEW
def fig_equity():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=PF.capital_in, name="Capital in (opening balance + deposits)",
                             line=dict(color=C["grey"], dash="dot"), hovertemplate=EUR))
    fig.add_trace(go.Scatter(x=idx, y=PF.equity, name="Equity", mode="lines+markers",
                             line=dict(color=C["blue"], width=3), fill="tonexty",
                             fillcolor="rgba(76,155,232,0.15)", hovertemplate=EUR))
    style(fig, "Equity vs money you put in", ytitle="Equity (€)")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


def fig_composition():
    """Where does the equity come from: starting balance, your deposits, or investment profit."""
    fig = go.Figure([
        go.Bar(x=idx, y=pd.Series(PF.equity.iloc[0], index=idx), name="Starting balance",
               marker_color=C["grey"], hovertemplate=EUR),
        go.Bar(x=idx, y=PF.net_dep - PF.flows.iloc[0], name="Deposits since start",
               marker_color=C["orange"], hovertemplate=EUR),
        go.Bar(x=idx, y=PF.cum_profit, name="Investment profit",
               marker_color=C["teal"], hovertemplate=EUR),
        go.Scatter(x=idx, y=PF.equity, name="Equity", mode="lines+markers",
                   line=dict(color="white", width=2), hovertemplate=EUR),
    ])
    style(fig, "What built your equity: starting balance + deposits + profit", ytitle="€")
    fig.update_layout(barmode="relative")
    return fig


def fig_returns():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=PF.ret * 100, name="Monthly return",
                         marker_color=bar_colors(PF.ret), hovertemplate=PCT), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=PF.cum_ret * 100, name="Cumulative return", mode="lines+markers",
                             line=dict(color=C["gold"], width=3), hovertemplate=PCT), secondary_y=True)
    style(fig, "Returns excluding deposits/withdrawals")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    fig.update_yaxes(title_text="Monthly (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True, showgrid=False)
    return fig


def fig_cashflows():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=PF.flows, name="Deposit / withdrawal",
                         marker_color=bar_colors(PF.flows), hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=PF.net_dep, name="Cumulative net deposits", mode="lines",
                             line=dict(color=C["orange"], width=3), hovertemplate=EUR), secondary_y=True)
    style(fig, "Deposits & withdrawals (Trading212)")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (€)", secondary_y=True, showgrid=False)
    return fig


# =============================================================== FIGURES - PERFORMANCE
def fig_pl():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if P.has("realised"):
        fig.add_trace(go.Bar(x=idx, y=P.series("realised"), name="Realised P&L",
                             marker_color=C["blue"], hovertemplate=EUR), secondary_y=False)
    if P.unreal_flow is not None:
        fig.add_trace(go.Bar(x=idx, y=P.unreal_flow,
                             name="Unrealised P&L (monthly change)" if P.unreal_is_balance else "Unrealised P&L",
                             marker_color=C["brown"], hovertemplate=EUR), secondary_y=False)
    if P.total_pl.notna().any():                      # equity-only sheets have no reported P&L rows to total up
        fig.add_trace(go.Bar(x=idx, y=P.total_pl, name="Total P&L",
                             marker_color=C["teal"], hovertemplate=EUR), secondary_y=False)
        fig.add_trace(go.Scatter(x=idx, y=P.total_pl.fillna(0).cumsum(), name="Cumulative total P&L",
                                 line=dict(color=C["gold"], width=3), hovertemplate=EUR), secondary_y=True)
    style(fig, "Realised vs unrealised vs total P&L")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (€)", secondary_y=True, showgrid=False)
    return fig


def fig_heatmap():
    r = PF.ret.iloc[1:] * 100
    t = pd.DataFrame({"y": r.index.year, "m": r.index.month, "r": r.values})
    piv = t.pivot(index="y", columns="m", values="r").reindex(columns=range(1, 13))
    text = [[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in piv.values]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=months, y=[str(y) for y in piv.index], text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmid=0, xgap=3, ygap=3, colorbar=dict(title="%"),
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>"))
    style(fig, "Monthly return heatmap", height=300, time_axis=False, hovermode="closest")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


def fig_distribution():
    r = (PF.ret.iloc[1:] * 100).dropna()
    fig = go.Figure(go.Histogram(x=r, nbinsx=14, marker_color=C["blue"],
                                 hovertemplate="%{x}%: %{y} months<extra></extra>"))
    fig.add_vline(x=0, line_color="white")
    fig.add_vline(x=r.mean(), line_dash="dash", line_color=C["gold"],
                  annotation_text=f"mean {r.mean():.1f}%", annotation_font_color=C["gold"])
    if np.isfinite(PF.var5):
        fig.add_vline(x=PF.var5 * 100, line_dash="dot", line_color=C["red"],
                      annotation_text=f"5th pct {PF.var5 * 100:.1f}%", annotation_position="top left",
                      annotation_font_color=C["red"])
    style(fig, f"Distribution of monthly returns (win rate {(r > 0).mean() * 100:.0f}%)",
          ytitle="Months", time_axis=False, hovermode="closest", height=300)
    fig.update_xaxes(title_text="Monthly return (%)")
    return fig


def _best_worst_frame():
    """Monthly returns in long format, keeping only years that have at least 2 months (else best == worst)."""
    r = PF.ret.iloc[1:] * 100
    t = pd.DataFrame({"year": r.index.year, "month": r.index.strftime("%b"), "r": r.values, "date": r.index})
    counts = t.groupby("year")["r"].count()
    return t[t["year"].isin(counts[counts >= 2].index)]


def fig_best_worst():
    t = _best_worst_frame()
    if t.empty:
        return empty_fig("Best & worst month per year (not enough data)")

    best = t.loc[t.groupby("year")["r"].idxmax()].set_index("year").sort_index()
    worst = t.loc[t.groupby("year")["r"].idxmin()].set_index("year").sort_index()
    years_ = [str(y) for y in best.index]

    def trace(d, name, color):
        return go.Bar(
            x=years_, y=d["r"], name=name, marker_color=color,
            text=[f"{m}<br>{v:+.1f}%" for m, v in zip(d["month"], d["r"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x} · " + name + ": %{y:.2f}%<extra></extra>")

    fig = go.Figure([trace(best, "Best month", C["green"]), trace(worst, "Worst month", C["red"])])
    style(fig, "Best & worst month per year", ytitle="Return (%)",
          time_axis=False, hovermode="closest", height=360)
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    return fig


def table_best_worst():
    t = _best_worst_frame()
    if t.empty:
        return html.Div("No years with at least 2 months of data.", className="text-muted text-center py-2")

    best = t.loc[t.groupby("year")["r"].idxmax()].set_index("year").sort_index()
    worst = t.loc[t.groupby("year")["r"].idxmin()].set_index("year").sort_index()

    header = html.Thead(
        html.Tr([
            html.Th("Year", style={"textAlign": "center"}),
            html.Th("Best month", style={"textAlign": "center", "color": C["green"]}),
            html.Th("Best %", style={"textAlign": "right", "color": C["green"]}),
            html.Th("Best P&L (€)", style={"textAlign": "right", "color": C["green"]}),
            html.Th("Worst month", style={"textAlign": "center", "color": C["red"]}),
            html.Th("Worst %", style={"textAlign": "right", "color": C["red"]}),
            html.Th("Worst P&L (€)", style={"textAlign": "right", "color": C["red"]}),
        ]),
        style={"backgroundColor": "#242424", "borderBottom": "2px solid #555"}
    )

    rows = []
    for y in sorted(best.index):
        b_row, w_row = best.loc[y], worst.loc[y]
        # euro P&L from the equity curve itself (always available, unlike reported rows)
        b_pl, w_pl = PF.monthly_pl.get(b_row["date"], np.nan), PF.monthly_pl.get(w_row["date"], np.nan)
        rows.append(html.Tr([
            html.Td(str(y), style={"textAlign": "center", "fontWeight": "bold"}),
            html.Td(b_row["month"], style={"textAlign": "center", "color": C["green"]}),
            html.Td(f"{b_row['r']:+.2f}%", style={"textAlign": "right", "color": C["green"], "fontWeight": "bold"}),
            html.Td(fnum(b_pl, ",.2f", " €", plus=True), style={"textAlign": "right", "color": C["green"]}),
            html.Td(w_row["month"], style={"textAlign": "center", "color": C["red"]}),
            html.Td(f"{w_row['r']:+.2f}%", style={"textAlign": "right", "color": C["red"], "fontWeight": "bold"}),
            html.Td(fnum(w_pl, ",.2f", " €", plus=True), style={"textAlign": "right", "color": C["red"]}),
        ]))

    return dbc.Table([header, html.Tbody(rows)], bordered=True, hover=True, responsive=True,
                     className="mb-4",
                     style={"backgroundColor": "#2e2e2e", "color": "#d0d0d0", "fontFamily": FONT,
                            "fontSize": "13px", "borderColor": "#444444"})


def fig_rolling():
    r3 = ((1 + PF.ret).rolling(3).apply(np.prod, raw=True) - 1) * 100
    r6 = ((1 + PF.ret).rolling(6).apply(np.prod, raw=True) - 1) * 100
    fig = go.Figure([
        go.Scatter(x=idx, y=r3, name="Rolling 3M", line=dict(color=C["blue"], width=2), hovertemplate=PCT),
        go.Scatter(x=idx, y=r6, name="Rolling 6M", line=dict(color=C["orange"], width=2), hovertemplate=PCT),
    ])
    style(fig, "Rolling returns", ytitle="Return (%)")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


def fig_growth_of_100():
    """Growth of 100 invested (time-weighted) - the 'pure' strategy curve."""
    fig = go.Figure([
        go.Scatter(x=idx, y=PF.perf_index * 100, name="Growth of €100 (TWR)", mode="lines+markers",
                   line=dict(color=C["teal"], width=3), fill="tozeroy",
                   fillcolor="rgba(26,188,156,0.10)", hovertemplate="%{y:,.2f}<extra></extra>"),
        go.Scatter(x=idx, y=PF.perf_index.cummax() * 100, name="High-water mark",
                   line=dict(color=C["grey"], width=1, dash="dot"), hovertemplate="%{y:,.2f}<extra></extra>"),
    ])
    style(fig, "Growth of €100 (deposits removed)", ytitle="Value of €100")
    fig.update_yaxes(rangemode="normal")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


# =============================================================== FIGURES - ATTRIBUTION
def fig_waterfall():
    tot = P.attr.sum()
    labels = list(tot.index) + ["Total profit"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=list(tot.values) + [tot.sum()],
        measure=["relative"] * len(tot) + ["total"],
        text=[f"{v:+,.0f} €" for v in list(tot.values) + [tot.sum()]], textposition="outside",
        increasing=dict(marker=dict(color=C["green"])),
        decreasing=dict(marker=dict(color=C["red"])),
        totals=dict(marker=dict(color=C["teal"])),
        connector=dict(line=dict(color="#888888")),
        hovertemplate="%{x}: %{y:,.2f} €<extra></extra>"))
    style(fig, "Where did the profit come from? (whole period)", ytitle="€",
          time_axis=False, hovermode="closest", height=420)
    return fig


def fig_attr_monthly():
    fig = go.Figure([go.Bar(x=idx, y=P.attr[c], name=c, marker_color=ATTR_COLORS[c], hovertemplate=EUR)
                     for c in P.attr.columns])
    fig.add_trace(go.Scatter(x=idx, y=PF.monthly_pl, name="Total (from equity curve)", mode="lines+markers",
                             line=dict(color="white", width=2, dash="dot"), hovertemplate=EUR))
    style(fig, "Monthly profit split: realised, income, fees, open positions")
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="€")
    return fig


def fig_attr_cumulative():
    cum = P.attr.cumsum()
    fig = go.Figure([go.Scatter(x=idx, y=cum[c], name=c, mode="lines",
                                line=dict(color=ATTR_COLORS[c], width=2.5), hovertemplate=EUR)
                     for c in cum.columns])
    fig.add_trace(go.Scatter(x=idx, y=cum.sum(axis=1), name="Total profit", mode="lines+markers",
                             line=dict(color="white", width=3), hovertemplate=EUR))
    style(fig, "Cumulative profit by source")
    fig.update_yaxes(title_text="€")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


def fig_win_loss_pl():
    """Euro P&L per month sorted - shows how lopsided the profit is."""
    s = PF.r_pl.sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=[d.strftime("%b-%y") for d in s.index], y=s.values,
                           marker_color=bar_colors(s), hovertemplate="%{x}: %{y:,.2f} €<extra></extra>"))
    style(fig, "Months ranked by euro profit", ytitle="€", time_axis=False, hovermode="closest", height=360)
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


# =============================================================== FIGURES - INCOME / COSTS
def fig_income_monthly():
    if not P.income_fields:
        return empty_fig("Income (no data)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for f in P.income_fields:
        fig.add_trace(go.Bar(x=P.df_income.index, y=P.df_income[f], name=INCOME_LABELS[f],
                             marker_color=INCOME_COLORS[f], hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=P.df_income.index, y=P.df_income.sum(axis=1).cumsum(),
                             name="Cumulative income", line=dict(color="white", width=2, dash="dot"),
                             hovertemplate=EUR), secondary_y=True)
    style(fig, "Monthly income (dividends, interest, cashback, lending)")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (€)", secondary_y=True, showgrid=False)
    return fig


def fig_income_yield():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=P.income_yield, name="Income (% of equity, monthly)",
                         marker_color=C["gold"], hovertemplate="%{y:.3f}%<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Bar(x=idx, y=P.fee_drag, name="Fees (% of equity, monthly)",
                         marker_color=C["red"], hovertemplate="%{y:.3f}%<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=P.income_yield_ann, name="Trailing-3M income yield (annualised)",
                             line=dict(color="white", width=2, dash="dot"),
                             hovertemplate="%{y:.2f}%<extra></extra>"), secondary_y=True)
    style(fig, "Income yield vs fee drag")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Monthly (%)", secondary_y=False)
    fig.update_yaxes(title_text="Annualised (%)", secondary_y=True, showgrid=False)
    return fig


def fig_costs():
    """Income against fees; the two fee rows of the new sheet get their own bar so you can see which one costs you."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    fig.add_trace(go.Bar(x=idx, y=P.income_total, name="Income", marker_color=C["green"], hovertemplate=EUR), 1, 1)
    for f in P.fee_fields:
        fig.add_trace(go.Bar(x=idx, y=P.fees_by_type[f], name=FEE_LABELS[f],
                             marker_color=FEE_COLORS[f], hovertemplate=EUR), 1, 1)
    fig.add_trace(go.Scatter(x=idx, y=P.income_total.cumsum(), name="Cumulative income",
                             line=dict(color=C["green"], width=2), hovertemplate=EUR), 2, 1)
    fig.add_trace(go.Scatter(x=idx, y=P.fees_abs.cumsum(), name="Cumulative fees",
                             line=dict(color=C["red"], width=2), hovertemplate=EUR), 2, 1)
    fig.add_trace(go.Scatter(x=idx, y=(P.income_total - P.fees_abs).cumsum(), name="Net (income − fees)",
                             line=dict(color="white", width=3, dash="dot"), hovertemplate=EUR), 2, 1)
    style(fig, "Income vs commissions & fees", height=680)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Monthly (€)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative (€)", row=2, col=1)
    return fig


def fig_cash():
    if P.cash is None:
        return empty_fig("Cash (missing data)")
    fig = go.Figure([
        go.Scatter(x=idx, y=P.cash, name="Cash", mode="lines+markers",
                   line=dict(color=C["teal"], width=3), hovertemplate=EUR),
        go.Scatter(x=idx, y=PF.equity, name="Equity", line=dict(color=C["blue"], width=1, dash="dot"),
                   hovertemplate=EUR),
    ])
    return style(fig, "Cash balance vs equity", ytitle="Amount (€)")


def fig_cash_alloc():
    if P.cash_pct is None:
        return empty_fig("Cash allocation (missing data)")
    cp = P.cash_pct.clip(lower=0, upper=100)
    fig = go.Figure([
        go.Scatter(x=idx, y=100 - cp, name="Invested", stackgroup="a", mode="lines",
                   line=dict(width=0.5, color=C["blue"]), fillcolor="rgba(76,155,232,0.55)",
                   hovertemplate="%{y:.1f}%<extra></extra>"),
        go.Scatter(x=idx, y=cp, name="Cash", stackgroup="a", mode="lines",
                   line=dict(width=0.5, color=C["teal"]), fillcolor="rgba(26,188,156,0.55)",
                   hovertemplate="%{y:.1f}%<extra></extra>"),
    ])
    style(fig, "Allocation: invested vs cash (% of equity)", ytitle="% of equity", height=360)
    fig.update_yaxes(range=[0, 100])
    return fig


# =============================================================== FIGURES - RISK / BEHAVIOUR
def fig_drawdown():
    fig = go.Figure([
        go.Scatter(x=idx, y=PF.equity_dd * 100, name="Raw equity drawdown (includes withdrawals)",
                   line=dict(color=C["grey"], width=2), fill="tozeroy",
                   fillcolor="rgba(154,160,166,0.15)", hovertemplate=PCT),
        go.Scatter(x=idx, y=PF.perf_dd * 100, name="Performance drawdown (cash flows removed)",
                   line=dict(color=C["red"], width=3), fill="tozeroy",
                   fillcolor="rgba(231,76,60,0.20)", hovertemplate=PCT),
    ])
    style(fig, "Drawdown: raw equity vs performance only", ytitle="Drawdown (%)")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


def fig_rolling_risk():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=PF.roll_vol3.index, y=PF.roll_vol3, name="Volatility 3M (ann.)",
                             line=dict(color=C["blue"], width=2), hovertemplate=PCT), 1, 1)
    fig.add_trace(go.Scatter(x=PF.roll_vol6.index, y=PF.roll_vol6, name="Volatility 6M (ann.)",
                             line=dict(color=C["orange"], width=2), hovertemplate=PCT), 1, 1)
    fig.add_trace(go.Scatter(x=PF.roll_sharpe6.index, y=PF.roll_sharpe6, name="Sharpe 6M",
                             line=dict(color=C["purple"], width=2),
                             hovertemplate="%{y:.2f}<extra></extra>"), 2, 1)
    fig.add_hline(y=0, line_color="#888888", row=2, col=1)
    style(fig, "Rolling risk: volatility and Sharpe ratio", height=620)
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    return fig


def _scatter_with_trend(x, y, labels, color, hover, title_stub, xtitle, ytitle):
    """Shared scatter + OLS trend line + correlation in the title."""
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="Months",
                               marker=dict(size=11, color=color, line=dict(width=1, color="white")),
                               text=labels, hovertemplate=hover))
    corr = np.nan
    if len(x) >= 4 and np.std(x) > 0 and np.std(y) > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 20)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                 line=dict(color=C["gold"], dash="dash"), hoverinfo="skip"))
        corr = np.corrcoef(x, y)[0, 1]
    style(fig, f"{title_stub} (corr {fnum(corr, '.2f')})", ytitle=ytitle,
          time_axis=False, hovermode="closest", height=380)
    fig.update_xaxes(title_text=xtitle)
    return fig


def fig_return_vs_flow():
    """Do you deposit more AFTER good months? previous month's return vs this month's deposit."""
    x = PF.r_m.shift(1).dropna() * 100       # previous month's return (index idx[2:])
    y = PF.flows.iloc[2:]                    # this month's deposit    (index idx[2:])
    return _scatter_with_trend(
        x.values, y.values, [d.strftime("%b-%y") for d in y.index], C["blue"],
        "%{text}<br>previous month %{x:.2f}%<br>deposit %{y:,.0f} €<extra></extra>",
        "Do good months trigger deposits? Previous return vs this month's deposit",
        "Previous month's return (%)", "Deposit / withdrawal (€)")


def fig_return_vs_lag():
    x, y = PF.r_m.shift(1).dropna() * 100, PF.r_m.iloc[1:] * 100
    return _scatter_with_trend(
        x.values, y.values, [d.strftime("%b-%y") for d in y.index], C["purple"],
        "%{text}<br>prev %{x:.2f}%<br>this %{y:.2f}%<extra></extra>",
        "Momentum check: previous vs this month",
        "Previous month (%)", "This month (%)")


def fig_deposit_timing():
    """Monthly deposits next to that month's return - visual check of deposit timing."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=PF.flows, name="Deposit / withdrawal", marker_color=C["orange"],
                         opacity=0.7, hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=PF.ret * 100, name="Return that month (%)", mode="lines+markers",
                             line=dict(color=C["teal"], width=2.5), hovertemplate=PCT), secondary_y=True)
    style(fig, "Deposit timing vs returns")
    fig.update_yaxes(title_text="Deposit (€)", secondary_y=False)
    fig.update_yaxes(title_text="Return (%)", secondary_y=True, showgrid=False)
    return fig


# =============================================================== FIGURES - NET WORTH (NEW)
def fig_networth():
    """Stacked bars of every account that makes up your net worth, with the total as a line on top."""
    if not P.has_networth:
        return empty_fig("Net worth (the sheet only has Trading212 rows)")
    fig = go.Figure([go.Bar(x=idx, y=P.nw_parts[k], name=NW_LABELS[k], marker_color=NW_COLORS[k],
                            hovertemplate=EUR) for k in P.nw_parts.columns])
    fig.add_trace(go.Scatter(x=idx, y=P.nw_line, name="Net worth (total)", mode="lines+markers",
                             line=dict(color="white", width=3), hovertemplate=EUR))
    style(fig, "Net worth across all accounts")
    fig.update_layout(barmode="relative")      # 'relative' lets a negative component (Revolut) hang below zero
    fig.update_yaxes(title_text="€")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


def fig_cashflow_breakdown():
    """The sheet's 'Cash Flow per month' split into its ingredients, so you can see what drove each month."""
    if not P.has_cashflow:
        return empty_fig("Cash flow (no extra cash-flow rows in the sheet)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for k in P.cf_parts.columns:
        fig.add_trace(go.Bar(x=idx, y=P.cf_parts[k], name=CF_LABELS[k], marker_color=CF_COLORS[k],
                             hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=P.cf_line, name="Cash flow per month (total)", mode="lines+markers",
                             line=dict(color="white", width=2, dash="dot"), hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=P.cf_line.cumsum(), name="Cumulative cash flow", mode="lines",
                             line=dict(color=C["gold"], width=3), hovertemplate=EUR), secondary_y=True)
    style(fig, "Monthly cash flow and what it is made of")
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (€)", secondary_y=True, showgrid=False)
    return fig


def fig_spending():
    """Spending rows next to the Revolut money moving in and out."""
    if not (P.has("spending") or P.has("spending_ex_invest")):
        return empty_fig("Spending (no data)")
    fig = go.Figure()
    if P.has("spending"):
        fig.add_trace(go.Bar(x=idx, y=P.series("spending"), name="Spending", marker_color=C["red"],
                             hovertemplate=EUR))
    if P.has("spending_ex_invest"):
        fig.add_trace(go.Bar(x=idx, y=P.series("spending_ex_invest"), name="Spendings excl. investments",
                             marker_color=C["orange"], hovertemplate=EUR))
    if P.has("revolut_in"):
        fig.add_trace(go.Scatter(x=idx, y=P.series("revolut_in"), name="Revolut deposits", mode="lines+markers",
                                 line=dict(color=C["green"], width=2, dash="dot"), hovertemplate=EUR))
    if P.has("revolut_out"):
        fig.add_trace(go.Scatter(x=idx, y=P.series("revolut_out"), name="Revolut withdrawals", mode="lines+markers",
                                 line=dict(color=C["blue"], width=2, dash="dot"), hovertemplate=EUR))
    style(fig, "Spending & Revolut flows", ytitle="€", height=400)
    fig.update_layout(barmode="group")
    return fig


def fig_wallet():
    """Physical cash you hold: balance and what was added each month. Blank months stay gaps, not zeros."""
    if not P.has("wallet_cash"):
        return empty_fig("Cash held by me (no data)")
    fig = go.Figure()
    if P.has("wallet_new"):
        fig.add_trace(go.Bar(x=idx, y=P.series("wallet_new"), name="New cash in wallet (month)",
                             marker_color=C["teal"], opacity=0.7, hovertemplate=EUR))
    fig.add_trace(go.Scatter(x=idx, y=P.series("wallet_cash"), name="Cash held by me (balance)",
                             mode="lines+markers", line=dict(color=C["gold"], width=3), hovertemplate=EUR))
    style(fig, "Cash held by me", ytitle="€", height=400)
    return fig


def fig_polivios():
    """Account value against everything deposited into it; the bars show the difference."""
    if not P.has_polivios:
        return empty_fig("Polivios account (no data)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=idx, y=P.poli_cum_dep, name="Cumulative deposits",
                             line=dict(color=C["grey"], dash="dot"), hovertemplate=EUR), 1, 1)
    fig.add_trace(go.Scatter(x=idx, y=P.poli_value, name="Account value", mode="lines+markers",
                             line=dict(color=C["purple"], width=3), fill="tonexty",
                             fillcolor="rgba(166,120,222,0.15)", hovertemplate=EUR), 1, 1)
    fig.add_trace(go.Bar(x=idx, y=P.poli_gap, name="Value − cumulative deposits",
                         marker_color=bar_colors(P.poli_gap), hovertemplate=EUR), 2, 1)
    style(fig, "Polivios account: value vs money deposited", height=int(round(GRAPH_HEIGHT * 1.1)))
    fig.update_yaxes(title_text="€", row=1, col=1)
    fig.update_yaxes(title_text="Gap (€)", row=2, col=1)
    return fig


def fig_polivios_flows():
    """Does each deposit show up in the account value? Compare the month's deposit with the month's value change."""
    if not P.has_polivios:
        return empty_fig("Polivios account (no data)")
    fig = go.Figure([
        go.Bar(x=idx, y=P.poli_dep, name="Deposit / withdrawal", marker_color=C["orange"], hovertemplate=EUR),
        go.Bar(x=idx, y=P.poli_dvalue, name="Change in account value", marker_color=C["purple"], hovertemplate=EUR),
    ])
    style(fig, "Polivios account: deposits vs change in value", ytitle="€", height=380)
    fig.update_layout(barmode="group")
    return fig


# =============================================================== TABLES
TABLE_STYLE = {"backgroundColor": "#2e2e2e", "color": "#d0d0d0", "fontFamily": FONT,
               "fontSize": "13px", "borderColor": "#444444"}


def stat_table(title, rows):
    body = [html.Tr([html.Td(k, className="text-muted"),
                     html.Td(v, style={"textAlign": "right", "fontWeight": "bold"})]) for k, v in rows]
    return html.Div([
        html.H6(title, className="mb-2", style={"fontWeight": "600"}),
        dbc.Table([html.Tbody(body)], bordered=True, hover=True, size="sm", style=TABLE_STYLE),
    ])


def table_return_stats():
    r = PF.r_m
    return stat_table("Return statistics (monthly)", [
        ("Months analysed", f"{PF.n_months}"),
        ("Mean", fnum(r.mean() * 100, ".2f", "%", plus=True)),
        ("Median", fnum(r.median() * 100, ".2f", "%", plus=True)),
        ("Std deviation", fnum(r.std(ddof=1) * 100 if PF.n_months > 1 else np.nan, ".2f", "%")),
        ("Skewness", fnum(r.skew(), ".2f")),
        ("Excess kurtosis", fnum(r.kurt(), ".2f")),
        ("5th percentile (monthly VaR-ish)", fnum(PF.var5 * 100, ".2f", "%")),
        ("Average winning month", fnum(PF.avg_win * 100, ".2f", "%", plus=True)),
        ("Average losing month", fnum(PF.avg_loss * 100, ".2f", "%", plus=True)),
        ("Payoff ratio (avg win / avg loss)", fnum(PF.payoff, ".2f")),
        ("Profit factor (€ gains / € losses)", fnum(PF.profit_factor, ".2f")),
        ("Longest win / loss streak", f"{PF.longest_win} / {PF.longest_loss} months"),
        ("Current streak", (f"{PF.current_streak} win(s)" if PF.current_streak > 0
                            else f"{-PF.current_streak} loss(es)" if PF.current_streak < 0 else "flat")),
    ])


def table_risk_stats():
    return stat_table(f"Risk-adjusted metrics (risk-free {RISK_FREE_ANNUAL * 100:.1f}% p.a.)", [
        ("Annualised return (CAGR, TWR)", fnum(PF.cagr * 100, ".1f", "%", plus=True)),
        ("Annualised return (money-weighted)", fnum(PF.mwr_ann * 100, ".1f", "%", plus=True)),
        ("Annualised volatility", fnum(PF.vol_ann * 100, ".1f", "%")),
        ("Downside deviation (ann.)", fnum(PF.downside_dev * 100, ".1f", "%")),
        ("Sharpe ratio", fnum(PF.sharpe, ".2f")),
        ("Sortino ratio", fnum(PF.sortino, ".2f")),
        ("Calmar ratio (CAGR / max DD)", fnum(PF.calmar, ".2f")),
        ("Max drawdown (performance)", fnum(PF.max_dd * 100, ".1f", "%")),
        ("Current drawdown", fnum(PF.cur_dd * 100, ".1f", "%")),
        ("Max drawdown (raw equity)", fnum(PF.equity_dd.min() * 100, ".1f", "%")),
    ])


def table_drawdowns():
    if not PF.dd_periods:
        return html.Div("No drawdowns in the data - equity-only-up so far.", className="text-muted py-2")
    head = html.Thead(html.Tr([html.Th(h, style={"textAlign": "center"}) for h in
                               ["Peak", "Trough", "Depth", "Months down", "Recovered", "Months to recover"]]),
                      style={"backgroundColor": "#242424", "borderBottom": "2px solid #555"})
    rows = []
    for d in PF.dd_periods[:6]:
        rec = d["end"]
        rows.append(html.Tr([
            html.Td(f"{d['peak']:%b-%y}", style={"textAlign": "center"}),
            html.Td(f"{d['trough']:%b-%y}", style={"textAlign": "center"}),
            html.Td(f"{d['depth'] * 100:.1f}%", style={"textAlign": "center", "color": C["red"], "fontWeight": "bold"}),
            html.Td(str(mdiff(d["peak"], d["trough"])), style={"textAlign": "center"}),
            html.Td(f"{rec:%b-%y}" if rec is not None else "not yet",
                    style={"textAlign": "center", "color": C["green"] if rec is not None else C["orange"]}),
            html.Td(str(mdiff(d["trough"], rec)) if rec is not None else "–", style={"textAlign": "center"}),
        ]))
    return html.Div([html.H6("Worst drawdown episodes", className="mb-2", style={"fontWeight": "600"}),
                     dbc.Table([head, html.Tbody(rows)], bordered=True, hover=True, responsive=True,
                               size="sm", style=TABLE_STYLE)])


# =============================================================== KPIs & INSIGHT CARDS
def kpi(label, value, color="#d0d0d0", hint=None):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(label, title=hint, style={"fontSize": "12px", "color": "#9AA0A6", "whiteSpace": "nowrap"}),
                html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": color, "whiteSpace": "nowrap"}),
            ], className="p-2 text-center"),
            style={"backgroundColor": "#3a3a3a", "border": "1px solid #444", "borderRadius": "8px"},
            className="h-100",
        ),
        xs=6, md=4, xl=True, className="h-100"
    )


def signed_color(v):
    if v is None or not np.isfinite(v):
        return "#d0d0d0"
    return C["green"] if v >= 0 else C["red"]


kpis_row1 = dbc.Row([
                        kpi("Current equity", f"{PF.equity.iloc[-1]:,.0f} €"),
                        kpi("Capital in", f"{PF.capital_in.iloc[-1]:,.0f} €", hint="Opening balance + all deposits/withdrawals since"),
                        kpi("Profit / loss", f"{PF.total_profit:+,.0f} €", signed_color(PF.total_profit),
                            "Equity gain not explained by deposits (equity - capital in)"),
                        kpi("Return (TWR)", fnum(PF.twr_total * 100, ".1f", "%", plus=True), signed_color(PF.twr_total),
                            "Time-weighted, deposits removed"),
                        kpi("Annualised (CAGR)", fnum(PF.cagr * 100, ".1f", "%", plus=True), signed_color(PF.cagr)),
                        kpi("Money-weighted (ann.)", fnum(PF.mwr_ann * 100, ".1f", "%", plus=True), signed_color(PF.mwr_ann),
                            "IRR (deposits mid-month, as in the TWR): also reflects the timing/size of your deposits"),
                        kpi("Income / Fees", f"{P.income_sum:,.0f} € / {P.fees_sum:,.2f} €", hint="Excludes the first month (no baseline)"),
                    ] + ([kpi("Net worth (all accounts)", f"{P.nw_line.iloc[-1]:,.0f} €",
                              hint="Trading212 + Polivios + cash held + Revolut, as in your 'Account value total' row")]
                         if P.has_networth else []), className="g-2 mb-2")

kpis_row2 = dbc.Row([
    kpi("Max drawdown", fnum(PF.max_dd * 100, ".1f", "%"), C["red"], "Performance only, deposits removed"),
    kpi("Current drawdown", fnum(PF.cur_dd * 100, ".1f", "%"), C["red"] if PF.cur_dd < -0.001 else C["green"],
        "Distance below the performance high-water mark"),
    kpi("Volatility (ann.)", fnum(PF.vol_ann * 100, ".1f", "%")),
    kpi("Sharpe", fnum(PF.sharpe, ".2f"), signed_color(PF.sharpe), f"Risk-free {RISK_FREE_ANNUAL * 100:.1f}%"),
    kpi("Sortino", fnum(PF.sortino, ".2f"), signed_color(PF.sortino)),
    kpi("Win rate", fnum(PF.win_rate * 100, ".0f", "% of months")),
    kpi("Profit factor", fnum(PF.profit_factor, ".2f"),
        signed_color(PF.profit_factor - 1 if np.isfinite(PF.profit_factor) else np.nan),
        "Sum of winning-month € / sum of losing-month €"),
], className="g-2 mb-3")


def insight_card(title, text, color_key):
    """One insight card; the data layer only names the colour ('green', 'red', ...), the palette lives here."""
    color = C[color_key]
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(title, style={"fontSize": "12px", "color": color, "fontWeight": "700",
                               "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        html.Div(text, style={"fontSize": "13px", "lineHeight": "1.5", "marginTop": "4px"}),
    ], className="p-3"), style={"backgroundColor": "#2e2e2e", "border": "1px solid #444",
                                "borderLeft": f"4px solid {color}", "borderRadius": "8px"},
        className="h-100"), xs=12, md=6, xl=4)


insights_block = html.Div([
    html.H5("Key insights", className="mb-2", style={"fontWeight": "600"}),
    dbc.Row([insight_card(*i) for i in P.insights], className="g-2 mb-4"),
])

# =============================================================== MONTHLY DATA TABLE
monthly = P.monthly_table()


def data_table():
    return dash_table.DataTable(
        data=monthly.to_dict("records"),
        columns=[{"name": MONTHLY_NAMES[c], "id": c} for c in monthly.columns],
        sort_action="native", page_size=36, style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#242424", "color": "#e0e0e0", "fontWeight": "bold",
                      "border": "1px solid #444", "fontFamily": FONT, "fontSize": "12px"},
        style_cell={"backgroundColor": "#2e2e2e", "color": "#d0d0d0", "border": "1px solid #3a3a3a",
                    "fontFamily": FONT, "fontSize": "12px", "padding": "6px 10px", "textAlign": "right",
                    "minWidth": "90px"},
        style_cell_conditional=[{"if": {"column_id": "month"}, "textAlign": "left", "fontWeight": "bold"}],
        style_data_conditional=[
            {"if": {"filter_query": "{ret_pct} < 0", "column_id": "ret_pct"}, "color": C["red"]},
            {"if": {"filter_query": "{ret_pct} > 0", "column_id": "ret_pct"}, "color": C["green"]},
            {"if": {"filter_query": "{implied_pl} < 0", "column_id": "implied_pl"}, "color": C["red"]},
            {"if": {"filter_query": "{implied_pl} > 0", "column_id": "implied_pl"}, "color": C["green"]},
            {"if": {"filter_query": "{dd_pct} < 0", "column_id": "dd_pct"}, "color": C["orange"]},
            {"if": {"filter_query": "{cash_flow} < 0", "column_id": "cash_flow"}, "color": C["red"]},
            {"if": {"filter_query": "{cash_flow} > 0", "column_id": "cash_flow"}, "color": C["green"]},
        ],
    )


# =============================================================== LAYOUT HELPERS
def graph(fig, gid=None):
    kw = {"id": gid} if gid else {}
    h = GRAPH_HEIGHT
    if fig is not None:
        try:
            h = fig.layout.height or GRAPH_HEIGHT
        except Exception:
            pass
    return dcc.Graph(
        figure=fig,
        responsive=True,
        config={"displaylogo": False, "responsive": True},
        style={"height": f"{h}px", "width": "100%", "minWidth": "0", "maxWidth": "100%",
               "marginBottom": "18px"},
        **kw
    )


def row(*figs):
    n = len(figs) or 1
    col_width = 12 // n
    return dbc.Row([dbc.Col(graph(f) if isinstance(f, go.Figure) else f, xs=12, md=col_width) for f in figs],
                   className="g-3")


def tab(label, tab_id, *children):
    return dbc.Tab(dcc.Loading(type="circle", color=C["blue"],
                               children=html.Div(list(children), className="pt-3")),
                   label=label, tab_id=tab_id)


def note(text, cls="note mb-3"):
    """Small grey explanatory paragraph; returns an empty Div for empty text so callers need no guard."""
    return html.Div(text, className=cls) if text else html.Div()


CUSTOM_CSS = """
body {
    background-color: #1f1f1f !important;
    color: #d0d0d0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.card {
    background-color: #2e2e2e !important;
    border-color: #444444 !important;
}
.nav-tabs { flex-wrap: nowrap; overflow-x: auto; overflow-y: hidden; }
.nav-tabs .nav-link {
    color: #9AA0A6 !important;
    background-color: #282828 !important;
    border-color: #444444 !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin-right: 4px;
    white-space: nowrap;
}
.nav-tabs .nav-link:hover {
    color: #ffffff !important;
    border-color: #555555 !important;
}
.nav-tabs .nav-link.active {
    color: #ffffff !important;
    background-color: #2e2e2e !important;
    border-top: 3px solid #4C9BE8 !important;
    border-color: #444444 #444444 #2e2e2e !important;
    font-weight: 600;
}
.form-select, .form-control {
    background-color: #2e2e2e !important;
    color: #ffffff !important;
    border-color: #555555 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.form-select:focus, .form-control:focus {
    border-color: #4C9BE8 !important;
    box-shadow: 0 0 0 0.25rem rgba(76, 155, 232, 0.25) !important;
}
.form-check-input:checked {
    background-color: #4C9BE8 !important;
    border-color: #4C9BE8 !important;
}
.form-check-label {
    color: #d0d0d0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.table {
    --bs-table-bg: #2e2e2e !important;
    --bs-table-striped-bg: #343434 !important;
    --bs-table-hover-bg: #383838 !important;
    --bs-table-color: #d0d0d0 !important;
    border-color: #444444 !important;
}
.note { font-size: 12px; color: #9AA0A6; line-height: 1.5; }
.warn { font-size: 12px; color: #FFA726; line-height: 1.5; }
"""

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap",
    ],
)
app.title = "Trading dashboard"
app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
{CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""

header = dbc.Row([
    dbc.Col(html.H2("Trading dashboard", className="mb-0", style={"fontWeight": "600"}),
            xs=12, sm=7, md=8, className="d-flex align-items-center"),
    dbc.Col(html.Span(f"Data: {idx.min():%b-%y} → {idx.max():%b-%y}  ·  {PF.n_months} monthly returns",
                      className="text-muted", style={"fontSize": "14px", "fontFamily": FONT}),
            xs=12, sm=5, md=4, className="d-flex align-items-center justify-content-sm-end mt-2 mt-sm-0"),
], className="mb-3 pb-2 border-bottom border-secondary", align="center")

gap_warning = note("Warning: some statement months are missing, so a few monthly returns span several months "
                   "and annualised figures are approximate." if P.has_gaps else "", "warn mb-2")

income_controls = html.Div([
    html.Label("Period:", className="me-2 text-muted fw-bold", style={"fontFamily": FONT}),
    dbc.Select(
        id="income-period", value="Q",
        options=[{"label": "Quarterly", "value": "Q"},
                 {"label": "Half-yearly", "value": "H"},
                 {"label": "Yearly", "value": "Y"}],
        style={"width": "180px"}, className="me-4",
    ),
    dbc.RadioItems(
        id="income-mode", value="amount", inline=True,
        options=[{"label": " Amount (€)", "value": "amount"},
                 {"label": " Share (%)", "value": "share"}],
        className="d-flex align-items-center", inputClassName="me-1", labelClassName="me-3 text-light",
    ),
], className="d-flex align-items-center my-3 p-2 rounded", style={"backgroundColor": "#282828", "border": "1px solid #444"})

projection_controls = html.Div([
    html.Label("Horizon (months):", className="me-2 text-muted fw-bold", style={"fontFamily": FONT}),
    dbc.Input(id="proj-horizon", type="number", min=1, max=60, step=1, value=12, debounce=True,
              style={"width": "110px"}, className="me-4"),
    html.Label("Monthly deposit (€):", className="me-2 text-muted fw-bold", style={"fontFamily": FONT}),
    dbc.Input(id="proj-contrib", type="number", step=10, value=P.default_contribution, debounce=True,
              style={"width": "140px"}),
], className="d-flex align-items-center flex-wrap my-3 p-2 rounded", style={"backgroundColor": "#282828", "border": "1px solid #444"})


def networth_tab():
    """The new tab. Each block only appears if its sheet rows exist, and each total is cross-checked against the sheet."""
    wallet_first = P.series("wallet_cash").first_valid_index()
    wallet_note = (f"'Cash held by me' is blank before {wallet_first:%b-%y} and counts as 0 in the net worth total. "
                   if wallet_first is not None and wallet_first > idx[0] else "")
    blocks = [
        note("Net worth = Trading212 equity + Polivios value + cash held by me + Revolut deposits − Revolut "
             "withdrawals (the formula of your 'Account value total' row). The Revolut part is that month's net "
             "flow, so it can be negative. " + wallet_note + P.nw_note),
        graph(fig_networth()),
    ]
    if P.has_cashflow:
        blocks += [graph(fig_cashflow_breakdown()), note(P.cf_note)]
    blocks.append(row(fig_spending(), fig_wallet()))
    if P.has_polivios:
        blocks += [
            note("Polivios is shown as value versus money deposited, with no return calculated: the value moves "
                 "far less than the deposits, so a time-weighted return would be meaningless unless every "
                 "withdrawal from the account is in the sheet."),
            graph(fig_polivios()),
            graph(fig_polivios_flows()),
        ]
    return tab("Net worth", "t-networth", *blocks)


tabs = [
    tab("Overview", "t-overview",
        insights_block,
        graph(fig_equity(), gid="graph-equity"), graph(fig_composition()), graph(fig_returns()), graph(fig_cashflows())),
    tab("Performance", "t-perf",
        graph(fig_growth_of_100()),
        graph(fig_pl()),
        row(fig_heatmap(), fig_distribution()),
        graph(fig_best_worst()),
        table_best_worst(),
        graph(fig_rolling())),
    tab("Attribution", "t-attr",
        graph(fig_waterfall()),
        html.Div("Realised P&L, income and fees come from your statement rows. "
                 "'Open positions / other' is the remainder needed to match the equity curve "
                 "(price moves on open positions, FX, anything unreported). If your realised figure "
                 "is already net of fees, the fees bar double-counts and the remainder absorbs the difference.",
                 className="note mb-3"),
        note(P.recon_note),
        graph(fig_attr_monthly()),
        graph(fig_attr_cumulative()),
        graph(fig_win_loss_pl())),
    tab("Income", "t-income",
        graph(fig_income_monthly()),
        graph(fig_income_yield()),
        income_controls,
        graph(None, gid="income-aggregated")),
    tab("Costs & cash", "t-costs",
        graph(fig_costs()),
        graph(fig_cash()),
        graph(fig_cash_alloc())),
    tab("Risk", "t-risk",
        graph(fig_drawdown()),
        dbc.Row([dbc.Col(table_drawdowns(), xs=12, lg=12)], className="mb-4"),
        graph(fig_rolling_risk()),
        dbc.Row([dbc.Col(table_return_stats(), xs=12, lg=6), dbc.Col(table_risk_stats(), xs=12, lg=6)],
                className="g-4 mb-4")),
    tab("Behaviour", "t-behaviour",
        html.Div("Left: do you deposit more after good or bad months (previous month's return vs this month's "
                 "deposit)? Right: does a good month predict the next one? With a small sample the "
                 "correlations are weak evidence - look for obvious patterns only.",
                 className="note mb-3"),
        graph(fig_deposit_timing()),
        row(fig_return_vs_flow(), fig_return_vs_lag())),
    tab("Projection", "t-proj",
        html.Div("Illustrative bootstrap simulation: monthly returns are resampled from your own history "
                 f"({N_SIMS:,} paths). It is not a forecast; a short, recent history can badly mis-state risk "
                 "and expected return.", className="note"),
        projection_controls,
        html.Div(id="proj-text", className="mb-3", style={"fontSize": "13px"}),
        graph(None, gid="proj-graph")),
]
if P.has_networth:
    tabs.append(networth_tab())                       # only when the sheet has more than the Trading212 rows
tabs.append(tab("Data", "t-data",
                html.Div([dbc.Button("Download CSV", id="btn-csv", color="primary", outline=True, size="sm"),
                          dcc.Download(id="dl-csv")], className="mb-3"),
                data_table()))

app.layout = dbc.Container([
    header,
    gap_warning,
    kpis_row1,
    kpis_row2,
    dbc.Tabs(tabs, active_tab="t-overview", className="mb-3"),
    html.Div("Methodology: returns are time-weighted (Modified Dietz, deposits assumed mid-month). "
             "Money-weighted return is the IRR with the same mid-month deposit timing. 'Capital in' = opening "
             "balance + deposits. Drawdown, volatility, Sharpe and Sortino use monthly returns, annualised with "
             "√12. 'Equity-implied P&L' = Δequity − deposits. Performance figures describe the Trading212 "
             "account only; the Net worth tab covers all accounts.",
             className="note mt-4 pb-3"),
], fluid=True, style={
    "backgroundColor": "#1f1f1f",
    "color": "#d0d0d0",
    "fontFamily": FONT,
    "minHeight": "100vh",
    "padding": "20px",
    "maxWidth": "100%",
    "overflowX": "hidden",
})


# =============================================================== CALLBACKS
@app.callback(Output("income-aggregated", "figure"),
              Input("income-period", "value"), Input("income-mode", "value"))
def update_income(period, mode):
    """Re-group income by quarter / half-year / year; '*' marks periods with fewer months than a full period."""
    if not P.income_fields:
        return empty_fig("Income breakdown (missing data)")

    d = P.df_income
    if period == "Q":
        keys, full = [f"Q{t.quarter} {t.year}" for t in d.index], 3
    elif period == "H":
        keys, full = [f"H{1 if t.month <= 6 else 2} {t.year}" for t in d.index], 6
    else:
        keys, full = [str(t.year) for t in d.index], 12

    grp = d.groupby(np.array(keys), sort=False)
    agg, months = grp.sum(), grp.size()
    labels = [k + ("*" if months[k] < full else "") for k in agg.index]
    totals = agg.sum(axis=1).replace(0, np.nan)

    fig = go.Figure()
    for f in P.income_fields:
        share = (agg[f] / totals * 100).fillna(0)
        fig.add_trace(go.Bar(x=labels, y=agg[f], name=INCOME_LABELS[f], marker_color=INCOME_COLORS[f],
                             customdata=share,
                             hovertemplate="%{y:,.2f} € (%{customdata:.1f}%)<extra></extra>"))
    style(fig, "Income breakdown per period (* = incomplete period)", time_axis=False,
          ytitle="€" if mode == "amount" else "% of income")
    fig.update_layout(barmode="stack", barnorm="percent" if mode == "share" else None)
    fig.update_xaxes(type="category")
    return fig


@app.callback(Output("proj-graph", "figure"), Output("proj-text", "children"),
              Input("proj-horizon", "value"), Input("proj-contrib", "value"))
def update_projection(horizon, contrib):
    """Bootstrap Monte-Carlo: resample your own monthly returns, add the monthly deposit, read off percentiles."""
    horizon = int(np.clip(horizon if horizon is not None else 12, 1, 60))
    contrib = float(contrib or 0)
    rr = PF.r_m.dropna().values
    if len(rr) < MIN_PROJECTION_MONTHS:
        return empty_fig(f"Projection needs at least {MIN_PROJECTION_MONTHS} months of returns "
                         f"(you have {len(rr)})"), ""

    rng = np.random.default_rng(42)                                   # fixed seed -> same chart on every refresh
    sims = rng.choice(rr, size=(N_SIMS, horizon))
    paths = np.empty((N_SIMS, horizon + 1))
    paths[:, 0] = PF.equity.iloc[-1]
    for h in range(horizon):
        paths[:, h + 1] = paths[:, h] * (1 + sims[:, h]) + contrib
    p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
    future = pd.date_range(idx[-1], periods=horizon + 1, freq="MS")
    flat = PF.equity.iloc[-1] + contrib * np.arange(horizon + 1)      # 0% return line = break-even
    total_contributed = PF.capital_in.iloc[-1] + contrib * horizon

    fig = go.Figure([
        go.Scatter(x=idx, y=PF.equity, name="History", mode="lines+markers",
                   line=dict(color=C["blue"], width=3), hovertemplate=EUR),
        go.Scatter(x=future, y=p90, name="90th percentile", line=dict(color=C["green"], width=1, dash="dot"),
                   hovertemplate=EUR),
        go.Scatter(x=future, y=p10, name="10th–90th range", line=dict(color=C["red"], width=1, dash="dot"),
                   fill="tonexty", fillcolor="rgba(166,120,222,0.15)", hovertemplate=EUR),
        go.Scatter(x=future, y=p50, name="Median", line=dict(color=C["gold"], width=3), hovertemplate=EUR),
        go.Scatter(x=future, y=flat, name="Deposits only (0% return)",
                   line=dict(color=C["grey"], width=2, dash="dash"), hovertemplate=EUR),
    ])
    style(fig, f"Bootstrap projection - next {horizon} months", ytitle="Equity (€)", time_axis=False)
    fig.update_xaxes(tickformat="%b-%y", tickangle=-45)

    # Compare with break-even (today's equity + new deposits), i.e. the grey line, and also with
    # everything you will have put in (opening balance + all deposits).
    end = paths[:, -1]
    p_below_flat = (end < flat[-1]).mean() * 100
    p_below_capital = (end < total_contributed).mean() * 100
    text = (f"Median outcome {np.median(end):,.0f} € (10th–90th percentile {p10[-1]:,.0f} – {p90[-1]:,.0f} €). "
            f"In {p_below_flat:.0f}% of simulations you end below break-even from today "
            f"({flat[-1]:,.0f} € = current equity + new deposits), and in {p_below_capital:.0f}% below the "
            f"{total_contributed:,.0f} € you would have put in in total. "
            f"Resampled mean monthly return {rr.mean() * 100:+.2f}% · volatility {rr.std(ddof=1) * 100:.2f}%.")
    return fig, text


@app.callback(Output("dl-csv", "data"), Input("btn-csv", "n_clicks"), prevent_initial_call=True)
def download_csv(_):
    return dcc.send_data_frame(monthly.rename(columns=MONTHLY_NAMES).to_csv, "monthly_summary.csv", index=False)


app.clientside_callback(
    """
    function(relayoutData, figure) {
        if (!relayoutData || !figure || !figure.data) {
            return window.dash_clientside.no_update;
        }

        var hasY = ('yaxis.range[0]' in relayoutData) || ('yaxis.range' in relayoutData) || ('yaxis.autorange' in relayoutData);
        var hasX = ('xaxis.range[0]' in relayoutData) || ('xaxis.range' in relayoutData) || ('xaxis.autorange' in relayoutData);
        if (hasY && !hasX) {
            return window.dash_clientside.no_update;
        }

        if (relayoutData['xaxis.autorange'] || relayoutData['autosize']) {
            var newFig = JSON.parse(JSON.stringify(figure));
            newFig.layout = newFig.layout || {};
            newFig.layout.xaxis = newFig.layout.xaxis || {};
            newFig.layout.yaxis = newFig.layout.yaxis || {};
            newFig.layout.xaxis.autorange = true;
            delete newFig.layout.xaxis.range;
            newFig.layout.yaxis.autorange = true;
            delete newFig.layout.yaxis.range;
            return newFig;
        }

        var rawMin, rawMax;
        if ('xaxis.range[0]' in relayoutData) {
            rawMin = relayoutData['xaxis.range[0]'];
            rawMax = relayoutData['xaxis.range[1]'];
        } else if ('xaxis.range' in relayoutData) {
            rawMin = relayoutData['xaxis.range'][0];
            rawMax = relayoutData['xaxis.range'][1];
        } else {
            return window.dash_clientside.no_update;
        }

        var xMin = typeof rawMin === 'number' ? rawMin : new Date(rawMin).getTime();
        var xMax = typeof rawMax === 'number' ? rawMax : new Date(rawMax).getTime();

        if (isNaN(xMin) || isNaN(xMax)) {
            return window.dash_clientside.no_update;
        }

        // Allow 15 days margin so points at monthly boundaries are included
        var marginMs = 15 * 86400 * 1000;
        var filterMin = xMin - marginMs;
        var filterMax = xMax + marginMs;

        var yVals = [];
        figure.data.forEach(function(trace) {
            if (!trace.x || !trace.y) return;
            if (trace.visible === 'legendonly' || trace.visible === false) return;
            for (var i = 0; i < trace.x.length; i++) {
                var xv = trace.x[i];
                var d = typeof xv === 'number' ? xv : new Date(xv).getTime();
                if (!isNaN(d) && d >= filterMin && d <= filterMax) {
                    var y = trace.y[i];
                    if (typeof y === 'number' && !isNaN(y)) {
                        yVals.push(y);
                    }
                }
            }
        });

        if (yVals.length === 0) {
            return window.dash_clientside.no_update;
        }

        var yMin = Math.min.apply(null, yVals);
        var yMax = Math.max.apply(null, yVals);
        var span = yMax - yMin;
        var pad = span > 0 ? span * 0.08 : (Math.abs(yMax) * 0.08 || 1);

        var newFig = JSON.parse(JSON.stringify(figure));
        newFig.layout = newFig.layout || {};
        newFig.layout.xaxis = newFig.layout.xaxis || {};
        newFig.layout.yaxis = newFig.layout.yaxis || {};

        newFig.layout.xaxis.range = [rawMin, rawMax];
        newFig.layout.xaxis.autorange = false;

        newFig.layout.yaxis.range = [yMin - pad, yMax + pad];
        newFig.layout.yaxis.autorange = false;

        return newFig;
    }
    """,
    Output("graph-equity", "figure"),
    Input("graph-equity", "relayoutData"),
    State("graph-equity", "figure"),
    prevent_initial_call=True
)


# =============================================================== RUN
if __name__ == "__main__":
    app.run(debug=True)