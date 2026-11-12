"""
Trading dashboard (Dash + Plotly) - improved version.

Run:   python trading_dashboard.py            (uses the path in EXCEL_FILE)
       python trading_dashboard.py "C:\\path\\to\\statements.xlsx"
"""
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

# =============================================================== CONFIG
EXCEL_FILE = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\anton\Downloads\try\Trading121 monthly statements december 2023-december 2024.xlsx"
)
START_DATE = "2023-10-01"
GRAPH_HEIGHT = 520

BG, GRID, FONT = "#2e2e2e", "#444444", "JetBrains Mono, monospace"
C = dict(blue="#4C9BE8", green="#2ECC71", red="#E74C3C", gold="#F1C40F",
         orange="#FFA726", teal="#1ABC9C", purple="#A678DE", grey="#9AA0A6",
         brown="#C47A3A")
EUR = "%{y:,.2f} €<extra></extra>"
PCT = "%{y:.2f}%<extra></extra>"


# =============================================================== LOAD & CLEAN
def load_data(path):
    raw = pd.read_excel(path, header=0)
    raw = raw.dropna(subset=[raw.columns[0]]).set_index(raw.columns[0])
    d = raw.T
    d.columns = ["_".join(str(c).strip().replace(":", "").replace("/", "_").split())
                 for c in d.columns]
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[d.index.notna()]
    d.index = d.index.to_period("M").to_timestamp()          # normalise to month start
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d[d.index >= START_DATE]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                             errors="coerce")
    return d.dropna(how="all")


df = load_data(EXCEL_FILE)


def find_col(keywords, exact=None, exclude=()):
    """Exact (case-insensitive) match first, then substring match."""
    if exact:
        for c in df.columns:
            if c.lower() == exact.lower():
                return c
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords) and not any(x in low for x in exclude):
            return c
    return None


col_equity     = find_col(["equity", "account_value"])
col_realised   = find_col(["realised", "realized"])
col_unrealised = find_col(["unrealised", "unrealized"])
col_deposits   = find_col(["deposit", "withdrawal"])
col_dividends  = find_col(["dividend"])
col_cashback   = find_col(["cashback"])
col_interest   = find_col(["interest"])
col_sharelend  = find_col(["lending", "lend"])
col_cash       = find_col(["cash"], exact="Cash", exclude=("cashback", "interest"))
col_fees       = find_col(["commission", "fees"])

if not col_equity:
    raise SystemExit(f"Could not find an equity column. Columns found: {list(df.columns)}")

income_cols = [c for c in [col_dividends, col_cashback, col_interest, col_sharelend] if c]
pretty = lambda c: c.replace("_", " ")
INCOME_COLORS = {"dividend": C["gold"], "interest": C["blue"],
                 "cashback": C["green"], "lend": C["purple"]}


def color_for(name):
    low = name.lower()
    return next((v for k, v in INCOME_COLORS.items() if k in low), C["grey"])


# =============================================================== DERIVED METRICS
idx = df.index
equity = df[col_equity]
cf = df[col_deposits].fillna(0) if col_deposits else pd.Series(0.0, index=idx)

# Monthly return with deposits/withdrawals removed (Modified Dietz, flows mid-month)
prev = equity.shift(1)
denom = prev + 0.5 * cf
ret = ((equity - prev - cf) / denom).where(denom > 0)
ret.iloc[0] = 0.0
ret = ret.fillna(0.0)
cum_ret = (1 + ret).cumprod() - 1
perf_index = 1 + cum_ret
perf_dd = perf_index / perf_index.cummax() - 1          # drawdown of performance only
equity_dd = equity / equity.cummax() - 1                # raw equity drawdown

net_dep = cf.cumsum()
cum_profit = (equity - equity.iloc[0]) - (net_dep - cf.iloc[0])   # equity gain not explained by deposits

realised_unrealised = [c for c in [col_realised, col_unrealised] if c]
df["Total_PL"] = df[realised_unrealised].sum(axis=1, min_count=1) if realised_unrealised else np.nan

full_idx = pd.date_range(idx.min(), idx.max(), freq="MS")
df_income = df[income_cols].reindex(full_idx).fillna(0) if income_cols else pd.DataFrame(index=full_idx)
income_total = df_income.sum(axis=1).reindex(idx).fillna(0)
fees_abs = df[col_fees].abs().reindex(idx).fillna(0) if col_fees else pd.Series(0.0, index=idx)

x_min, x_max = idx.min(), idx.max()
XRANGE = [(x_min - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
          (x_max + pd.Timedelta(days=20)).strftime("%Y-%m-%d")]


# =============================================================== STYLE HELPERS
def style(fig, title, ytitle=None, height=GRAPH_HEIGHT, time_axis=True, hovermode="x unified"):
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.01, xanchor="left", y=0.96, yanchor="top", font=dict(size=18)),
        font=dict(family=FONT, color="#d0d0d0", size=12),
        paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        margin=dict(l=70, r=40, t=100, b=70), hovermode=hovermode,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.04, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#888888", zerolinewidth=1.5)
    if time_axis:
        fig.update_xaxes(tickformat="%b-%y", dtick="M2", tickangle=-45, range=XRANGE,
                         showspikes=True, spikemode="across", spikethickness=1, spikecolor="#888888")
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    return fig


def empty_fig(msg):
    return style(go.Figure(), msg, height=300)


def bar_colors(series):
    return [C["green"] if v >= 0 else C["red"] for v in series.fillna(0)]


# =============================================================== FIGURES
def fig_equity():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=idx, y=net_dep, name="Net deposited",
                             line=dict(color=C["grey"], dash="dot"), hovertemplate=EUR), 1, 1)
    fig.add_trace(go.Scatter(x=idx, y=equity, name="Equity", mode="lines+markers",
                             line=dict(color=C["blue"], width=3), fill="tonexty",
                             fillcolor="rgba(76,155,232,0.15)", hovertemplate=EUR), 1, 1)
    fig.add_trace(go.Bar(x=idx, y=cum_profit, name="Cumulative P/L (equity − net deposits)",
                         marker_color=bar_colors(cum_profit), hovertemplate=EUR), 2, 1)
    style(fig, "Equity vs money you put in")
    fig.update_yaxes(title_text="Equity (€)", row=1, col=1)
    fig.update_yaxes(title_text="Cum. P/L (€)", row=2, col=1)
    return fig


def fig_returns():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=ret * 100, name="Monthly return",
                         marker_color=bar_colors(ret), hovertemplate=PCT), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=cum_ret * 100, name="Cumulative return", mode="lines+markers",
                             line=dict(color=C["gold"], width=3), hovertemplate=PCT), secondary_y=True)
    style(fig, "Returns excluding deposits/withdrawals")
    fig.update_yaxes(title_text="Monthly (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True, showgrid=False)
    return fig


def fig_cashflows():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=cf, name="Deposit / withdrawal",
                         marker_color=bar_colors(cf), hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=net_dep, name="Cumulative net deposits", mode="lines",
                             line=dict(color=C["orange"], width=3), hovertemplate=EUR), secondary_y=True)
    style(fig, "Deposits & withdrawals")
    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (€)", secondary_y=True, showgrid=False)
    return fig


def fig_pl():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if col_realised:
        fig.add_trace(go.Bar(x=idx, y=df[col_realised], name="Realised P&L",
                             marker_color=C["blue"], hovertemplate=EUR), secondary_y=False)
    if col_unrealised:
        fig.add_trace(go.Bar(x=idx, y=df[col_unrealised], name="Unrealised P&L",
                             marker_color=C["brown"], hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Bar(x=idx, y=df["Total_PL"], name="Total P&L",
                         marker_color=C["teal"], hovertemplate=EUR), secondary_y=False)
    # assumes the statement's P&L columns are per-month amounts (as your original chart implied)
    fig.add_trace(go.Scatter(x=idx, y=df["Total_PL"].fillna(0).cumsum(), name="Cumulative total P&L",
                             line=dict(color=C["gold"], width=3), hovertemplate=EUR), secondary_y=True)
    style(fig, "Realised vs unrealised vs total P&L")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (€)", secondary_y=True, showgrid=False)
    return fig


def fig_heatmap():
    r = (ret.iloc[1:] * 100)
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
    r = (ret.iloc[1:] * 100).dropna()
    fig = go.Figure(go.Histogram(x=r, nbinsx=14, marker_color=C["blue"],
                                 hovertemplate="%{x}%: %{y} months<extra></extra>"))
    fig.add_vline(x=0, line_color="white")
    fig.add_vline(x=r.mean(), line_dash="dash", line_color=C["gold"],
                  annotation_text=f"mean {r.mean():.1f}%", annotation_font_color=C["gold"])
    style(fig, f"Distribution of monthly returns (win rate {(r > 0).mean() * 100:.0f}%)",
          ytitle="Months", time_axis=False, hovermode="closest", height=300)
    fig.update_xaxes(title_text="Monthly return (%)")
    return fig


def fig_rolling():
    r3 = ((1 + ret).rolling(3).apply(np.prod, raw=True) - 1) * 100
    r6 = ((1 + ret).rolling(6).apply(np.prod, raw=True) - 1) * 100
    fig = go.Figure([
        go.Scatter(x=idx, y=r3, name="Rolling 3M", line=dict(color=C["blue"], width=2), hovertemplate=PCT),
        go.Scatter(x=idx, y=r6, name="Rolling 6M", line=dict(color=C["orange"], width=2), hovertemplate=PCT),
    ])
    return style(fig, "Rolling returns", ytitle="Return (%)")


def fig_income_monthly():
    if not income_cols:
        return empty_fig("Income (no data)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for c in income_cols:
        fig.add_trace(go.Bar(x=df_income.index, y=df_income[c], name=pretty(c),
                             marker_color=color_for(c), hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_income.index, y=df_income.sum(axis=1).cumsum(),
                             name="Cumulative income", line=dict(color="white", width=2, dash="dot"),
                             hovertemplate=EUR), secondary_y=True)
    style(fig, "Monthly income (dividends, interest, cashback, lending)")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (€)", secondary_y=True, showgrid=False)
    return fig


def fig_costs():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    fig.add_trace(go.Bar(x=idx, y=income_total, name="Income", marker_color=C["green"], hovertemplate=EUR), 1, 1)
    fig.add_trace(go.Bar(x=idx, y=fees_abs, name="Commissions & fees", marker_color=C["red"], hovertemplate=EUR), 1, 1)
    fig.add_trace(go.Scatter(x=idx, y=income_total.cumsum(), name="Cumulative income",
                             line=dict(color=C["green"], width=2), hovertemplate=EUR), 2, 1)
    fig.add_trace(go.Scatter(x=idx, y=fees_abs.cumsum(), name="Cumulative fees",
                             line=dict(color=C["red"], width=2), hovertemplate=EUR), 2, 1)
    fig.add_trace(go.Scatter(x=idx, y=(income_total - fees_abs).cumsum(), name="Net (income − fees)",
                             line=dict(color="white", width=3, dash="dot"), hovertemplate=EUR), 2, 1)
    style(fig, "Income vs commissions & fees", height=680)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Monthly (€)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative (€)", row=2, col=1)
    return fig


def fig_cash():
    if not col_cash:
        return empty_fig("Cash (missing data)")
    fig = go.Figure([
        go.Scatter(x=idx, y=df[col_cash], name="Cash", mode="lines+markers",
                   line=dict(color=C["teal"], width=3), hovertemplate=EUR),
        go.Scatter(x=idx, y=equity, name="Equity", line=dict(color=C["blue"], width=1, dash="dot"),
                   hovertemplate=EUR),
    ])
    return style(fig, "Cash balance vs equity", ytitle="Amount (€)")


def fig_drawdown():
    fig = go.Figure([
        go.Scatter(x=idx, y=equity_dd * 100, name="Raw equity drawdown (includes withdrawals)",
                   line=dict(color=C["grey"], width=2), fill="tozeroy",
                   fillcolor="rgba(154,160,166,0.15)", hovertemplate=PCT),
        go.Scatter(x=idx, y=perf_dd * 100, name="Performance drawdown (cash flows removed)",
                   line=dict(color=C["red"], width=3), fill="tozeroy",
                   fillcolor="rgba(231,76,60,0.20)", hovertemplate=PCT),
    ])
    return style(fig, "Drawdown: raw equity vs performance only", ytitle="Drawdown (%)")


# =============================================================== KPIs
def kpi(label, value, color="#d0d0d0"):
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": "#9AA0A6"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
    ], style={"background": "#3a3a3a", "padding": "12px 18px", "borderRadius": "8px",
              "minWidth": "150px", "flex": "1"})


def signed_color(v):
    return C["green"] if v >= 0 else C["red"]


r_m = ret.iloc[1:]
total_profit = cum_profit.iloc[-1]
kpis = html.Div([
    kpi("Current equity", f"{equity.iloc[-1]:,.0f} €"),
    kpi("Net deposited", f"{net_dep.iloc[-1]:,.0f} €"),
    kpi("Profit / loss", f"{total_profit:+,.0f} €", signed_color(total_profit)),
    kpi("Return (excl. deposits)", f"{cum_ret.iloc[-1] * 100:+.1f}%", signed_color(cum_ret.iloc[-1])),
    kpi("Max drawdown (perf.)", f"{perf_dd.min() * 100:.1f}%", C["red"]),
    kpi("Best month", f"{r_m.max() * 100:+.1f}% ({r_m.idxmax():%b-%y})", C["green"]),
    kpi("Worst month", f"{r_m.min() * 100:+.1f}% ({r_m.idxmin():%b-%y})", C["red"]),
    kpi("Win rate", f"{(r_m > 0).mean() * 100:.0f}% of months"),
    kpi("Income / Fees", f"{income_total.sum():,.0f} € / {fees_abs.sum():,.0f} €"),
], style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginBottom": "18px"})


# =============================================================== LAYOUT
def graph(fig, gid=None):
    kw = {"id": gid} if gid else {}
    return dcc.Graph(figure=fig, config={"displaylogo": False}, style={"marginBottom": "18px"}, **kw)


def row(*figs):
    return html.Div([html.Div(graph(f), style={"flex": "1", "minWidth": "0"}) for f in figs],
                    style={"display": "flex", "gap": "16px"})


TAB = dict(backgroundColor=BG, color="#d0d0d0", border="1px solid #444", padding="10px")
TAB_SEL = dict(backgroundColor="#3a3a3a", color="white", border="1px solid #444",
               borderTop=f"3px solid {C['blue']}", padding="10px")


def tab(label, children):
    return dcc.Tab(label=label, children=children, style=TAB, selected_style=TAB_SEL)


app = Dash(__name__)
app.title = "Trading dashboard"

app.layout = html.Div([
    html.H2("Trading dashboard", style={"margin": "0 0 14px 0"}),
    kpis,
    dcc.Tabs(children=[
        tab("Overview", [graph(fig_equity()), graph(fig_returns()), graph(fig_cashflows())]),
        tab("Performance", [graph(fig_pl()), row(fig_heatmap(), fig_distribution()), graph(fig_rolling())]),
        tab("Income", [
            graph(fig_income_monthly()),
            html.Div([
                html.Label("Period:", style={"marginRight": "8px"}),
                dcc.Dropdown(id="income-period", clearable=False, value="Q", style={"width": "180px", "color": "#000"},
                             options=[{"label": "Quarterly", "value": "Q"},
                                      {"label": "Half-yearly", "value": "H"},
                                      {"label": "Yearly", "value": "Y"}]),
                dcc.RadioItems(id="income-mode", value="amount", inline=True,
                               options=[{"label": " Amount (€)", "value": "amount"},
                                        {"label": " Share (%)", "value": "share"}],
                               style={"marginLeft": "24px"}, inputStyle={"marginLeft": "14px"}),
            ], style={"display": "flex", "alignItems": "center", "margin": "10px 0"}),
            dcc.Graph(id="income-aggregated", config={"displaylogo": False}),
        ]),
        tab("Costs & cash", [graph(fig_costs()), graph(fig_cash())]),
        tab("Risk", [graph(fig_drawdown())]),
    ], colors=dict(border="#444", primary=C["blue"], background=BG)),
], style={"background": "#1f1f1f", "color": "#d0d0d0", "fontFamily": FONT, "padding": "20px"})


# =============================================================== CALLBACK
@app.callback(Output("income-aggregated", "figure"),
              Input("income-period", "value"), Input("income-mode", "value"))
def update_income(period, mode):
    if not income_cols:
        return empty_fig("Income breakdown (missing data)")

    d = df_income
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
    for c in income_cols:
        share = (agg[c] / totals * 100).fillna(0)
        fig.add_trace(go.Bar(x=labels, y=agg[c], name=pretty(c), marker_color=color_for(c),
                             customdata=share,
                             hovertemplate="%{y:,.2f} € (%{customdata:.1f}%)<extra></extra>"))
    style(fig, "Income breakdown per period (* = incomplete period)", time_axis=False,
          ytitle="€" if mode == "amount" else "% of income")
    fig.update_layout(barmode="stack", barnorm="percent" if mode == "share" else None)
    fig.update_xaxes(type="category")
    return fig


# =============================================================== RUN
if __name__ == "__main__":
    app.run(debug=True)