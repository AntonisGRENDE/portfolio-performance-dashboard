"""
Trading dashboard (Dash + Plotly) - improved version with dash-bootstrap-components.

Dependencies:
    pip install dash dash-bootstrap-components pandas plotly openpyxl

Run:   python trading_dashboard.py            (uses the path in EXCEL_FILE)
       python trading_dashboard.py "C:\\path\\to\\statements.xlsx"
"""
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
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


# =============================================================== STYLE HELPERS
def style(fig, title, ytitle=None, height=GRAPH_HEIGHT, time_axis=True, hovermode="x unified"):
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
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08, bgcolor="#242424", bordercolor="#444444"), row=2, col=1)
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
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
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
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
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


def fig_best_worst():
    r = ret.iloc[1:] * 100
    t = pd.DataFrame({"year": r.index.year, "month": r.index.strftime("%b"), "r": r.values})
    counts = t.groupby("year")["r"].count()
    valid_years = counts[counts >= 2].index
    t = t[t["year"].isin(valid_years)]
    if t.empty:
        return empty_fig("Best & worst month per year (not enough data)")

    best = t.loc[t.groupby("year")["r"].idxmax()].set_index("year").sort_index()
    worst = t.loc[t.groupby("year")["r"].idxmin()].set_index("year").sort_index()
    years = [str(y) for y in best.index]

    def trace(d, name, color):
        return go.Bar(
            x=years, y=d["r"], name=name, marker_color=color,
            text=[f"{m}<br>{v:+.1f}%" for m, v in zip(d["month"], d["r"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x} · " + name + ": %{y:.2f}%<extra></extra>")

    fig = go.Figure([trace(best, "Best month", C["green"]),
                     trace(worst, "Worst month", C["red"])])
    style(fig, "Best & worst month per year", ytitle="Return (%)",
          time_axis=False, hovermode="closest", height=360)
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    return fig


def table_best_worst():
    r = ret.iloc[1:] * 100
    t = pd.DataFrame({"year": r.index.year, "month": r.index.strftime("%b"), "r": r.values, "date": r.index})
    counts = t.groupby("year")["r"].count()
    valid_years = counts[counts >= 2].index
    t = t[t["year"].isin(valid_years)]
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
    has_total_pl = "Total_PL" in df.columns
    for y in sorted(best.index):
        b_row = best.loc[y]
        w_row = worst.loc[y]

        b_pl = df.loc[b_row["date"], "Total_PL"] if has_total_pl and b_row["date"] in df.index else np.nan
        w_pl = df.loc[w_row["date"], "Total_PL"] if has_total_pl and w_row["date"] in df.index else np.nan

        b_pl_str = f"{b_pl:+,.2f} €" if pd.notna(b_pl) else "–"
        w_pl_str = f"{w_pl:+,.2f} €" if pd.notna(w_pl) else "–"

        rows.append(html.Tr([
            html.Td(str(y), style={"textAlign": "center", "fontWeight": "bold"}),
            html.Td(b_row["month"], style={"textAlign": "center", "color": C["green"]}),
            html.Td(f"{b_row['r']:+.2f}%", style={"textAlign": "right", "color": C["green"], "fontWeight": "bold"}),
            html.Td(b_pl_str, style={"textAlign": "right", "color": C["green"]}),
            html.Td(w_row["month"], style={"textAlign": "center", "color": C["red"]}),
            html.Td(f"{w_row['r']:+.2f}%", style={"textAlign": "right", "color": C["red"], "fontWeight": "bold"}),
            html.Td(w_pl_str, style={"textAlign": "right", "color": C["red"]}),
        ]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        hover=True,
        responsive=True,
        className="mb-4",
        style={
            "backgroundColor": "#2e2e2e",
            "color": "#d0d0d0",
            "fontFamily": FONT,
            "fontSize": "13px",
            "borderColor": "#444444",
        }
    )


def fig_rolling():
    r3 = ((1 + ret).rolling(3).apply(np.prod, raw=True) - 1) * 100
    r6 = ((1 + ret).rolling(6).apply(np.prod, raw=True) - 1) * 100
    fig = go.Figure([
        go.Scatter(x=idx, y=r3, name="Rolling 3M", line=dict(color=C["blue"], width=2), hovertemplate=PCT),
        go.Scatter(x=idx, y=r6, name="Rolling 6M", line=dict(color=C["orange"], width=2), hovertemplate=PCT),
    ])
    style(fig, "Rolling returns", ytitle="Return (%)")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


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
    style(fig, "Drawdown: raw equity vs performance only", ytitle="Drawdown (%)")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


# =============================================================== KPIs
def kpi(label, value, color="#d0d0d0"):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(label, style={"fontSize": "12px", "color": "#9AA0A6", "whiteSpace": "nowrap"}),
                html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": color, "whiteSpace": "nowrap"}),
            ], className="p-2 text-center"),
            style={"backgroundColor": "#3a3a3a", "border": "1px solid #444", "borderRadius": "8px"},
            className="h-100",
        ),
        xs=6, md=4, xl=True, className="h-100"
    )


def signed_color(v):
    return C["green"] if v >= 0 else C["red"]


r_m = ret.iloc[1:]
total_profit = cum_profit.iloc[-1]
kpis = dbc.Row([
    kpi("Current equity", f"{equity.iloc[-1]:,.0f} €"),
    kpi("Net deposited", f"{net_dep.iloc[-1]:,.0f} €"),
    kpi("Profit / loss", f"{total_profit:+,.0f} €", signed_color(total_profit)),
    kpi("Return (excl. deposits)", f"{cum_ret.iloc[-1] * 100:+.1f}%", signed_color(cum_ret.iloc[-1])),
    kpi("Max drawdown (perf.)", f"{perf_dd.min() * 100:.1f}%", C["red"]),
    kpi("Best month", f"{r_m.max() * 100:+.1f}% ({r_m.idxmax():%b-%y})", C["green"]),
    kpi("Worst month", f"{r_m.min() * 100:+.1f}% ({r_m.idxmin():%b-%y})", C["red"]),
    kpi("Win rate", f"{(r_m > 0).mean() * 100:.0f}% of months"),
    kpi("Income / Fees", f"{income_total.sum():,.0f} € / {fees_abs.sum():,.0f} €"),
], className="g-2 mb-3")


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
        style={
            "height": f"{h}px",
            "width": "100%",
            "minWidth": "0",
            "maxWidth": "100%",
            "marginBottom": "18px",
        },
        **kw
    )


def row(*figs):
    n = len(figs) or 1
    col_width = 12 // n
    return dbc.Row(
        [dbc.Col(graph(f), xs=12, md=col_width) for f in figs],
        className="g-3"
    )


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
.nav-tabs .nav-link {
    color: #9AA0A6 !important;
    background-color: #282828 !important;
    border-color: #444444 !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin-right: 4px;
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
    dbc.Col(
        html.H2("Trading dashboard", className="mb-0", style={"fontWeight": "600"}),
        xs=12, sm=7, md=8, className="d-flex align-items-center"
    ),
    dbc.Col(
        html.Span(
            f"Data: {idx.min():%b-%y} → {idx.max():%b-%y}",
            className="text-muted",
            style={"fontSize": "14px", "fontFamily": FONT}
        ),
        xs=12, sm=5, md=4, className="d-flex align-items-center justify-content-sm-end mt-2 mt-sm-0"
    ),
], className="mb-3 pb-2 border-bottom border-secondary", align="center")

income_controls = html.Div([
    html.Label("Period:", className="me-2 text-muted fw-bold", style={"fontFamily": FONT}),
    dbc.Select(
        id="income-period",
        value="Q",
        options=[
            {"label": "Quarterly", "value": "Q"},
            {"label": "Half-yearly", "value": "H"},
            {"label": "Yearly", "value": "Y"},
        ],
        style={"width": "180px"},
        className="me-4",
    ),
    dbc.RadioItems(
        id="income-mode",
        value="amount",
        inline=True,
        options=[
            {"label": " Amount (€)", "value": "amount"},
            {"label": " Share (%)", "value": "share"},
        ],
        className="d-flex align-items-center",
        inputClassName="me-1",
        labelClassName="me-3 text-light",
    ),
], className="d-flex align-items-center my-3 p-2 rounded", style={"backgroundColor": "#282828", "border": "1px solid #444"})

app.layout = dbc.Container([
    header,
    kpis,
    dbc.Tabs([
        dbc.Tab(
            dcc.Loading(
                type="circle",
                color=C["blue"],
                children=html.Div([
                    graph(fig_equity()),
                    graph(fig_returns()),
                    graph(fig_cashflows()),
                ], className="pt-3")
            ),
            label="Overview",
        ),
        dbc.Tab(
            dcc.Loading(
                type="circle",
                color=C["blue"],
                children=html.Div([
                    graph(fig_pl()),
                    row(fig_heatmap(), fig_distribution()),
                    graph(fig_best_worst()),
                    table_best_worst(),
                    graph(fig_rolling()),
                ], className="pt-3")
            ),
            label="Performance",
        ),
        dbc.Tab(
            dcc.Loading(
                type="circle",
                color=C["blue"],
                children=html.Div([
                    graph(fig_income_monthly()),
                    income_controls,
                    graph(None, gid="income-aggregated"),
                ], className="pt-3")
            ),
            label="Income",
        ),
        dbc.Tab(
            dcc.Loading(
                type="circle",
                color=C["blue"],
                children=html.Div([
                    graph(fig_costs()),
                    graph(fig_cash()),
                ], className="pt-3")
            ),
            label="Costs & cash",
        ),
        dbc.Tab(
            dcc.Loading(
                type="circle",
                color=C["blue"],
                children=html.Div([
                    graph(fig_drawdown()),
                ], className="pt-3")
            ),
            label="Risk",
        ),
    ], className="mb-3"),
], fluid=True, style={
    "backgroundColor": "#1f1f1f",
    "color": "#d0d0d0",
    "fontFamily": FONT,
    "minHeight": "100vh",
    "padding": "20px",
    "maxWidth": "100%",
    "overflowX": "hidden",
})


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