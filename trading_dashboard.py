"""
Trading dashboard (Dash + Plotly + dash-bootstrap-components) - analytics edition.

Dependencies:
    pip install dash dash-bootstrap-components pandas plotly openpyxl numpy

Run:   python trading_dashboard.py            (uses the path in DEFAULT_EXCEL_FILE)
       python trading_dashboard.py "C:\\path\\to\\statements.xlsx"

Tabs: Overview | Performance | Attribution | Income | Costs & cash | Risk | Behaviour | Projection | Data
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

# =============================================================== CONFIG
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL_FILE = BASE_DIR;
EXCEL_FILE = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXCEL_FILE
START_DATE = "2023-10-01"
GRAPH_HEIGHT = 520
RISK_FREE_ANNUAL = 0.02     # used for Sharpe / Sortino. Set to your cash/T-bill rate.
N_SIMS = 3000               # Monte-Carlo paths in the Projection tab

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


# =============================================================== SMALL HELPERS
def fnum(v, spec=",.0f", suffix="", plus=False):
    """Format a number, returning 'n/a' for NaN/inf/None."""
    try:
        if v is None or not np.isfinite(v):
            return "n/a"
    except TypeError:
        return "n/a"
    return format(v, ("+" if plus else "") + spec) + suffix


def safe_div(a, b):
    try:
        return a / b if b not in (0, None) and np.isfinite(a) and np.isfinite(b) else np.nan
    except Exception:
        return np.nan


def mdiff(a, b):
    return (b.year - a.year) * 12 + b.month - a.month


def monthly_irr(flows):
    """Monthly internal rate of return by bisection (money-weighted return)."""
    flows = np.asarray(flows, float)
    t = np.arange(len(flows))
    npv = lambda rt: np.sum(flows / (1 + rt) ** t)
    lo, hi = -0.9, 1.0
    f_lo, f_hi = npv(lo), npv(hi)
    if not np.isfinite(f_lo * f_hi) or f_lo * f_hi > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def streaks(s):
    """Longest win streak, longest loss streak, current streak (+ win / - loss)."""
    best_w = best_l = cur = 0
    for v in s:
        if v > 0:
            cur = cur + 1 if cur > 0 else 1
        elif v < 0:
            cur = cur - 1 if cur < 0 else -1
        else:
            cur = 0
        best_w, best_l = max(best_w, cur), max(best_l, -cur)
    return best_w, best_l, cur


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

# Euro profit implied by the equity curve itself (independent of reported P&L columns)
monthly_pl = (equity - prev - cf).fillna(0.0)
monthly_pl.iloc[0] = 0.0

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

# ---- return statistics (months excluding the first, which has no prior equity)
r_m = ret.iloc[1:]
r_pl = monthly_pl.iloc[1:]
n_months = len(r_m)
years = n_months / 12
rf_m = (1 + RISK_FREE_ANNUAL) ** (1 / 12) - 1

total_profit = cum_profit.iloc[-1]
twr_total = cum_ret.iloc[-1]
cagr = (1 + twr_total) ** (1 / years) - 1 if years > 0 and (1 + twr_total) > 0 else np.nan
vol_ann = r_m.std(ddof=1) * np.sqrt(12) if n_months > 1 else np.nan
sharpe = safe_div((r_m - rf_m).mean() * 12, vol_ann)
downside_dev = np.sqrt(np.mean(np.minimum(r_m - rf_m, 0) ** 2)) * np.sqrt(12) if n_months else np.nan
sortino = safe_div((r_m - rf_m).mean() * 12, downside_dev)
max_dd = perf_dd.min()
cur_dd = perf_dd.iloc[-1]
calmar = safe_div(cagr, abs(max_dd)) if max_dd < 0 else np.nan

win_rate = (r_m > 0).mean() if n_months else np.nan
avg_win = r_m[r_m > 0].mean() if (r_m > 0).any() else np.nan
avg_loss = r_m[r_m < 0].mean() if (r_m < 0).any() else np.nan
payoff = safe_div(avg_win, abs(avg_loss)) if np.isfinite(avg_loss) else np.nan
gross_gain, gross_loss = r_pl[r_pl > 0].sum(), abs(r_pl[r_pl < 0].sum())
profit_factor = safe_div(gross_gain, gross_loss)
longest_win, longest_loss, current_streak = streaks(r_m.values)
var5 = np.percentile(r_m, 5) if n_months >= 5 else np.nan

# money-weighted return (IRR) vs time-weighted: shows whether deposit timing helped
_flows = np.zeros(len(idx))
_flows[0] = -equity.iloc[0]
_flows[1:] = -cf.iloc[1:].values
_flows[-1] += equity.iloc[-1]
_irr_m = monthly_irr(_flows)
mwr_ann = (1 + _irr_m) ** 12 - 1 if np.isfinite(_irr_m) else np.nan

# ---- profit attribution (monthly, euros). Residual = open-position moves & anything not reported.
_real = df[col_realised].reindex(idx).fillna(0) if col_realised else pd.Series(0.0, index=idx)
attr = pd.DataFrame({
    "Realised P&L": _real,
    "Income": income_total,
    "Fees": -fees_abs,
})
attr["Open positions / other"] = monthly_pl - attr.sum(axis=1)
attr.iloc[0] = 0.0                                    # first month has no prior equity baseline
attr_colors = {"Realised P&L": C["blue"], "Income": C["gold"], "Fees": C["red"],
               "Open positions / other": C["brown"]}

# ---- cash / allocation, yields
cash_pct = (df[col_cash] / equity * 100) if col_cash else None
avg_eq = ((equity + prev) / 2).fillna(equity)
income_yield = income_total / avg_eq * 100
fee_drag = fees_abs / avg_eq * 100
income_yield_ann = (income_total.rolling(3).sum() / avg_eq.rolling(3).mean()) * 4 * 100

# ---- rolling risk
roll_vol3 = r_m.rolling(3).std() * np.sqrt(12) * 100
roll_vol6 = r_m.rolling(6).std() * np.sqrt(12) * 100
roll_sharpe6 = ((r_m - rf_m).rolling(6).mean() / r_m.rolling(6).std()) * np.sqrt(12)


def drawdown_periods(pidx):
    """List drawdown episodes (peak -> trough -> recovery) sorted by depth."""
    dd = pidx / pidx.cummax() - 1
    dates, out, start = list(pidx.index), [], None
    for i, (t, v) in enumerate(dd.items()):
        if v < 0 and start is None:
            start = dates[i - 1] if i > 0 else t
        if v >= 0 and start is not None:
            seg = dd.loc[start:t]
            out.append(dict(peak=start, trough=seg.idxmin(), end=t, depth=seg.min()))
            start = None
    if start is not None:
        seg = dd.loc[start:]
        out.append(dict(peak=start, trough=seg.idxmin(), end=None, depth=seg.min()))
    return sorted(out, key=lambda d: d["depth"])


dd_periods = drawdown_periods(perf_index)


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
    return style(go.Figure(), msg, height=300, time_axis=False)


def bar_colors(series):
    return [C["green"] if v >= 0 else C["red"] for v in series.fillna(0)]


# =============================================================== FIGURES - OVERVIEW
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
    style(fig, "Equity vs money you put in", height=int(round(GRAPH_HEIGHT * 1.2)))
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08, bgcolor="#242424", bordercolor="#444444"), row=2, col=1)
    fig.update_yaxes(title_text="Equity (€)", row=1, col=1)
    fig.update_yaxes(title_text="Cum. P/L (€)", row=2, col=1)
    return fig


def fig_composition():
    """Where does the equity come from: starting balance, your deposits, or investment profit."""
    fig = go.Figure([
        go.Bar(x=idx, y=pd.Series(equity.iloc[0], index=idx), name="Starting balance",
               marker_color=C["grey"], hovertemplate=EUR),
        go.Bar(x=idx, y=net_dep - cf.iloc[0], name="Deposits since start",
               marker_color=C["orange"], hovertemplate=EUR),
        go.Bar(x=idx, y=cum_profit, name="Investment profit",
               marker_color=C["teal"], hovertemplate=EUR),
        go.Scatter(x=idx, y=equity, name="Equity", mode="lines+markers",
                   line=dict(color="white", width=2), hovertemplate=EUR),
    ])
    style(fig, "What built your equity: starting balance + deposits + profit", ytitle="€")
    fig.update_layout(barmode="relative")
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


# =============================================================== FIGURES - PERFORMANCE
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
    if np.isfinite(var5):
        fig.add_vline(x=var5 * 100, line_dash="dot", line_color=C["red"],
                      annotation_text=f"5th pct {var5 * 100:.1f}%", annotation_position="top left",
                      annotation_font_color=C["red"])
    style(fig, f"Distribution of monthly returns (win rate {(r > 0).mean() * 100:.0f}%)",
          ytitle="Months", time_axis=False, hovermode="closest", height=300)
    fig.update_xaxes(title_text="Monthly return (%)")
    return fig


def _best_worst_frame():
    r = ret.iloc[1:] * 100
    t = pd.DataFrame({"year": r.index.year, "month": r.index.strftime("%b"), "r": r.values, "date": r.index})
    counts = t.groupby("year")["r"].count()
    t = t[t["year"].isin(counts[counts >= 2].index)]
    return t


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
        # euro P&L from the equity curve itself (always available, unlike reported columns)
        b_pl, w_pl = monthly_pl.get(b_row["date"], np.nan), monthly_pl.get(w_row["date"], np.nan)
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
    r3 = ((1 + ret).rolling(3).apply(np.prod, raw=True) - 1) * 100
    r6 = ((1 + ret).rolling(6).apply(np.prod, raw=True) - 1) * 100
    fig = go.Figure([
        go.Scatter(x=idx, y=r3, name="Rolling 3M", line=dict(color=C["blue"], width=2), hovertemplate=PCT),
        go.Scatter(x=idx, y=r6, name="Rolling 6M", line=dict(color=C["orange"], width=2), hovertemplate=PCT),
    ])
    style(fig, "Rolling returns", ytitle="Return (%)")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


def fig_return_vs_benchmark_free():
    """Growth of 100 invested (time-weighted) - the 'pure' strategy curve."""
    fig = go.Figure([
        go.Scatter(x=idx, y=perf_index * 100, name="Growth of €100 (TWR)", mode="lines+markers",
                   line=dict(color=C["teal"], width=3), fill="tozeroy",
                   fillcolor="rgba(26,188,156,0.10)", hovertemplate="%{y:,.2f}<extra></extra>"),
        go.Scatter(x=idx, y=perf_index.cummax() * 100, name="High-water mark",
                   line=dict(color=C["grey"], width=1, dash="dot"), hovertemplate="%{y:,.2f}<extra></extra>"),
    ])
    style(fig, "Growth of €100 (deposits removed)", ytitle="Value of €100")
    fig.update_yaxes(rangemode="normal")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


# =============================================================== FIGURES - ATTRIBUTION
def fig_waterfall():
    tot = attr.sum()
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
    fig = go.Figure([go.Bar(x=idx, y=attr[c], name=c, marker_color=attr_colors[c], hovertemplate=EUR)
                     for c in attr.columns])
    fig.add_trace(go.Scatter(x=idx, y=monthly_pl, name="Total (from equity curve)", mode="lines+markers",
                             line=dict(color="white", width=2, dash="dot"), hovertemplate=EUR))
    style(fig, "Monthly profit split: realised, income, fees, open positions")
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="€")
    return fig


def fig_attr_cumulative():
    cum = attr.cumsum()
    fig = go.Figure([go.Scatter(x=idx, y=cum[c], name=c, mode="lines",
                                line=dict(color=attr_colors[c], width=2.5), hovertemplate=EUR)
                     for c in cum.columns])
    fig.add_trace(go.Scatter(x=idx, y=cum.sum(axis=1), name="Total profit", mode="lines+markers",
                             line=dict(color="white", width=3), hovertemplate=EUR))
    style(fig, "Cumulative profit by source")
    fig.update_yaxes(title_text="€")
    fig.update_xaxes(rangeselector=RANGESELECTOR, rangeslider=dict(visible=False))
    return fig


def fig_win_loss_pl():
    """Euro P&L per month sorted - shows how lopsided the profit is."""
    s = r_pl.sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=[d.strftime("%b-%y") for d in s.index], y=s.values,
                           marker_color=bar_colors(s), hovertemplate="%{x}: %{y:,.2f} €<extra></extra>"))
    style(fig, "Months ranked by euro profit", ytitle="€", time_axis=False, hovermode="closest", height=360)
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


# =============================================================== FIGURES - INCOME / COSTS
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


def fig_income_yield():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=income_yield, name="Income (% of equity, monthly)",
                         marker_color=C["gold"], hovertemplate="%{y:.3f}%<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Bar(x=idx, y=fee_drag, name="Fees (% of equity, monthly)",
                         marker_color=C["red"], hovertemplate="%{y:.3f}%<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=income_yield_ann, name="Trailing-3M income yield (annualised)",
                             line=dict(color="white", width=2, dash="dot"),
                             hovertemplate="%{y:.2f}%<extra></extra>"), secondary_y=True)
    style(fig, "Income yield vs fee drag")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Monthly (%)", secondary_y=False)
    fig.update_yaxes(title_text="Annualised (%)", secondary_y=True, showgrid=False)
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


def fig_cash_alloc():
    if cash_pct is None:
        return empty_fig("Cash allocation (missing data)")
    cp = cash_pct.clip(lower=0, upper=100)
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


def fig_rolling_risk():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=roll_vol3.index, y=roll_vol3, name="Volatility 3M (ann.)",
                             line=dict(color=C["blue"], width=2), hovertemplate=PCT), 1, 1)
    fig.add_trace(go.Scatter(x=roll_vol6.index, y=roll_vol6, name="Volatility 6M (ann.)",
                             line=dict(color=C["orange"], width=2), hovertemplate=PCT), 1, 1)
    fig.add_trace(go.Scatter(x=roll_sharpe6.index, y=roll_sharpe6, name="Sharpe 6M",
                             line=dict(color=C["purple"], width=2),
                             hovertemplate="%{y:.2f}<extra></extra>"), 2, 1)
    fig.add_hline(y=0, line_color="#888888", row=2, col=1)
    style(fig, "Rolling risk: volatility and Sharpe ratio", height=620)
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    return fig


def fig_return_vs_flow():
    x, y = cf.iloc[1:], r_m * 100
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="Months",
                               marker=dict(size=11, color=C["blue"], line=dict(width=1, color="white")),
                               text=[d.strftime("%b-%y") for d in x.index],
                               hovertemplate="%{text}<br>deposit %{x:,.0f} €<br>return %{y:.2f}%<extra></extra>"))
    corr = np.nan
    if len(x) >= 4 and x.std() > 0 and y.std() > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                 line=dict(color=C["gold"], dash="dash"), hoverinfo="skip"))
        corr = np.corrcoef(x, y)[0, 1]
    style(fig, f"Deposit size vs that month's return (corr {fnum(corr, '.2f')})",
          ytitle="Return (%)", time_axis=False, hovermode="closest", height=380)
    fig.update_xaxes(title_text="Deposit / withdrawal (€)")
    return fig


def fig_return_vs_lag():
    x, y = r_m.shift(1).dropna() * 100, r_m.iloc[1:] * 100
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", name="Months",
                               marker=dict(size=11, color=C["purple"], line=dict(width=1, color="white")),
                               text=[d.strftime("%b-%y") for d in y.index],
                               hovertemplate="%{text}<br>prev %{x:.2f}%<br>this %{y:.2f}%<extra></extra>"))
    corr = np.nan
    if len(x) >= 4 and x.std() > 0 and y.std() > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                 line=dict(color=C["gold"], dash="dash"), hoverinfo="skip"))
        corr = np.corrcoef(x, y)[0, 1]
    style(fig, f"Momentum check: previous vs this month (corr {fnum(corr, '.2f')})",
          ytitle="This month (%)", time_axis=False, hovermode="closest", height=380)
    fig.update_xaxes(title_text="Previous month (%)")
    return fig


def fig_deposit_timing():
    """Cumulative deposits vs cumulative equity profit - were deposits made before gains or losses?"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=idx, y=cf, name="Deposit / withdrawal", marker_color=C["orange"],
                         opacity=0.7, hovertemplate=EUR), secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=ret * 100, name="Return that month (%)", mode="lines+markers",
                             line=dict(color=C["teal"], width=2.5), hovertemplate=PCT), secondary_y=True)
    style(fig, "Deposit timing vs returns")
    fig.update_yaxes(title_text="Deposit (€)", secondary_y=False)
    fig.update_yaxes(title_text="Return (%)", secondary_y=True, showgrid=False)
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
    return stat_table("Return statistics (monthly)", [
        ("Months analysed", f"{n_months}"),
        ("Mean", fnum(r_m.mean() * 100, ".2f", "%", plus=True)),
        ("Median", fnum(r_m.median() * 100, ".2f", "%", plus=True)),
        ("Std deviation", fnum(r_m.std(ddof=1) * 100 if n_months > 1 else np.nan, ".2f", "%")),
        ("Skewness", fnum(r_m.skew(), ".2f")),
        ("Excess kurtosis", fnum(r_m.kurt(), ".2f")),
        ("5th percentile (monthly VaR-ish)", fnum(var5 * 100, ".2f", "%")),
        ("Average winning month", fnum(avg_win * 100, ".2f", "%", plus=True)),
        ("Average losing month", fnum(avg_loss * 100, ".2f", "%", plus=True)),
        ("Payoff ratio (avg win / avg loss)", fnum(payoff, ".2f")),
        ("Profit factor (€ gains / € losses)", fnum(profit_factor, ".2f")),
        ("Longest win / loss streak", f"{longest_win} / {longest_loss} months"),
        ("Current streak", (f"{current_streak} win(s)" if current_streak > 0
                            else f"{-current_streak} loss(es)" if current_streak < 0 else "flat")),
    ])


def table_risk_stats():
    return stat_table(f"Risk-adjusted metrics (risk-free {RISK_FREE_ANNUAL * 100:.1f}% p.a.)", [
        ("Annualised return (CAGR, TWR)", fnum(cagr * 100, ".1f", "%", plus=True)),
        ("Annualised return (money-weighted)", fnum(mwr_ann * 100, ".1f", "%", plus=True)),
        ("Annualised volatility", fnum(vol_ann * 100, ".1f", "%")),
        ("Downside deviation (ann.)", fnum(downside_dev * 100, ".1f", "%")),
        ("Sharpe ratio", fnum(sharpe, ".2f")),
        ("Sortino ratio", fnum(sortino, ".2f")),
        ("Calmar ratio (CAGR / max DD)", fnum(calmar, ".2f")),
        ("Max drawdown (performance)", fnum(max_dd * 100, ".1f", "%")),
        ("Current drawdown", fnum(cur_dd * 100, ".1f", "%")),
        ("Max drawdown (raw equity)", fnum(equity_dd.min() * 100, ".1f", "%")),
    ])


def table_drawdowns():
    if not dd_periods:
        return html.Div("No drawdowns in the data - equity-only-up so far.", className="text-muted py-2")
    head = html.Thead(html.Tr([html.Th(h, style={"textAlign": "center"}) for h in
                               ["Peak", "Trough", "Depth", "Months down", "Recovered", "Months to recover"]]),
                      style={"backgroundColor": "#242424", "borderBottom": "2px solid #555"})
    rows = []
    for d in dd_periods[:6]:
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


# =============================================================== KPIs & INSIGHTS
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
    kpi("Current equity", f"{equity.iloc[-1]:,.0f} €"),
    kpi("Net deposited", f"{net_dep.iloc[-1]:,.0f} €"),
    kpi("Profit / loss", f"{total_profit:+,.0f} €", signed_color(total_profit),
        "Equity gain not explained by deposits"),
    kpi("Return (TWR)", fnum(twr_total * 100, ".1f", "%", plus=True), signed_color(twr_total),
        "Time-weighted, deposits removed"),
    kpi("Annualised (CAGR)", fnum(cagr * 100, ".1f", "%", plus=True), signed_color(cagr)),
    kpi("Money-weighted (ann.)", fnum(mwr_ann * 100, ".1f", "%", plus=True), signed_color(mwr_ann),
        "IRR: also reflects the timing/size of your deposits"),
    kpi("Income / Fees", f"{income_total.sum():,.0f} € / {fees_abs.sum():,.0f} €"),
], className="g-2 mb-2")

kpis_row2 = dbc.Row([
    kpi("Max drawdown", fnum(max_dd * 100, ".1f", "%"), C["red"], "Performance only, deposits removed"),
    kpi("Current drawdown", fnum(cur_dd * 100, ".1f", "%"), C["red"] if cur_dd < -0.001 else C["green"],
        "Distance below the performance high-water mark"),
    kpi("Volatility (ann.)", fnum(vol_ann * 100, ".1f", "%")),
    kpi("Sharpe", fnum(sharpe, ".2f"), signed_color(sharpe), f"Risk-free {RISK_FREE_ANNUAL * 100:.1f}%"),
    kpi("Sortino", fnum(sortino, ".2f"), signed_color(sortino)),
    kpi("Win rate", fnum(win_rate * 100, ".0f", "% of months")),
    kpi("Profit factor", fnum(profit_factor, ".2f"), signed_color(profit_factor - 1 if np.isfinite(profit_factor) else np.nan),
        "Sum of winning-month € / sum of losing-month €"),
], className="g-2 mb-3")


def build_insights():
    out = []

    # 1) headline + timing effect
    gap = (mwr_ann - cagr) * 100 if np.isfinite(mwr_ann) and np.isfinite(cagr) else np.nan
    timing = ""
    if np.isfinite(gap) and abs(gap) >= 1:
        timing = (f" Your deposit timing {'added' if gap > 0 else 'cost'} about {abs(gap):.1f} pts a year "
                  f"(money-weighted {fnum(mwr_ann * 100, '.1f', '%')} vs time-weighted {fnum(cagr * 100, '.1f', '%')}).")
    out.append(("Headline", f"{fnum(total_profit, ',.0f', ' €', plus=True)} profit on {net_dep.iloc[-1]:,.0f} € deposited "
                            f"({fnum(twr_total * 100, '.1f', '%', plus=True)} excl. deposits over {n_months} months).{timing}",
                signed_color(total_profit)))

    # 2) where equity growth came from
    growth = equity.iloc[-1] - equity.iloc[0]
    dep_since = net_dep.iloc[-1] - cf.iloc[0]
    if growth > 0 and np.isfinite(dep_since):
        out.append(("Growth driver", f"{dep_since / growth * 100:.0f}% of your equity growth since the first month came from "
                                     f"deposits, {total_profit / growth * 100:.0f}% from investment profit.", C["orange"]))

    # 3) consistency / payoff
    out.append(("Consistency", f"{fnum(win_rate * 100, '.0f', '%')} of months were positive. Average win "
                               f"{fnum(avg_win * 100, '.1f', '%', plus=True)} vs average loss {fnum(avg_loss * 100, '.1f', '%', plus=True)} "
                               f"(payoff {fnum(payoff, '.2f')}, profit factor {fnum(profit_factor, '.2f')}).",
                C["green"] if np.isfinite(profit_factor) and profit_factor >= 1 else C["red"]))

    # 4) concentration
    if n_months >= 3 and total_profit > 0 and r_pl.max() > 0:
        best_d = r_pl.idxmax()
        share = r_pl.max() / total_profit * 100
        ex_best = ((1 + r_m.drop(r_m.idxmax())).prod() - 1) * 100
        out.append(("Profit concentration", f"{best_d:%b-%y} alone produced {share:.0f}% of total profit. Without the best "
                                            f"month the return would be {fnum(ex_best, '.1f', '%', plus=True)} instead of "
                                            f"{fnum(twr_total * 100, '.1f', '%', plus=True)}.", C["gold"]))

    # 5) risk
    worst = dd_periods[0] if dd_periods else None
    rec_txt = ""
    if worst:
        rec_txt = (f" It bottomed in {worst['trough']:%b-%y} and "
                   + (f"recovered by {worst['end']:%b-%y}." if worst["end"] is not None else "has not fully recovered yet."))
    out.append(("Risk", f"Max drawdown {fnum(max_dd * 100, '.1f', '%')}, now {fnum(cur_dd * 100, '.1f', '%')} from the high. "
                        f"Annualised volatility {fnum(vol_ann * 100, '.1f', '%')}, Sharpe {fnum(sharpe, '.2f')}.{rec_txt}",
                C["red"] if cur_dd < -0.05 else C["blue"]))

    # 6) income vs fees
    inc, fee = income_total.sum(), fees_abs.sum()
    if inc > 0 or fee > 0:
        cover = safe_div(inc, fee)
        yld = income_yield.iloc[1:].mean() * 12 if n_months else np.nan
        out.append(("Income vs fees", f"Income {inc:,.0f} € vs fees {fee:,.0f} € "
                                      f"({'covers fees ' + fnum(cover, '.1f', '×') if np.isfinite(cover) else 'no fees recorded'}). "
                                      f"Average income yield ≈ {fnum(yld, '.1f', '%')} per year on equity; income is "
                                      f"{fnum(safe_div(inc, total_profit) * 100, '.0f', '%')} of total profit.", C["gold"]))

    # 7) cash
    if cash_pct is not None:
        out.append(("Cash drag", f"Cash averaged {cash_pct.mean():.0f}% of equity (now {cash_pct.iloc[-1]:.0f}%). "
                                 f"If the invested part earned your average return, idle cash cost roughly "
                                 f"{fnum(max(0.0, cash_pct.mean() / 100 * max(cagr, 0) * 100) if np.isfinite(cagr) else np.nan, '.1f', ' pts/yr')} "
                                 f"of annual return - a rough estimate, not a prediction.", C["teal"]))

    # 8) streak
    out.append(("Streaks", f"Longest winning run {longest_win} month(s), longest losing run {longest_loss}. "
                           f"Currently {'on a ' + str(current_streak) + '-month winning run' if current_streak > 0 else 'on a ' + str(-current_streak) + '-month losing run' if current_streak < 0 else 'flat'}.",
                C["purple"]))

    # 9) caveat
    if n_months < 24:
        out.append(("Sample size", f"Only {n_months} monthly observations: ratios such as Sharpe, Sortino, skew and the "
                                   "projection are noisy. Treat them as descriptive, not predictive.", C["grey"]))
    return out


def insight_card(title, text, color):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(title, style={"fontSize": "12px", "color": color, "fontWeight": "700",
                               "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        html.Div(text, style={"fontSize": "13px", "lineHeight": "1.5", "marginTop": "4px"}),
    ], className="p-3"), style={"backgroundColor": "#2e2e2e", "border": "1px solid #444",
                                "borderLeft": f"4px solid {color}", "borderRadius": "8px"},
        className="h-100"), xs=12, md=6, xl=4)


insights_block = html.Div([
    html.H5("Key insights", className="mb-2", style={"fontWeight": "600"}),
    dbc.Row([insight_card(*i) for i in build_insights()], className="g-2 mb-4"),
])


# =============================================================== MONTHLY DATA TABLE
monthly = pd.DataFrame({
    "month": idx.strftime("%Y-%m"),
    "equity": equity.round(2).values,
    "deposits": cf.round(2).values,
    "implied_pl": monthly_pl.round(2).values,
    "ret_pct": (ret * 100).round(2).values,
    "cum_ret_pct": (cum_ret * 100).round(2).values,
    "dd_pct": (perf_dd * 100).round(2).values,
    "realised": (_real.round(2).values),
    "unrealised": df[col_unrealised].round(2).values if col_unrealised else np.nan,
    "income": income_total.round(2).values,
    "fees": fees_abs.round(2).values,
    "cash_pct": cash_pct.round(1).values if cash_pct is not None else np.nan,
})
monthly = monthly.replace({np.nan: None})
MONTHLY_NAMES = {"month": "Month", "equity": "Equity €", "deposits": "Deposit €", "implied_pl": "P&L € (equity-implied)",
                 "ret_pct": "Return %", "cum_ret_pct": "Cum. return %", "dd_pct": "Drawdown %",
                 "realised": "Realised €", "unrealised": "Unrealised €", "income": "Income €",
                 "fees": "Fees €", "cash_pct": "Cash %"}


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
    dbc.Col(html.Span(f"Data: {idx.min():%b-%y} → {idx.max():%b-%y}  ·  {n_months} monthly returns",
                      className="text-muted", style={"fontSize": "14px", "fontFamily": FONT}),
            xs=12, sm=5, md=4, className="d-flex align-items-center justify-content-sm-end mt-2 mt-sm-0"),
], className="mb-3 pb-2 border-bottom border-secondary", align="center")

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

_pos_dep = cf.iloc[1:][cf.iloc[1:] > 0]
default_contrib = int(round(_pos_dep.mean(), -1)) if len(_pos_dep) else 0

projection_controls = html.Div([
    html.Label("Horizon (months):", className="me-2 text-muted fw-bold", style={"fontFamily": FONT}),
    dbc.Input(id="proj-horizon", type="number", min=1, max=60, step=1, value=12, debounce=True,
              style={"width": "110px"}, className="me-4"),
    html.Label("Monthly deposit (€):", className="me-2 text-muted fw-bold", style={"fontFamily": FONT}),
    dbc.Input(id="proj-contrib", type="number", step=10, value=default_contrib, debounce=True,
              style={"width": "140px"}),
], className="d-flex align-items-center flex-wrap my-3 p-2 rounded", style={"backgroundColor": "#282828", "border": "1px solid #444"})

app.layout = dbc.Container([
    header,
    kpis_row1,
    kpis_row2,
    dbc.Tabs([
        tab("Overview", "t-overview",
            insights_block,
            graph(fig_equity()), graph(fig_composition()), graph(fig_returns()), graph(fig_cashflows())),
        tab("Performance", "t-perf",
            graph(fig_return_vs_benchmark_free()),
            graph(fig_pl()),
            row(fig_heatmap(), fig_distribution()),
            graph(fig_best_worst()),
            table_best_worst(),
            graph(fig_rolling())),
        tab("Attribution", "t-attr",
            graph(fig_waterfall()),
            html.Div("Realised P&L, income and fees come from your statement columns. "
                     "'Open positions / other' is the remainder needed to match the equity curve "
                     "(price moves on open positions, FX, anything unreported). If your realised figure "
                     "is already net of fees, the fees bar double-counts and the remainder absorbs the difference.",
                     className="note mb-3"),
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
            html.Div("Do you deposit more after good or bad months, and does a good month predict the next one? "
                     "With a small sample the correlations are weak evidence - look for obvious patterns only.",
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
        tab("Data", "t-data",
            html.Div([dbc.Button("Download CSV", id="btn-csv", color="primary", outline=True, size="sm"),
                      dcc.Download(id="dl-csv")], className="mb-3"),
            data_table()),
    ], active_tab="t-overview", className="mb-3"),
    html.Div("Methodology: returns are time-weighted (Modified Dietz, deposits assumed mid-month). "
             "Money-weighted return is the IRR of month-end flows. Drawdown, volatility, Sharpe and Sortino "
             "use monthly returns, annualised with √12. 'Equity-implied P&L' = Δequity − deposits.",
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


@app.callback(Output("proj-graph", "figure"), Output("proj-text", "children"),
              Input("proj-horizon", "value"), Input("proj-contrib", "value"))
def update_projection(horizon, contrib):
    horizon = int(np.clip(horizon if horizon is not None else 12, 1, 60))
    contrib = float(contrib or 0)
    rr = r_m.dropna().values
    if len(rr) < 6:
        return empty_fig("Projection needs at least 6 months of returns"), ""

    rng = np.random.default_rng(42)
    sims = rng.choice(rr, size=(N_SIMS, horizon))
    paths = np.empty((N_SIMS, horizon + 1))
    paths[:, 0] = equity.iloc[-1]
    for h in range(horizon):
        paths[:, h + 1] = paths[:, h] * (1 + sims[:, h]) + contrib
    p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
    future = pd.date_range(idx[-1], periods=horizon + 1, freq="MS")
    flat = equity.iloc[-1] + contrib * np.arange(horizon + 1)
    invested_total = net_dep.iloc[-1] + contrib * horizon

    fig = go.Figure([
        go.Scatter(x=idx, y=equity, name="History", mode="lines+markers",
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

    end = paths[:, -1]
    p_loss = (end < invested_total).mean() * 100
    text = (f"Median outcome {np.median(end):,.0f} € (10th–90th percentile {p10[-1]:,.0f} – {p90[-1]:,.0f} €). "
            f"Total deposited would be {invested_total:,.0f} €; in {p_loss:.0f}% of simulations you end below it. "
            f"Resampled mean monthly return {rr.mean() * 100:+.2f}% · volatility {rr.std(ddof=1) * 100:.2f}%.")
    return fig, text


@app.callback(Output("dl-csv", "data"), Input("btn-csv", "n_clicks"), prevent_initial_call=True)
def download_csv(_):
    return dcc.send_data_frame(monthly.rename(columns=MONTHLY_NAMES).to_csv, "monthly_summary.csv", index=False)


# =============================================================== RUN
if __name__ == "__main__":
    app.run(debug=True)