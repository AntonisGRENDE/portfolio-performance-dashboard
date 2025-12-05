# trading_dashboard.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots  # <-- add this

from dash.dependencies import Input, Output




# === CONFIG ===
excel_file = r"C:\Users\anton\Downloads\try\Trading121 monthly statements december 2023-december 2024.xlsx"
GRAPH_HEIGHT = 650  # ~30% taller than default


# === LOAD & CLEAN DATA ===
df_raw = pd.read_excel(excel_file, header=0)
df_raw = df_raw.set_index(df_raw.columns[0])
df = df_raw.T

df.columns = [str(c).strip().replace(":", "").replace("/", "_").replace(" ", "_") for c in df.columns]
df.index = pd.to_datetime(df.index, format="%b-%y", errors="coerce")
# Keep only rows from October 2023 onward
df = df[df.index >= "2023-10-01"]


for col in df.columns:
    # Keep only digits, dot, minus
    series = df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)

    # Convert to numeric first
    series = pd.to_numeric(series, errors="coerce")

    # Replace NaN placeholders if needed (optional, now already numeric)
    series = series.replace({np.nan: np.nan})

    df[col] = series



# Keep only months where at least one column has data
df = df[df.notna().any(axis=1)]

# === HELPER: FIND COLUMNS ===
def find_col(possible_names):
    for name in df.columns:
        if any(alt.lower() in name.lower() for alt in possible_names):
            return name
    return None


col_equity     = find_col(["equity", "account_value"])
col_realised   = find_col(["realised"])
col_unrealised = find_col(["unrealised"])
col_deposits   = find_col(["deposit", "withdrawal"])
col_dividends  = find_col(["dividend"])
col_cashback   = find_col(["cashback"])
col_interest   = find_col(["interest"])
col_sharelend  = find_col(["share", "lending", "lend"])
col_cash       = find_col(["cash"])



income_cols = [c for c in [col_dividends, col_cashback, col_interest, col_sharelend] if c]

# === Monthly income amounts ===
# Build a dedicated income DataFrame with a COMPLETE monthly index and 0s for missing months
if income_cols:
    df_income = df[income_cols].copy()
    full_income_index = pd.date_range(df.index.min(), df.index.max(), freq="MS")  # month start
    df_income = df_income.reindex(full_income_index).fillna(0)
else:
    df_income = pd.DataFrame()


# === Equity drawdown ===
if col_equity:
    equity_cummax = df[col_equity].cummax()
    drawdown = df[col_equity] / equity_cummax - 1

# === x-axis limits for actual data ===
x_min = df.index.min()
x_max = df.index.max()
# Create Total P/L = Realised + Unrealised
if col_realised and col_unrealised:
    df["Total_PL"] = df[col_realised].fillna(0) + df[col_unrealised].fillna(0)
else:
    df["Total_PL"] = np.nan

# === Equity % change and Total P&L % of equity (monthly) ===
if col_equity:
    df["Equity_Pct_Change"] = df[col_equity].pct_change() * 100

    # Total P&L relative to previous equity
    df["Total_PL_pct"] = (df["Total_PL"] / df[col_equity].shift(1)) * 100

    # Replace NaN or inf in the first row with 0
    if not df.empty:
        first_idx = df.index[0]
        df.loc[first_idx, "Equity_Pct_Change"] = 0
        df.loc[first_idx, "Total_PL_pct"] = 0
else:
    df["Equity_Pct_Change"] = np.nan
    df["Total_PL_pct"] = np.nan

# === Deposits/Withdrawals as % of previous equity ===
if col_deposits and col_equity:
    df["Deposits_pct"] = (df[col_deposits] / df[col_equity].shift(1)) * 100
    # First month should be 0, not NaN
    if not df.empty:
        first_idx = df.index[0]
        df.loc[first_idx, "Deposits_pct"] = 0
else:
    df["Deposits_pct"] = np.nan






# === DASH APP ===
app = Dash(__name__)

dark_layout = dict(
    font=dict(family="JetBrains Mono, monospace", color="#d0d0d0", size=12),
    plot_bgcolor="#2e2e2e",
    paper_bgcolor="#2e2e2e",
    # Move legend inside the graph (top-right corner)
    legend=dict(
        x=0.99,
        y=0.99,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(0,0,0,0.5)",  # Semi-transparent background
        bordercolor="#444444",
        borderwidth=1
    ),
    xaxis=dict(
        color="#d0d0d0",
        tickangle=45,
        gridcolor="#444444",
        zerolinecolor="#444444",
        tickformat="%b-%y",
        dtick="M1",
        title_font=dict(size=16),
        tickfont=dict(size=12),
        range=[x_min, x_max],
        fixedrange=False   # prevent infinite pan
    ),
    yaxis=dict(
        color="#d0d0d0",
        gridcolor="#444444",
        zerolinecolor="#666666",  # Make 0 line lighter/more visible by default
        zerolinewidth=2,          # Make it thicker
        title_font=dict(size=16),
        tickfont=dict(size=14),
    )
)

# === DASH LAYOUT ===
app.layout = html.Div([

    # Equity curve
    dcc.Graph(
        id="equity-curve",
        figure=go.Figure([
            go.Scatter(
                x=df.index, y=df[col_equity], mode="lines+markers", name="Equity",
                hovertemplate="%{y:.2f}€<extra></extra>"
            )
        ]).update_layout(
            title="Account Equity Over Time",
            xaxis_title="Date",
            yaxis_title="Equity (€)",
            dragmode="zoom",
            hovermode="x",
            height=GRAPH_HEIGHT,
            **dark_layout
        ).update_yaxes(
            # Fix for spike: set range manually if data exists, ignoring extreme outliers if necessary
            # Or simply use range based on the 10th to 100th percentile to cut off initial spikes
            range=[
                df[col_equity].min() * 0.95,
                df[col_equity].max() * 1.05
            ] if col_equity else None
        ) if col_equity else go.Figure().update_layout(title="Equity Curve (missing data)")
    ),

    # Equity % Change vs Total P&L (% of equity per month)
    dcc.Graph(
        id="equity-pct-totalpl",
        figure=go.Figure([
            go.Scatter(
                x=df.index,
                y=df["Equity_Pct_Change"],
                mode="lines+markers",
                name="Equity % Change",
                line=dict(color="#1f77b4"),
                hovertemplate="%{y:.2f}%<extra></extra>"
            ),
            go.Scatter(
                x=df.index,
                y=df["Total_PL_pct"],
                name="Total P&L (% of equity)",
                marker_color="#32CD32",
                hovertemplate="%{y:.2f}%<extra></extra>"
            ),
            go.Scatter(
                x=df.index,
                y=df["Deposits_pct"],
                name="Deposits/Withdrawals (% of equity)",
                marker_color="#FFD700",
                hovertemplate="%{y:.2f}%<extra></extra>"
            )
        ]).update_layout(
            title="Equity % Change vs Total P&L (monthly % of equity) and Deposits/Withdrawals",
            xaxis_title="Date",
            yaxis_title="Percentage (%)",
            height=GRAPH_HEIGHT,
            barmode="group",
            hovermode="x unified",
            **dark_layout
        ).update_yaxes(
            # Make 0 line very prominent specifically for this graph
            zeroline=True,
            zerolinecolor="white",
            zerolinewidth=3,
            # Fix spike: Limit y-axis if there's a massive outlier (e.g., > 100%)
            # We cap visualization at -100% to +100% roughly, or dynamically based on median
            range=[-35, 35] # Example hard cap, or calculated:
            # range=[df["Total_PL_pct"].quantile(0.05), df["Total_PL_pct"].quantile(0.95)]
        )
    ),



    # Realized vs Unrealized P&L Per Month
    dcc.Graph(
        id="realised-unrealised",
        figure=go.Figure([
            go.Bar(x=df.index,
                   y=df[col_realised],
                   name="Realised P&L Per Month",
                   hovertemplate="%{y:.2f}€<extra></extra>") if col_realised else None,
            go.Bar(x=df.index, y=df[col_unrealised],
                   name="Unrealised P&L",
                   marker_color="#8B4513",
                   hovertemplate="%{y:.2f}€<extra></extra>") if col_unrealised else None,
            go.Bar(
                x=df.index,
                y=df["Total_PL"],
                name="Total P&L per Month",
                marker_color="#00CED1",  # turquoise, can change
                hovertemplate="%{y:.2f}€<extra></extra>"
            )
        ]).update_layout(
            barmode="group",
            title="Realized vs Unrealized P&L vs Total P&L per Month",
            xaxis_title="Date",
            yaxis_title="Amount (€)",
            height=GRAPH_HEIGHT,
            **dark_layout)
        if col_realised or col_unrealised else go.Figure().update_layout(title="Realized vs Unrealized P&L (missing data)")
    ),

    # # Total P/L per month
    # dcc.Graph(
    #     id="total-pl",
    #     figure=go.Figure([
    #         go.Bar(
    #             x=df.index,
    #             y=df["Total_PL"],
    #             name="Total P&L per Month",
    #             marker_color="#00CED1",  # turquoise, can change
    #             hovertemplate="%{y:.2f}€<extra></extra>"
    #         )
    #     ]).update_layout(
    #         title="Total P&L (Realized + Unrealized) per Month",
    #         xaxis_title="Date",
    #         yaxis_title="Amount (€)",
    #         height=GRAPH_HEIGHT,
    #         barmode="group",
    #         **dark_layout
    #     )
    # ),



    # Income breakdown as multiple pies per period
    html.Div([
        html.Label("Select Income Aggregation Period:"),
        dcc.Dropdown(
            id="income-period-dropdown",
            options=[
                {"label": "Quarterly", "value": "QE"},
                {"label": "6 Months", "value": "6ME"},
                {"label": "Yearly", "value": "YE"},
            ],
            value="QE",  # default: quarterly
            clearable=False,
        ),
        dcc.Graph(id="income-aggregated"),
    ]),
    # ... existing code ...

    # Monthly income (actual amounts)
    dcc.Graph(
        id="income-monthly",
        figure=go.Figure([
            go.Bar(
                x=df_income.index,
                y=df_income[c],
                name=c.replace("_", " "),
                hovertemplate="%{y:.2f}€<extra></extra>"
            )
            for c in income_cols
        ]).update_layout(
            barmode="stack",
            title="Monthly Income without deposits and selling stock profits (actual amounts)",
            xaxis_title="Date",
            yaxis_title="Amount (€)",
            height=GRAPH_HEIGHT,
            dragmode="zoom",
            **dark_layout
        )
        if income_cols else go.Figure().update_layout(title="Income Breakdown (missing data)")
    ),

    # Deposits / Withdrawals
    dcc.Graph(
        id="deposits",
        figure=go.Figure([
            go.Bar(x=df.index, y=df[col_deposits], name="Deposits/Withdrawals", marker_color="orange",
                   hovertemplate="%{y:.2f}€<extra></extra>")
        ]).update_layout(title="Deposits and Withdrawals", xaxis_title="Date", yaxis_title="Amount (€)", height=GRAPH_HEIGHT, **dark_layout)
        if col_deposits else go.Figure().update_layout(title="Deposits/Withdrawals (missing data)")
    ),

    # Equity drawdown
    dcc.Graph(
        id="equity-drawdown",
        figure=go.Figure([
            go.Scatter(x=df.index, y=drawdown*100, mode="lines+markers", name="Drawdown",
                       hovertemplate="%{y:.2f}%<extra></extra>")
        ]).update_layout(title="Equity Drawdown Over Time", xaxis_title="Date", yaxis_title="Drawdown (%)", **dark_layout)
        if col_equity else go.Figure().update_layout(title="Drawdown (missing data)")
    ),

    # New Graph: Cash over time
    dcc.Graph(
        id="cash",
        figure=go.Figure([go.Scatter(
            x=df.index,
            y=df.get("Cash", pd.Series(np.nan, index=df.index)),
            name="Cash",
            mode="lines+markers",
            hovertemplate="%{y}<extra></extra>"
        )]).update_layout(
            title="Cash Over Time",
            xaxis_title="Date",
            yaxis_title="Amount (€)",
            height=GRAPH_HEIGHT,
            **dark_layout
        )
    ),


    # New Graph: Commissions and Fees
    dcc.Graph(
        id="commissions-fees",
        figure=go.Figure([go.Bar(
            x=df.index, y=df.get("Commissions_and_fees", pd.Series(np.nan, index=df.index)), name="Commissions and Fees", marker_color="red"
        )]).update_layout(
            title="Commissions and Fees",
            xaxis_title="Date",
            yaxis_title="Amount (€)",
            height=GRAPH_HEIGHT,
            **dark_layout
        )
    ),
])

@app.callback(
    Output("income-aggregated", "figure"),
    Input("income-period-dropdown", "value")
)
def update_income_graph(period):
    if not income_cols:
        return go.Figure().update_layout(
            title="Income Breakdown (missing data)",
            height=600,
            **dark_layout
        )

    # Map dropdown values to pandas resample frequencies
    freq_map = {
        "QE": "Q",    # Quarterly
        "6ME": "6M",  # 6 months
        "YE": "Y",    # Yearly
    }
    freq = freq_map.get(period, "Q")

    # Aggregate income by selected period
    df_agg = df_income.resample(freq).sum()

    if df_agg.empty:
        return go.Figure().update_layout(
            title="Income Breakdown (no data for selected period)",
            height=600,
            **dark_layout
        )

    n_periods = len(df_agg.index)
    # Safety: avoid rendering absurdly many pies
    if n_periods > 8:
        df_agg = df_agg.tail(8)
        n_periods = len(df_agg.index)

    # Nice labels per period (e.g. "Q1 2024", "2024", "2024-01–2024-06", etc.)
    period_labels = []
    for idx in df_agg.index:
        if freq == "Q":
            period_labels.append(f"Q{idx.quarter} {idx.year}")
        elif freq == "Y":
            period_labels.append(str(idx.year))
        else:  # 6M or other
            start = idx.strftime("%b %Y")
            end = (idx + pd.offsets.MonthEnd(5)).strftime("%b %Y")
            period_labels.append(f"{start} – {end}")

    fig = make_subplots(
        rows=1,
        cols=n_periods,
        specs=[[{"type": "domain"}] * n_periods],
        subplot_titles=period_labels,
    )

    for i, (idx, row) in enumerate(df_agg.iterrows(), start=1):
        fig.add_trace(
            go.Pie(
                labels=[c.replace("_", " ") for c in income_cols],
                values=row[income_cols],
                name=period_labels[i - 1],
                hovertemplate="%{label}: %{value:.2f}€ (%{percent})<extra></extra>",
                textinfo="percent",
            ),
            row=1,
            col=i,
        )

    # Use dark_layout but drop its legend to avoid conflict
    layout_no_legend = {k: v for k, v in dark_layout.items() if k != "legend"}

    fig.update_layout(
        title="Income Breakdown per Period",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        **layout_no_legend,
    )
    return fig

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)