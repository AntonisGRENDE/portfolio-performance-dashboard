# trading_dashboard.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html

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
df_income = df[income_cols].copy()

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
        zerolinecolor="#444444",
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
                   marker_color="#8B4513",  # <-- change this color
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


    # Replace the previous append with this inside app.layout:
    html.Div([
        html.Label("Select Income Aggregation Period:"),
        dcc.Dropdown(
            id="income-period-dropdown",
            options=[
                {"label": "Monthly", "value": "ME"},
                {"label": "Quarterly", "value": "QE"},
                {"label": "6 Months", "value": "6ME"},
                {"label": "Yearly", "value": "YE"}
            ],
            value="6ME",  # default 6 months
            clearable=False
        ),
        dcc.Graph(id="income-aggregated")
    ]),

    # Monthly income (actual amounts)
    dcc.Graph(
        id="income-monthly",
        figure=go.Figure([
            go.Bar(x=df.index, y=df[c], name=c.replace("_", " "), hovertemplate="%{y:.2f}€<extra></extra>")
            for c in income_cols
        ]).update_layout(barmode="stack",
                         title="Monthly Income without deposits and selling stock profits (actual amounts)",
                         xaxis_title="Date",
                         yaxis_title="Amount (€)",
                         height=GRAPH_HEIGHT,
                         dragmode="zoom",
                         **dark_layout)
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

    # Cash vs Interest graph
    # dcc.Graph(
    #     id="cash-interest",
    #     figure=go.Figure([
    #         go.Scatter(
    #             x=df.index,
    #             y=df.get(col_cash, pd.Series(np.nan, index=df.index)),
    #             name="Cash",
    #             mode="lines+markers",
    #             hovertemplate="%{y:.2f}€<extra></extra>",
    #             line=dict(color="#1f77b4"),  # darker blue for dark theme
    #         ),
    #         go.Scatter(
    #             x=df.index,
    #             y=df.get(col_interest, pd.Series(np.nan, index=df.index)),
    #             name="Interest on Cash",
    #             mode="lines+markers",
    #             hovertemplate="%{y:.2f}€<extra></extra>",
    #             line=dict(color="#ffa500"),  # darker orange
    #             yaxis="y2"
    #         )
    #     ]).update_layout(
    #         title="Cash vs Interest on Cash",
    #         height=GRAPH_HEIGHT,
    #         xaxis=dict(
    #             title="Date",
    #             color="#d0d0d0",
    #             tickangle=45,
    #             gridcolor="#444444",
    #             zerolinecolor="#444444",
    #             tickformat="%b-%y",
    #             dtick="M1",
    #             range=[x_min, x_max],
    #             fixedrange=False
    #         ),
    #         yaxis=dict(
    #             title="Cash (€)",
    #             color="#d0d0d0",
    #             gridcolor="#444444",
    #             zerolinecolor="#444444"
    #         ),
    #         yaxis2=dict(
    #             title="Interest (€)",
    #             overlaying="y",
    #             side="right",
    #             color="#ffa500",
    #             gridcolor="#444444",
    #             zerolinecolor="#444444"
    #         ),
    #         hovermode="x unified",
    #         plot_bgcolor="#2e2e2e",
    #         paper_bgcolor="#2e2e2e",
    #         font=dict(family="JetBrains Mono, monospace", color="#d0d0d0", size=12)
    #     )
    # ),

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
        return go.Figure().update_layout(title="Income Breakdown (missing data)")

    # Resample & sum for selected period
    df_agg = df[income_cols].resample(period).sum()

    fig = go.Figure([
        go.Bar(x=df_agg.index, y=df_agg[c], name=c.replace("_", " "), hovertemplate="%{y:.2f}€<extra></extra>")
        for c in income_cols
    ])
    fig.update_layout(
        title=f"Income Breakdown Aggregated ({period})",
        barmode="stack",
        xaxis_title="Date",
        yaxis_title="Amount (€)",
        height=GRAPH_HEIGHT,
        **dark_layout
    )
    return fig

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)