import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# === CONFIG ===
excel_file = r"C:\Users\anton\Downloads\try\Trading121 monthly statements december 2023-december 2024.xlsx"
date_range = ("2023-10-01", "2025-08-31")
IMG_SIZE = (16, 9)
DPI = 400
CURRENCY = "€"   # change to "$" if USD account

# === LOAD & CLEAN DATA ===
df_raw = pd.read_excel(excel_file, header=0)

# First column = metrics
df = df_raw.set_index(df_raw.columns[0]).T

# Now the index are months, columns are metrics
df.index = pd.to_datetime(df.index, format="%b-%y", errors="coerce")
df.columns = [str(c).strip().replace(":", "").replace("/", "_").replace(" ", "_") for c in df.columns]

# Clean values
for col in df.columns:
    df[col] = (
        df[col].astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace({"": np.nan, "?": np.nan})
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_index()
df = df.loc[pd.to_datetime(date_range[0]):pd.to_datetime(date_range[1])]


plt.style.use("seaborn-v0_8")

def format_xaxis(ax, series):
    valid = series.dropna()
    ax.set_xticks(valid.index)
    ax.set_xticklabels(valid.index.strftime("%b-%y"), rotation=45, ha="right")

def save_with_caption(ax, series, filename, caption, percent=False):
    format_xaxis(ax, series)
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(f"{CURRENCY}{{x:,.0f}}"))
    plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()

# === GRAPHS ===

# 1. Equity curve
if "Account_value_Equity" in df.columns:
    fig, ax = plt.subplots(figsize=IMG_SIZE)
    df["Account_value_Equity"].dropna().plot(ax=ax, color="blue")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity Value")
    save_with_caption(ax, df["Account_value_Equity"], "equity_curve.png",
                      "Total account equity (investments + cash) over time.")

# 2. Cumulative income contributions
income_cols = [c for c in ["Dividends", "Cashback", "Share_lending", "Interest_on_cash"] if c in df.columns]
if income_cols:
    df_income = df[income_cols].cumsum()
    fig, ax = plt.subplots(figsize=IMG_SIZE)
    df_income.plot(ax=ax, marker="o")
    ax.set_title("Cumulative Income Contributions")
    ax.set_ylabel("Cumulative Income")
    save_with_caption(ax, df_income.sum(axis=1), "cumulative_income.png",
                      "Cumulative contributions from dividends, cashback, lending, and interest.")

# 3. Percentage change in cumulative income
if income_cols:
    df_income_change = df_income.pct_change() * 100
    fig, ax = plt.subplots(figsize=IMG_SIZE)
    df_income_change.plot(ax=ax, marker="o")
    ax.set_title("Percentage Change in Cumulative Income")
    ax.set_ylabel("% Change")
    save_with_caption(ax, df_income_change.sum(axis=1), "cumulative_income_pct_change.png",
                      "Monthly percentage growth in total cumulative income sources.", percent=True)

# 4. Realised + Unrealised P&L
if {"Realised_PL", "Unrealised_PL"}.issubset(df.columns):
    fig, ax = plt.subplots(figsize=IMG_SIZE)
    df[["Realised_PL", "Unrealised_PL"]].dropna().plot(ax=ax, marker="o")
    ax.set_title("Monthly P&L (Realised & Unrealised)")
    ax.set_ylabel("Profit / Loss")
    save_with_caption(ax, df["Realised_PL"], "pl_monthly.png",
                      "Monthly realised and unrealised profits/losses.")

# 5. Rolling 3-month average P&L
if "Realised_PL" in df.columns:
    fig, ax = plt.subplots(figsize=IMG_SIZE)
    df["Realised_PL"].rolling(3).mean().plot(ax=ax, color="purple")
    ax.set_title("3-Month Rolling Average of Realised P&L")
    ax.set_ylabel("Average P&L (3M)")
    save_with_caption(ax, df["Realised_PL"], "rolling_pl.png",
                      "Smoothed view of realised P&L with a 3-month moving average.")

# 6. Deposits vs Equity growth
if {"Deposit_Withdrawal", "Account_value_Equity"}.issubset(df.columns):
    fig, ax = plt.subplots(figsize=IMG_SIZE)
    df["Deposit_Withdrawal"].cumsum().plot(ax=ax, label="Cumulative Deposits", color="orange")
    df["Account_value_Equity"].plot(ax=ax, label="Equity", color="blue")
    ax.set_title("Cumulative Deposits vs Equity Growth")
    ax.set_ylabel("Value")
    ax.legend()
    save_with_caption(ax, df["Account_value_Equity"], "deposits_vs_equity.png",
                      "Comparison of net deposits and total account equity over time.")

# 7. Income breakdown (stacked bar)
if income_cols:
    fig, ax = plt.subplots(figsize=IMG_SIZE)
    df[income_cols].plot.bar(stacked=True, ax=ax)
    ax.set_title("Monthly Income Breakdown")
    ax.set_ylabel("Income")
    save_with_caption(ax, df[income_cols].sum(axis=1), "income_breakdown.png",
                      "Stacked breakdown of monthly income sources: dividends, cashback, interest, lending.")

# 8. Heatmap of monthly realised P&L
if "Realised_PL" in df.columns:
    pivot = df["Realised_PL"].copy()
    pivot.index = pd.to_datetime(pivot.index)
    heatmap = pivot.groupby([pivot.index.year, pivot.index.month]).sum().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto")
    ax.set_title("Heatmap of Monthly Realised P&L")
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.get_level_values(0).unique())
    fig.colorbar(im, ax=ax, label="P&L")
    plt.savefig("pl_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()

print("✅ High-resolution (4K) charts with detailed axes and new insights saved.")