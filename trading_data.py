"""
trading_data.py - DATA LAYER of the trading dashboard (no Dash / Plotly imports in here).

What this file does
    1. Reads the Excel statement sheet (one row per metric, one column per month).
    2. Maps your row labels to stable internal field names (see ROW_PATTERNS below).
    3. Computes every number the dashboard shows: returns, risk, attribution, income, costs,
       net worth across accounts, cash flow, and the plain-English insights.

trading_dashboard.py imports build_portfolio() from here and only has to DRAW what comes back.
If you change the layout of your sheet again, ROW_PATTERNS is the only place that needs editing.

Dependencies:  pip install pandas numpy openpyxl
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


# =============================================================== SHEET LAYOUT
# field name -> regex that is tested against the *normalised* row label
# (lower-case, every run of punctuation/spaces collapsed into one space, so
#  "Interest on  cash:" -> "interest on cash" and "Account value/Equity:" -> "account value equity").
# Only `equity` is mandatory; every other row is optional and its charts simply disappear if it is missing.
ROW_PATTERNS = {
    # ---- Trading212 account (same meaning as before)
    "deposits":           r"^deposits? withdrawals?$",
    "realised":           r"^reali[sz]ed p l$",
    "unrealised":         r"^unreali[sz]ed p l$",
    "equity":             r"^account value equity$|^equity$",
    "dividends":          r"^dividends?$",
    "interest":           r"^interest",
    "cashback":           r"^cashback",
    "share_lending":      r"^share lending|^lending",
    "cash":               r"^cash( trading ?212)?$",
    "equity_fees":        r"^equity charges",       # NEW: fees are now split over two rows ...
    "commission_fees":    r"^commissions?",         # ... and both must be counted
    "stock_performance":  r"^stock performance",    # NEW: reported realised + unrealised
    "spending":           r"^spending$",
    # ---- NEW: other accounts and cash outside Trading212
    "polivios_deposits":  r"^polivios account deposits? withdrawals?$",
    "polivios_value":     r"^polivios account value$",
    "wallet_cash":        r"^cash held by me",
    "wallet_new":         r"^new cash on my wallet",
    "revolut_in":         r"^revolut deposits",
    "revolut_out":        r"^revolut withdrawals",
    "spending_ex_invest": r"^spendings excluding investments",
    # ---- NEW: totals that you calculate in the sheet (used to cross-check the dashboard)
    "cash_flow_reported": r"^cash flow per month",
    "net_worth_reported": r"^account value total",
}

# Human-readable names, shared with the dashboard so labels never drift apart.
INCOME_LABELS = {"dividends": "Dividends", "interest": "Interest on cash",
                 "cashback": "Cashback", "share_lending": "Share lending"}
FEE_LABELS = {"equity_fees": "Equity charges and fees", "commission_fees": "Commissions and fees"}
NW_LABELS = {"t212": "Trading212 equity", "polivios": "Polivios account",
             "wallet": "Cash held by me", "revolut": "Revolut net flow"}
CF_LABELS = {"t212_flow": "Trading212 deposits / withdrawals", "spending": "Spending (−)",
             "polivios": "Polivios deposits", "wallet_new": "New cash in wallet",
             "revolut_in": "Revolut deposits", "revolut_out": "Revolut withdrawals (−)"}

# Column titles of the Data tab / CSV export.
MONTHLY_NAMES = {"month": "Month", "equity": "Equity €", "deposits": "Deposit €",
                 "implied_pl": "P&L € (equity-implied)", "ret_pct": "Return %",
                 "cum_ret_pct": "Cum. return %", "dd_pct": "Drawdown %", "realised": "Realised €",
                 "unrealised": "Unrealised € (as reported)", "income": "Income €", "fees": "Fees €",
                 "cash_pct": "Cash %", "net_worth": "Net worth € (all accounts)",
                 "cash_flow": "Cash flow €", "spending": "Spending €", "polivios": "Polivios value €"}


# =============================================================== SMALL HELPERS
def norm(label) -> str:
    """Normalise a row label so tiny formatting changes (colons, double spaces, case) don't matter."""
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def fnum(v, spec=",.0f", suffix="", plus=False):
    """Format a number, returning 'n/a' for NaN/inf/None."""
    try:
        if v is None or not np.isfinite(v):
            return "n/a"
    except TypeError:
        return "n/a"
    return format(v, ("+" if plus else "") + spec) + suffix


def safe_div(a, b):
    """a / b that yields NaN instead of raising when b is 0/None/NaN/inf."""
    try:
        return a / b if b not in (0, None) and np.isfinite(a) and np.isfinite(b) else np.nan
    except Exception:
        return np.nan


def mdiff(a, b):
    """Whole months between two timestamps."""
    return (b.year - a.year) * 12 + b.month - a.month


def monthly_irr(flows, times=None):
    """Monthly internal rate of return by bisection (money-weighted return).
    `times` are flow dates in months (fractions allowed); default = 0, 1, 2, ..."""
    flows = np.asarray(flows, float)
    t = np.arange(len(flows), dtype=float) if times is None else np.asarray(times, float)
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


# =============================================================== READ THE EXCEL SHEET
def _to_number(x):
    """Cell -> float. Numbers pass through; text such as '1,234.50 €' is stripped; '?' or blanks become NaN."""
    if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
        return float(x)
    try:
        return float(re.sub(r"[^\d.\-]", "", str(x)))
    except ValueError:
        return np.nan


def map_rows(labels):
    """Match sheet row labels to the internal field names in ROW_PATTERNS.
    Every sheet row is claimed by at most one field, so a label can never be counted twice."""
    found, used = {}, set()
    for field, pattern in ROW_PATTERNS.items():
        for lab in labels:
            if lab not in used and re.search(pattern, norm(lab)):
                found[field] = lab
                used.add(lab)
                break
    return found, [lab for lab in labels if lab not in used]


def read_statements(path, start_date):
    """Return (wide, mapping, unmatched): one row per month, one column per *internal field name*."""
    raw = pd.read_excel(path, header=0)
    label_col = raw.columns[0]
    raw = raw.dropna(subset=[label_col])                       # blank spacer rows in the sheet
    raw[label_col] = raw[label_col].astype(str).str.strip()
    raw = raw.drop_duplicates(subset=label_col, keep="first")  # a repeated label would break the transpose

    mapping, unmatched = map_rows(raw[label_col].tolist())
    if "equity" not in mapping:
        raise SystemExit("Could not find the equity row (expected something like 'Account value/Equity:').\n"
                         f"Row labels found: {raw[label_col].tolist()}")

    wide = raw.set_index(label_col).T[list(mapping.values())]  # months become rows
    wide.columns = list(mapping.keys())                        # from here on: stable names, not sheet labels
    wide.index = pd.to_datetime(wide.index, errors="coerce")
    wide = wide[wide.index.notna()]
    wide.index = wide.index.to_period("M").to_timestamp()      # normalise to month start
    wide = wide[~wide.index.duplicated(keep="last")].sort_index()
    wide = wide[wide.index >= start_date]
    if wide.empty:
        raise SystemExit(f"No statement months on or after START_DATE = {start_date}. Check START_DATE in the config.")
    wide = wide.apply(lambda col: col.map(_to_number))         # '?' placeholders etc. -> NaN
    wide = wide.dropna(how="all")

    if wide["equity"].notna().sum() < 2:
        raise SystemExit("The equity row has fewer than 2 numeric months. If the cells are formulas, open the "
                         "workbook in Excel and save it once so the calculated values are stored in the file.")
    return wide, mapping, unmatched


# =============================================================== PERFORMANCE OF ONE EQUITY CURVE
class Performance:
    """Flow-adjusted performance analytics for ONE equity curve.

    Pure maths, independent of the sheet layout: give it month-end equity and net deposits and it returns
    time-weighted returns (Modified Dietz, flows mid-month), money-weighted return, drawdowns and risk ratios.
    """

    def __init__(self, equity: pd.Series, flows: pd.Series, rf_annual: float):
        self.equity, self.flows = equity, flows
        prev = equity.shift(1)

        # --- time-weighted monthly return, deposits/withdrawals removed
        denom = prev + 0.5 * flows
        ret = ((equity - prev - flows) / denom).where(denom > 0)
        ret.iloc[0] = 0.0                              # first month has no opening equity -> no return
        self.ret = ret.fillna(0.0)
        self.cum_ret = (1 + self.ret).cumprod() - 1
        self.perf_index = 1 + self.cum_ret
        self.perf_dd = self.perf_index / self.perf_index.cummax() - 1    # drawdown of performance only
        self.equity_dd = equity / equity.cummax() - 1                    # raw equity drawdown

        # --- money you put in (opening balance counts: equity - capital_in == cumulative profit)
        self.net_dep = flows.cumsum()
        self.base_capital = equity.iloc[0] - flows.iloc[0]
        self.capital_in = self.net_dep + self.base_capital
        self.cum_profit = equity - self.capital_in

        # --- euro P&L implied by the equity curve itself (independent of any reported P&L row)
        pl = (equity - prev - flows).fillna(0.0)
        pl.iloc[0] = 0.0
        self.monthly_pl = pl
        self.avg_eq = ((equity + prev) / 2).fillna(equity)

        # --- statistics (the first month is excluded: it has no baseline)
        r = self.r_m = self.ret.iloc[1:]
        self.r_pl = pl.iloc[1:]
        n = self.n_months = len(r)
        self.years = n / 12
        self.rf_m = (1 + rf_annual) ** (1 / 12) - 1

        self.total_profit = self.cum_profit.iloc[-1]
        self.twr_total = self.cum_ret.iloc[-1]
        self.cagr = ((1 + self.twr_total) ** (1 / self.years) - 1
                     if self.years > 0 and (1 + self.twr_total) > 0 else np.nan)
        self.vol_ann = r.std(ddof=1) * np.sqrt(12) if n > 1 else np.nan
        excess_ann = (r - self.rf_m).mean() * 12
        self.sharpe = safe_div(excess_ann, self.vol_ann)
        self.downside_dev = np.sqrt(np.mean(np.minimum(r - self.rf_m, 0) ** 2)) * np.sqrt(12) if n else np.nan
        self.sortino = safe_div(excess_ann, self.downside_dev)
        self.max_dd, self.cur_dd = self.perf_dd.min(), self.perf_dd.iloc[-1]
        self.calmar = safe_div(self.cagr, abs(self.max_dd)) if self.max_dd < 0 else np.nan

        self.win_rate = (r > 0).mean() if n else np.nan
        self.avg_win = r[r > 0].mean() if (r > 0).any() else np.nan
        self.avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan
        self.payoff = safe_div(self.avg_win, abs(self.avg_loss)) if np.isfinite(self.avg_loss) else np.nan
        gross_gain, gross_loss = self.r_pl[self.r_pl > 0].sum(), abs(self.r_pl[self.r_pl < 0].sum())
        self.profit_factor = safe_div(gross_gain, gross_loss)
        self.longest_win, self.longest_loss, self.current_streak = streaks(r.values)
        self.var5 = np.percentile(r, 5) if n >= 5 else np.nan
        self.mwr_ann = self._money_weighted_return()
        self.dd_periods = drawdown_periods(self.perf_index)

        # --- rolling risk
        self.roll_vol3 = r.rolling(3).std() * np.sqrt(12) * 100
        self.roll_vol6 = r.rolling(6).std() * np.sqrt(12) * 100
        self.roll_sharpe6 = ((r - self.rf_m).rolling(6).mean() / r.rolling(6).std()) * np.sqrt(12)

    def _money_weighted_return(self):
        """Annualised IRR using the SAME mid-month deposit timing as the time-weighted return:
        opening balance at t=0, the deposit of month k at k-0.5, closing equity at the last month-end."""
        eq, fl = self.equity, self.flows
        times = [0.0] + [k - 0.5 for k in range(1, len(eq))] + [len(eq) - 1.0]
        cash = [-eq.iloc[0]] + [-fl.iloc[k] for k in range(1, len(eq))] + [eq.iloc[-1]]
        irr_m = monthly_irr(cash, times)
        return (1 + irr_m) ** 12 - 1 if np.isfinite(irr_m) else np.nan


# =============================================================== EVERYTHING THE DASHBOARD NEEDS
class Portfolio:
    """All sheet data plus every derived series/number, grouped by topic.

    Attribute groups: raw/idx (data) | perf (Trading212 performance) | income, fees, cash |
    attribution (where profit came from) | nw_* (net worth) | cf_* (cash flow) | poli_* (Polivios) | insights.
    """

    def __init__(self, wide, mapping, unmatched, *, risk_free_annual, unrealised_is_balance):
        self.raw, self.mapping, self.unmatched = wide, mapping, unmatched
        self.idx = wide.index
        self.risk_free_annual = risk_free_annual
        self._check_gaps()

        flows = self.series("deposits").fillna(0)
        self.perf = Performance(wide["equity"], flows, risk_free_annual)
        self._build_income_costs_cash()
        self._build_attribution(unrealised_is_balance)
        self._build_networth_and_cashflow()

        x_min, x_max = self.idx.min(), self.idx.max()
        self.x_range = [(x_min - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
                        (x_max + pd.Timedelta(days=20)).strftime("%Y-%m-%d")]
        pos = flows.iloc[1:][flows.iloc[1:] > 0]       # suggested monthly deposit for the Projection tab
        self.default_contribution = int(round(pos.mean(), -1)) if len(pos) else 0
        self.insights = build_insights(self)

    # ---- access helpers --------------------------------------------------------------
    def has(self, field) -> bool:
        return field in self.raw.columns

    def series(self, field) -> pd.Series:
        """A sheet row on the main month index; all-NaN if the row doesn't exist, so callers never need a guard."""
        return self.raw[field].reindex(self.idx) if self.has(field) else pd.Series(np.nan, index=self.idx)

    # ---- data quality ----------------------------------------------------------------
    def _check_gaps(self):
        """Returns are computed row-to-row, so a missing month silently becomes a multi-month 'return'."""
        self.full_idx = pd.date_range(self.idx.min(), self.idx.max(), freq="MS")
        self.has_gaps = len(self.full_idx) != len(self.idx)
        if self.has_gaps:
            missing = [f"{t:%b-%y}" for t in self.full_idx.difference(self.idx)]
            print(f"WARNING: missing statement months ({', '.join(missing)}). "
                  "Returns across a gap span several months and annualised figures will be off.")

    # ---- income, fees, cash ----------------------------------------------------------
    def _build_income_costs_cash(self):
        idx, pf = self.idx, self.perf
        self.income_fields = [f for f in INCOME_LABELS if self.has(f)]
        # reindexed to every calendar month so period grouping (quarter/half-year) sees gaps as zero income
        self.df_income = self.raw[self.income_fields].reindex(self.full_idx).fillna(0)
        self.income_total = self.df_income.sum(axis=1).reindex(idx).fillna(0).astype(float)

        # fees now come from TWO rows; keep them apart for the cost chart and add them up for everything else
        self.fee_fields = [f for f in FEE_LABELS if self.has(f)]
        self.fees_by_type = self.raw[self.fee_fields].abs().reindex(idx).fillna(0)
        self.fees_abs = self.fees_by_type.sum(axis=1).astype(float)

        # income / fee totals exclude month 1 (no baseline), consistent with attribution and total profit
        self.income_sum = self.income_total.iloc[1:].sum()
        self.fees_sum = self.fees_abs.iloc[1:].sum()

        self.cash = self.raw["cash"].reindex(idx) if self.has("cash") else None
        self.cash_pct = (self.cash / pf.equity * 100) if self.cash is not None else None
        self.income_yield = self.income_total / pf.avg_eq * 100
        self.fee_drag = self.fees_abs / pf.avg_eq * 100
        self.income_yield_ann = (self.income_total.rolling(3).sum() / pf.avg_eq.rolling(3).mean()) * 4 * 100

    # ---- profit attribution ----------------------------------------------------------
    def _build_attribution(self, forced_balance):
        """Split equity-implied profit into realised / income / fees / remainder.
        Also decides whether the 'unrealised' row is a month-end BALANCE or a monthly FLOW by testing
        which reading reconciles better with the equity curve."""
        idx, pf = self.idx, self.perf
        self.real = self.series("realised").fillna(0)
        self.unreal_raw = self.series("unrealised") if self.has("unrealised") else None

        self.unreal_is_balance, self.recon_note = False, ""
        if self.has("realised") and self.has("unrealised"):
            u = self.unreal_raw.fillna(0)
            base = self.real + self.income_total - self.fees_abs
            gap_flow = (pf.monthly_pl - (base + u)).iloc[1:].abs().median()
            gap_bal = (pf.monthly_pl - (base + u.diff().fillna(0))).iloc[1:].abs().median()
            if forced_balance is None:
                self.unreal_is_balance = bool(np.isfinite(gap_bal) and (not np.isfinite(gap_flow) or gap_bal < gap_flow))
            else:
                self.unreal_is_balance = bool(forced_balance)
            self.recon_note = (
                f"Reconciliation: median monthly gap between equity-implied P&L and realised + income - fees "
                f"+ unrealised is {gap_bal:,.2f} € if unrealised is a balance (its monthly change is used) "
                f"vs {gap_flow:,.2f} € if it is a monthly flow. Using: "
                f"{'change in unrealised' if self.unreal_is_balance else 'unrealised as reported'}"
                f"{' (forced in CONFIG)' if forced_balance is not None else ' (auto-detected)'}.")
            print(self.recon_note)

        if self.unreal_raw is None:
            self.unreal_flow = None
        else:
            self.unreal_flow = self.unreal_raw.diff().fillna(0) if self.unreal_is_balance else self.unreal_raw
        parts = [s for s in (self.real if self.has("realised") else None, self.unreal_flow) if s is not None]
        self.total_pl = pd.concat(parts, axis=1).sum(axis=1, min_count=1) if parts else pd.Series(np.nan, index=idx)

        self.attr = pd.DataFrame({"Realised P&L": self.real, "Income": self.income_total, "Fees": -self.fees_abs})
        self.attr["Open positions / other"] = pf.monthly_pl - self.attr.sum(axis=1)   # remainder
        self.attr.iloc[0] = 0.0                                  # first month has no prior equity baseline

    # ---- NEW: net worth, cash flow, Polivios -----------------------------------------
    def _build_networth_and_cashflow(self):
        pf, s = self.perf, self.series

        # Net worth = Trading212 equity + Polivios + cash held by me + Revolut (deposits - withdrawals).
        # This is the formula of your own 'Account value total' row; blanks count as 0.
        parts = {"t212": pf.equity}
        if self.has("polivios_value"):
            parts["polivios"] = s("polivios_value")
        if self.has("wallet_cash"):
            parts["wallet"] = s("wallet_cash")
        if self.has("revolut_in") or self.has("revolut_out"):
            parts["revolut"] = s("revolut_in").sub(s("revolut_out"), fill_value=0)
        self.nw_parts = pd.DataFrame(parts).fillna(0.0)
        self.nw_total = self.nw_parts.sum(axis=1)
        self.has_networth = len(parts) > 1
        self.nw_reported = s("net_worth_reported") if self.has("net_worth_reported") else None
        self.nw_line = self.nw_reported if self.nw_reported is not None else self.nw_total
        self.nw_note = self._cross_check("Net worth", self.nw_total, self.nw_reported, "Account value total")

        # Cash flow = Trading212 deposits - spending + Polivios deposits + new wallet cash + Revolut in - Revolut out
        cf = {"t212_flow": pf.flows}
        for key, field, sign in [("spending", "spending", -1), ("polivios", "polivios_deposits", 1),
                                 ("wallet_new", "wallet_new", 1), ("revolut_in", "revolut_in", 1),
                                 ("revolut_out", "revolut_out", -1)]:
            if self.has(field):
                cf[key] = sign * s(field)
        self.cf_parts = pd.DataFrame(cf).fillna(0.0)
        self.cf_total = self.cf_parts.sum(axis=1)
        self.has_cashflow = len(cf) > 1
        self.cf_reported = s("cash_flow_reported") if self.has("cash_flow_reported") else None
        self.cf_line = self.cf_reported if self.cf_reported is not None else self.cf_total
        self.cf_note = self._cross_check("Cash flow", self.cf_total, self.cf_reported, "Cash Flow per month")

        # Polivios: shown as value vs cumulative deposits. No return is computed because deposits and value
        # do not move together (see the insight card), so a time-weighted return would be meaningless.
        self.has_polivios = self.has("polivios_value") and self.has("polivios_deposits")
        if self.has_polivios:
            self.poli_value = s("polivios_value")
            self.poli_dep = s("polivios_deposits").fillna(0)
            self.poli_cum_dep = self.poli_dep.cumsum()
            self.poli_gap = self.poli_value - self.poli_cum_dep
            self.poli_dvalue = self.poli_value.diff()

    @staticmethod
    def _cross_check(name, mine, reported, row_label):
        """Compare a total rebuilt from components with the total typed into the sheet."""
        if reported is None:
            return ""
        gap = (mine - reported).abs().max()
        if not np.isfinite(gap) or gap < 0.01:
            return f"{name}: the components reproduce your '{row_label}' row exactly."
        return (f"{name}: the components differ from your '{row_label}' row by up to {gap:,.2f} € - "
                "the sheet formula probably contains something extra. The line shows your sheet's own figure.")

    # ---- Data tab --------------------------------------------------------------------
    def monthly_table(self) -> pd.DataFrame:
        pf, nan = self.perf, pd.Series(np.nan, index=self.idx)
        d = pd.DataFrame({
            "month": self.idx.strftime("%Y-%m"),
            "equity": pf.equity.round(2).values,
            "deposits": pf.flows.round(2).values,
            "implied_pl": pf.monthly_pl.round(2).values,
            "ret_pct": (pf.ret * 100).round(2).values,
            "cum_ret_pct": (pf.cum_ret * 100).round(2).values,
            "dd_pct": (pf.perf_dd * 100).round(2).values,
            "realised": self.real.round(2).values,
            "unrealised": (self.unreal_raw if self.unreal_raw is not None else nan).round(2).values,
            "income": self.income_total.round(2).values,
            "fees": self.fees_abs.round(2).values,
            "cash_pct": (self.cash_pct if self.cash_pct is not None else nan).round(1).values,
            "net_worth": (self.nw_line if self.has_networth else nan).round(2).values,
            "cash_flow": (self.cf_line if self.has_cashflow else nan).round(2).values,
            "spending": self.series("spending").round(2).values,
            "polivios": self.series("polivios_value").round(2).values,
        })
        return d.astype(object).where(d.notna(), None)          # None -> empty cell in the DataTable


# =============================================================== INSIGHTS (plain-English cards)
def build_insights(p: Portfolio):
    """Return [(title, text, colour_key)]. colour_key is a key of the dashboard's palette ('green', 'red', ...)
    so this file stays free of any styling code."""
    pf, idx = p.perf, p.idx
    sign = lambda v: "green" if (v is not None and np.isfinite(v) and v >= 0) else "red"
    out = []

    # 1) headline + timing effect
    gap = (pf.mwr_ann - pf.cagr) * 100 if np.isfinite(pf.mwr_ann) and np.isfinite(pf.cagr) else np.nan
    timing = ""
    if np.isfinite(gap) and abs(gap) >= 1:
        timing = (f" Your deposit timing {'added' if gap > 0 else 'cost'} about {abs(gap):.1f} pts a year "
                  f"(money-weighted {fnum(pf.mwr_ann * 100, '.1f', '%')} vs time-weighted {fnum(pf.cagr * 100, '.1f', '%')}).")
    out.append(("Headline", f"{fnum(pf.total_profit, ',.0f', ' €', plus=True)} profit on {pf.capital_in.iloc[-1]:,.0f} € "
                            f"contributed (opening balance + deposits; "
                            f"{fnum(pf.twr_total * 100, '.1f', '%', plus=True)} excl. deposits over {pf.n_months} months).{timing}",
                sign(pf.total_profit)))

    # 2) where equity growth came from
    growth = pf.equity.iloc[-1] - pf.equity.iloc[0]
    dep_since = pf.net_dep.iloc[-1] - pf.flows.iloc[0]
    if growth > 0 and np.isfinite(dep_since):
        out.append(("Growth driver", f"{dep_since / growth * 100:.0f}% of your equity growth since the first month came from "
                                     f"deposits, {pf.total_profit / growth * 100:.0f}% from investment profit.", "orange"))

    # 3) consistency / payoff
    out.append(("Consistency", f"{fnum(pf.win_rate * 100, '.0f', '%')} of months were positive. Average win "
                               f"{fnum(pf.avg_win * 100, '.1f', '%', plus=True)} vs average loss {fnum(pf.avg_loss * 100, '.1f', '%', plus=True)} "
                               f"(payoff {fnum(pf.payoff, '.2f')}, profit factor {fnum(pf.profit_factor, '.2f')}).",
                "green" if np.isfinite(pf.profit_factor) and pf.profit_factor >= 1 else "red"))

    # 4) concentration: drop the SAME month that is named (the best euro month)
    if pf.n_months >= 3 and pf.total_profit > 0 and pf.r_pl.max() > 0:
        best_d = pf.r_pl.idxmax()
        share = pf.r_pl.max() / pf.total_profit * 100
        ex_best = ((1 + pf.r_m.drop(best_d)).prod() - 1) * 100
        out.append(("Profit concentration", f"{best_d:%b-%y} alone produced {share:.0f}% of total profit. Without that "
                                            f"month the return would be {fnum(ex_best, '.1f', '%', plus=True)} instead of "
                                            f"{fnum(pf.twr_total * 100, '.1f', '%', plus=True)}.", "gold"))

    # 5) risk
    worst = pf.dd_periods[0] if pf.dd_periods else None
    rec_txt = ""
    if worst:
        rec_txt = (f" It bottomed in {worst['trough']:%b-%y} and "
                   + (f"recovered by {worst['end']:%b-%y}." if worst["end"] is not None else "has not fully recovered yet."))
    out.append(("Risk", f"Max drawdown {fnum(pf.max_dd * 100, '.1f', '%')}, now {fnum(pf.cur_dd * 100, '.1f', '%')} from the high. "
                        f"Annualised volatility {fnum(pf.vol_ann * 100, '.1f', '%')}, Sharpe {fnum(pf.sharpe, '.2f')}.{rec_txt}",
                "red" if pf.cur_dd < -0.05 else "blue"))

    # 6) income vs fees
    inc, fee = p.income_sum, p.fees_sum
    if inc > 0 or fee > 0:
        cover = safe_div(inc, fee)
        yld = p.income_yield.iloc[1:].mean() * 12 if pf.n_months else np.nan
        # a "share of profit" only makes sense when there IS a profit; otherwise the % would be negative nonsense
        share_txt = (f"; income is {fnum(safe_div(inc, pf.total_profit) * 100, '.0f', '%')} of total profit"
                     if pf.total_profit > 0 else "")
        out.append(("Income vs fees", f"Income {inc:,.0f} € vs fees {fee:,.2f} € "
                                      f"({'covers fees ' + fnum(cover, '.1f', '×') if np.isfinite(cover) else 'no fees recorded'}). "
                                      f"Average income yield ≈ {fnum(yld, '.1f', '%')} per year on equity{share_txt}.", "gold"))

    # 7) cash drag = w * (R - rf) / (1 - w)
    if p.cash_pct is not None:
        w = p.cash_pct.mean() / 100
        drag = (max(0.0, w * (pf.cagr - p.risk_free_annual) / (1 - w)) * 100
                if np.isfinite(pf.cagr) and np.isfinite(w) and w < 1 else np.nan)
        out.append(("Cash drag", f"Cash averaged {p.cash_pct.mean():.0f}% of equity (now {p.cash_pct.iloc[-1]:.0f}%). "
                                 f"Treating cash as earning the risk-free rate and the invested part as earning whatever "
                                 f"lifts you to your CAGR, being fully invested would have added roughly "
                                 f"{fnum(drag, '.1f', ' pts/yr')} - a rough estimate, not a prediction.", "teal"))

    # 8) NEW: net worth across all accounts
    if p.has_networth:
        nw = p.nw_line
        t212_share = safe_div(pf.equity.iloc[-1], nw.iloc[-1]) * 100
        out.append(("Net worth", f"{nw.iloc[-1]:,.0f} € across all accounts "
                                 f"({fnum(nw.iloc[-1] - nw.iloc[0], ',.0f', ' €', plus=True)} since {idx[0]:%b-%y}). "
                                 f"Trading212 is {fnum(t212_share, '.0f', '%')} of it.", "blue"))

    # 9) NEW: Polivios account - value vs deposits
    if p.has_polivios:
        dep_total, gap_now = p.poli_cum_dep.iloc[-1], p.poli_gap.iloc[-1]
        hint = (" A gap this large usually means withdrawals or spending from this account are missing in the sheet - "
                "worth checking before reading it as a loss." if gap_now < -0.1 * max(dep_total, 1) else "")
        out.append(("Polivios account", f"Value {p.poli_value.iloc[-1]:,.0f} € vs {dep_total:,.0f} € deposited since "
                                        f"{idx[0]:%b-%y} ({fnum(gap_now, ',.0f', ' €', plus=True)}).{hint}", "purple"))

    # 10) streak
    cs = pf.current_streak
    now = f"on a {cs}-month winning run" if cs > 0 else f"on a {-cs}-month losing run" if cs < 0 else "flat"
    out.append(("Streaks", f"Longest winning run {pf.longest_win} month(s), longest losing run {pf.longest_loss}. Currently {now}.",
                "purple"))

    # 11) data-quality / caveats
    if p.has_gaps:
        out.append(("Data gap", "Some statement months are missing, so a few 'monthly' returns span several months; "
                                "annualised figures are approximate.", "orange"))
    if pf.n_months < 24:
        out.append(("Sample size", f"Only {pf.n_months} monthly observations: ratios such as Sharpe, Sortino, skew and the "
                                   "projection are noisy. Treat them as descriptive, not predictive.", "grey"))
    return out


# =============================================================== ENTRY POINT
def build_portfolio(path, start_date="2023-10-01", risk_free_annual=0.02, unrealised_is_balance=None) -> Portfolio:
    """Read the workbook and return a fully computed Portfolio.
    start_date            ignore statement months before this date
    risk_free_annual      cash / T-bill rate used for Sharpe, Sortino and the cash-drag estimate
    unrealised_is_balance True = month-end balance, False = monthly flow, None = detect automatically"""
    wide, mapping, unmatched = read_statements(path, start_date)
    print("Detected rows:")
    for field, label in mapping.items():
        print(f"  {field:<20} <- {label[:60]!r}")
    if unmatched:
        print("Ignored rows (no pattern in ROW_PATTERNS):", [u[:40] for u in unmatched])
    return Portfolio(wide, mapping, unmatched, risk_free_annual=risk_free_annual,
                     unrealised_is_balance=unrealised_is_balance)
