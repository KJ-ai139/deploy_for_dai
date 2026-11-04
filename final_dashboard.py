"""Startup funding dashboard - step 3 (problem statement + Q1 + Q2 + Q3).
Run:  streamlit run app_v3_q3.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ----------------------------- editable constants -----------------------------
COURSE = "Data Analysis & Interpretation"
TEAM = "TEAM-06"
DATA_DIR = "Original_dataset_cleaned"   # searched recursively for Apr_2015.csv style files
CONF = 0.95                             # confidence level (Q1 and Q2)
ALPHA = 1 - CONF                        # significance level for the tests
MAX_AMOUNT = 1e10                       # above this = impossible
MIN_AMOUNT = 1000                       # below this = impossible
WINDOW = 12                             # months before and after the event (event month left out)

DEMON_DATE = pd.Timestamp("2016-11-08")
COVID_DATE = pd.Timestamp("2020-03-25")
DEMON_SECTORS = {                       # normalized sector key -> name shown on the page
    "consumerinternet": "Consumer Internet",
    "technology": "Technology",
    "ecommerce": "eCommerce",
}
COVID_SECTORS = {
    "ecommerce": "eCommerce",
    "fintech": "FinTech",
    "edtech": "EdTech",
}
# colours that stay visible in both light and dark themes
COLORS = {"Consumer Internet": "#4C78A8", "Technology": "#F58518", "eCommerce": "#54A24B",
          "FinTech": "#B279A2", "EdTech": "#E45756"}
REF_COLOR = "#9E9E9E"                   # reference / event lines (grey works on light and dark)

st.set_page_config(page_title="Indian startup funding", layout="wide")


def show(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def usd(x):
    """Readable dollar amount. The dollar sign is escaped so Streamlit does not read it as maths."""
    if x >= 1e9:
        return f"\\${x / 1e9:.2f} billion"
    if x >= 1e6:
        return f"\\${x / 1e6:.2f} million"
    return f"\\${x:,.0f}"


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def norm_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def pooled_t(after, before):
    """Pooled two-sample t-test on log10 amounts (difference = after - before)."""
    na, nb = len(after), len(before)
    ma, mb = after.mean(), before.mean()
    va, vb = after.var(ddof=1), before.var(ddof=1)
    df = na + nb - 2
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / df)
    se = sp * np.sqrt(1 / na + 1 / nb)
    d = ma - mb
    t = d / se
    p = 2 * stats.t.sf(abs(t), df)
    tc = stats.t.ppf(1 - ALPHA / 2, df)
    return dict(na=na, nb=nb, ma=ma, mb=mb, sa=np.sqrt(va), sb=np.sqrt(vb),
                d=d, t=t, df=df, p=p, lo=d - tc * se, hi=d + tc * se)


# ------------------------------- load and clean -------------------------------
@st.cache_data(show_spinner="Reading and cleaning the data...")
def load_clean(folder):
    files = sorted(Path(folder).rglob("*.csv"))
    frames = []
    for f in files:
        try:
            month = pd.to_datetime(f.stem, format="%b_%Y")
        except ValueError:
            continue
        d = pd.read_csv(f)
        d = d[["startup_name", "industry_vertical", "amount"]].copy()
        d["month"] = month
        frames.append(d)
    if not frames:
        return None, None, None
    df = pd.concat(frames, ignore_index=True)
    log = {"files": len(frames), "rows_read": len(df)}

    # amounts as numbers (commas removed); rows with no amount cannot be used
    df["amount"] = pd.to_numeric(
        df["amount"].astype("string").str.replace(",", "", regex=False), errors="coerce"
    )
    log["missing_amount"] = int(df["amount"].isna().sum())
    df = df[df["amount"].notna()]

    # 1. missing / empty sector
    df["industry_vertical"] = df["industry_vertical"].astype("string").str.strip()
    bad = df["industry_vertical"].isna() | (df["industry_vertical"] == "")
    log["missing_sector"] = int(bad.sum())
    df = df[~bad].copy()

    # 2. merge sector spelling variants (key = lowercase letters/digits; label = most frequent spelling)
    df["sector_key"] = df["industry_vertical"].map(norm_key)
    label = df.groupby("sector_key")["industry_vertical"].agg(lambda s: s.value_counts().index[0])
    df["sector"] = df["sector_key"].map(label)

    # 3. duplicates (rows with a missing name are kept)
    df["name_key"] = df["startup_name"].astype("string").map(
        lambda s: norm_key(s) if pd.notna(s) else ""
    )
    has_name = df["name_key"] != ""
    dup = has_name & df.duplicated(["name_key", "sector_key", "month", "amount"], keep="first")
    log["duplicates"] = int(dup.sum())
    df = df[~dup]

    # 4. impossible amounts
    out = (df["amount"] > MAX_AMOUNT) | (df["amount"] < MIN_AMOUNT)
    dropped = df.loc[out, ["startup_name", "amount"]].sort_values("amount", ascending=False)
    log["impossible"] = int(out.sum())
    df = df[~out].copy()

    log["final"] = len(df)
    return df, log, dropped


# ---------------------------------- the page ----------------------------------
st.title("Effects of Demonetization and COVID on funding of Indian startups")
st.caption(f"{COURSE}  |  {TEAM}")

st.header("Problem statement")
st.markdown(
    """
**1. What we are studying.** We look at how much money Indian startups received in
funding deals between April 2015 and December 2021. Funding amounts vary hugely, from a few
thousand dollars to over a billion, so a handful of giant deals would dominate an ordinary average.

**2. Working with the log of the amount (base 10).** The log of an amount roughly counts its
digits (1,000 becomes 3, one million becomes 6). The histogram of the raw amounts piles up near zero
with a very long tail. The histogram of the log amounts is a symmetric bell, close to the red normal
curve, so the log of the amount is **roughly normal**. That lets us use the mean, confidence
intervals and the t-test.

**3. Questions.**
- **Q1.** How much does a typical deal receive, and how sure are we? (All sectors together.)
- **Q2.** Did typical funding per startup change after demonetization (8 Nov 2016) in
  Consumer Internet, Technology and eCommerce?
- **Q3.** Did it change after the covid lockdown (25 Mar 2020) in eCommerce, FinTech and EdTech?

**4. How Q2 and Q3 are answered.** We compare the average log amount before and after the event,
using equal 12-month windows with the event month left out. We test whether the difference is
bigger than chance, and report it as "about 2 times (or 0.5 times) the level before".

**5. What the answers can and cannot tell us.** The results show an association, not a cause. A result
that is not significant means we *could not detect* a change, not that nothing changed.
"""
)

if not Path(DATA_DIR).exists():
    st.error(f"Data folder '{DATA_DIR}' not found. Put it next to this file or edit DATA_DIR at the top.")
    st.stop()

df, log, dropped = load_clean(DATA_DIR)
if df is None:
    st.error(f"No files named like Apr_2015.csv were found in '{DATA_DIR}'.")
    st.stop()

# ------------------------------------- Q1 -------------------------------------
st.header("Q1. How much does a typical deal receive?")

amount = df["amount"].to_numpy(dtype=float)
x = np.log10(amount)
n = len(x)

# MLE for a normal sample, plus the t confidence interval for the mean
mu_hat = x.mean()
sigma2_hat = ((x - mu_hat) ** 2).sum() / n
sigma_hat = np.sqrt(sigma2_hat)
s = x.std(ddof=1)
tcrit = stats.t.ppf(1 - (1 - CONF) / 2, n - 1)
half = tcrit * s / np.sqrt(n)
lo, hi = mu_hat - half, mu_hat + half

gm, gm_lo, gm_hi = 10**mu_hat, 10**lo, 10**hi
median = float(np.median(amount))

st.markdown(
    f"We use all **{n:,}** cleaned deals from every sector, so this describes a typical deal "
    "over 2015-2021."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Amount")
    raw_m = amount / 1e6
    fig_raw = go.Figure(go.Histogram(x=raw_m, nbinsx=60, marker_color="#4C78A8", name="deals"))
    fig_raw.update_layout(
        xaxis_title="Funding amount (USD million)", yaxis_title="Number of deals",
        bargap=0.05, margin=dict(t=30, b=40), showlegend=False,
    )
    show(fig_raw)
    st.caption(
        f"Most deals sit near zero and a few giant ones stretch the tail "
        f"(skewness {stats.skew(amount):.1f}). This is not a bell."
    )

with col2:
    st.subheader("Log (base 10) of the amount")
    grid = np.linspace(x.min() - 0.3, x.max() + 0.3, 400)
    fig_log = go.Figure()
    fig_log.add_trace(go.Histogram(
        x=x, nbinsx=50, histnorm="probability density",
        marker_color="#4C78A8", name="deals",
    ))
    fig_log.add_trace(go.Scatter(
        x=grid, y=stats.norm.pdf(grid, mu_hat, sigma_hat), mode="lines",
        line=dict(color="red", width=3), name="normal curve (fitted by MLE)",
    ))
    fig_log.update_layout(
        xaxis_title="log10 of funding amount", yaxis_title="Density",
        bargap=0.05, margin=dict(t=30, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    show(fig_log)
    st.caption(
        f"The bars follow the red normal curve closely (skewness {stats.skew(x):.2f}), "
        "so the log of the amount is roughly normal."
    )

st.subheader("Formulas used")
st.markdown("For the log amounts $x_1,\\dots,x_n$, assumed normal, the maximum likelihood estimates are:")
st.latex(r"\hat\mu=\bar x=\frac1n\sum_{i=1}^n x_i,\qquad \hat\sigma^2=\frac1n\sum_{i=1}^n (x_i-\bar x)^2")
st.markdown("The confidence interval for the mean uses the t distribution with $n-1$ degrees of freedom:")
st.latex(r"\bar x \pm t_{n-1,\,0.975}\,\frac{s}{\sqrt n},\qquad s^2=\frac1{n-1}\sum_{i=1}^n (x_i-\bar x)^2")

st.subheader("Conclusion")
st.success(
    f"The MLE of the mean log10 amount is **{mu_hat:.3f}** and the MLE of the standard deviation "
    f"is **{sigma_hat:.3f}** (n = {n:,}). A {CONF:.0%} confidence interval for the mean log10 amount "
    f"is **({lo:.3f}, {hi:.3f})**.\n\n"
    f"Converted back to dollars, a typical deal receives about **{usd(gm)}** "
    f"(the geometric mean), and we are {CONF:.0%} confident that it lies between "
    f"**{usd(gm_lo)}** and **{usd(gm_hi)}**. This is close to the median deal "
    f"({usd(median)}) and much lower than the raw average, which a few giant deals pull up."
)

with st.expander("What was removed in cleaning"):
    st.write(
        f"{log['files']} files and {log['rows_read']:,} rows were read. Removed: "
        f"{log['missing_amount']} rows with no amount, {log['missing_sector']} with no sector, "
        f"{log['duplicates']} duplicate deals, and {log['impossible']} impossible amounts "
        f"(above \\$10 billion or below \\$1,000). **{log['final']:,} deals remain.**"
    )
    if len(dropped):
        d = dropped.copy()
        d["amount"] = d["amount"].map(lambda v: f"{v:,.1f}")
        st.dataframe(d.rename(columns={"startup_name": "Startup", "amount": "Amount (USD)"}),
                     hide_index=True)

# ------------------------- one event section (Q2 and Q3) -------------------------
def render_event(qid, heading, event_date, event_label, phrase, lead, sectors):
    """Hypotheses, formulas, p-value table, CI plot, monthly line plot and conclusion for one event."""
    st.header(heading)

    ev = event_date.replace(day=1)                      # month of the event (left out)
    b_start = ev - pd.DateOffset(months=WINDOW)
    a_end = ev + pd.DateOffset(months=WINDOW)
    before_txt = f"{b_start:%b %Y} to {(ev - pd.DateOffset(months=1)):%b %Y}"
    after_txt = f"{(ev + pd.DateOffset(months=1)):%b %Y} to {a_end:%b %Y}"

    st.markdown(
        f"{lead} {event_date:%d %b %Y}. For each of the three sectors we compare "
        f"the **{WINDOW} months before** ({before_txt}) with the **{WINDOW} months after** ({after_txt}). "
        f"{ev:%B %Y}, the month of the event, is left out."
    )

    st.subheader("Hypotheses")
    st.markdown(
        f"For each sector, let $\\mu_{{\\text{{before}}}}$ and $\\mu_{{\\text{{after}}}}$ be the mean log10 funding "
        f"of a deal before and after {phrase}."
    )
    st.latex(r"H_0:\ \mu_{\text{after}}=\mu_{\text{before}}\qquad H_1:\ \mu_{\text{after}}\neq\mu_{\text{before}}\qquad \alpha=0.05")
    st.markdown(f"In words, the null hypothesis says that typical funding per deal did not change after {phrase}.")

    st.subheader("Formulas used")
    st.markdown(
        "Let $n_a, n_b$ be the numbers of deals after and before, $\\bar x_a, \\bar x_b$ the means of the log "
        "amounts and $s_a^2, s_b^2$ their variances. We assume the two groups have equal variance."
    )
    st.latex(r"s_p^2=\frac{(n_a-1)s_a^2+(n_b-1)s_b^2}{n_a+n_b-2},\qquad SE=s_p\sqrt{\frac1{n_a}+\frac1{n_b}}")
    st.latex(r"t=\frac{\bar x_a-\bar x_b}{SE},\qquad df=n_a+n_b-2")
    st.latex(r"p\text{-value}=2\,P\left(T_{df}\ge |t|\right)")
    st.markdown(
        "We reject $H_0$ when the p-value is below 0.05. The confidence interval for the difference "
        "$d=\\bar x_a-\\bar x_b$ is $d \\pm t_{df,\\,0.975}\\,SE$, and $10^{d}$ is the factor by which typical funding "
        "changed."
    )

    # run the test for each sector
    rows, res = [], {}
    for key, name in sectors.items():
        sec = df[df["sector_key"] == key]
        xb = np.log10(sec.loc[(sec["month"] >= b_start) & (sec["month"] < ev), "amount"].to_numpy(float))
        xa = np.log10(sec.loc[(sec["month"] > ev) & (sec["month"] <= a_end), "amount"].to_numpy(float))
        if len(xa) < 2 or len(xb) < 2:
            st.warning(f"{name}: not enough deals in the windows (before {len(xb)}, after {len(xa)}).")
            continue
        r = pooled_t(xa, xb)
        r["factor"], r["f_lo"], r["f_hi"] = 10 ** r["d"], 10 ** r["lo"], 10 ** r["hi"]
        r["sig"] = r["p"] < ALPHA
        res[name] = r
        rows.append({
            "Sector": name,
            "Deals before": r["nb"],
            "Deals after": r["na"],
            "Difference d (log10)": round(r["d"], 3),
            "t": round(r["t"], 3),
            "df": r["df"],
            "p-value": fmt_p(r["p"]),
            "Decision at 5%": "Reject H0 (significant change)" if r["sig"] else "Do not reject H0 (no significant change)",
        })

    if not res:
        return

    st.subheader("p-values")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    ratio = max(max(r["sa"], r["sb"]) / min(r["sa"], r["sb"]) for r in res.values())
    if ratio < 2:
        st.caption(f"The before and after standard deviations differ by a factor of at most {ratio:.2f}, "
                   "so the equal-variance assumption looks reasonable.")
    else:
        st.caption(f"The before and after standard deviations differ by a factor of up to {ratio:.2f}, "
                   "so the equal-variance assumption should be read with care.")

    # confidence interval plot (one trace per sector so each bar gets its own visible colour)
    st.subheader("Confidence intervals for the change")
    names = list(res)
    pos = list(range(len(names)))
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(                         # reference line at factor 1 (no change)
        x=[1, 1], y=[-0.6, len(names) - 0.4], mode="lines",
        line=dict(color=REF_COLOR, dash="dash", width=2), name="no change (factor 1)",
        showlegend=False,
    ))
    for i, nm in enumerate(names):
        r = res[nm]
        fig_ci.add_trace(go.Scatter(
            x=[r["factor"]], y=[i], mode="markers+text", name=nm, showlegend=False,
            marker=dict(size=12, color=COLORS[nm]),
            error_x=dict(
                type="data", symmetric=False, thickness=3, width=8, color=COLORS[nm],
                array=[r["f_hi"] - r["factor"]], arrayminus=[r["factor"] - r["f_lo"]],
            ),
            text=[f"{r['factor']:.2f}x  ({r['f_lo']:.2f} to {r['f_hi']:.2f})"],
            textposition="top center",
        ))
    fig_ci.update_layout(
        xaxis=dict(type="log", title="Funding after / funding before (typical deal, log scale)"),
        yaxis=dict(tickvals=pos, ticktext=names, range=[-0.6, len(names) - 0.4]),
        margin=dict(t=30, b=40),
    )
    show(fig_ci, key=f"{qid}_ci")
    st.caption(
        f"Each dot is the estimated factor and the bar is its {CONF:.0%} confidence interval. "
        "A bar that does not cross the dashed line at 1 matches a significant change."
    )

    # monthly line plot (continuous lines)
    st.subheader("Monthly funding per deal")
    months = pd.date_range(b_start, a_end, freq="MS")
    fig_m = go.Figure()
    ymins, ymaxs = [], []
    for key, name in sectors.items():
        if name not in res:
            continue
        sec = df[df["sector_key"] == key]
        sec = sec[((sec["month"] >= b_start) & (sec["month"] < ev)) | ((sec["month"] > ev) & (sec["month"] <= a_end))]
        mm = (10 ** np.log10(sec["amount"]).groupby(sec["month"]).mean()).reindex(months)
        ymins.append(mm.min()); ymaxs.append(mm.max())
        fig_m.add_trace(go.Scatter(
            x=months, y=mm.values, mode="lines+markers", name=name,
            line=dict(color=COLORS[name], width=2), connectgaps=True,
        ))
    fig_m.add_trace(go.Scatter(                          # event line drawn as a trace
        x=[event_date, event_date], y=[np.nanmin(ymins) / 1.5, np.nanmax(ymaxs) * 1.5], mode="lines",
        line=dict(color=REF_COLOR, dash="dash", width=2), name=f"{event_label} ({event_date:%d %b %Y})",
    ))
    fig_m.update_layout(
        xaxis_title="Month", yaxis=dict(type="log", title="Funding per deal (USD, log scale)"),
        margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.15),
    )
    show(fig_m, key=f"{qid}_monthly")
    st.caption(
        "Each point is the average of the log amounts of that month's deals, converted back to dollars "
        "(the same quantity the test compares). Months with no deals in a sector are bridged by the line, "
        "and the month of the event is left out."
    )

    st.subheader("Conclusion")
    lines = []
    for nm, r in res.items():
        ci = f"{CONF:.0%} CI {r['f_lo']:.2f} to {r['f_hi']:.2f}, p = {fmt_p(r['p'])}"
        if r["sig"]:
            lines.append(f"- **{nm}:** typical funding per deal after {phrase} was about "
                         f"**{r['factor']:.2f} times** the level before ({ci}). This is a significant change.")
        else:
            lines.append(f"- **{nm}:** typical funding per deal after {phrase} was about "
                         f"{r['factor']:.2f} times the level before ({ci}). There is no significant change, so we "
                         "cannot rule out that funding stayed the same.")
    st.success(
        "\n".join(lines)
        + f"\n\nThese results show an association with the period after {phrase}, not proof that "
        f"{phrase} caused the change."
    )


render_event("q2", "Q2. Did funding change after demonetization?", DEMON_DATE,
             "Demonetization", "demonetization", "Demonetization was announced on", DEMON_SECTORS)
render_event("q3", "Q3. Did funding change after the covid lockdown?", COVID_DATE,
             "Covid lockdown", "the covid lockdown", "The covid lockdown began on", COVID_SECTORS)