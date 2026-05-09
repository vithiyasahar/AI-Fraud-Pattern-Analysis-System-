import time
import tracemalloc
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mlxtend.frequent_patterns import (
    apriori,
    fpgrowth,
    association_rules
)

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="AI Fraud Analytics System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<div style="
background: linear-gradient(90deg,#0f172a,#1e293b);
padding:20px;
border-radius:15px;
margin-bottom:20px;
">
<h1 style="color:white;text-align:center;">
🛡️Fraud Pattern Mining using Association Rule Mining
</h1>

<p style="color:#cbd5e1;text-align:center;font-size:18px;">
IEEE-CIS Fraud Detection using Apriori, FP-Growth & ECLAT
</p>
</div>
""", unsafe_allow_html=True)

# CUSTOM CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        180deg,
        #020617,
        #0f172a
    );
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.kpi-card {
    background: linear-gradient(145deg,#1e293b,#0f172a);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    transition: 0.3s;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

.kpi-card:hover{
    transform: translateY(-5px);
    border-color:#38bdf8;
}

.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}

.kpi-label {
    color: #94a3b8;
    margin-top: 10px;
}

.section-title {
    font-size: 1.3rem;
    font-weight: bold;
    color: #38bdf8;
    margin-top: 20px;
}

.rule-card {
    background: #111827;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 12px;
}

.rule-text {
    color: #facc15;
    font-weight: bold;
}

.rule-meta {
    color: #cbd5e1;
    margin-top: 8px;
}

.stButton > button {
    background: linear-gradient(90deg,#0ea5e9,#2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.7rem 2rem;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg,#2563eb,#0ea5e9);
}

</style>
""", unsafe_allow_html=True)



if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None

if "item_matrix" not in st.session_state:
    st.session_state["item_matrix"] = None

if "rules" not in st.session_state:
    st.session_state["rules"] = None


# ECLAT ALGORITHM


class ECLAT:

    def __init__(self, min_support=0.01, max_len=3):
        self.min_support = min_support
        self.max_len = max_len

    def fit(self, df):

        self._n = len(df)

        min_count = int(self.min_support * self._n)

        results = []

        vdb = {
            col: frozenset(df.index[df[col]])
            for col in df.columns
            if df[col].sum() >= min_count
        }

        for item, tids in vdb.items():

            results.append({
                "support": len(tids) / self._n,
                "itemsets": frozenset([item])
            })

        self._dfs(
            [],
            None,
            list(vdb.items()),
            results,
            min_count,
            1
        )

        self.freq_itemsets_ = pd.DataFrame(results)

        return self

    def _dfs(
        self,
        prefix,
        prefix_tids,
        items,
        results,
        min_count,
        depth
    ):

        if depth >= self.max_len:
            return

        for i, (item, tids) in enumerate(items):

            new_tids = (
                prefix_tids & tids
                if prefix_tids is not None
                else tids
            )

            if len(new_tids) < min_count:
                continue

            new_prefix = prefix + [item]

            results.append({
                "support": len(new_tids) / self._n,
                "itemsets": frozenset(new_prefix)
            })

            self._dfs(
                new_prefix,
                new_tids,
                items[i+1:],
                results,
                min_count,
                depth + 1
            )

    def get_rules(self, min_confidence=0.05):

        sup_map = {
            r["itemsets"]: r["support"]
            for _, r in self.freq_itemsets_.iterrows()
        }

        rows = []

        for _, row in self.freq_itemsets_.iterrows():

            iset = row["itemsets"]

            if len(iset) < 2:
                continue

            sup_iset = row["support"]

            for r in range(1, len(iset)):

                for ant in combinations(list(iset), r):

                    ant = frozenset(ant)

                    con = iset - ant

                    sup_ant = sup_map.get(ant)
                    sup_con = sup_map.get(con)

                    if sup_ant is None or sup_con is None:
                        continue

                    conf = sup_iset / sup_ant

                    if conf < min_confidence:
                        continue

                    lift = conf / sup_con if sup_con > 0 else 0

                    rows.append({
                        "antecedents": ant,
                        "consequents": con,
                        "support": sup_iset,
                        "confidence": conf,
                        "lift": lift
                    })

        return pd.DataFrame(rows)


# LOAD DATA


@st.cache_data
def load_real_data():

    trans = pd.read_csv(
        "train_transaction.csv"
    )

    identity = pd.read_csv(
        "train_identity.csv"
    )

    df = trans.merge(
        identity,
        on="TransactionID",
        how="left"
    )

    return df


# PREPROCESSING


@st.cache_data
def preprocess(df):

    keep = [
        "TransactionAmt",
        "card4",
        "card6",
        "ProductCD",
        "P_emaildomain",
        "DeviceType",
        "isFraud"
    ]

    keep = [c for c in keep if c in df.columns]

    df = df[keep].copy()

    # Fill missing values
    for col in df.columns:

        if col == "isFraud":
            continue

        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")

    # Amount binning
    df["AMT_BIN"] = pd.cut(
        df["TransactionAmt"],
        bins=[0,50,100,500,1000,5000,np.inf],
        labels=[
            "AMT_SMALL",
            "AMT_MEDIUM",
            "AMT_LARGE",
            "AMT_XLARGE",
            "AMT_XXLARGE",
            "AMT_EXTREME"
        ]
    )

    # SAFE COLUMN HANDLING
    cols = [
        "AMT_BIN",
        "card4",
        "card6",
        "ProductCD",
        "P_emaildomain",
        "DeviceType"
    ]

    cols = [c for c in cols if c in df.columns]

    dummies = pd.get_dummies(
        df[cols].astype(str)
    )

    dummies["FRAUD"] = df["isFraud"]

    return dummies.astype(bool)


# RUN ALGORITHM
def run_algorithm(
    df,
    algo,
    min_support,
    min_confidence
):

    tracemalloc.start()

    t0 = time.perf_counter()

    if algo == "Apriori":

        freq = apriori(
            df,
            min_support=min_support,
            use_colnames=True,
            max_len=3
        )

        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=min_confidence
        )

    elif algo == "FP-Growth":

        freq = fpgrowth(
            df,
            min_support=min_support,
            use_colnames=True,
            max_len=3
        )

        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=min_confidence
        )

    else:

        ec = ECLAT(
            min_support=min_support,
            max_len=3
        )

        ec.fit(df)

        freq = ec.freq_itemsets_

        rules = ec.get_rules(
            min_confidence=min_confidence
        )

    elapsed = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    return (
        freq,
        rules,
        elapsed,
        peak / 1e6
    )

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("""
# 🛡️ Fraud pattern Analytics

### Association Rule Mining
""")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Run Algorithms",
            "Rule Explorer"
        ]
    )

    st.markdown("---")

    algo = st.selectbox(
        "Algorithm",
        [
            "FP-Growth",
            "Apriori",
            "ECLAT"
        ]
    )

    min_support = st.select_slider(
    "Min Support",
    options=[
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05
    ],
    value=0.005
)

    min_confidence = st.slider(
        "Min Confidence",
        0.01,
        0.8,
        0.05,
        0.01
    )

# LOAD DATA

if st.session_state["raw_df"] is None:

    with st.spinner("Loading IEEE-CIS Dataset..."):

        st.session_state["raw_df"] = load_real_data()

        st.session_state["item_matrix"] = preprocess(
            st.session_state["raw_df"]
        )

        st.session_state["rules"] = None


# DASHBOARD
if page == "Dashboard":

    df = st.session_state["raw_df"]

    im = st.session_state["item_matrix"]

    fraud_rate = df["isFraud"].mean() * 100

    n_fraud = df["isFraud"].sum()

    n_legit = len(df) - n_fraud

    st.error(f"""
⚠️ Fraud Risk Alert

Detected Fraud Rate: {fraud_rate:.2f}%

High-risk transaction patterns identified.
""")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
<div class="kpi-card">
<div class="kpi-value">{len(df):,}</div>
<div class="kpi-label">Transactions</div>
</div>
""", unsafe_allow_html=True)

    c2.markdown(f"""
<div class="kpi-card">
<div class="kpi-value">{n_fraud:,}</div>
<div class="kpi-label">Fraud Cases</div>
</div>
""", unsafe_allow_html=True)

    c3.markdown(f"""
<div class="kpi-card">
<div class="kpi-value">{n_legit:,}</div>
<div class="kpi-label">Legitimate</div>
</div>
""", unsafe_allow_html=True)

    c4.markdown(f"""
<div class="kpi-card">
<div class="kpi-value">{fraud_rate:.2f}%</div>
<div class="kpi-label">Fraud Rate</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()

    legit = np.log1p(
        df[df["isFraud"] == 0]["TransactionAmt"]
    )

    fraud = np.log1p(
        df[df["isFraud"] == 1]["TransactionAmt"]
    )

    fig.add_trace(go.Histogram(
        x=legit,
        name="Legitimate",
        opacity=0.7
    ))

    fig.add_trace(go.Histogram(
        x=fraud,
        name="Fraud",
        opacity=0.7
    ))

    fig.update_layout(
        title="Fraud vs Legitimate Transaction Distribution",
        barmode="overlay",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# RUN ALGORITHMS
elif page == "Run Algorithms":

    st.subheader("⚙️ Run Association Rule Mining")

    im = st.session_state["item_matrix"]

    sample_size = st.slider(
        "Sample Size",
        5000,
        50000,
        10000,
        5000
    )

    if st.button(f"Run {algo}"):

        sample = im.sample(
            sample_size,
            random_state=42
        )

        with st.spinner(f"Running {algo}..."):

            freq, rules, elapsed, mem = run_algorithm(
                sample,
                algo,
                min_support,
                min_confidence
            )

            st.session_state["rules"] = rules

        st.success("Algorithm Completed Successfully")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Runtime",
            f"{elapsed:.2f}s"
        )

        c2.metric(
            "Memory",
            f"{mem:.1f} MB"
        )

        c3.metric(
            "Rules",
            len(rules)
        )

        fraud_rules = rules[
            rules["consequents"].apply(
                lambda x: "FRAUD" in x
            )
        ]

        st.markdown("## 🎯 Top Fraud Rules")

        top = fraud_rules.sort_values(
            "lift",
            ascending=False
        ).head(10)

        for _, row in top.iterrows():

            ant = " + ".join(
                sorted(row["antecedents"])
            )

            con = " + ".join(
                sorted(row["consequents"])
            )

            st.markdown(f"""
<div class="rule-card">

<div class="rule-text">
{ant} → {con}
</div>

<div class="rule-meta">
Support: {row['support']:.4f}
|
Confidence: {row['confidence']:.3f}
|
Lift: {row['lift']:.2f}
</div>

</div>
""", unsafe_allow_html=True)

# RULE EXPLORER

elif page == "Rule Explorer":

    st.subheader("📋 Rule Explorer")

    rules = st.session_state["rules"]

    if rules is None:

        st.warning(
            "Run algorithm first."
        )

    else:

        rules["rule"] = rules.apply(
            lambda r:
            " + ".join(sorted(r["antecedents"]))
            + " → " +
            " + ".join(sorted(r["consequents"])),
            axis=1
        )

        search = st.text_input(
            "Search Rule"
        )

        if search:

            rules = rules[
                rules["rule"].str.contains(
                    search,
                    case=False
                )
            ]

        st.dataframe(
            rules[[
                "rule",
                "support",
                "confidence",
                "lift"
            ]],
            use_container_width=True
        )

        csv = rules.to_csv(
            index=False
        ).encode()

        st.download_button(
            "Download Rules CSV",
            csv,
            "fraud_rules.csv",
            "text/csv"
        )
