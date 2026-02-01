

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from mrmr import mrmr_classif

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ThunderBio | Gene Feature Selection",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<style>
.block-container { padding-top: 2.5rem; }

.thunder-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 18px 26px;
    border-radius: 18px;
    background: linear-gradient(90deg, #5B7CFA, #2EC7F0);
    box-shadow: 0 10px 26px rgba(0,0,0,0.08);
    margin-bottom: 1.6rem;
}

.thunder-logo {
    width: 54px;
    height: 54px;
    border-radius: 14px;
    background: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}

.thunder-title {
    font-size: 24px;
    font-weight: 700;
    color: white;
}

.thunder-subtitle {
    font-size: 13px;
    color: #E0F2FE;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="thunder-header">
    <div class="thunder-logo">🧬</div>
    <div>
        <div class="thunder-title">Gene Feature Selection</div>
        <div class="thunder-subtitle">ThunderBio</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR – FILE UPLOAD
# =====================================================
st.sidebar.header(" Upload Input Files")

sig_file = st.sidebar.file_uploader(
    "Significant genes (ENSEMBL)", type=["csv", "tsv"]
)



sig_df = None
sig_col = None

if sig_file is not None:
    sig_df = pd.read_csv(sig_file)

    st.sidebar.markdown("### 🧬 Significant Gene Column")
    sig_col = st.sidebar.selectbox(
        "Select gene ID column",
        sig_df.columns,
        key="sig_gene_col"
    )




expr_file = st.sidebar.file_uploader(
    "Expression matrix", type=["csv", "tsv"]
)

meta_file = st.sidebar.file_uploader(
    "Metadata (Condition column required)", type=["csv", "tsv"]
)





# =====================================================
# FORM (THIS IS THE KEY FIX)
# =====================================================
with st.form("feature_selection_form"):

    st.markdown(
        "<div style='font-size:20px; font-weight:600;'>Select Feature Selection Methods</div>",
        unsafe_allow_html=True
    )

    methods = st.multiselect(
        "Choose methods",
        ["LASSO", "mRMR", "Random Forest"],
        default=["LASSO", "mRMR", "Random Forest"]
    )

    st.markdown(
        "<div style='font-size:20px; font-weight:600;'>Set Gene Selection Limits</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        lasso_top_n = st.slider("LASSO | Top N genes", 10, 500, 10)

    with col2:
        mrmr_top_n = st.slider("mRMR | Top N genes", 10, 500, 10)

    with col3:
        rf_top_n = st.slider("RF | Top N genes", 10, 500, 10)

    run_button = st.form_submit_button(" Run Feature Selection")

# =====================================================
# MAIN LOGIC (UNCHANGED)
# =====================================================
if run_button:

    if not (sig_file and expr_file and meta_file):
        st.error("❌ Please upload all required files")
        st.stop()

    with st.spinner("Running feature selection..."):

        # -----------------------------
        # LOAD DATA
        # -----------------------------
        # sig_genes = pd.read_csv(sig_file)["ENSEMBL"].astype(str)
        if run_button:

            if sig_df is None or sig_col is None:
                st.error("❌ Please upload significant genes file and select gene column")
                st.stop()

            sig_genes = sig_df[sig_col].astype(str)

        # sig_df = pd.read_csv(sig_file)

        # sig_col = st.selectbox(
        #     "Select gene ID column from significant genes file",
        #     sig_df.columns
        # )

        # sig_genes = sig_df[sig_col].astype(str)


        expr = pd.read_csv(expr_file, index_col=0)
        meta = pd.read_csv(meta_file, index_col=0)

        genes = list(set(sig_genes) & set(expr.index))
        st.success(f" {len(genes)} genes matched")

        X = expr.loc[genes].T
        y = meta.loc[X.index, "Condition"]

        X = X.loc[:, X.notna().all(axis=0)]

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )

        # =============================
        # LASSO
        # =============================
        if "LASSO" in methods:
            st.subheader("🔹 LASSO")

            lasso = LogisticRegressionCV(
                penalty="l1",
                solver="liblinear",
                cv=StratifiedKFold(5),
                max_iter=5000,
                n_jobs=-1
            )
            lasso.fit(X_scaled, y_enc)

            coef_df = pd.DataFrame({
                "Gene": X.columns,
                "Coefficient": lasso.coef_[0],
                "Abs_Coefficient": np.abs(lasso.coef_[0])
            }).sort_values("Abs_Coefficient", ascending=False)

            lasso_sel = coef_df.head(lasso_top_n)

            st.dataframe(lasso_sel)
            st.download_button(
                "Download LASSO CSV",
                lasso_sel.to_csv(index=False),
                "LASSO_Selected_Genes.csv"
            )

        # =============================
        # mRMR
        # =============================
        if "mRMR" in methods:
            st.subheader("🔹 mRMR")

            mrmr_genes = mrmr_classif(X=X, y=y, K=mrmr_top_n)

            mrmr_df = pd.DataFrame({
                "Gene": mrmr_genes,
                "Rank": range(1, len(mrmr_genes) + 1)
            })

            st.dataframe(mrmr_df)
            st.download_button(
                "Download mRMR CSV",
                mrmr_df.to_csv(index=False),
                "MRMR_Selected_Genes.csv"
            )

        # =============================
        # RANDOM FOREST
        # =============================


        # =============================
        # RANDOM FOREST (FAST & STABLE)
        # =============================
        if "Random Forest" in methods:
            st.subheader("🔹 Random Forest")

            rf = RandomForestClassifier(
                n_estimators=300,        # reduced
                random_state=42,
                n_jobs=-1
            )

            with st.spinner("Training Random Forest..."):
                rf.fit(X, y_enc)

            rf_df = pd.DataFrame({
                "Gene": X.columns,
                "Importance": rf.feature_importances_
            }).sort_values("Importance", ascending=False)

            rf_sel = rf_df.head(rf_top_n)

            st.dataframe(rf_sel)
            st.download_button(
                "Download RF CSV",
                rf_sel.to_csv(index=False),
                "RF_Selected_Genes.csv"
            )


        # if "Random Forest" in methods:
        #     st.subheader("🔹 Random Forest")

        #     rf = RandomForestClassifier(
        #         n_estimators=500,
        #         random_state=42,
        #         n_jobs=-1
        #     )
        #     rf.fit(X, y_enc)

        #     perm = permutation_importance(
        #         rf, X, y_enc,
        #         n_repeats=20,
        #         random_state=42,
        #         n_jobs=-1
        #     )

        #     rf_df = pd.DataFrame({
        #         "Gene": X.columns,
        #         "Importance": perm.importances_mean
        #     }).sort_values("Importance", ascending=False)

        #     rf_sel = rf_df.head(rf_top_n)

        #     st.dataframe(rf_sel)
        #     st.download_button(
        #         "Download RF CSV",
        #         rf_sel.to_csv(index=False),
        #         "RF_Selected_Genes.csv"
        #     )

else:
    st.info("⬅ Upload files → select options → click Run")


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f9fafb;
        color: #6b7280;
        text-align: center;
        padding: 8px 0;
        font-size: 12px;
        border-top: 1px solid #red;
        z-index: 100;
    }
    </style>

    <div class="footer">
        📧 Contact: <a href="mailto:premkumart.2402@gmail.com">premkumart.2402@gmail.com</a> |
        🧬 ThunderBio – Gene Feature Selection Platform
    </div>
    """,
    unsafe_allow_html=True
)