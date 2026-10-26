"""Streamlit web application for semantic schema matching.

Provides interactive evaluation, side-by-side feature inspection,
explainability breakdowns, and similarity heatmaps.
"""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so package imports work
# regardless of the directory Streamlit is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from service import match_tables, MatchResult  # 'service' lives in app/, which Streamlit puts on sys.path[0]
from person_a.name_embedder import get_name_embedder
from evaluation.metrics import compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SCHEMAS_DIR = REPO_ROOT / "data" / "test_schemas"
REPORTS_DIR = REPO_ROOT / "reports"

st.set_page_config(
    page_title="Semantic Schema Matching",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_cached_embedder():
    """Cache the embedding model instance across user sessions."""
    return get_name_embedder()


@st.cache_data
def run_matching(
    source_csv_bytes: bytes,
    target_csv_bytes: bytes,
    method: str,
    threshold: float,
    sample_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, MatchResult]:
    """Execute table matching with cached results for repeated parameter sets."""
    import io

    df_src = pd.read_csv(io.BytesIO(source_csv_bytes))
    df_tgt = pd.read_csv(io.BytesIO(target_csv_bytes))

    res = match_tables(
        source_df=df_src,
        target_df=df_tgt,
        method=method,
        threshold=threshold,
        sample_size=sample_size,
    )
    return df_src, df_tgt, res


def load_demo_pair(pair_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load bundled demonstration dataset from test_schemas."""
    pair_dir = TEST_SCHEMAS_DIR / pair_name
    src_path = pair_dir / "source.csv"
    tgt_path = pair_dir / "target.csv"
    gt_path = pair_dir / "ground_truth.csv"

    if not (src_path.exists() and tgt_path.exists()):
        raise FileNotFoundError(f"Demo files missing in {pair_dir}")

    df_src = pd.read_csv(src_path)
    df_tgt = pd.read_csv(tgt_path)
    df_gt = pd.read_csv(gt_path) if gt_path.exists() else None
    return df_src, df_tgt, df_gt


# Initialize model in background
embedder = get_cached_embedder()
backend_badge = embedder.backend_name

# Header
st.title("Semantic Schema Matching")
st.caption(
    "Automated column correspondence for federated databases via adaptive name-value signal fusion. "
    f"Active backend: `{backend_badge}`"
)

# Sidebar inputs
st.sidebar.header("Input Schemas")
input_mode = st.sidebar.radio("Data Source", ["Bundled Demo Pair", "Upload Custom CSVs"], index=0)

df_source: Optional[pd.DataFrame] = None
df_target: Optional[pd.DataFrame] = None
df_gt: Optional[pd.DataFrame] = None

if input_mode == "Bundled Demo Pair":
    available_pairs = [
        ("schema_pair_1", "Pair 1: Easy (Abbreviations)"),
        ("schema_pair_2", "Pair 2: Medium (Synonyms)"),
        ("schema_pair_3", "Pair 3: Hard (Generic Names)"),
        ("schema_pair_4", "Pair 4: Hard (Conflicting Dtypes)"),
        ("schema_pair_5", "Pair 5: Real-World (Asymmetric)"),
    ]
    selected_pair_id = st.sidebar.selectbox(
        "Choose Demo Pair",
        options=[p[0] for p in available_pairs],
        format_func=lambda x: dict(available_pairs).get(x, x),
        index=0,
    )
    try:
        df_source, df_target, df_gt = load_demo_pair(selected_pair_id)
        st.sidebar.success(f"Loaded {df_source.shape[1]} source cols, {df_target.shape[1]} target cols.")
    except Exception as exc:
        st.sidebar.error(f"Error loading demo: {exc}")
else:
    src_file = st.sidebar.file_uploader("Upload Source CSV", type=["csv"], key="src_upload")
    tgt_file = st.sidebar.file_uploader("Upload Target CSV", type=["csv"], key="tgt_upload")
    gt_file = st.sidebar.file_uploader("Optional Ground Truth CSV", type=["csv"], key="gt_upload")

    if src_file and tgt_file:
        try:
            df_source = pd.read_csv(src_file)
            df_target = pd.read_csv(tgt_file)
            if gt_file:
                df_gt = pd.read_csv(gt_file)
        except Exception as exc:
            st.sidebar.error(f"Error reading CSVs: {exc}")

st.sidebar.header("Matching Configuration")
method_options = {
    "Adaptive Fusion": "adaptive",
    "Name Only (alpha=1.0)": "name_only",
    "Value Only (alpha=0.0)": "value_only",
    "Fixed Fusion (alpha=0.5)": "fixed",
}
method_label = st.sidebar.selectbox("Matching Method", list(method_options.keys()), index=0)
selected_method = method_options[method_label]

threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help="Scores below this threshold are marked unaccepted/unmatched.",
)

sample_size = st.sidebar.slider(
    "Sample Size for Value Profiling",
    min_value=50,
    max_value=1000,
    value=500,
    step=50,
    help="Maximum non-null values sampled per column to construct statistical profiles.",
)

if df_source is None or df_target is None:
    st.info("Please upload Source and Target CSV files or select a bundled demo pair to begin.")
    st.stop()

# Execution with spinner
with st.spinner("Processing column profiles and computing match signals..."):
    try:
        src_bytes = df_source.to_csv(index=False).encode("utf-8")
        tgt_bytes = df_target.to_csv(index=False).encode("utf-8")
        _, _, match_result = run_matching(
            source_csv_bytes=src_bytes,
            target_csv_bytes=tgt_bytes,
            method=selected_method,
            threshold=threshold,
            sample_size=sample_size,
        )
    except Exception as exc:
        st.error(f"Failed to match schemas: {exc}")
        st.stop()

# Tabs
tab_matches, tab_explain, tab_matrix, tab_eval = st.tabs(
    ["Matches", "Why This Match", "Similarity Matrix", "Evaluation"]
)

# -----------------------------------------------------------------------------
# TAB 1: Matches
# -----------------------------------------------------------------------------
with tab_matches:
    accepted_matches = [m for m in match_result.matches if m.accepted]
    unaccepted_matches = [m for m in match_result.matches if not m.accepted]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Source Columns", len(match_result.source_columns))
    col2.metric("Target Columns", len(match_result.target_columns))
    col3.metric("Accepted Matches", len(accepted_matches))
    col4.metric("Unmatched Source", len(match_result.unmatched_source))

    st.subheader("Predicted Correspondences")
    if accepted_matches:
        match_table_data = []
        for m in accepted_matches:
            match_table_data.append({
                "Source Column": m.column_a,
                "Target Column": m.column_b,
                "Confidence": f"{m.score:.4f}",
                "Name Similarity": f"{m.name_similarity:.4f}",
                "Value Similarity": f"{m.value_similarity:.4f}",
                "Alpha": f"{m.alpha:.2f}",
                "Decision": "Accepted",
            })
        df_matches_ui = pd.DataFrame(match_table_data)
        st.dataframe(df_matches_ui, use_container_width=True, hide_index=True)

        csv_download = df_matches_ui.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Matches as CSV",
            data=csv_download,
            file_name="schema_matches.csv",
            mime="text/csv",
        )
    else:
        st.warning("No matches met the confidence threshold.")

    if unaccepted_matches:
        with st.expander("Low-Confidence Candidates Below Threshold"):
            low_conf_data = []
            for m in unaccepted_matches:
                low_conf_data.append({
                    "Source Column": m.column_a,
                    "Target Column": m.column_b,
                    "Confidence": f"{m.score:.4f}",
                    "Name Sim": f"{m.name_similarity:.4f}",
                    "Value Sim": f"{m.value_similarity:.4f}",
                    "Alpha": f"{m.alpha:.2f}",
                    "Decision": "Rejected",
                })
            st.dataframe(pd.DataFrame(low_conf_data), use_container_width=True, hide_index=True)

    if match_result.unmatched_source or match_result.unmatched_target:
        st.subheader("Unmatched Columns")
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            st.write("**Unmatched Source Columns:**")
            if match_result.unmatched_source:
                for c in match_result.unmatched_source:
                    st.markdown(f"- `{c}`")
            else:
                st.caption("All source columns matched.")
        with u_col2:
            st.write("**Unmatched Target Columns:**")
            if match_result.unmatched_target:
                for c in match_result.unmatched_target:
                    st.markdown(f"- `{c}`")
            else:
                st.caption("All target columns matched.")

# -----------------------------------------------------------------------------
# TAB 2: Why This Match (Explainability)
# -----------------------------------------------------------------------------
with tab_explain:
    st.subheader("Match Decision Breakdown")
    if not match_result.matches:
        st.info("No candidate matches available.")
    else:
        pair_labels = [f"{m.column_a}  ⟷  {m.column_b} (Score: {m.score:.3f})" for m in match_result.matches]
        selected_idx = st.selectbox("Select Column Pair to Explain", range(len(pair_labels)), format_func=lambda i: pair_labels[i])
        selected_match = match_result.matches[selected_idx]

        # Retrieve profiles
        src_prof = next((p for p in match_result.source_profiles if p["raw_name"] == selected_match.column_a), {})
        tgt_prof = next((p for p in match_result.target_profiles if p["raw_name"] == selected_match.column_b), {})

        expl = selected_match.explanation
        st.markdown(f"### `{selected_match.column_a}` ⟷ `{selected_match.column_b}`")

        # Explainability summary card
        status_color = "green" if selected_match.accepted else "orange"
        st.markdown(
            f"**Status:** :{status_color}[{'ACCEPTED' if selected_match.accepted else 'BELOW THRESHOLD'}] "
            f"| **Confidence Score:** `{selected_match.score:.4f}` (Threshold: `{threshold}`)"
        )

        st.info(
            f"**Fused Formula:** `Score = {selected_match.alpha:.2f} * {selected_match.name_similarity:.3f} (Name) + "
            f"{1.0 - selected_match.alpha:.2f} * {selected_match.value_similarity:.3f} (Value) = {selected_match.score:.3f}`"
        )

        if expl and expl.reasons:
            st.markdown("**Signal Reliability & Weighting Decisions:**")
            for r in expl.reasons:
                st.markdown(f"- {r}")

        st.divider()

        # Side by side profile details
        p_col1, p_col2 = st.columns(2)
        vp_src = src_prof.get("value_profile", {})
        vp_tgt = tgt_prof.get("value_profile", {})

        with p_col1:
            st.markdown(f"#### Source: `{selected_match.column_a}`")
            st.write(f"**Normalized Tokens:** `{src_prof.get('normalized_tokens', [])}`")
            st.write(f"**Inferred Type:** `{vp_src.get('dtype', 'unknown')}`")
            st.write(f"**Null Ratio:** `{vp_src.get('null_ratio', 0.0):.2%}`")
            st.write(f"**Cardinality Ratio:** `{vp_src.get('cardinality_ratio', 0.0):.4f}`")
            if vp_src.get("numeric_stats"):
                st.write("**Numeric Range:**", f"[{vp_src['numeric_stats']['min']}, {vp_src['numeric_stats']['max']}], Mean: {vp_src['numeric_stats']['mean']:.2f}")
            elif vp_src.get("length_stats"):
                st.write("**String Length:**", f"Min {vp_src['length_stats']['min']}, Max {vp_src['length_stats']['max']}, Mean {vp_src['length_stats']['mean']:.1f}")

            st.write("**Sample Values:**")
            st.code("\n".join(str(v) for v in src_prof.get("sample_values", [])[:8]))

        with p_col2:
            st.markdown(f"#### Target: `{selected_match.column_b}`")
            st.write(f"**Normalized Tokens:** `{tgt_prof.get('normalized_tokens', [])}`")
            st.write(f"**Inferred Type:** `{vp_tgt.get('dtype', 'unknown')}`")
            st.write(f"**Null Ratio:** `{vp_tgt.get('null_ratio', 0.0):.2%}`")
            st.write(f"**Cardinality Ratio:** `{vp_tgt.get('cardinality_ratio', 0.0):.4f}`")
            if vp_tgt.get("numeric_stats"):
                st.write("**Numeric Range:**", f"[{vp_tgt['numeric_stats']['min']}, {vp_tgt['numeric_stats']['max']}], Mean: {vp_tgt['numeric_stats']['mean']:.2f}")
            elif vp_tgt.get("length_stats"):
                st.write("**String Length:**", f"Min {vp_tgt['length_stats']['min']}, Max {vp_tgt['length_stats']['max']}, Mean {vp_tgt['length_stats']['mean']:.1f}")

            st.write("**Sample Values:**")
            st.code("\n".join(str(v) for v in tgt_prof.get("sample_values", [])[:8]))

# -----------------------------------------------------------------------------
# TAB 3: Similarity Matrix Heatmap
# -----------------------------------------------------------------------------
with tab_matrix:
    st.subheader("Interactive Similarity Heatmap")
    matrix_mode = st.radio("Signal View", ["Fused Score Matrix", "Semantic Name Matrix", "Value Distribution Matrix"], horizontal=True)

    if matrix_mode == "Fused Score Matrix":
        display_mat = match_result.similarity_matrix
        title_text = "Fused Similarity Matrix (Source x Target)"
    elif matrix_mode == "Semantic Name Matrix":
        display_mat = match_result.name_matrix
        title_text = "Cosine Semantic Name Matrix"
    else:
        display_mat = match_result.value_matrix
        title_text = "Statistical Value Profile Matrix"

    # Matched pairs set
    matched_coords = set()
    for m in match_result.matches:
        if m.accepted and m.column_a in match_result.source_columns and m.column_b in match_result.target_columns:
            r = match_result.source_columns.index(m.column_a)
            c = match_result.target_columns.index(m.column_b)
            matched_coords.add((r, c))

    fig = px.imshow(
        display_mat,
        labels=dict(x="Target Columns", y="Source Columns", color="Score"),
        x=match_result.target_columns,
        y=match_result.source_columns,
        color_continuous_scale="Blues",
        zmin=0.0,
        zmax=1.0,
        text_auto=".2f",
    )
    fig.update_layout(
        title=title_text,
        xaxis_tickangle=-45,
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    # Add marker symbols on matched cells
    for r, c in matched_coords:
        fig.add_annotation(
            x=match_result.target_columns[c],
            y=match_result.source_columns[r],
            text="*",
            showarrow=False,
            font=dict(color="red", size=22),
            align="center",
        )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red asterisk (*) indicates globally optimal 1-to-1 correspondence assigned by Hungarian algorithm.")

# -----------------------------------------------------------------------------
# TAB 4: Evaluation
# -----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Performance Evaluation")

    if df_gt is not None:
        gt_tuples = list(zip(df_gt["source_column"].astype(str), df_gt["target_column"].astype(str)))
        pred_tuples = [m.to_dict() for m in match_result.matches]
        curr_metrics = compute_metrics(pred_tuples, gt_tuples)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{curr_metrics['precision']:.2%}")
        m2.metric("Recall", f"{curr_metrics['recall']:.2%}")
        m3.metric("F1 Score", f"{curr_metrics['f1']:.4f}")
        m4.metric("True Positives", f"{curr_metrics['tp']} / {len(gt_tuples)}")

        # Method comparative chart on this pair
        st.markdown("#### Comparative Baseline Performance on this Pair")
        baseline_methods = [
            ("Name Only", "name_only"),
            ("Value Only", "value_only"),
            ("Fixed Fusion (0.5)", "fixed"),
            ("Adaptive Fusion", "adaptive"),
        ]
        comp_records = []
        for bl_label, bl_key in baseline_methods:
            bl_res = match_tables(
                source_df=df_source,
                target_df=df_target,
                method=bl_key,
                threshold=threshold,
                sample_size=sample_size,
            )
            bl_metrics = compute_metrics([m.to_dict() for m in bl_res.matches], gt_tuples)
            comp_records.append({
                "Method": bl_label,
                "Precision": bl_metrics["precision"],
                "Recall": bl_metrics["recall"],
                "F1 Score": bl_metrics["f1"],
            })

        df_bar = pd.DataFrame(comp_records)
        fig_bar = px.bar(
            df_bar,
            x="Method",
            y=["Precision", "Recall", "F1 Score"],
            barmode="group",
            title="Comparison Across Matcher Architectures",
            color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6"],
        )
        fig_bar.update_layout(yaxis_range=[0.0, 1.05], height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Match categorization
        gt_set = set(gt_tuples)
        pred_set = {(m.column_a, m.column_b) for m in match_result.matches if m.accepted}

        correct_pairs = pred_set & gt_set
        missed_pairs = gt_set - pred_set
        false_pairs = pred_set - gt_set

        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            st.markdown(f"**Correct Matches ({len(correct_pairs)}):**")
            for sc, tc in sorted(correct_pairs):
                st.write(f"- `{sc}` ⟷ `{tc}`")
        with e_col2:
            st.markdown(f"**Missed Matches ({len(missed_pairs)}):**")
            for sc, tc in sorted(missed_pairs):
                st.write(f"- `{sc}` ⟷ `{tc}`")
        with e_col3:
            st.markdown(f"**False Matches ({len(false_pairs)}):**")
            for sc, tc in sorted(false_pairs):
                st.write(f"- `{sc}` ⟷ `{tc}`")
    else:
        st.info("No ground-truth mapping provided for this table pair. Upload ground_truth.csv to inspect precision and recall.")

    st.divider()
    st.subheader("Global Benchmark Comparison Across All Schema Pairs")
    comp_file = REPORTS_DIR / "comparison_table.csv"
    if comp_file.exists():
        df_all_comp = pd.read_csv(comp_file)
        st.dataframe(df_all_comp, use_container_width=True, hide_index=True)
    else:
        st.caption("Run `python -m evaluation.run_evaluation` to generate global benchmark tables.")
