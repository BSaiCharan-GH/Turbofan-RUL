"""
AeroPulse: NASA C-MAPSS multi-subset RUL dashboard.

Expected subset-specific artifacts in models/:
  FD001_data_scaler.pkl
  FD001_lstm_weights.pth
  FD001_transformer_weights.pth
  ...repeat for FD002, FD003, FD004.

For backward compatibility, FD001 may use the original names:
  data_scaler.pkl, lstm_weights.pth, transformer_weights.pth

Each model must have been trained for that subset with the same 24 features,
30-cycle window, MinMaxScaler preprocessing, and RUL target scaling (/125).
This app intentionally does not reuse FD001 artifacts for other subsets.
"""

from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from src.model_lstm import RUL_LSTM
from src.model_transformer import RUL_Transformer


# ----------------------------- Configuration -----------------------------

st.set_page_config(
    page_title="AeroPulse | Turbofan RUL",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
WINDOW_SIZE = 30
RUL_CAP = 125.0

FEATURES = ["setting_1", "setting_2", "setting_3"] + [
    f"s_{i}" for i in range(1, 22)
]
COLUMNS = ["unit_number", "time_cycles"] + FEATURES

DATASETS = {
    "FD001": {
        "description": "1 operating condition · 1 fault mode",
        "train": "train_FD001.txt",
        "test": "test_FD001.txt",
        "rul": "RUL_FD001.txt",
    },
    "FD002": {
        "description": "6 operating conditions · 1 fault mode",
        "train": "train_FD002.txt",
        "test": "test_FD002.txt",
        "rul": "RUL_FD002.txt",
    },
    "FD003": {
        "description": "1 operating condition · 2 fault modes",
        "train": "train_FD003.txt",
        "test": "test_FD003.txt",
        "rul": "RUL_FD003.txt",
    },
    "FD004": {
        "description": "6 operating conditions · 2 fault modes",
        "train": "train_FD004.txt",
        "test": "test_FD004.txt",
        "rul": "RUL_FD004.txt",
    },
}


# -------------------------------- Styling ---------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp {
        background: radial-gradient(ellipse at 80% 0%, rgba(40,91,146,.22), transparent 38%), #08111f;
        color: #eaf1ff;
    }
    [data-testid="stSidebar"] { background: #0c1728; border-right: 1px solid rgba(160,183,220,.16); }
    h1,h2,h3 { font-family: 'Space Grotesk',sans-serif !important; letter-spacing: -.03em; color: #f4f7ff; }
    p,label { color: #c6d2e8; }
    .hero { padding: 1.8rem; border: 1px solid rgba(104,224,207,.25); border-radius: 24px; background: linear-gradient(120deg,#142b45,#0d192b); margin-bottom: 1.2rem; }
    .eyebrow { color: #68e0cf; font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .7rem; }
    .hero h1 { font-size: clamp(2.2rem,4vw,3.4rem); line-height: 1.03; margin: 0 0 .8rem; }
    .hero p,.subtle { color: #b8c7df; line-height: 1.7; }
    .pill { display: inline-block; border: 1px solid rgba(104,224,207,.35); color: #8af0df; background: rgba(104,224,207,.08); border-radius: 99px; padding: .3rem .7rem; font-size: .76rem; margin: .25rem .25rem 0 0; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg,#132137,#0d1829); border: 1px solid rgba(160,183,220,.16); padding: 1rem; border-radius: 16px; }
    div[data-testid="stMetricValue"] { color: #f4f7ff; font-family: 'Space Grotesk',sans-serif; }
    .stButton > button { background: #68e0cf; color: #071320; border: 0; border-radius: 12px; min-height: 3rem; font-weight: 700; width: 100%; }
    .stButton > button:hover { background: #9af4e6; color: #071320; }
    [data-testid="stFileUploader"] { background: rgba(17,28,49,.7); border: 1px dashed rgba(104,224,207,.38); border-radius: 16px; }
    .stTabs [aria-selected="true"] { color: #68e0cf !important; }
    hr { border-color: rgba(160,183,220,.16); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- Data I/O ---------------------------------

def parse_cmapss_test(file_bytes: bytes) -> pd.DataFrame:
    """Read an original NASA C-MAPSS test file (whitespace-separated, 26 cols)."""
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=r"\s+",
            header=None,
            engine="python",
        ).dropna(axis=1, how="all")
    except Exception as exc:
        raise ValueError(f"Could not parse the uploaded text file: {exc}") from exc

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns in the raw NASA test file, "
            f"but found {df.shape[1]}. Upload the original test_FD00X.txt file."
        )
    df.columns = COLUMNS
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.isna().any().any():
        raise ValueError("The dataset contains missing or non-numeric values.")
    if df.empty:
        raise ValueError("The uploaded file contains no rows.")
    if (df["unit_number"] <= 0).any() or (df["time_cycles"] <= 0).any():
        raise ValueError("Engine IDs and cycle numbers must be positive.")
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    return df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)


def parse_rul_labels(file_bytes: bytes) -> np.ndarray:
    """Read the NASA RUL_FD00X.txt file: one final RUL label per test engine."""
    try:
        labels = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=r"\s+",
            header=None,
            engine="python",
        ).to_numpy(dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Could not parse the RUL labels file: {exc}") from exc
    if labels.size == 0 or not np.isfinite(labels).all():
        raise ValueError("RUL labels file is empty or contains invalid values.")
    if (labels < 0).any():
        raise ValueError("RUL labels cannot be negative.")
    return labels


# -------------------------- Model/artifact loading -------------------------

def artifact_paths(subset: str) -> dict:
    """Return exact subset-specific artifact paths; FD001 supports legacy names."""
    paths = {
        "scaler": MODEL_DIR / f"{subset}_data_scaler.pkl",
        "lstm": MODEL_DIR / f"{subset}_lstm_weights.pth",
        "transformer": MODEL_DIR / f"{subset}_transformer_weights.pth",
    }
    if subset == "FD001":
        legacy = {
            "scaler": MODEL_DIR / "data_scaler.pkl",
            "lstm": MODEL_DIR / "lstm_weights.pth",
            "transformer": MODEL_DIR / "transformer_weights.pth",
        }
        for key, path in paths.items():
            if not path.exists() and legacy[key].exists():
                paths[key] = legacy[key]
    return paths


def missing_artifacts(subset: str) -> list[str]:
    return [str(p.relative_to(ROOT)) for p in artifact_paths(subset).values() if not p.exists()]


@st.cache_resource(show_spinner=False)
def load_models(subset: str):
    """Load the matching scaler and models for the selected subset."""
    paths = artifact_paths(subset)
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"No complete trained artifact set for {subset}. Missing: "
            + ", ".join(missing)
        )

    scaler = joblib.load(paths["scaler"])
    lstm = RUL_LSTM(input_size=len(FEATURES))
    transformer = RUL_Transformer(input_size=len(FEATURES))

    try:
        lstm_state = torch.load(paths["lstm"], map_location="cpu", weights_only=True)
        transformer_state = torch.load(
            paths["transformer"], map_location="cpu", weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        lstm_state = torch.load(paths["lstm"], map_location="cpu")
        transformer_state = torch.load(paths["transformer"], map_location="cpu")

    lstm.load_state_dict(lstm_state)
    transformer.load_state_dict(transformer_state)
    lstm.eval()
    transformer.eval()
    return scaler, lstm, transformer


# -------------------------------- Inference --------------------------------

def make_engine_window(engine_df: pd.DataFrame, scaler) -> torch.Tensor:
    engine_df = engine_df.sort_values("time_cycles")
    if len(engine_df) < WINDOW_SIZE:
        raise ValueError(
            f"Engine has {len(engine_df)} cycles; at least {WINDOW_SIZE} are required."
        )
    raw = engine_df[FEATURES].tail(WINDOW_SIZE).to_numpy(dtype=np.float64)
    scaled = scaler.transform(raw)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaling produced invalid values. Check the selected scaler.")
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)


def predict_engine(engine_df: pd.DataFrame, scaler, lstm, transformer) -> tuple[float, float]:
    """Return LSTM and Transformer estimates in cycles."""
    x = make_engine_window(engine_df, scaler)
    with torch.inference_mode():
        pred_lstm = float(lstm(x).reshape(-1)[0].item()) * RUL_CAP
        pred_transformer = float(transformer(x).reshape(-1)[0].item()) * RUL_CAP
    # RUL cannot be negative. This is a display constraint, not model calibration.
    return max(0.0, pred_lstm), max(0.0, pred_transformer)


def evaluate_all_engines(
    data: pd.DataFrame,
    rul_labels: np.ndarray,
    scaler,
    lstm,
    transformer,
) -> pd.DataFrame:
    """Evaluate final-cycle predictions against NASA's per-engine test RUL labels."""
    engines = sorted(data["unit_number"].unique().tolist())
    if len(engines) != len(rul_labels):
        raise ValueError(
            f"The uploaded RUL file has {len(rul_labels)} labels, but the test file "
            f"contains {len(engines)} engines. Use the matching RUL_FD00X.txt file."
        )

    rows = []
    for engine_id, true_rul in zip(engines, rul_labels):
        engine_df = data[data["unit_number"] == engine_id]
        pred_lstm, pred_transformer = predict_engine(
            engine_df, scaler, lstm, transformer
        )
        rows.append({
            "Engine": int(engine_id),
            "Actual RUL": float(true_rul),
            "LSTM prediction": pred_lstm,
            "Transformer prediction": pred_transformer,
            "LSTM absolute error": abs(pred_lstm - float(true_rul)),
            "Transformer absolute error": abs(pred_transformer - float(true_rul)),
        })
    return pd.DataFrame(rows)


# --------------------------------- Sidebar ---------------------------------

with st.sidebar:
    st.markdown("## ✈️ AeroPulse")
    st.caption("ENGINE HEALTH INTELLIGENCE")
    st.divider()

    selected_subset = st.selectbox(
        "NASA C-MAPSS subset",
        options=list(DATASETS.keys()),
        format_func=lambda key: f"{key} — {DATASETS[key]['description']}",
    )
    st.caption(
        "Choose the same subset for test data, RUL labels, scaler, and model weights."
    )
    st.divider()

    st.markdown("### Upload test data")
    test_file = st.file_uploader(
        f"Upload {DATASETS[selected_subset]['test']}",
        type=["txt"],
        key=f"test_{selected_subset}",
        help="Original NASA C-MAPSS whitespace-separated test file.",
    )
    rul_file = st.file_uploader(
        f"Optional: upload {DATASETS[selected_subset]['rul']}",
        type=["txt"],
        key=f"rul_{selected_subset}",
        help="Upload the matching NASA RUL file to calculate RMSE.",
    )

    st.divider()
    st.markdown("**Models**")
    st.markdown('<span class="pill">LSTM</span><span class="pill">Transformer</span>',
                unsafe_allow_html=True)
    st.caption("Input window: latest 30 cycles")
    st.caption("Inference: CPU")


# ----------------------------------- Hero ----------------------------------

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Predictive maintenance · Research demo</div>
      <h1>See engine health<br>before failure.</h1>
      <p>Explore sensor behaviour, estimate remaining useful life, and compare
      LSTM and Transformer models across NASA C-MAPSS benchmark subsets.</p>
      <span class="pill">01 / Select dataset</span>
      <span class="pill">02 / Analyze engine</span>
      <span class="pill">03 / Evaluate predictions</span>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected subset", selected_subset)
m2.metric("Operating conditions", "6" if selected_subset in ("FD002", "FD004") else "1")
m3.metric("Fault modes", "2" if selected_subset in ("FD003", "FD004") else "1")
m4.metric("Input features", "24")


# ----------------------------------- Tabs ----------------------------------

analysis_tab, evaluation_tab, methodology_tab = st.tabs(
    ["⚡ Engine analysis", "📈 Evaluation", "ⓘ Methodology"]
)

with analysis_tab:
    st.subheader(f"{selected_subset} · Engine analysis")
    st.markdown(
        f'<p class="subtle">{DATASETS[selected_subset]["description"]}. '
        "Upload the matching test file to inspect engines and sensor trends.</p>",
        unsafe_allow_html=True,
    )

    missing = missing_artifacts(selected_subset)
    if missing:
        st.warning(
            f"No complete model artifact set is available for {selected_subset}. "
            "The app will not substitute FD001 models for this subset."
        )
        st.code("\n".join(missing), language="text")
        st.info(
            "After training this subset, save the scaler and weights using the "
            f"filenames shown above. FD001 can use the legacy filenames if needed."
        )

    if test_file is None:
        st.info(f"Upload the original {DATASETS[selected_subset]['test']} file from the sidebar.")
    else:
        try:
            data = parse_cmapss_test(test_file.getvalue())
            engine_ids = sorted(data["unit_number"].unique().tolist())
            left, right = st.columns([1, 2])
            with left:
                selected_engine = st.selectbox(
                    "Engine ID", engine_ids, key=f"engine_{selected_subset}"
                )
            engine = data[data["unit_number"] == selected_engine].sort_values("time_cycles")
            with right:
                st.metric(
                    "Selected engine",
                    f"#{int(selected_engine):02d}",
                    f"{len(engine)} observed cycles · {len(engine_ids)} engines in file",
                )

            st.markdown("#### Sensor trends")
            sensor_choices = [f"s_{i}" for i in range(1, 22)]
            selected_sensors = st.multiselect(
                "Sensor channels",
                sensor_choices,
                default=["s_2", "s_3", "s_4"],
                key=f"sensors_{selected_subset}",
            )
            if selected_sensors:
                plot_df = engine[["time_cycles"] + selected_sensors].melt(
                    id_vars="time_cycles", var_name="Sensor", value_name="Reading"
                )
                fig = px.line(
                    plot_df,
                    x="time_cycles",
                    y="Reading",
                    color="Sensor",
                    template="plotly_dark",
                    labels={"time_cycles": "Operating cycle", "Reading": "Raw sensor reading"},
                )
                fig.update_layout(
                    height=390,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,20,35,.6)",
                    margin=dict(l=8, r=8, t=20, b=8),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one sensor to display a trend.")

            with st.expander("View latest sensor records"):
                st.dataframe(
                    engine[["time_cycles"] + selected_sensors].tail(10),
                    use_container_width=True,
                    hide_index=True,
                )

            if len(engine) < WINDOW_SIZE:
                st.warning(f"At least {WINDOW_SIZE} cycles are required for inference.")
            elif not missing:
                if st.button("Run LSTM + Transformer", type="primary", key=f"predict_{selected_subset}"):
                    try:
                        with st.spinner(f"Loading {selected_subset} artifacts and predicting..."):
                            scaler, lstm, transformer = load_models(selected_subset)
                            pred_lstm, pred_transformer = predict_engine(
                                engine, scaler, lstm, transformer
                            )
                        st.session_state["prediction"] = {
                            "subset": selected_subset,
                            "engine": int(selected_engine),
                            "lstm": pred_lstm,
                            "transformer": pred_transformer,
                        }
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

            result = st.session_state.get("prediction")
            if result and result["subset"] == selected_subset and result["engine"] == int(selected_engine):
                st.markdown("#### Prediction results")
                c1, c2, c3 = st.columns(3)
                c1.metric("LSTM estimate", f'{result["lstm"]:.1f} cycles')
                c2.metric("Transformer estimate", f'{result["transformer"]:.1f} cycles')
                c3.metric("Absolute difference", f'{abs(result["transformer"] - result["lstm"]):.1f} cycles')

                pred_df = pd.DataFrame({
                    "Model": ["LSTM", "Transformer"],
                    "Predicted RUL (cycles)": [result["lstm"], result["transformer"]],
                })
                pred_fig = px.bar(
                    pred_df, x="Model", y="Predicted RUL (cycles)",
                    color="Model", text="Predicted RUL (cycles)",
                    color_discrete_map={"LSTM": "#6f8dbb", "Transformer": "#68e0cf"},
                    template="plotly_dark",
                )
                pred_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                pred_fig.update_layout(
                    height=320, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,20,35,.6)", margin=dict(l=8, r=8, t=24, b=8),
                )
                st.plotly_chart(pred_fig, use_container_width=True)
                st.caption("Estimates are for research demonstration only, not maintenance or flight-safety decisions.")

        except Exception as exc:
            st.error(f"Could not read this dataset: {exc}")
            st.caption("Use the original NASA whitespace-separated test_FD00X.txt file, not a spreadsheet export.")


with evaluation_tab:
    st.subheader(f"{selected_subset} · Held-out evaluation")
    st.markdown(
        "Upload the matching `RUL_FD00X.txt` file to compare final-cycle predictions "
        "against NASA's reference RUL labels and calculate RMSE."
    )

    if test_file is None:
        st.info("Upload the matching test file in the sidebar first.")
    elif rul_file is None:
        st.info(f"Upload {DATASETS[selected_subset]['rul']} in the sidebar to calculate RMSE.")
    elif missing:
        st.warning("Evaluation is unavailable until matching models and scaler are trained for this subset.")
    else:
        if st.button("Evaluate all test engines", type="primary", key=f"evaluate_{selected_subset}"):
            try:
                with st.spinner(f"Evaluating all {selected_subset} test engines..."):
                    data = parse_cmapss_test(test_file.getvalue())
                    labels = parse_rul_labels(rul_file.getvalue())
                    scaler, lstm, transformer = load_models(selected_subset)
                    results = evaluate_all_engines(data, labels, scaler, lstm, transformer)
                    rmse_lstm = float(np.sqrt(np.mean((results["LSTM prediction"] - results["Actual RUL"]) ** 2)))
                    rmse_transformer = float(np.sqrt(np.mean((results["Transformer prediction"] - results["Actual RUL"]) ** 2)))
                    st.session_state["evaluation"] = {
                        "subset": selected_subset,
                        "results": results,
                        "rmse_lstm": rmse_lstm,
                        "rmse_transformer": rmse_transformer,
                    }
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

        evaluation = st.session_state.get("evaluation")
        if evaluation and evaluation["subset"] == selected_subset:
            e1, e2, e3 = st.columns(3)
            e1.metric("LSTM RMSE", f'{evaluation["rmse_lstm"]:.2f} cycles')
            e2.metric("Transformer RMSE", f'{evaluation["rmse_transformer"]:.2f} cycles')
            improvement = evaluation["rmse_lstm"] - evaluation["rmse_transformer"]
            e3.metric("RMSE difference", f"{improvement:+.2f} cycles")

            results = evaluation["results"]
            comparison = results[["Engine", "Actual RUL", "LSTM prediction", "Transformer prediction"]].melt(
                id_vars=["Engine", "Actual RUL"], var_name="Model", value_name="Predicted RUL"
            )
            eval_fig = px.scatter(
                comparison, x="Actual RUL", y="Predicted RUL", color="Model",
                hover_data=["Engine"], template="plotly_dark",
                title="Predicted vs actual RUL across test engines",
            )
            eval_fig.add_shape(
                type="line", x0=0, y0=0,
                x1=max(float(results["Actual RUL"].max()), 1),
                y1=max(float(results["Actual RUL"].max()), 1),
                line=dict(dash="dash"),
            )
            eval_fig.update_layout(
                height=420, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,20,35,.6)",
            )
            st.plotly_chart(eval_fig, use_container_width=True)
            st.dataframe(results.round(2), use_container_width=True, hide_index=True)
            st.download_button(
                "Download evaluation results (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_subset}_evaluation_results.csv",
                mime="text/csv",
            )


with methodology_tab:
    st.subheader("How the application works")
    st.markdown(
        """
        **1. Dataset selection.** FD001–FD004 differ in operating conditions and fault modes.
        Select the subset that matches your uploaded test and RUL files.

        **2. Input validation.** The app expects the original NASA C-MAPSS text format:
        26 columns per cycle (engine ID, cycle, 3 settings, 21 sensors).

        **3. Preprocessing.** The last 30 cycles for each engine are transformed using
        the scaler fitted during training for that same subset.

        **4. Inference.** LSTM and Transformer predict normalized RUL; outputs are converted
        to cycles using the project's target scale of 125.

        **5. Evaluation.** If the matching NASA RUL label file is uploaded, the app evaluates
        the final prediction for every test engine and computes RMSE.
        """
    )
    st.warning(
        "Research demonstration only. Do not use this application to make real aviation "
        "maintenance or flight-safety decisions."
    )
    st.markdown("**Stack:** PyTorch · Streamlit · Plotly · scikit-learn · NASA C-MAPSS")


st.divider()
st.markdown(
    f'<div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;'
    f'color:#7184a5;font-size:.78rem;"><span>AEROPULSE · TURBOFAN RUL RESEARCH DEMO</span>'
    f'<span>NASA C-MAPSS {selected_subset} · LSTM + TRANSFORMER</span></div>',
    unsafe_allow_html=True,
)
