
import io
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from src.model_lstm import RUL_LSTM
from src.model_transformer import RUL_Transformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroPulse | Turbofan RUL",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

WINDOW_SIZE = 30

FEATURES = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

COLUMNS = ["unit_number", "time_cycles"] + FEATURES


# ============================================================
# CUSTOM FRONTEND STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                ellipse at 80% 0%,
                rgba(40, 91, 146, 0.22),
                transparent 38%
            ),
            #08111f;
        color: #eaf1ff;
    }

    [data-testid="stSidebar"] {
        background: #0c1728;
        border-right: 1px solid rgba(160, 183, 220, 0.16);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.03em;
        color: #f4f7ff;
    }

    p, label {
        color: #c6d2e8;
    }

    .hero {
        padding: 1.8rem;
        border: 1px solid rgba(104, 224, 207, 0.25);
        border-radius: 24px;
        background: linear-gradient(
            120deg,
            #142b45,
            #0d192b
        );
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color: #68e0cf;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero h1 {
        font-size: clamp(2.2rem, 4vw, 3.4rem);
        line-height: 1.03;
        margin: 0 0 0.8rem;
    }

    .hero p {
        max-width: 760px;
        line-height: 1.7;
        color: #b8c7df;
    }

    .pill {
        display: inline-block;
        border: 1px solid rgba(104, 224, 207, 0.35);
        color: #8af0df;
        background: rgba(104, 224, 207, 0.08);
        border-radius: 99px;
        padding: 0.3rem 0.7rem;
        font-size: 0.76rem;
        margin: 0.25rem 0.25rem 0 0;
    }

    .subtle {
        color: #98a8c5;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #132137,
            #0d1829
        );
        border: 1px solid rgba(160, 183, 220, 0.16);
        padding: 1rem;
        border-radius: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #98a8c5;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f7ff;
        font-family: 'Space Grotesk', sans-serif;
    }

    .stButton > button {
        background: #68e0cf;
        color: #071320;
        border: 0;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
        width: 100%;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #9af4e6;
        color: #071320;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        background: rgba(17, 28, 49, 0.7);
        border: 1px dashed rgba(104, 224, 207, 0.38);
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        color: #68e0cf !important;
    }

    hr {
        border-color: rgba(160, 183, 220, 0.16);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained models and scaler once per app session."""

    scaler_path = MODEL_DIR / "data_scaler.pkl"
    lstm_path = MODEL_DIR / "lstm_weights.pth"
    transformer_path = MODEL_DIR / "transformer_weights.pth"

    required_files = [
        scaler_path,
        lstm_path,
        transformer_path,
    ]

    missing_files = [
        str(path.relative_to(ROOT))
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing_files)
        )

    scaler = joblib.load(scaler_path)

    lstm = RUL_LSTM()
    lstm_weights = torch.load(
        lstm_path,
        map_location="cpu",
        weights_only=True,
    )
    lstm.load_state_dict(lstm_weights)
    lstm.eval()

    transformer = RUL_Transformer()
    transformer_weights = torch.load(
        transformer_path,
        map_location="cpu",
        weights_only=True,
    )
    transformer.load_state_dict(transformer_weights)
    transformer.eval()

    return scaler, lstm, transformer


# ============================================================
# DATA VALIDATION
# ============================================================

def parse_dataset(file_content):
    """Parse the raw NASA CMAPSS FD001 test file."""

    data = pd.read_csv(
        io.BytesIO(file_content),
        sep=r"\s+",
        header=None,
        engine="python",
    )

    # Remove entirely empty columns caused by trailing whitespace.
    data = data.dropna(axis=1, how="all")

    if data.shape[1] != 26:
        raise ValueError(
            f"Expected 26 columns but found {data.shape[1]}. "
            "Upload the original NASA test_FD001.txt file."
        )

    data.columns = COLUMNS

    for column in COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data.empty:
        raise ValueError("The uploaded file contains no records.")

    if data["unit_number"].isna().any():
        raise ValueError("Engine IDs contain missing values.")

    return data


# ============================================================
# PREDICTION
# ============================================================

def predict_engine(engine, scaler, lstm, transformer):
    """
    Predict RUL using both models on the same latest
    30-cycle sensor window.
    """

    engine = engine.sort_values("time_cycles")

    if len(engine) < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} cycles are required. "
            f"This engine has {len(engine)}."
        )

    sensor_values = engine[FEATURES].to_numpy(dtype=float)

    # Use the latest operating cycles.
    sensor_values = sensor_values[-WINDOW_SIZE:]

    # Apply the scaler saved during model training.
    sensor_values = scaler.transform(sensor_values)

    tensor = torch.tensor(
        sensor_values,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        lstm_output = lstm(tensor)
        transformer_output = transformer(tensor)

        lstm_prediction = (
            float(lstm_output.reshape(-1)[0].item()) * 125.0
        )

        transformer_prediction = (
            float(transformer_output.reshape(-1)[0].item())
            * 125.0
        )

    return (
        max(0.0, lstm_prediction),
        max(0.0, transformer_prediction),
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ✈️ AeroPulse")
    st.caption("ENGINE HEALTH INTELLIGENCE")

    st.divider()

    st.markdown("### Start an analysis")

    st.write(
        "Upload the original NASA C-MAPSS FD001 test file "
        "to begin analyzing an engine."
    )

    uploaded_file = st.file_uploader(
        "test_FD001.txt",
        type=["txt"],
        help="Raw whitespace-separated file containing 26 columns.",
    )

    st.divider()

    st.markdown("**Model architecture**")

    st.markdown(
        '<span class="pill">LSTM</span>'
        '<span class="pill">Transformer</span>',
        unsafe_allow_html=True,
    )

    st.caption("Dataset: NASA C-MAPSS · FD001")
    st.caption("Input window: latest 30 cycles")
    st.caption("Inference device: CPU")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            Predictive maintenance · Research demo
        </div>

        <h1>
            See engine health<br>
            before failure.
        </h1>

        <p>
            Explore sensor behaviour, estimate remaining useful
            life, and compare two sequence-learning models in one
            workspace. Built around the NASA C-MAPSS FD001
            turbofan benchmark.
        </p>

        <span class="pill">01 / Upload data</span>
        <span class="pill">02 / Select engine</span>
        <span class="pill">03 / Compare predictions</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP-LEVEL PROJECT METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Transformer RMSE",
    "14.59 cycles",
    help="Reported test result from the project README.",
)

metric2.metric(
    "LSTM RMSE",
    "14.74 cycles",
    help="Reported test result from the project README.",
)

metric3.metric(
    "Input channels",
    "24",
    "3 settings + 21 sensors",
)

metric4.metric(
    "Inference window",
    "30 cycles",
)


# ============================================================
# MAIN TABS
# ============================================================

analysis_tab, benchmark_tab, methodology_tab = st.tabs(
    [
        "⚡ Live engine analysis",
        "📈 Model benchmark",
        "ⓘ Methodology",
    ]
)


# ============================================================
# TAB 1: LIVE ENGINE ANALYSIS
# ============================================================

with analysis_tab:
    st.subheader("Analyze an engine")

    st.markdown(
        """
        <p class="subtle">
            Select an engine, inspect its sensor trends, and run
            both trained models on the same recent sequence.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("### Your workspace is ready")

            st.info(
                "Upload the raw test_FD001.txt file using the "
                "sidebar to begin."
            )

            st.write(
                "The application will validate the data, let you "
                "select an engine, and compare the two model estimates."
            )

        with right:
            st.markdown("### Included in the analysis")

            st.markdown(
                "- Side-by-side RUL predictions\n"
                "- Interactive sensor trend charts\n"
                "- Recent sensor records\n"
                "- LSTM versus Transformer comparison"
            )

    else:
        try:
            data = parse_dataset(
                uploaded_file.getvalue()
            )

            engine_ids = sorted(
                data["unit_number"].unique().tolist()
            )

            select_col, summary_col = st.columns([1, 2])

            with select_col:
                selected_engine = st.selectbox(
                    "Select engine ID",
                    engine_ids,
                )

            engine = data[
                data["unit_number"] == selected_engine
            ].sort_values("time_cycles")

            with summary_col:
                st.markdown(" ")

                st.metric(
                    "Selected engine",
                    f"#{int(selected_engine):02d}",
                    f"{len(engine)} recorded cycles · "
                    f"{len(engine_ids)} engines in file",
                )

            if len(engine) < WINDOW_SIZE:
                st.warning(
                    f"This engine has {len(engine)} cycles. "
                    f"At least {WINDOW_SIZE} cycles are needed "
                    "for the model input window."
                )

            else:
                if st.button(
                    "Run LSTM + Transformer",
                    type="primary",
                ):
                    with st.spinner(
                        "Loading trained weights and analyzing "
                        "sensor data..."
                    ):
                        scaler, lstm, transformer = load_models()

                        pred_lstm, pred_transformer = predict_engine(
                            engine,
                            scaler,
                            lstm,
                            transformer,
                        )

                    st.session_state["rul_result"] = {
                        "engine_id": int(selected_engine),
                        "lstm": pred_lstm,
                        "transformer": pred_transformer,
                    }

                result = st.session_state.get("rul_result")

                if (
                    result
                    and result["engine_id"]
                    == int(selected_engine)
                ):
                    st.markdown("#### Prediction results")

                    p1, p2, p3 = st.columns(3)

                    p1.metric(
                        "LSTM estimate",
                        f'{result["lstm"]:.1f} cycles',
                    )

                    p2.metric(
                        "Transformer estimate",
                        f'{result["transformer"]:.1f} cycles',
                    )

                    difference = abs(
                        result["transformer"] - result["lstm"]
                    )

                    p3.metric(
                        "Absolute difference",
                        f"{difference:.1f} cycles",
                    )

                    prediction_data = pd.DataFrame({
                        "Model": ["LSTM", "Transformer"],
                        "Predicted RUL (cycles)": [
                            result["lstm"],
                            result["transformer"],
                        ],
                    })

                    prediction_fig = px.bar(
                        prediction_data,
                        x="Model",
                        y="Predicted RUL (cycles)",
                        color="Model",
                        text="Predicted RUL (cycles)",
                        color_discrete_map={
                            "LSTM": "#6f8dbb",
                            "Transformer": "#68e0cf",
                        },
                        template="plotly_dark",
                    )

                    prediction_fig.update_traces(
                        texttemplate="%{text:.1f}",
                        textposition="outside",
                    )

                    prediction_fig.update_layout(
                        height=320,
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,20,35,.6)",
                        margin=dict(
                            l=8,
                            r=8,
                            t=24,
                            b=8,
                        ),
                        font=dict(color="#c6d2e8"),
                    )

                    st.plotly_chart(
                        prediction_fig,
                        use_container_width=True,
                    )

                    st.caption(
                        "These are model estimates, not confidence "
                        "intervals or aviation safety determinations."
                    )

                st.divider()

                st.markdown("#### Sensor behaviour over time")

                sensor_options = [
                    f"s_{i}" for i in range(1, 22)
                ]

                selected_sensors = st.multiselect(
                    "Choose sensor channels",
                    sensor_options,
                    default=["s_2", "s_3", "s_4"],
                )

                if selected_sensors:
                    plot_data = engine[
                        ["time_cycles"] + selected_sensors
                    ].melt(
                        id_vars="time_cycles",
                        var_name="Sensor",
                        value_name="Reading",
                    )

                    sensor_fig = px.line(
                        plot_data,
                        x="time_cycles",
                        y="Reading",
                        color="Sensor",
                        template="plotly_dark",
                        labels={
                            "time_cycles": "Operating cycle",
                            "Reading": "Raw sensor reading",
                        },
                    )

                    sensor_fig.update_layout(
                        height=390,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,20,35,.6)",
                        margin=dict(
                            l=8,
                            r=8,
                            t=20,
                            b=8,
                        ),
                        font=dict(color="#c6d2e8"),
                        legend_title_text="Channel",
                    )

                    sensor_fig.update_xaxes(
                        gridcolor="rgba(160,183,220,.12)"
                    )

                    sensor_fig.update_yaxes(
                        gridcolor="rgba(160,183,220,.12)"
                    )

                    st.plotly_chart(
                        sensor_fig,
                        use_container_width=True,
                    )

                else:
                    st.info(
                        "Choose at least one sensor channel "
                        "to display its trend."
                    )

                with st.expander("View latest sensor records"):
                    visible_columns = (
                        ["time_cycles"] + selected_sensors
                    )

                    st.dataframe(
                        engine[visible_columns].tail(10),
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as error:
            st.error(
                f"Could not analyze this file: {error}"
            )

            st.caption(
                "Use the original whitespace-separated NASA "
                "C-MAPSS FD001 test file, not a CSV export."
            )


# ============================================================
# TAB 2: MODEL BENCHMARK
# ============================================================

with benchmark_tab:
    st.subheader("LSTM vs Transformer")

    st.markdown(
        """
        <p class="subtle">
            Reported held-out evaluation results from the project
            README. Lower RMSE indicates lower average prediction error.
        </p>
        """,
        unsafe_allow_html=True,
    )

    benchmark_data = pd.DataFrame({
        "Model": ["LSTM", "Transformer"],
        "Test RMSE (cycles)": [14.74, 14.59],
    })

    benchmark_fig = px.bar(
        benchmark_data,
        x="Model",
        y="Test RMSE (cycles)",
        color="Model",
        text="Test RMSE (cycles)",
        color_discrete_map={
            "LSTM": "#6f8dbb",
            "Transformer": "#68e0cf",
        },
        template="plotly_dark",
    )

    benchmark_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        cliponaxis=False,
    )

    benchmark_fig.update_layout(
        height=340,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,20,35,.6)",
        margin=dict(
            l=8,
            r=8,
            t=24,
            b=8,
        ),
        yaxis_title="RMSE · cycles (lower is better)",
        xaxis_title="",
        font=dict(color="#c6d2e8"),
    )

    benchmark_fig.update_yaxes(
        rangemode="tozero",
        gridcolor="rgba(160,183,220,.12)",
    )

    st.plotly_chart(
        benchmark_fig,
        use_container_width=True,
    )

    relative_gain = (
        (14.74 - 14.59) / 14.74
    ) * 100

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "LSTM RMSE",
        "14.74 cycles",
    )

    b2.metric(
        "Transformer RMSE",
        "14.59 cycles",
        delta="-0.15 cycles",
    )

    b3.metric(
        "Relative RMSE reduction",
        f"{relative_gain:.2f}%",
    )

    saved_graph = (
        MODEL_DIR / "final_evaluation_graph.png"
    )

    if saved_graph.exists():
        with st.expander("View original evaluation plot"):
            st.image(
                str(saved_graph),
                caption=(
                    "Saved evaluation plot from the repository. "
                    "It is not recalculated from the currently "
                    "uploaded engine."
                ),
                use_container_width=True,
            )

    st.caption(
        "Benchmark values are reported in the project README. "
        "They are not recalculated from the uploaded data."
    )


# ============================================================
# TAB 3: METHODOLOGY
# ============================================================

with methodology_tab:
    st.subheader(
        "From sensor telemetry to an estimate"
    )

    st.markdown(
        """
        ### 01 · Input

        Upload the raw NASA C-MAPSS FD001 test file. Each row
        represents an engine operating cycle with three operating
        settings and 21 sensor readings.

        ### 02 · Preprocessing

        The application validates the 26-column layout, sorts
        operating cycles, and applies the saved feature scaler to
        the latest 30 cycles for the selected engine.

        ### 03 · Model inference

        The same sensor sequence is passed through the saved LSTM
        and Transformer weights. Both outputs are converted to the
        cycle scale used by the project's evaluation code.

        ### 04 · Interpretation

        The difference between model estimates is a comparison aid.
        It is not a probability, calibrated confidence interval,
        or measure of flight safety.
        """
    )

    st.warning(
        "Research demonstration only. This application is not "
        "certified for maintenance decisions or aviation safety."
    )

    st.markdown(
        "**Technology stack:** PyTorch · Streamlit · Plotly · "
        "scikit-learn · NASA C-MAPSS FD001"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        display:flex;
        justify-content:space-between;
        gap:1rem;
        flex-wrap:wrap;
        color:#7184a5;
        font-size:.78rem;
    ">
        <span>AEROPULSE · TURBOFAN RUL RESEARCH DEMO</span>
        <span>NASA C-MAPSS FD001 · LSTM + TRANSFORMER</span>
    </div>
    """,
    unsafe_allow_html=True,
)
