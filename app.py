
import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch

from src.model_transformer import RUL_Transformer

st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Turbofan Engine RUL Predictor")
st.write(
    "Estimate an aircraft engine's remaining useful life "
    "using a trained Transformer model."
)

ROOT = Path(__file__).resolve().parent
FEATURES = ["setting_1", "setting_2", "setting_3"] + [
    f"s_{i}" for i in range(1, 22)
]
COLUMNS = ["unit_number", "time_cycles"] + FEATURES


@st.cache_resource
def load_model():
    scaler = joblib.load(ROOT / "models" / "data_scaler.pkl")
    model = RUL_Transformer()

    weights = torch.load(
        ROOT / "models" / "transformer_weights.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()

    return scaler, model


uploaded = st.file_uploader(
    "Upload NASA test_FD001.txt",
    type=["txt"],
)

if uploaded is not None:
    try:
        df = pd.read_csv(
            io.BytesIO(uploaded.getvalue()),
            sep=r"\s+",
            header=None,
        )

        if df.shape[1] != 26:
            st.error(
                "Expected 26 columns. Upload the standard "
                "NASA CMAPSS FD001 test file."
            )
            st.stop()

        df.columns = COLUMNS
        df[FEATURES] = df[FEATURES].apply(
            pd.to_numeric, errors="raise"
        )

        engine_ids = sorted(df["unit_number"].unique())
        engine_id = st.selectbox("Select engine", engine_ids)

        engine = df[df["unit_number"] == engine_id].sort_values(
            "time_cycles"
        )

        st.write(f"Recorded cycles: {len(engine)}")

        if len(engine) < 30:
            st.warning("This engine has fewer than 30 recorded cycles.")

        elif st.button("Predict Remaining Useful Life"):
            with st.spinner("Analyzing sensor data..."):
                scaler, model = load_model()

                values = engine[FEATURES].to_numpy(dtype=float)[-30:]
                values = scaler.transform(values)

                x = torch.tensor(
                    values, dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():
                    prediction = model(x).item() * 125.0

            prediction = max(0.0, prediction)

            st.success("Prediction completed")
            st.metric(
                "Estimated Remaining Useful Life",
                f"{prediction:.1f} cycles",
            )

            st.caption(
                "For research and demonstration only. "
                "Not an aviation safety determination."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

else:
    st.info(
        "Upload the standard NASA CMAPSS FD001 test_FD001.txt "
        "dataset to begin."
    )

st.divider()
st.caption("Model: Transformer | Dataset: NASA CMAPSS FD001")
