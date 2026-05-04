from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cnn_xai_lab.config import IMAGE_SIZE, MODELS_DIR, OUTPUTS_DIR
from cnn_xai_lab.face_detection import detect_primary_face, make_portrait_focus_crop
from cnn_xai_lab.xai import (
    compute_saliency_map,
    make_gradcam_heatmap,
    overlay_map_on_image,
    predict_probabilities,
    prepare_image,
)

st.set_page_config(
    page_title="CNNs-XAI Lab",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink-strong: #f5efff;
        --ink: #ddd3f5;
        --ink-soft: #a99bc8;
        --accent: #8f5cff;
        --accent-soft: #d5a8ff;
        --panel: rgba(20, 15, 31, 0.74);
        --panel-strong: rgba(22, 16, 36, 0.92);
        --line: rgba(173, 141, 255, 0.16);
        --shadow: 0 22px 55px rgba(4, 2, 10, 0.34);
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(149, 92, 255, 0.22), transparent 24%),
            radial-gradient(circle at 85% 8%, rgba(214, 122, 255, 0.14), transparent 22%),
            radial-gradient(circle at 50% 100%, rgba(80, 32, 150, 0.16), transparent 30%),
            linear-gradient(180deg, #09070f 0%, #100b1d 38%, #140d23 100%);
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(185, 162, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(185, 162, 255, 0.035) 1px, transparent 1px);
        background-size: 36px 36px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.18), transparent 78%);
    }
    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top, rgba(143, 92, 255, 0.16), transparent 28%),
            linear-gradient(180deg, rgba(14, 10, 24, 0.96), rgba(20, 13, 32, 0.98));
        border-right: 1px solid rgba(173, 141, 255, 0.10);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }
    .hero-shell {
        position: relative;
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(290px, 0.85fr);
        gap: 1.1rem;
        padding: 1.35rem;
        border-radius: 30px;
        background:
            radial-gradient(circle at top right, rgba(173, 141, 255, 0.18), transparent 24%),
            linear-gradient(135deg, rgba(20, 14, 34, 0.98) 0%, rgba(13, 10, 23, 0.96) 42%, rgba(34, 19, 57, 0.98) 100%);
        border: 1px solid rgba(173, 141, 255, 0.16);
        box-shadow: var(--shadow);
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        right: -90px;
        top: -90px;
        width: 240px;
        height: 240px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(143, 92, 255, 0.25), transparent 65%);
    }
    .hero-copy {
        position: relative;
        z-index: 1;
        padding: 0.25rem 0.2rem 0.2rem 0.1rem;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(143, 92, 255, 0.14);
        color: #d7c3ff;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
    }
    .hero-title {
        margin: 1rem 0 0.65rem 0;
        color: var(--ink-strong);
        font-size: clamp(2.4rem, 4vw, 4.1rem);
        line-height: 0.94;
        letter-spacing: -0.05em;
        max-width: 10.5ch;
    }
    .hero-body {
        margin: 0;
        color: var(--ink);
        font-size: 1.03rem;
        line-height: 1.75;
        max-width: 50rem;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.2rem;
    }
    .meta-pill {
        padding: 0.65rem 0.82rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(173, 141, 255, 0.14);
        color: var(--ink-soft);
        font-size: 0.86rem;
        min-width: 150px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(10px);
    }
    .meta-pill strong {
        display: block;
        color: var(--ink-strong);
        font-size: 0.96rem;
        margin-bottom: 0.22rem;
    }
    .hero-rail {
        position: relative;
        z-index: 1;
        display: grid;
        gap: 0.8rem;
        align-content: stretch;
    }
    .rail-card {
        padding: 1rem 1.05rem;
        border-radius: 22px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(173, 141, 255, 0.14);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
    }
    .rail-label {
        color: #b79cff;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .rail-value {
        color: var(--ink-strong);
        font-size: 1.95rem;
        line-height: 1;
        font-weight: 700;
        margin-bottom: 0.28rem;
    }
    .rail-note {
        color: var(--ink-soft);
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .story-ribbon {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.2rem 0;
    }
    .story-card {
        padding: 1rem 1.05rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(173, 141, 255, 0.10);
        box-shadow: 0 18px 34px rgba(4, 2, 10, 0.18);
    }
    .story-card h3 {
        margin: 0 0 0.45rem 0;
        color: var(--ink-strong);
        font-size: 1rem;
    }
    .story-card p {
        margin: 0;
        color: var(--ink-soft);
        line-height: 1.6;
        font-size: 0.92rem;
    }
    .eyebrow {
        display: inline-block;
        padding: 0.32rem 0.68rem;
        border-radius: 999px;
        background: rgba(143, 92, 255, 0.12);
        color: #cfbbff;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }
    .section-shell {
        position: relative;
        padding: 1.25rem 1.35rem;
        border-radius: 24px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
        border: 1px solid var(--line);
        box-shadow: 0 16px 32px rgba(4, 2, 10, 0.18);
        overflow: hidden;
        backdrop-filter: blur(14px);
    }
    .section-shell::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #8f5cff, #d5a8ff, rgba(213, 168, 255, 0.10));
    }
    .section-shell h2 {
        margin: 0.05rem 0 0.32rem 0;
        color: var(--ink-strong);
        font-size: 1.75rem;
        letter-spacing: -0.03em;
    }
    .section-shell p {
        margin: 0;
        color: var(--ink-soft);
        line-height: 1.6;
    }
    .info-card {
        height: 100%;
        padding: 1.1rem 1.15rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(173, 141, 255, 0.12);
        box-shadow: 0 16px 36px rgba(4, 2, 10, 0.16);
        backdrop-filter: blur(12px);
    }
    .info-card h3 {
        margin: 0 0 0.55rem 0;
        color: var(--ink-strong);
        font-size: 1.12rem;
    }
    .info-card p, .info-card li {
        color: var(--ink-soft);
        line-height: 1.58;
        font-size: 0.95rem;
    }
    .info-card ul {
        margin: 0;
        padding-left: 1.1rem;
    }
    .metric-card {
        height: 100%;
        padding: 1.08rem 1.1rem;
        border-radius: 22px;
        background:
            linear-gradient(140deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(173, 141, 255, 0.12);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 14px 28px rgba(4, 2, 10, 0.18);
        backdrop-filter: blur(12px);
    }
    .metric-label {
        display: block;
        color: #bb9cff;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .metric-value {
        color: var(--ink-strong);
        font-size: 2.1rem;
        line-height: 1.1;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-caption {
        color: var(--ink-soft);
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.85rem;
    }
    .chip {
        padding: 0.55rem 0.82rem;
        border-radius: 999px;
        background: rgba(143, 92, 255, 0.10);
        color: #d5c3ff;
        font-size: 0.85rem;
        border: 1px solid rgba(173, 141, 255, 0.12);
    }
    .process-grid, .architecture-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.85rem;
        margin-top: 0.9rem;
    }
    .process-step, .arch-card {
        position: relative;
        min-height: 150px;
        padding: 1.08rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
        border: 1px solid rgba(173, 141, 255, 0.10);
        overflow: hidden;
        box-shadow: 0 18px 32px rgba(4, 2, 10, 0.18);
        transition: transform 160ms ease, box-shadow 160ms ease;
        backdrop-filter: blur(12px);
    }
    .process-step::before, .arch-card::before {
        content: "";
        position: absolute;
        inset: auto auto 0 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #8f5cff, #d5a8ff);
    }
    .process-step:hover, .arch-card:hover, .metric-card:hover, .info-card:hover, .story-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 22px 38px rgba(64, 40, 27, 0.08);
    }
    .step-index, .arch-index {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        background: rgba(143, 92, 255, 0.16);
        color: #e4d8ff;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }
    .process-step h4, .arch-card h4 {
        margin: 0 0 0.4rem 0;
        color: var(--ink-strong);
        font-size: 1rem;
    }
    .process-step p, .arch-card p {
        margin: 0;
        color: var(--ink-soft);
        line-height: 1.55;
        font-size: 0.92rem;
    }
    .arch-shape {
        margin-top: 0.55rem;
        font-size: 0.83rem;
        color: #c5adff;
        font-weight: 700;
    }
    .note-box {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(143, 92, 255, 0.10), rgba(255,255,255,0.03));
        border: 1px solid rgba(173, 141, 255, 0.10);
        color: var(--ink);
        line-height: 1.55;
        box-shadow: 0 14px 28px rgba(4, 2, 10, 0.16);
    }
    .insight-box {
        padding: 1rem 1.1rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(143, 92, 255, 0.12), rgba(255,255,255,0.04));
        border: 1px solid rgba(173, 141, 255, 0.12);
        color: var(--ink);
        line-height: 1.6;
        box-shadow: 0 16px 34px rgba(4, 2, 10, 0.18);
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 22px;
        background: rgba(20, 15, 31, 0.82);
        margin-top: 0.35rem;
        box-shadow: 0 18px 32px rgba(4, 2, 10, 0.20);
    }
    .report-table th {
        background: rgba(143, 92, 255, 0.16);
        color: var(--ink-strong);
        font-weight: 700;
        text-align: left;
        padding: 0.8rem 0.9rem;
        font-size: 0.9rem;
    }
    .report-table td {
        padding: 0.8rem 0.9rem;
        border-top: 1px solid rgba(173, 141, 255, 0.08);
        color: var(--ink);
        font-size: 0.92rem;
    }
    .report-table tr:nth-child(even) td {
        background: rgba(255,255,255,0.025);
    }
    .divider-title {
        margin-top: 0.1rem;
        color: var(--ink-strong);
        font-size: 1.22rem;
        font-weight: 700;
    }
    .demo-shell {
        padding: 1.15rem 1.2rem;
        border-radius: 24px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(173, 141, 255, 0.10);
        box-shadow: 0 18px 36px rgba(4, 2, 10, 0.18);
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .demo-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .demo-header h3 {
        margin: 0;
        color: var(--ink-strong);
        font-size: 1.45rem;
        letter-spacing: -0.03em;
    }
    .demo-header p {
        margin: 0.35rem 0 0 0;
        color: var(--ink-soft);
        line-height: 1.6;
        max-width: 48rem;
    }
    .demo-badge {
        padding: 0.55rem 0.75rem;
        border-radius: 14px;
        background: rgba(143, 92, 255, 0.12);
        border: 1px solid rgba(173, 141, 255, 0.12);
        color: #dacbff;
        font-size: 0.85rem;
        white-space: nowrap;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(173, 141, 255, 0.10);
        border-radius: 22px;
        padding: 1rem 1rem 0.85rem 1rem;
        box-shadow: 0 16px 30px rgba(4, 2, 10, 0.18);
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] p {
        color: #b99bff;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        color: #f6f1ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        padding: 0.4rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(173, 141, 255, 0.10);
        width: fit-content;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.7rem 1rem;
        border-radius: 16px;
        color: #b6a7d7;
        background: transparent;
        border: 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(143, 92, 255, 0.18), rgba(213, 168, 255, 0.18));
        color: #f5efff;
        box-shadow: inset 0 0 0 1px rgba(173, 141, 255, 0.12);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.04);
        border: 1px dashed rgba(173, 141, 255, 0.35);
        border-radius: 22px;
    }
    [data-testid="stImage"] img {
        border-radius: 22px;
        border: 1px solid rgba(173, 141, 255, 0.12);
        box-shadow: 0 18px 30px rgba(4, 2, 10, 0.18);
    }
    [data-testid="stSelectbox"] > div {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
    }
    [data-testid="stMarkdownContainer"], .stMarkdown, p, li, label, span {
        color: var(--ink);
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--ink-soft) !important;
    }
    h1, h2, h3, h4 {
        color: var(--ink-strong);
    }
    .stAlert {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(173, 141, 255, 0.12);
    }
    @media (max-width: 980px) {
        .hero-shell {
            grid-template-columns: 1fr;
        }
        .story-ribbon {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_file_version(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def load_json_file(path_str: str, version: int):
    path = Path(path_str)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_model(model_path_str: str, version: int):
    return keras.models.load_model(model_path_str)


@st.cache_data(show_spinner=False)
def load_experiment_table(path_str: str, version: int):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_experiment_registry(experiments_dir_str: str, signature: tuple[str, ...]):
    experiments_dir = Path(experiments_dir_str)
    registry: dict[str, dict] = {}
    for metrics_path in sorted(experiments_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        experiment_name = metrics.get("name", metrics_path.parent.name)
        registry[experiment_name] = metrics
    return registry


def read_json_file(path: Path):
    return load_json_file(str(path), get_file_version(path))


def read_experiment_table(path: Path):
    return load_experiment_table(str(path), get_file_version(path))


def read_model(path: Path):
    return load_model(str(path), get_file_version(path))


def get_experiment_signature(experiments_dir: Path) -> tuple[str, ...]:
    signature_parts = []
    for metrics_path in sorted(experiments_dir.glob("*/metrics.json")):
        signature_parts.append(f"{metrics_path.parent.name}:{metrics_path.stat().st_mtime_ns}")
    return tuple(signature_parts)


def resolve_model_path(model_metrics: dict | None) -> Path:
    fallback = MODELS_DIR / "model.keras"
    if not model_metrics:
        return fallback

    model_path_value = model_metrics.get("model_path")
    if not model_path_value:
        return fallback

    candidate = Path(str(model_path_value))
    if candidate.exists():
        return candidate
    return fallback


def summarize_from_male_probability(male_probability: float) -> dict:
    male_probability = float(np.clip(male_probability, 0.0, 1.0))
    female_probability = 1.0 - male_probability
    predicted_index = 1 if male_probability >= 0.5 else 0
    return {
        "female_probability": female_probability,
        "male_probability": male_probability,
        "predicted_index": predicted_index,
        "predicted_label": "male" if predicted_index == 1 else "female",
        "confidence": male_probability if predicted_index == 1 else female_probability,
    }


def build_demo_views(image: Image.Image) -> dict[str, dict]:
    auto_detection = detect_primary_face(image)
    portrait_detection = make_portrait_focus_crop(image)

    full_view = {
        "image": image,
        "boxed_image": image,
        "label": "Imagen completa",
        "method_label": "Imagen completa",
    }
    auto_view = (
        {
            "image": auto_detection.cropped_face,
            "boxed_image": auto_detection.boxed_image,
            "label": "Deteccion de rostro",
            "method_label": "Deteccion de rostro",
        }
        if auto_detection is not None
        else full_view
    )
    portrait_view = {
        "image": portrait_detection.cropped_face,
        "boxed_image": portrait_detection.boxed_image,
        "label": "Recorte retrato",
        "method_label": "Recorte retrato",
    }
    return {
        "full": full_view,
        "auto": auto_view,
        "portrait": portrait_view,
        "auto_found": auto_detection is not None,
    }


def run_single_inference(model_metrics: dict, selected_view: dict) -> tuple[dict, dict]:
    model = read_model(resolve_model_path(model_metrics))
    image_batch = prepare_image(selected_view["image"], image_size=IMAGE_SIZE)
    prediction = predict_probabilities(model, image_batch)
    anchor_vote = {
        "model_name": model_metrics.get("name", "modelo"),
        "view_key": selected_view["label"],
        "view_label": selected_view["label"],
        "weight": 1.0,
        "prediction": prediction,
        "image_batch": image_batch,
        "model": model,
    }
    return prediction, anchor_vote


def run_robust_inference(experiment_registry: dict[str, dict], views: dict[str, dict]) -> tuple[dict, dict, list[dict]]:
    vote_specs = [
        ("face_crop_k5", "auto", 0.34),
        ("compact_k5_dropout", "auto", 0.28),
        ("wider_k5_dropout", "auto", 0.22),
        ("wider_k5_dropout", "full", 0.16),
    ]
    votes: list[dict] = []

    for experiment_name, view_key, weight in vote_specs:
        metrics = experiment_registry.get(experiment_name)
        if metrics is None:
            continue
        selected_view = views[view_key]
        model = read_model(resolve_model_path(metrics))
        image_batch = prepare_image(selected_view["image"], image_size=IMAGE_SIZE)
        prediction = predict_probabilities(model, image_batch)
        votes.append(
            {
                "model_name": experiment_name,
                "view_key": view_key,
                "view_label": selected_view["label"],
                "weight": weight,
                "prediction": prediction,
                "image_batch": image_batch,
                "model": model,
                "metrics": metrics,
            }
        )

    if not votes:
        raise FileNotFoundError("No hay modelos disponibles para la inferencia robusta.")

    total_weight = sum(vote["weight"] for vote in votes) or 1.0
    male_probability = sum(
        vote["weight"] * vote["prediction"]["male_probability"] for vote in votes
    ) / total_weight
    ensemble_prediction = summarize_from_male_probability(male_probability)
    anchor_vote = min(
        votes,
        key=lambda vote: (
            abs(vote["prediction"]["male_probability"] - ensemble_prediction["male_probability"]),
            -vote["weight"],
        ),
    )
    return ensemble_prediction, anchor_vote, votes


def get_architecture_animation_path() -> Path:
    preferred = OUTPUTS_DIR / "animations" / "cnn_architecture_manim.mp4"
    fallback = OUTPUTS_DIR / "animations" / "videos" / "render_architecture_manim" / "720p30" / "ArchitectureFlowScene.mp4"
    if preferred.exists():
        return preferred
    return fallback


def render_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">{title}</span>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="eyebrow">{eyebrow}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(
    raw_dataset_summary: dict | None,
    dataset_summary: dict | None,
    best_metrics: dict | None,
) -> None:
    raw_total_images = raw_dataset_summary["num_images"] if raw_dataset_summary else 0
    unique_images = dataset_summary["num_images"] if dataset_summary else 0
    duplicates = max(raw_total_images - unique_images, 0)
    best_name = best_metrics["best_experiment"] if best_metrics else "N/D"
    best_auc = f'{best_metrics["test_auc"]:.4f}' if best_metrics else "N/D"
    best_acc = f'{best_metrics["test_accuracy"]:.4f}' if best_metrics else "N/D"

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-copy">
                <div class="hero-kicker">Laboratorio CNNs + XAI</div>
                <h1 class="hero-title">Clasificacion de rostros con explicabilidad visual</h1>
                <p class="hero-body">
                    Una experiencia de laboratorio convertida en pieza interactiva: auditoria de datos,
                    CNN entrenada desde cero, interpretabilidad con Saliency Map y Grad-CAM, y una demo
                    que deja ver tanto los aciertos como las limitaciones reales del modelo.
                </p>
                <div class="hero-meta">
                    <div class="meta-pill">
                        <strong>Dataset auditado</strong>
                        {raw_total_images} imagenes crudas, {unique_images} unicas tras deduplicacion.
                    </div>
                    <div class="meta-pill">
                        <strong>Evaluacion honesta</strong>
                        Split estricto y sin fuga entre entrenamiento, validacion y prueba.
                    </div>
                    <div class="meta-pill">
                        <strong>Demo interactiva</strong>
                        Comparacion entre imagen completa, rostro detectado y recorte retrato.
                    </div>
                </div>
            </div>
            <div class="hero-rail">
                <div class="rail-card">
                    <div class="rail-label">Mejor experimento</div>
                    <div class="rail-value">{best_name}</div>
                    <div class="rail-note">Configuracion final elegida tras la correccion del pipeline.</div>
                </div>
                <div class="rail-card">
                    <div class="rail-label">Test accuracy</div>
                    <div class="rail-value">{best_acc}</div>
                    <div class="rail-note">Desempeno realista despues de eliminar duplicados y fuga entre splits.</div>
                </div>
                <div class="rail-card">
                    <div class="rail-label">Test AUC</div>
                    <div class="rail-value">{best_auc}</div>
                    <div class="rail-note">{duplicates} imagenes duplicadas o casi duplicadas fueron aisladas del analisis.</div>
                </div>
            </div>
        </div>
        <div class="story-ribbon">
            <div class="story-card">
                <h3>Lo que impresiona aqui</h3>
                <p>No solo muestra predicciones: cuenta la historia completa del modelo, desde la limpieza del dataset hasta la interpretabilidad visual.</p>
            </div>
            <div class="story-card">
                <h3>Lo metodologicamente valioso</h3>
                <p>La pagina deja claro que un score alto puede ser enganoso cuando el protocolo de evaluacion tiene fuga entre splits.</p>
            </div>
            <div class="story-card">
                <h3>Lo que se puede explorar</h3>
                <p>La demo permite contrastar como cambia la salida cuando el modelo ve el fondo completo frente a una region mas facial.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_html_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Aun no hay resultados tabulados para mostrar.")
        return
    st.markdown(frame.to_html(index=False, classes="report-table", border=0), unsafe_allow_html=True)


def render_process_simulation() -> None:
    steps = [
        (
            "01",
            "Lectura RGB",
            "La imagen se mantiene en color para conservar patrones utiles de iluminacion, tono y contraste.",
        ),
        (
            "02",
            "Recorte opcional",
            "En la demo se puede usar la imagen completa, deteccion de rostro o recorte retrato para reducir fondo.",
        ),
        (
            "03",
            "Resize 224x224",
            "Todas las entradas se redimensionan a un tamano fijo para que la CNN reciba tensores consistentes.",
        ),
        (
            "04",
            "Normalizacion [0,1]",
            "Los pixeles se escalan dividiendo entre 255 para estabilizar el entrenamiento y la inferencia.",
        ),
        (
            "05",
            "Tensor 224x224x3",
            "La imagen final entra como tensor numerico con tres canales RGB listo para ser procesado por la red.",
        ),
        (
            "06",
            "Bloques CNN",
            "La red aplica filtros convolucionales y max pooling para extraer patrones faciales cada vez mas abstractos.",
        ),
    ]

    for row_start in range(0, len(steps), 3):
        row_steps = steps[row_start : row_start + 3]
        columns = st.columns(len(row_steps))
        for column, (index, title, body) in zip(columns, row_steps):
            with column:
                st.markdown(
                    f"""
                    <div class="process-step">
                        <div class="step-index">{index}</div>
                        <h4>{title}</h4>
                        <p>{body}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_architecture_simulation(best_row: pd.Series | None) -> None:
    base_filters = int(best_row["base_filters"]) if best_row is not None else 32
    dense_units = int(best_row["dense_units"]) if best_row is not None else 96
    dropout = float(best_row["dropout"]) if best_row is not None else 0.50
    kernel_size = int(best_row["kernel_size"]) if best_row is not None else 5

    layers = [
        ("01", "Input", "Imagen RGB normalizada", "224 x 224 x 3"),
        ("02", "Conv2D", f"{base_filters} filtros, kernel {kernel_size}x{kernel_size}, ReLU", f"224 x 224 x {base_filters}"),
        ("03", "BatchNorm + Pool", "Estabiliza activaciones y reduce resolucion", f"112 x 112 x {base_filters}"),
        ("04", "Conv2D", f"{base_filters * 2} filtros con ReLU", f"112 x 112 x {base_filters * 2}"),
        ("05", "BatchNorm + Pool", "Compresion espacial intermedia", f"56 x 56 x {base_filters * 2}"),
        ("06", "Conv2D", f"{base_filters * 4} filtros con ReLU", f"56 x 56 x {base_filters * 4}"),
        ("07", "BatchNorm + Pool", "Extraccion profunda de rasgos", f"28 x 28 x {base_filters * 4}"),
        ("08", "Conv2D", f"{base_filters * 6} filtros con ReLU", f"28 x 28 x {base_filters * 6}"),
        ("09", "GlobalAveragePooling2D", "Reduce parametros y evita Flatten masivo", f"{base_filters * 6}"),
        ("10", "Dense", f"{dense_units} neuronas con ReLU", f"{dense_units}"),
        ("11", "Dropout", f"Regularizacion {dropout:.2f}", f"{dense_units}"),
        ("12", "Salida", "Sigmoid para clasificacion binaria", "1"),
    ]

    for row_start in range(0, len(layers), 4):
        row_layers = layers[row_start : row_start + 4]
        columns = st.columns(len(row_layers))
        for column, (index, name, body, shape) in zip(columns, row_layers):
            with column:
                st.markdown(
                    f"""
                    <div class="arch-card">
                        <div class="arch-index">{index}</div>
                        <h4>{name}</h4>
                        <p>{body}</p>
                        <div class="arch-shape">{shape}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_architecture_animation(best_row: pd.Series | None) -> None:
    animation_path = get_architecture_animation_path()
    if animation_path.exists():
        st.video(str(animation_path), autoplay=True, muted=True, loop=True)
        st.caption(
            "Animacion generada con Manim en Python. El recorrido respeta la arquitectura del mejor experimento y muestra el paso del tensor desde la entrada RGB hasta la salida sigmoide."
        )
    else:
        st.warning(
            "No se encontro el video de Manim. Ejecuta `python -m manim -q h --disable_caching --media_dir outputs/animations scripts/render_architecture_manim.py ArchitectureFlowScene` para regenerarlo."
        )

    st.markdown(
        """
        <div class="note-box">
            Debajo se conserva el desglose tecnico de capas para que la animacion tenga respaldo explicativo y siga cumpliendo con el nivel de detalle pedido en el informe.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_architecture_simulation(best_row)


def render_report_tab(
    raw_dataset_summary: dict | None,
    dataset_summary: dict | None,
    split_summary: dict | None,
    best_metrics: dict | None,
    comparison_frame: pd.DataFrame,
) -> None:
    render_section_header(
        "Informe del laboratorio",
        "Contexto del dataset y objetivos del taller",
        "Esta seccion resume el problema trabajado en el laboratorio, las caracteristicas del dataset y los objetivos academicos que guiaron el desarrollo del modelo y la app.",
    )

    raw_total_images = raw_dataset_summary["num_images"] if raw_dataset_summary else 0
    unique_images = dataset_summary["num_images"] if dataset_summary else 0
    female_count = raw_dataset_summary["classes"]["female"] if raw_dataset_summary else 0
    male_count = raw_dataset_summary["classes"]["male"] if raw_dataset_summary else 0
    best_accuracy = f'{best_metrics["test_accuracy"]:.4f}' if best_metrics else "N/D"
    best_auc = f'{best_metrics["test_auc"]:.4f}' if best_metrics else "N/D"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Dataset crudo", f"{raw_total_images}", "Imagenes originales detectadas antes de limpiar duplicados.")
    with col2:
        render_metric_card("Dataset unico", f"{unique_images}", "Imagenes retenidas despues de deduplicacion por huella visual.")
    with col3:
        render_metric_card("Test Accuracy", best_accuracy, "Rendimiento del mejor experimento.")
    with col4:
        render_metric_card("Test AUC", best_auc, "Capacidad de separacion entre clases.")

    st.write("")

    overview_col, goals_col = st.columns([1.15, 0.85], gap="large")
    with overview_col:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Contexto del dataset</h3>
                <p>
                    El laboratorio se desarrollo con el conjunto <strong>Male and Female Faces Dataset</strong>,
                    un dataset de rostros humanos con dos clases binarias. En bruto se detectaron <strong>{raw_total_images}</strong> imagenes
                    ({female_count} female, {male_count} male), pero tras la auditoria y deduplicacion estricta se trabajaron
                    <strong>{unique_images}</strong> ejemplos unicos para evitar fuga entre entrenamiento, validacion y prueba.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mosaic_path = OUTPUTS_DIR / "dataset_mosaic.png"
        if mosaic_path.exists():
            st.image(
                str(mosaic_path),
                caption="Mosaico del dataset usado en el laboratorio.",
                use_container_width=True,
            )

        if raw_dataset_summary:
            jpeg_ratio = 100 * raw_dataset_summary["formats"]["JPEG"] / raw_dataset_summary["num_images"]
            st.markdown(
                f"""
                <div class="chip-row">
                    <div class="chip">JPEG predominante: {jpeg_ratio:.1f}%</div>
                    <div class="chip">Ancho promedio: {raw_dataset_summary["width"]["mean"]:.2f}px</div>
                    <div class="chip">Alto promedio: {raw_dataset_summary["height"]["mean"]:.2f}px</div>
                    <div class="chip">Tamano maximo: {raw_dataset_summary["width"]["max"]}x{raw_dataset_summary["height"]["max"]}</div>
                    <div class="chip">Duplicados detectados: {raw_total_images - unique_images}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with goals_col:
        st.markdown(
            """
            <div class="info-card">
                <h3>Objetivos del taller</h3>
                <ul>
                    <li>Construir y entrenar una CNN desde cero sin modelos preentrenados.</li>
                    <li>Aplicar un flujo consistente de preprocesamiento, limpieza y particion del dataset.</li>
                    <li>Comparar hiperparametros para analizar su impacto en el desempeno.</li>
                    <li>Explicar decisiones del modelo con Saliency Map y Grad-CAM.</li>
                    <li>Desplegar una app interactiva en Streamlit para demostrar el modelo.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if split_summary:
            split_chips = []
            for split_name in ("train", "val", "test"):
                split = split_summary[split_name]
                split_chips.append(
                    f'<div class="chip">{split_name.upper()}: {split["count"]} imagenes</div>'
                )
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>Particion del dataset limpio</h3>
                    <div class="chip-row">{"".join(split_chips)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    render_section_header(
        "Preprocesamiento",
        "Simulacion grafica del flujo de entrada a la CNN",
        "Este bloque explica de forma visual como entra una imagen al sistema, como se transforma antes del modelo y por que esas decisiones fueron necesarias.",
    )
    render_process_simulation()

    process_col, rationale_col = st.columns([1.15, 0.85], gap="large")
    with process_col:
        st.markdown(
            """
            <div class="note-box">
                La consistencia de tamano y color es clave: sin resize, la red recibiria entradas incompatibles;
                sin normalizacion, el entrenamiento seria menos estable; y sin un encuadre razonable del rostro,
                el modelo puede empezar a depender de fondo, ropa o composicion de la escena. En esta version del laboratorio
                tambien se elimino fuga entre splits deduplicando antes de entrenar.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rationale_col:
        st.markdown(
            """
            <div class="info-card">
                <h3>Flujo aplicado</h3>
                <ul>
                    <li>Lectura y conversion a RGB.</li>
                    <li>Recorte opcional para reducir ruido de fondo en la demo.</li>
                    <li>Redimensionamiento uniforme a 224x224.</li>
                    <li>Normalizacion al rango [0, 1].</li>
                    <li>Conversion a tensor para inferencia y entrenamiento.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    render_section_header(
        "Arquitectura CNN",
        "Simulacion visual de como la imagen atraviesa la red",
        "La simulacion principal de esta seccion fue desarrollada con Manim en Python. Muestra en movimiento el recorrido del tensor y conserva las especificaciones reales de la CNN final.",
    )
    best_row = None
    if best_metrics and not comparison_frame.empty:
        selected = comparison_frame[comparison_frame["name"] == best_metrics["best_experiment"]]
        if not selected.empty:
            best_row = selected.iloc[0]
    render_architecture_animation(best_row)

    st.write("")
    render_section_header(
        "Resultados",
        "Desempeno, hiperparametros y hallazgos del entrenamiento",
        "Aqui se muestran las metricas reales obtenidas en las corridas del laboratorio, la comparacion de configuraciones y una lectura critica de los resultados.",
    )

    best_experiment_name = best_metrics["best_experiment"] if best_metrics else "N/D"
    best_experiment_metrics = (
        read_json_file(OUTPUTS_DIR / "experiments" / best_experiment_name / "metrics.json")
        if best_metrics
        else None
    )

    result_col_1, result_col_2, result_col_3 = st.columns(3)
    with result_col_1:
        render_metric_card(
            "Val Accuracy",
            f'{best_experiment_metrics["val_accuracy"]:.4f}' if best_experiment_metrics else "N/D",
            "Exactitud en validacion del experimento ganador.",
        )
    with result_col_2:
        render_metric_card(
            "Test Loss",
            f'{best_metrics["test_loss"]:.4f}' if best_metrics else "N/D",
            "Perdida medida en el conjunto de prueba.",
        )
    with result_col_3:
        render_metric_card(
            "Input",
            "224x224x3",
            "Forma fija utilizada durante entrenamiento e inferencia.",
        )

    left_results, right_results = st.columns([1.08, 0.92], gap="large")
    with left_results:
        curves_path = OUTPUTS_DIR / "experiments" / best_experiment_name / "training_curves.png"
        if curves_path.exists():
            st.image(
                str(curves_path),
                caption="Curvas de entrenamiento y validacion del mejor experimento.",
                use_container_width=True,
            )
    with right_results:
        interpretability_path = OUTPUTS_DIR / "interpretability" / "correct_prediction_panel.png"
        if interpretability_path.exists():
            st.image(
                str(interpretability_path),
                caption="Prediccion correcta con Saliency Map y Grad-CAM.",
                use_container_width=True,
            )

    st.markdown('<div class="divider-title">Comparacion de hiperparametros</div>', unsafe_allow_html=True)
    if not comparison_frame.empty:
        display_frame = comparison_frame[
            [
                "name",
                "base_filters",
                "kernel_size",
                "dropout",
                "learning_rate",
                "test_accuracy",
                "test_auc",
                "test_loss",
            ]
        ].copy()
        display_frame.columns = [
            "Experimento",
            "Filtros base",
            "Kernel",
            "Dropout",
            "Learning rate",
            "Test accuracy",
            "Test AUC",
            "Test loss",
        ]
        for column in ["Test accuracy", "Test AUC", "Test loss", "Learning rate", "Dropout"]:
            display_frame[column] = display_frame[column].map(lambda value: f"{value:.4f}")
        for column in ["Filtros base", "Kernel"]:
            display_frame[column] = display_frame[column].map(lambda value: f"{int(value)}")
        render_html_table(display_frame)


def render_demo_tab(
    demo_metrics: dict | None,
    crop_strategy: str,
    inference_mode: str,
    experiment_registry: dict[str, dict],
) -> None:
    st.markdown(
        """
        <div class="demo-shell">
            <div class="demo-header">
                <div>
                    <h3>Laboratorio interactivo</h3>
                    <p>
                        Sube una imagen, ejecuta la inferencia y compara como cambia la explicacion visual cuando
                        el modelo usa toda la escena o una region mas centrada en el rostro.
                    </p>
                </div>
                <div class="demo-badge">XAI en tiempo real</div>
            </div>
            <div class="note-box">
                El modelo fue entrenado con rostros bastante centrados. Si subes una foto con mucho fondo o cuerpo completo,
                prueba comparar la imagen completa contra el recorte de rostro o el recorte retrato.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Sube una imagen de rostro",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Sube una imagen para generar la prediccion y las visualizaciones.")
        return

    if inference_mode == "Modelo individual":
        model_path = Path(demo_metrics["model_path"]) if demo_metrics and demo_metrics.get("model_path") else MODELS_DIR / "model.keras"
    else:
        model_path = MODELS_DIR / "model.keras"
    if inference_mode == "Modelo individual" and not model_path.exists():
        st.error(
            "No se encontro el archivo del modelo seleccionado. Ejecuta primero `python scripts/train.py` "
            "o cambia el modelo de demo en la barra lateral."
        )
        return

    image = Image.open(uploaded_file).convert("RGB")
    view_map = build_demo_views(image)

    if crop_strategy == "Auto: detectar rostro" and not view_map["auto_found"]:
        st.warning(
            "No pude detectar un rostro claro. La prediccion se hara sobre la imagen completa."
        )

    selected_view_key = {
        "Auto: detectar rostro": "auto",
        "Imagen completa": "full",
        "Recorte retrato": "portrait",
    }[crop_strategy]
    selected_view = view_map[selected_view_key]

    demo_model_name = demo_metrics.get("name", demo_metrics.get("best_experiment", "modelo")) if demo_metrics else "modelo"
    if inference_mode == "Modelo individual":
        if demo_model_name in {"compact_k5_dropout", "wider_k5_dropout"} and crop_strategy == "Recorte retrato":
            st.warning(
                "Este modelo de demo suele funcionar mejor con `Auto: detectar rostro` o `Imagen completa` "
                "que con `Recorte retrato`."
            )
    if crop_strategy == "Recorte retrato":
        st.caption("Usando un recorte heuristico centrado en la zona superior de la imagen.")

    if inference_mode == "Robusto (ensemble)":
        prediction, anchor_vote, votes = run_robust_inference(experiment_registry, view_map)
    else:
        if not demo_metrics:
            st.error("No hay metricas disponibles para el modelo individual seleccionado.")
            return
        prediction, anchor_vote = run_single_inference(demo_metrics, selected_view)
        votes = [anchor_vote]

    image_batch = anchor_vote["image_batch"]
    image_for_model = selected_view["image"] if inference_mode == "Modelo individual" else view_map[anchor_vote["view_key"]]["image"]
    selected_display_view = selected_view if inference_mode == "Modelo individual" else view_map[anchor_vote["view_key"]]
    saliency = compute_saliency_map(anchor_vote["model"], image_batch, prediction["predicted_index"])
    gradcam = make_gradcam_heatmap(anchor_vote["model"], image_batch, prediction["predicted_index"])

    saliency_overlay = overlay_map_on_image(image_batch, saliency, colormap="magma")
    gradcam_overlay = overlay_map_on_image(image_batch, gradcam, colormap="viridis")

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Clase predicha", prediction["predicted_label"])
    metric_col_2.metric("Probabilidad female", f'{prediction["female_probability"]:.2%}')
    metric_col_3.metric("Probabilidad male", f'{prediction["male_probability"]:.2%}')

    if inference_mode == "Robusto (ensemble)":
        metric_name = "robusto_ensemble"
        st.caption(
            f'Modelo cargado: {metric_name} | '
            "Combina `face_crop_k5`, `compact_k5_dropout` y `wider_k5_dropout` "
            "sobre multiples encuadres."
        )
    elif demo_metrics:
        metric_name = demo_metrics.get("name", demo_metrics.get("best_experiment", "N/D"))
        st.caption(
            f'Modelo cargado: {metric_name} | '
            f'Test accuracy: {demo_metrics["test_accuracy"]:.4f} | '
            f'Test AUC: {demo_metrics["test_auc"]:.4f}'
        )

    st.caption(f'Region usada para inferencia: {selected_display_view["method_label"]}.')
    if inference_mode == "Robusto (ensemble)":
        vote_summary = " | ".join(
            f'{vote["model_name"]}:{vote["view_label"]}={vote["prediction"]["predicted_label"]} '
            f'({vote["prediction"]["confidence"]:.1%})'
            for vote in votes
        )
        st.caption(f"Votos del ensemble: {vote_summary}")

    col1, col2, col3, col4 = st.columns(4)
    col1.image(
        selected_display_view["boxed_image"].resize(IMAGE_SIZE),
        caption="Imagen original",
        use_container_width=True,
    )
    col2.image(
        image_for_model.resize(IMAGE_SIZE),
        caption="Region usada por el modelo",
        use_container_width=True,
    )
    col3.image(saliency_overlay, caption="Saliency Map", use_container_width=True)
    col4.image(gradcam_overlay, caption="Grad-CAM", use_container_width=True)

    st.markdown("### Interpretacion sugerida")
    st.write(
        "Compara el Saliency Map con Grad-CAM para identificar si la red se enfoca en rasgos "
        "faciales relevantes. Si la imagen original tiene mucho fondo, compara los resultados entre "
        "imagen completa, auto-deteccion de rostro y recorte retrato."
    )


def main() -> None:
    st.sidebar.header("Configuracion")
    st.sidebar.write(f"Tamano de entrada: `{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}`")
    st.sidebar.write("Clases: `female`, `male`")
    best_metrics = read_json_file(OUTPUTS_DIR / "experiments" / "best_experiment.json")
    experiment_registry = load_experiment_registry(
        str(OUTPUTS_DIR / "experiments"),
        get_experiment_signature(OUTPUTS_DIR / "experiments"),
    )
    experiment_options = list(experiment_registry.keys())
    default_experiment_name = (
        best_metrics["best_experiment"]
        if best_metrics and best_metrics.get("best_experiment") in experiment_registry
        else (experiment_options[0] if experiment_options else None)
    )
    selected_experiment_name = st.sidebar.selectbox(
        "Modelo para demo",
        experiment_options,
        index=experiment_options.index(default_experiment_name) if default_experiment_name else 0,
        key="demo_model_selector_v2",
    ) if experiment_options else None
    demo_metrics = experiment_registry.get(selected_experiment_name) if selected_experiment_name else None
    inference_mode = st.sidebar.selectbox(
        "Modo de inferencia",
        ("Robusto (ensemble)", "Modelo individual"),
        index=0,
        key="demo_inference_mode_v1",
    )

    default_crop_index = 0
    if demo_metrics and demo_metrics.get("name") == "compact_k5_dropout":
        default_crop_index = 2
    crop_strategy = st.sidebar.selectbox(
        "Region para inferencia",
        (
            "Auto: detectar rostro",
            "Imagen completa",
            "Recorte retrato",
        ),
        index=default_crop_index,
        key="demo_crop_strategy_v2",
    )

    raw_dataset_summary = read_json_file(OUTPUTS_DIR / "dataset_summary_raw.json")
    dataset_summary = read_json_file(OUTPUTS_DIR / "dataset_summary.json")
    split_summary = read_json_file(OUTPUTS_DIR / "split_summary.json")
    comparison_frame = read_experiment_table(OUTPUTS_DIR / "experiments" / "comparison.csv")

    render_hero(raw_dataset_summary, dataset_summary, best_metrics)

    if best_metrics:
        st.sidebar.markdown("### Mejor experimento")
        st.sidebar.write(f'Nombre: `{best_metrics["best_experiment"]}`')
        st.sidebar.write(f'Accuracy test: `{best_metrics["test_accuracy"]:.4f}`')
        st.sidebar.write(f'AUC test: `{best_metrics["test_auc"]:.4f}`')
    if demo_metrics:
        st.sidebar.markdown("### Modelo activo en demo")
        if inference_mode == "Robusto (ensemble)":
            st.sidebar.write("Nombre: `robusto_ensemble`")
            st.sidebar.write("Base: combinacion de checkpoints disponibles en el despliegue.")
            st.sidebar.write("Objetivo: reducir errores extremos en imagenes externas.")
        else:
            st.sidebar.write(f'Nombre: `{demo_metrics.get("name", "N/D")}`')
            st.sidebar.write(f'Accuracy test: `{demo_metrics["test_accuracy"]:.4f}`')
            st.sidebar.write(f'AUC test: `{demo_metrics["test_auc"]:.4f}`')

    report_tab, demo_tab = st.tabs(["Informe del laboratorio", "Demo interactiva"])

    with report_tab:
        render_report_tab(raw_dataset_summary, dataset_summary, split_summary, best_metrics, comparison_frame)

    with demo_tab:
        render_demo_tab(demo_metrics, crop_strategy, inference_mode, experiment_registry)


if __name__ == "__main__":
    main()
