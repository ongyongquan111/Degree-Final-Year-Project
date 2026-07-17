# Student Name: Ong Yong Quan
# Student ID: 243UT246XG
# FYP Title: Identifying Deepfake Images
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import io
import zipfile
import time
import shutil
import datetime
import gc
import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import threading
import ctypes
from ctypes import wintypes
import joblib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from safetensors.torch import load_file as load_safetensors
import re

# Fix matplotlib for Flask
plt.switch_backend('Agg')
sns.set_style("whitegrid")

# --------------------------
# Folder setup
# --------------------------
ROOT_FOLDER = os.getcwd()
MODELS_FOLDER = os.path.join(ROOT_FOLDER, "models")
OUTPUTS_FOLDER = os.path.join(ROOT_FOLDER, "outputs")
UPLOADS_FOLDER = os.path.join(ROOT_FOLDER, "uploads")
RESOURCES_STATISTIC_FOLDER = os.path.join(ROOT_FOLDER, "resources_statistic_result")

for folder in [MODELS_FOLDER, OUTPUTS_FOLDER, UPLOADS_FOLDER, RESOURCES_STATISTIC_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --------------------------
# Output cleanup configs
# --------------------------
# Used to store temporary result files, plots, and uploaded previews.
# It is safe to clear it, this is necessary because I encounter output not showup correctly.
OUTPUT_CLEANUP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif",
    ".csv", ".json", ".txt"
)
OUTPUT_CLEANUP_INTERVAL_SECONDS = 15 * 60     # periodic cleanup every 15 minutes
OUTPUT_MAX_FILE_AGE_SECONDS = 6 * 60 * 60     # delete files older than 6 hours
OUTPUT_MAX_FILES = 2000                       # large batches create many evidence files, so set it higher.
LAST_OUTPUT_CLEANUP_TS = 0

# --------------------------
# Safe file handling
# --------------------------
def sanitize_filename(filename, default="file"):
    """Return a safe filename for saving uploaded/generated files."""
    filename = os.path.basename(str(filename or default))
    safe = secure_filename(filename)
    return safe if safe else default


def make_batch_filename(original_name, index=None, prefix=""):
    """Create unique safe output filename while preserving original label words for metrics/debugging."""
    safe = sanitize_filename(original_name, default="file")
    if index is not None:
        safe = f"{int(index) + 1:04d}_{safe}"
    if prefix:
        safe = f"{prefix}_{safe}"
    return safe


def safe_extract_zip(zip_file_stream, extract_to):
    """
    Safely extract ZIP files while blocking path traversal entries.
    Returns the number of extracted files.
    """
    extracted = 0
    extract_root = os.path.abspath(extract_to)

    # Werkzeug provide spooled upload stream without ``seekable()``,
    # while zipfile expects complete seekable file object. Buffering the
    # uploaded ZIP can keeps extraction independent from request-stream state.
    if hasattr(zip_file_stream, "read"):
        try:
            zip_file_stream.seek(0)
        except (AttributeError, OSError):
            pass
        zip_source = io.BytesIO(zip_file_stream.read())
    else:
        zip_source = zip_file_stream

    with zipfile.ZipFile(zip_source) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue

            member_name = member.filename.replace("\\", "/")
            parts = [p for p in member_name.split("/") if p]

            if not parts or any(part == ".." for part in parts):
                print(f"Skipped unsafe ZIP entry: {member.filename}")
                continue

            target_path = os.path.abspath(os.path.join(extract_to, *parts))

            if not target_path.startswith(extract_root + os.sep):
                print(f"Skipped unsafe ZIP entry: {member.filename}")
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with zip_file.open(member, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted += 1

    return extracted


# --------------------------
# Large batch performance configs
# --------------------------
# Grad-CAM is the slowest XAI step. If use large batches, keep Grad-CAM for the first N images only.
MAX_GRADCAM_BATCH_IMAGES = 20
MAX_UPLOAD_MB = 1024

# --------------------------
# Global Variables
# --------------------------
# FYP2 core architecture: one lightweight EfficientNet-B0 CNN plus an
# validation-trained forensic calibration layer. This is not multi-CNN ensemble...
ACTIVE_MODELS = {
    "efficientnet": None
}
ACTIVE_MODEL_NAMES = {
    "efficientnet": None
}
ACTIVE_MODEL_SHA256 = None
ACTIVE_CALIBRATOR_BUNDLE = None
FORENSIC_CALIBRATOR_PATH = os.path.join(MODELS_FOLDER, "efficientnet_b0_forensic_calibrator.joblib")
SELECTED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Server-side result cache
# --------------------------
# Flask default session is cookie-based and too small for large batch results.
# Store large result objects on the server and keep only a small key in the session.
RESULT_CACHE = {}
RESULT_CACHE_MAX_ITEMS = 10

def cache_result_data(data):
    key = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    RESULT_CACHE[key] = data

    # Keep that newest cached result objects.
    if len(RESULT_CACHE) > RESULT_CACHE_MAX_ITEMS:
        oldest_keys = sorted(RESULT_CACHE.keys())[:-RESULT_CACHE_MAX_ITEMS]
        for old_key in oldest_keys:
            RESULT_CACHE.pop(old_key, None)

    return key


def get_cached_result_data(key):
    if not key:
        return None
    return RESULT_CACHE.get(key)


# --------------------------
# Image size config
# --------------------------
IMAGE_SIZE = (256, 256)  # Global

# --------------------------
# EfficientNet-B0 config
# --------------------------
# FYP2 uses one custom-trained EfficientNet-B0 CNN.
# DFT, Canny, Laplacian sharpness and noise-residual statistics will be feed to calibrator.
# Grad-CAM remains XAI only.
EFFICIENTNET_INPUT_SIZE = (224, 224)
EFFICIENTNET_VAL_MEAN = (0.485, 0.456, 0.406)
EFFICIENTNET_VAL_STD = (0.229, 0.224, 0.225)
EFFICIENTNET_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(EFFICIENTNET_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=list(EFFICIENTNET_VAL_MEAN),
        std=list(EFFICIENTNET_VAL_STD),
    ),
])

# Threshold modes control REAL / UNCERTAIN / FAKE decision range.
THRESHOLD_MODES = {
    "lenient": {
        "label": "Lenient",
        "real": 0.35,
        "fake": 0.65,
        "description": "Smoother prediction answer and fewer UNCERTAIN results."
    },
    "standard": {
        "label": "Standard",
        "real": 0.30,
        "fake": 0.70,
        "description": "Balanced mode for general deepfake detection."
    },
    "strict": {
        "label": "Strict",
        "real": 0.20,
        "fake": 0.80,
        "description": "Strict mode can lead to more UNCERTAIN results."
    }
}

CURRENT_THRESHOLD_MODE = "standard"
REAL_THRESHOLD = THRESHOLD_MODES[CURRENT_THRESHOLD_MODE]["real"]
FAKE_THRESHOLD = THRESHOLD_MODES[CURRENT_THRESHOLD_MODE]["fake"]
FORCE_BINARY_COMPARISON = False
GRADCAM_ALPHA = 0.45


def get_threshold_config():
    return THRESHOLD_MODES.get(CURRENT_THRESHOLD_MODE, THRESHOLD_MODES["standard"])


def set_threshold_mode(mode_name):
    global CURRENT_THRESHOLD_MODE, REAL_THRESHOLD, FAKE_THRESHOLD
    if mode_name not in THRESHOLD_MODES:
        return False
    CURRENT_THRESHOLD_MODE = mode_name
    REAL_THRESHOLD = THRESHOLD_MODES[mode_name]["real"]
    FAKE_THRESHOLD = THRESHOLD_MODES[mode_name]["fake"]
    return True


def probability_to_output_state(prob_fake):
    """Apply either the normal three-state decision or optional binary comparison mode."""
    prob_fake = float(prob_fake)
    if FORCE_BINARY_COMPARISON:
        prediction = "FAKE" if prob_fake >= 0.5 else "REAL"
        confidence = round(max(prob_fake, 1.0 - prob_fake) * 100, 2)
        return prediction, confidence
    return probability_to_three_state(prob_fake)


def get_active_decision_label():
    if FORCE_BINARY_COMPARISON:
        return "Forced REAL/FAKE (0.50 threshold)"
    return get_threshold_config()["label"]

# --------------------------
# Supported Models File Types
# --------------------------
SUPPORTED_MODELS = {
    ".pt": {
        "loader": lambda x: torch.load(x, map_location="cpu", weights_only=False),
        "post_processor": None
    },
    ".pth": {
        "loader": lambda x: torch.load(x, map_location="cpu", weights_only=False),
        "post_processor": None
    },
    ".safetensors": {
        "loader": lambda x: load_safetensors(x, device="cpu"),
        "post_processor": None
    }
}

# --------------------------
# Image Preprocessing Functions
# --------------------------
def preprocess_rgb_image(img_array, target_size=IMAGE_SIZE, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), apply_normalize=True):
    """
    RGB preprocessing used by the FYP2 custom-trained EfficientNet-B0 model.
    Converts BGR to RGB, resizes the image, converts it to tensor, and applies normalization.
    """
    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        img_pil = Image.fromarray(img_resized)

        steps = [transforms.ToTensor()]
        if apply_normalize:
            steps.append(transforms.Normalize(mean=list(mean), std=list(std)))
        transform_steps = transforms.Compose(steps)

        img_tensor = transform_steps(img_pil).unsqueeze(0)
        return img_tensor.to(SELECTED_DEVICE)
    except Exception as e:
        print(f"RGB preprocessing error: {e}")
        return None



def extract_canny_edges(img_array):
    """Extract Canny edges for visualization"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return canny

def extract_dft_magnitude(img_array):
    """Extract DFT magnitude for visualization"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
    magnitude_spectrum_view = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude_spectrum_view

# --------------------------
# Build Model Channel
# --------------------------
def build_efficientnet_b0_3channel():
    """
    Core model: EfficientNet-B0 with single binary logit, still using sigmoid, 2 states.

    Training label:
      0 = REAL
      1 = FAKE

    Preview Output:
      REAL / FAKE / UNCERTAIN based on probability thresholds.
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def preprocess_efficientnet_image(img_array):
    """Match the FYP2 notebook validation preprocessing exactly."""
    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return EFFICIENTNET_VAL_TRANSFORM(img_pil).unsqueeze(0).to(SELECTED_DEVICE)
    except Exception as e:
        print(f"EfficientNet preprocessing error: {e}")
        return None


def probability_to_three_state(prob_fake):
    """Convert fake probability into a state and model score.

    The score is the model's strongest class probability, not the probability
    that the prediction is correct. A wrong prediction can therefore still
    have a high score when the model is overconfident.
    """
    prob_fake = float(prob_fake)
    if prob_fake >= FAKE_THRESHOLD:
        prediction = "FAKE"
        confidence = round(prob_fake * 100, 2)
    elif prob_fake <= REAL_THRESHOLD:
        prediction = "REAL"
        confidence = round((1.0 - prob_fake) * 100, 2)
    else:
        prediction = "UNCERTAIN"
        # No class is selected, so report the strongest class score instead of
        # the old inverted uncertainty-band formula.
        confidence = round(max(prob_fake, 1.0 - prob_fake) * 100, 2)
    return prediction, confidence


def infer_efficientnet(model, rgb_tensor):
    """Run EfficientNet-B0 inference and return 3-state output."""
    with torch.no_grad():
        try:
            logit = model(rgb_tensor)
            if isinstance(logit, (tuple, list)):
                logit = logit[0]
            prob_fake = torch.sigmoid(logit).view(-1)[0].item()
            prediction, confidence = probability_to_three_state(prob_fake)
        except Exception as e:
            print(f"Error in EfficientNet inference: {e}")
            prob_fake = 0.5
            prediction = "UNCERTAIN"
            confidence = 50.0
    return prediction, confidence, prob_fake * 100.0


class EfficientNetGradCAM:
    """Simple Grad-CAM for torchvision EfficientNet-B0."""
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hooks = []
        try:
            target_layer = self.model.features[-1]
            self.hooks.append(target_layer.register_forward_hook(self._forward_hook))
            self.hooks.append(target_layer.register_full_backward_hook(self._backward_hook))
        except Exception as e:
            print(f"Grad-CAM hook setup failed: {e}")

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, original_bgr):
        try:
            self.model.zero_grad(set_to_none=True)
            output = self.model(input_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            score = output.view(-1)[0]
            score.backward()

            if self.activations is None or self.gradients is None:
                return None

            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze()
            cam = torch.relu(cam)
            cam = cam.detach().cpu().numpy()
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            cam = cv2.resize(cam, (original_bgr.shape[1], original_bgr.shape[0]))
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_bgr, 1 - GRADCAM_ALPHA, heatmap, GRADCAM_ALPHA, 0)
            return overlay
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return None
        finally:
            try:
                self.model.zero_grad(set_to_none=True)
            except Exception:
                pass

    def close(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass


def extract_noise_residual(img_array):
    """Noise residual visualization for XAI/forensic explanation."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, blur)
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return residual


def calculate_forensic_scores(img_array):
    """
    Lightweight forensic statistics used for explanation and calibrated fusion.
    They do not add a second neural model or a second CNN forward pass.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, IMAGE_SIZE).astype(np.float32)

        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        mag = np.log1p(np.abs(fft_shift))
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        low = float(np.mean(mag[dist <= 32]))
        high = float(np.mean(mag[dist > 80]))
        high_low_ratio = high / (low + 1e-8)

        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density = float(np.mean(edges > 0))

        # This is Laplacian variance: a local sharpness/focus measure.  Keep the
        # legacy bundle key ``blur_score`` so the saved calibrator remains valid.
        laplacian_variance = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
        denoised = cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0)
        residual = gray - denoised.astype(np.float32)
        noise_std = float(np.std(residual))

        return {
            "high_low_frequency_ratio": round(high_low_ratio, 4),
            "edge_density": round(edge_density, 4),
            "blur_score": round(laplacian_variance, 2),
            "noise_residual_std": round(noise_std, 4)
        }
    except Exception as e:
        print(f"Forensic score error: {e}")
        return {}


def load_matching_forensic_calibrator(model_sha256):
    """Load the validation-trained calibrator only for its matching CNN checkpoint."""
    global ACTIVE_CALIBRATOR_BUNDLE
    ACTIVE_CALIBRATOR_BUNDLE = None
    if not os.path.isfile(FORENSIC_CALIBRATOR_PATH):
        return False, "forensic calibrator not found; using raw EfficientNet probability"
    try:
        bundle = joblib.load(FORENSIC_CALIBRATOR_PATH)
        if not isinstance(bundle, dict) or not hasattr(bundle.get("estimator"), "predict_proba"):
            return False, "forensic calibrator format is invalid; using raw EfficientNet probability"
        if bundle.get("cnn_checkpoint_sha256") != model_sha256:
            return False, "forensic calibrator does not match this checkpoint; using raw EfficientNet probability"
        feature_names = bundle.get("feature_names")
        supported_features = {
            "fake_probability_logit",
            "high_low_frequency_ratio",
            "edge_density",
            "log_blur_score",
            "noise_residual_std",
        }
        if not feature_names or not set(feature_names).issubset(supported_features):
            return False, "forensic calibrator has unsupported features; using raw EfficientNet probability"
        ACTIVE_CALIBRATOR_BUNDLE = bundle
        return True, f"forensic calibration active ({bundle.get('selected_method', 'validated fusion')})"
    except Exception as exc:
        print(f"Forensic calibrator load error: {exc}")
        return False, "forensic calibrator could not be loaded; using raw EfficientNet probability"


def calibrate_fake_probability(raw_probability, forensic_scores):
    """Fuse the CNN score with lightweight forensic statistics when available."""
    raw_probability = float(np.clip(raw_probability, 1e-5, 1.0 - 1e-5))
    bundle = ACTIVE_CALIBRATOR_BUNDLE
    if not bundle or not forensic_scores:
        return raw_probability, False
    feature_values = {
        "fake_probability_logit": float(np.log(raw_probability / (1.0 - raw_probability))),
        "high_low_frequency_ratio": float(forensic_scores.get("high_low_frequency_ratio", np.nan)),
        "edge_density": float(forensic_scores.get("edge_density", np.nan)),
        "log_blur_score": float(np.log1p(max(0.0, forensic_scores.get("blur_score", np.nan)))),
        "noise_residual_std": float(forensic_scores.get("noise_residual_std", np.nan)),
    }
    feature_names = bundle["feature_names"]
    values = [feature_values[name] for name in feature_names]
    if not np.isfinite(values).all():
        return raw_probability, False
    try:
        feature_frame = pd.DataFrame([values], columns=feature_names)
        calibrated = float(bundle["estimator"].predict_proba(feature_frame)[0, 1])
        return float(np.clip(calibrated, 0.0, 1.0)), True
    except Exception as exc:
        print(f"Forensic calibration error: {exc}")
        return raw_probability, False


def build_text_explanation_legacy(prediction, prob_fake_percent, forensic_scores):
    """Generate human-readable XAI explanation."""
    reasons = []
    if prediction == "FAKE":
        reasons.append("EfficientNet-B0 produced a high fake probability.")
    elif prediction == "REAL":
        reasons.append("EfficientNet-B0 produced a low fake probability, so the image is closer to the learned real-image distribution.")
    else:
        reasons.append("The fake probability falls inside the uncertainty threshold, so the system avoids forcing a binary decision.")

    if forensic_scores:
        ratio = forensic_scores.get("high_low_frequency_ratio", 0)
        edge_density = forensic_scores.get("edge_density", 0)
        blur_score = forensic_scores.get("blur_score", 0)
        noise_std = forensic_scores.get("noise_residual_std", 0)

        if ratio < 0.55:
            reasons.append("Frequency analysis shows relatively weak high-frequency content, which can appear in generated or over-smoothed images.")
        elif ratio > 0.85:
            reasons.append("Frequency distribution retains stronger high-frequency content, which is more consistent with natural image details.")
        else:
            reasons.append("Frequency evidence is moderate and not decisive by itself.")

        if edge_density > 0.12:
            reasons.append("Canny evidence shows many edges or possible boundary transitions.")
        elif edge_density < 0.035:
            reasons.append("Canny evidence shows low edge density, which may indicate smoothing or low-detail regions.")
        else:
            reasons.append("Edge density is within a normal middle range.")

        if blur_score < 40:
            reasons.append("The Laplacian sharpness score is low, so the image may be smooth or compressed.")
        if noise_std < 2.0:
            reasons.append("Noise residual is weak, which may indicate denoising or synthetic smoothing.")

    cfg = get_threshold_config()
    reasons.append(
        f"Threshold mode: {cfg['label']} "
        f"(REAL ≤ {int(cfg['real'] * 100)}%, FAKE ≥ {int(cfg['fake'] * 100)}%)."
    )
    reasons.append("Grad-CAM highlights the image regions that most influenced the EfficientNet-B0 decision.")
    return " ".join(reasons)


def build_short_explanation_legacy(prediction, prob_fake_percent, forensic_scores):
    """Short explanation for compact batch table view."""
    cfg = get_threshold_config()
    if prediction == "FAKE":
        decision = "High fake probability."
    elif prediction == "REAL":
        decision = "Low fake probability."
    else:
        decision = "Falls within the uncertainty range."

    return (
        f"{decision} Mode: {cfg['label']} "
        f"(REAL ≤ {int(cfg['real'] * 100)}%, FAKE ≥ {int(cfg['fake'] * 100)}%). "
        "See XAI preview for Grad-CAM, DFT, Canny, and noise evidence."
    )

# Override the simple explanation builders above with more decision-specific wording.
def build_text_explanation(
    prediction, prob_fake_percent, forensic_scores,
    raw_prob_fake_percent=None, calibration_applied=False,
):
    """Generate a decision-specific XAI explanation."""
    cfg = get_threshold_config()
    real_threshold = 50 if FORCE_BINARY_COMPARISON else int(cfg["real"] * 100)
    fake_threshold = 50 if FORCE_BINARY_COMPARISON else int(cfg["fake"] * 100)
    prob_fake_percent = float(prob_fake_percent or 0.0)
    reasons = []

    if calibration_applied and raw_prob_fake_percent is not None:
        reasons.append(
            f"The validation-trained forensic calibration adjusted the raw EfficientNet-B0 fake score "
            f"from {float(raw_prob_fake_percent):.2f}% to {prob_fake_percent:.2f}%."
        )

    if prediction == "FAKE":
        distance = prob_fake_percent - fake_threshold
        if distance >= 15:
            strength = "strong"
        elif distance >= 5:
            strength = "moderate"
        else:
            strength = "near-threshold"
        reasons.append(
            f"The detector estimated {prob_fake_percent:.2f}% fake probability, above the {fake_threshold}% FAKE threshold. "
            f"This is a {strength} fake decision."
        )
    elif prediction == "REAL":
        distance = real_threshold - prob_fake_percent
        if distance >= 15:
            strength = "strong"
        elif distance >= 5:
            strength = "moderate"
        else:
            strength = "near-threshold"
        reasons.append(
            f"The detector estimated {prob_fake_percent:.2f}% fake probability, below the {real_threshold}% REAL threshold. "
            f"This is a {strength} real decision."
        )
    else:
        nearest_real_gap = abs(prob_fake_percent - real_threshold)
        nearest_fake_gap = abs(fake_threshold - prob_fake_percent)
        nearer_side = "REAL" if nearest_real_gap < nearest_fake_gap else "FAKE"
        reasons.append(
            f"The detector estimated {prob_fake_percent:.2f}% fake probability, between the REAL and FAKE thresholds. "
            f"It is closer to the {nearer_side} boundary, but not far enough for a confident label."
        )

    if forensic_scores:
        ratio = forensic_scores.get("high_low_frequency_ratio", 0)
        edge_density = forensic_scores.get("edge_density", 0)
        blur_score = forensic_scores.get("blur_score", 0)
        noise_std = forensic_scores.get("noise_residual_std", 0)
        evidence_notes = []

        if ratio < 0.55:
            evidence_notes.append(f"DFT frequency balance is low ({ratio:.2f}), which can indicate smoothing or synthetic-looking texture.")
        elif ratio > 0.85:
            evidence_notes.append(f"DFT frequency balance is high ({ratio:.2f}), showing stronger fine-detail content.")
        else:
            evidence_notes.append(f"DFT frequency balance is mid-range ({ratio:.2f}), so frequency evidence is not decisive by itself.")

        if edge_density > 0.12:
            evidence_notes.append(f"Canny edge density is high ({edge_density:.3f}), highlighting many sharp boundaries or transitions.")
        elif edge_density < 0.035:
            evidence_notes.append(f"Canny edge density is low ({edge_density:.3f}), which can happen with smooth or low-detail regions.")
        else:
            evidence_notes.append(f"Canny edge density is balanced ({edge_density:.3f}).")

        if blur_score < 40:
            evidence_notes.append(f"Laplacian sharpness score is low ({blur_score:.1f}), suggesting softer detail or compression.")
        elif blur_score > 180:
            evidence_notes.append(f"Laplacian sharpness score is high ({blur_score:.1f}), suggesting sharper local detail.")

        if noise_std < 2.0:
            evidence_notes.append(f"Noise residual is weak ({noise_std:.2f}), which may indicate denoising or synthetic smoothing.")
        elif noise_std > 8.0:
            evidence_notes.append(f"Noise residual is stronger ({noise_std:.2f}), showing more local pixel variation.")

        reasons.extend(evidence_notes[:3])

    if FORCE_BINARY_COMPARISON:
        reasons.append("Comparison mode forces a binary label at 0.50, so UNCERTAIN is disabled.")
    else:
        reasons.append(f"Threshold mode: {cfg['label']} (REAL ≤ {real_threshold}%, FAKE ≥ {fake_threshold}%).")
    if calibration_applied:
        reasons.append(
            "Grad-CAM explains the EfficientNet-B0 score; the DFT, Canny, Laplacian sharpness, and noise-residual statistics "
            "are inputs to the lightweight validation-trained calibration layer."
        )
    else:
        reasons.append("Grad-CAM highlights the regions that most influenced the EfficientNet-B0 score; DFT, Canny, and noise views are supporting evidence, not separate model votes.")
    return " ".join(reasons)


def build_short_explanation(
    prediction, prob_fake_percent, forensic_scores,
    raw_prob_fake_percent=None, calibration_applied=False,
):
    """Short explanation for compact batch table view."""
    cfg = get_threshold_config()
    real_threshold = 50 if FORCE_BINARY_COMPARISON else int(cfg["real"] * 100)
    fake_threshold = 50 if FORCE_BINARY_COMPARISON else int(cfg["fake"] * 100)
    prob_fake_percent = float(prob_fake_percent or 0.0)

    if prediction == "FAKE":
        decision = f"{prob_fake_percent:.2f}% fake probability is above the FAKE threshold."
    elif prediction == "REAL":
        decision = f"{prob_fake_percent:.2f}% fake probability is below the REAL threshold."
    else:
        decision = f"{prob_fake_percent:.2f}% fake probability falls inside the uncertainty range."

    forensic_hint = ""
    if forensic_scores:
        ratio = forensic_scores.get("high_low_frequency_ratio", 0)
        edge_density = forensic_scores.get("edge_density", 0)
        if ratio < 0.55:
            forensic_hint = " Low DFT fine-detail balance."
        elif edge_density < 0.035:
            forensic_hint = " Low edge density."
        elif edge_density > 0.12:
            forensic_hint = " High edge density."

    calibration_hint = " Validated forensic calibration is active." if calibration_applied else ""
    mode_text = (
        " Mode: Forced REAL/FAKE at 0.50."
        if FORCE_BINARY_COMPARISON
        else f" Mode: {cfg['label']} (REAL ≤ {real_threshold}%, FAKE ≥ {fake_threshold}%)."
    )
    return f"{decision}{calibration_hint}{forensic_hint}{mode_text} See XAI preview for Grad-CAM, DFT, Canny, and noise evidence."


# --------------------------
# Model Type Detection Function
# --------------------------
def detect_model_type_from_state_dict(state_dict, filename=None):
    """
    Accepts only custom-trained EfficientNet-B0 state dictionaries.
    """
    keys = list(state_dict.keys())

    # Filename hint is only used as supporting evidence.
    if filename:
        filename_lower = filename.lower()
        if not any(h in filename_lower for h in ["efficientnet", "effb0", "b0"]):
            # Still continue key checking, because users may rename the file.
            pass

    efficientnet_keys = [
        "features.0.0",
        "features.1.",
        "features.2.",
        "classifier.1",
        "stochastic_depth"
    ]

    if any(any(ek in key.lower() for ek in efficientnet_keys) for key in keys):
        return "efficientnet"

    return None


def detect_model_type_from_object(obj, filename=None):
    """Detect whether the uploaded file is an EfficientNet-B0 model."""
    if isinstance(obj, dict):
        return detect_model_type_from_state_dict(obj, filename)

    if isinstance(obj, nn.Module):
        module_name = obj.__class__.__name__.lower()
        if "efficientnet" in module_name:
            return "efficientnet"

    return None

# --------------------------
# Load Models with Auto-Detection
# --------------------------
def normalize_loaded_state_dict(loaded_data):
    """
    Accept common PyTorch checkpoint formats and normalize keys for EfficientNet-B0.
    Supported examples:
      - direct state_dict
      - {'state_dict': state_dict}
      - {'model_state_dict': state_dict}
      - keys prefixed with module., model., backbone., or _orig_mod.
    """
    if not isinstance(loaded_data, dict):
        return loaded_data

    for wrapper_key in ("state_dict", "model_state_dict", "model_state", "net_state_dict"):
        value = loaded_data.get(wrapper_key)
        if isinstance(value, dict):
            loaded_data = value
            break

    # Only normalize tensor-like state dictionaries.
    if not loaded_data or not all(isinstance(k, str) for k in loaded_data.keys()):
        return loaded_data

    prefixes = ("module.", "model.", "backbone.", "_orig_mod.")
    normalized = {}
    changed = False

    for key, value in loaded_data.items():
        new_key = key
        keep_checking = True
        while keep_checking:
            keep_checking = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                    keep_checking = True
        normalized[new_key] = value

    return normalized if changed else loaded_data


def release_active_model():
    """Release the currently active model and clear cached GPU memory."""
    global ACTIVE_MODELS, ACTIVE_MODEL_NAMES, ACTIVE_MODEL_SHA256, ACTIVE_CALIBRATOR_BUNDLE

    old_model = ACTIVE_MODELS.get("efficientnet")
    if old_model is not None:
        try:
            old_model.to("cpu")
        except Exception:
            pass
        del old_model

    ACTIVE_MODELS["efficientnet"] = None
    ACTIVE_MODEL_NAMES["efficientnet"] = None
    ACTIVE_MODEL_SHA256 = None
    ACTIVE_CALIBRATOR_BUNDLE = None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_with_auto_detect(model_file_path):
    """
    Only load custom-trained EfficientNet-B0 model.
    Replaces any currently loaded EfficientNet-B0 model after the new model validates.
    Returns: (success: bool, message: str)
    """
    global ACTIVE_MODELS, ACTIVE_MODEL_NAMES, ACTIVE_MODEL_SHA256

    if not os.path.exists(model_file_path):
        return False, f"File not found: {model_file_path}"

    file_ext = os.path.splitext(model_file_path)[1].lower()
    if file_ext not in SUPPORTED_MODELS:
        supported_exts = ", ".join(SUPPORTED_MODELS.keys())
        return False, f"Unsupported file type: {file_ext}. Supported: {supported_exts}"

    try:
        loader = SUPPORTED_MODELS[file_ext]["loader"]
        loaded_data = loader(model_file_path)
        filename = os.path.basename(model_file_path)

        # Before detection and loading, normalize common checkpoint formats and prefixes.
        loaded_data = normalize_loaded_state_dict(loaded_data)

        model_type = detect_model_type_from_object(loaded_data, filename)

        if model_type != "efficientnet":
            return False, (
                "This file is not recognized as a custom-trained EfficientNet-B0 model. "
                "Only accepts custom trained EfficientNet-B0 .pth/.pt/.safetensors model."
            )

        if isinstance(loaded_data, dict):
            model = build_efficientnet_b0_3channel()
            try:
                model.load_state_dict(loaded_data, strict=True)
            except Exception as e:
                return False, f"Failed to load EfficientNet-B0 state dict: {e}"
        else:
            model = loaded_data

        previous_model_name = ACTIVE_MODEL_NAMES.get("efficientnet")
        if ACTIVE_MODELS.get("efficientnet") is not None:
            release_active_model()

        model = model.to(SELECTED_DEVICE).eval()
        model_sha256 = benchmark_sha256(model_file_path)

        ACTIVE_MODELS["efficientnet"] = model
        ACTIVE_MODEL_NAMES["efficientnet"] = filename
        ACTIVE_MODEL_SHA256 = model_sha256
        calibration_active, calibration_message = load_matching_forensic_calibrator(model_sha256)

        if previous_model_name:
            return True, (
                f"Switched model from {previous_model_name} to {filename}; {calibration_message}"
            )
        return True, f"Successfully loaded EfficientNet-B0 model: {filename}; {calibration_message}"

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Failed to load model: {str(e)}"


def find_startup_model_path():
    """Return the newest supported model file from the models folder, if available."""
    candidates = []
    for filename in os.listdir(MODELS_FOLDER):
        model_path = os.path.join(MODELS_FOLDER, filename)
        if not os.path.isfile(model_path):
            continue
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_MODELS:
            continue
        candidates.append(model_path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def auto_load_startup_model():
    """Load one model automatically when the detector starts."""
    if ACTIVE_MODELS.get("efficientnet") is not None:
        return

    model_path = find_startup_model_path()
    if model_path is None:
        return

    load_model_with_auto_detect(model_path)

# --------------------------
# Inference Functions
# --------------------------
# Only infer_efficientnet() is used

# --------------------------
# Single-Model Decision Function
# --------------------------
def single_model_decision(model_prediction):
    """
    FYP2 uses one EfficientNet-B0 decision instead of multi-model fusion.
    No voting or per-model comparison is used.
    """
    if not model_prediction:
        return {
            "final_prediction": "ERROR",
            "final_confidence": 0.0,
            "fake_probability": 0.0,
            "model_count": 0
        }

    pred, conf, prob_fake = model_prediction
    return {
        "final_prediction": pred,
        "final_confidence": conf,
        "fake_probability": prob_fake,
        "model_count": 1
    }

# --------------------------
# Main Processing Function
# --------------------------
def process_image(img_array, enable_gradcam=True):
    """
    FYP2 processing pipeline.

    Kept from FYP1:
      - same Flask app flow
      - same upload/batch/video support
      - same DFT and Canny visual evidence
      - same plotting/statistics functions

    FYP2 Changes:
      - removed multi-model ensemble
      - one EfficientNet-B0 CNN forward pass
      - optional validation-trained forensic calibration layer
      - three-state decision: REAL / FAKE / UNCERTAIN
      - Grad-CAM + XAI text explanation
    """
    model = ACTIVE_MODELS.get("efficientnet")
    if model is None:
        return False, "No EfficientNet-B0 model loaded. Please load the FYP2 model first."

    efficientnet_tensor = preprocess_efficientnet_image(img_array)
    if efficientnet_tensor is None:
        return False, "Failed to preprocess image for EfficientNet-B0."

    # Standard forensic evidence.
    canny_raw = extract_canny_edges(img_array)
    dft_raw = extract_dft_magnitude(img_array)
    noise_raw = extract_noise_residual(img_array)

    # Main one-model score followed by the optional validation-trained,
    # lightweight forensic calibration layer.
    _, _, raw_prob_fake = infer_efficientnet(model, efficientnet_tensor)
    forensic_scores = calculate_forensic_scores(img_array)
    calibrated_probability, calibration_applied = calibrate_fake_probability(
        raw_prob_fake / 100.0, forensic_scores
    )
    pred, conf = probability_to_output_state(calibrated_probability)
    prob_fake = round(calibrated_probability * 100.0, 2)
    raw_prob_fake = round(raw_prob_fake, 2)
    result = single_model_decision((pred, conf, prob_fake))

    # XAI explanation keeps the raw CNN score and calibrated decision distinct.
    explanation = build_text_explanation(
        pred,
        prob_fake,
        forensic_scores,
        raw_prob_fake_percent=raw_prob_fake,
        calibration_applied=calibration_applied,
    )

    # Grad-CAM visual explanation.
    # For large batches, Grad-CAM can be disabled after the preview limit to prevent timeout.
    gradcam_raw = None
    if enable_gradcam:
        try:
            cam = EfficientNetGradCAM(model)
            gradcam_raw = cam.generate(efficientnet_tensor, img_array)
            cam.close()
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

    result["canny_raw"] = canny_raw
    result["dft_raw"] = dft_raw
    result["noise_raw"] = noise_raw
    result["gradcam_raw"] = gradcam_raw
    result["forensic_scores"] = forensic_scores
    result["raw_fake_probability"] = raw_prob_fake
    result["calibration_applied"] = calibration_applied
    result["decision_method"] = (
        "EfficientNet-B0 + validated forensic calibration"
        if calibration_applied else "EfficientNet-B0 raw probability"
    )
    result["threshold_mode"] = "forced_binary" if FORCE_BINARY_COMPARISON else CURRENT_THRESHOLD_MODE
    result["threshold_label"] = get_active_decision_label()
    result["real_threshold_percent"] = 50 if FORCE_BINARY_COMPARISON else int(REAL_THRESHOLD * 100)
    result["fake_threshold_percent"] = 50 if FORCE_BINARY_COMPARISON else int(FAKE_THRESHOLD * 100)
    result["xai_explanation"] = explanation
    result["xai_summary"] = build_short_explanation(
        pred,
        prob_fake,
        forensic_scores,
        raw_prob_fake_percent=raw_prob_fake,
        calibration_applied=calibration_applied,
    )

    return True, result

# --------------------------
# Results Handling
# --------------------------
def calculate_ml_metrics(true_labels, pred_labels):
    """
    Calculate metrics only when ground-truth labels are available.
    Ground truth is inferred from filenames containing REAL or FAKE.
    FAKE is treated as the positive class for precision/recall/F1.
    """
    if len(true_labels) == 0 or len(pred_labels) == 0:
        return {
            "available": False,
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "confusion_matrix": [[0, 0], [0, 0]],
            "labeled_count": 0,
            "message": "Metrics unavailable because no filename contains REAL or FAKE ground-truth labels."
        }

    if len(true_labels) != len(pred_labels):
        return {
            "available": False,
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "confusion_matrix": [[0, 0], [0, 0]],
            "labeled_count": len(true_labels),
            "message": "Metrics unavailable because labeled samples and confident predictions do not match."
        }

    accuracy = round(accuracy_score(true_labels, pred_labels) * 100, 2)
    precision = round(precision_score(true_labels, pred_labels, zero_division=0) * 100, 2)
    recall = round(recall_score(true_labels, pred_labels, zero_division=0) * 100, 2)
    f1 = round(f1_score(true_labels, pred_labels, zero_division=0) * 100, 2)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).tolist()

    return {
        "available": True,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "labeled_count": len(true_labels),
        "message": (
            "FAKE is positive; forced binary mode includes every labelled result."
            if FORCE_BINARY_COMPARISON
            else "FAKE is positive; UNCERTAIN results are excluded."
        )
    }


def calculate_raw_roc_auc(detection_results):
    """Calculate ROC-AUC using only the raw EfficientNet-B0 score and filename labels."""
    labels = []
    raw_scores = []
    for result in detection_results:
        if len(result) < 5 or not isinstance(result[4], dict):
            continue
        true_label = infer_true_label_from_filename(result[0])
        raw_score = result[4].get("raw_fake_probability")
        if true_label is None or raw_score is None:
            continue
        raw_score = float(raw_score)
        if raw_score > 1.0:
            raw_score /= 100.0
        labels.append(true_label)
        raw_scores.append(float(np.clip(raw_score, 0.0, 1.0)))

    if len(labels) == 0:
        return {
            "available": False,
            "value": "N/A",
            "labeled_count": 0,
            "message": "Raw ROC-AUC needs REAL and FAKE filename labels."
        }, labels, raw_scores
    if len(set(labels)) < 2:
        return {
            "available": False,
            "value": "N/A",
            "labeled_count": len(labels),
            "message": "Raw ROC-AUC needs both REAL and FAKE samples."
        }, labels, raw_scores

    auc_value = float(roc_auc_score(labels, raw_scores))
    return {
        "available": True,
        "value": round(auc_value, 4),
        "labeled_count": len(labels),
        "message": "Raw EfficientNet-B0 score only; forensic calibration is excluded."
    }, labels, raw_scores


def calculate_calibrated_roc_auc(detection_results):
    """Calculate full-system ROC-AUC from calibrated probabilities and filename labels."""
    labels = []
    calibrated_scores = []
    calibration_flags = []
    for result in detection_results:
        if len(result) < 5 or not isinstance(result[4], dict):
            continue
        true_label = infer_true_label_from_filename(result[0])
        calibrated_score = result[4].get("fake_probability")
        if true_label is None or calibrated_score is None:
            continue
        calibrated_score = float(calibrated_score)
        if calibrated_score > 1.0:
            calibrated_score /= 100.0
        labels.append(true_label)
        calibrated_scores.append(float(np.clip(calibrated_score, 0.0, 1.0)))
        calibration_flags.append(bool(result[4].get("calibration_applied", False)))

    if len(labels) == 0:
        return {
            "available": False,
            "value": "N/A",
            "labeled_count": 0,
            "message": "Calibrated ROC-AUC needs REAL and FAKE filename labels."
        }, labels, calibrated_scores
    if len(set(labels)) < 2:
        return {
            "available": False,
            "value": "N/A",
            "labeled_count": len(labels),
            "message": "Calibrated ROC-AUC needs both REAL and FAKE samples."
        }, labels, calibrated_scores
    if not calibration_flags or not all(calibration_flags):
        return {
            "available": False,
            "value": "N/A",
            "labeled_count": len(labels),
            "message": "Calibrated ROC-AUC is unavailable because calibration was not applied to every labelled sample."
        }, labels, calibrated_scores

    auc_value = float(roc_auc_score(labels, calibrated_scores))
    return {
        "available": True,
        "value": round(auc_value, 4),
        "labeled_count": len(labels),
        "message": "Full-system score after validation-trained forensic calibration."
    }, labels, calibrated_scores


def infer_true_label_from_filename(filename):
    """
    Infer ground-truth label from filename for demo metrics.
    Uses token-like matching so words such as 'unreal' do not accidentally count as REAL.
    """
    name = os.path.basename(str(filename or "")).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]

    if "fake" in tokens:
        return 1
    if "real" in tokens:
        return 0

    # Useful fallback for common dataset folder/file naming patterns after ZIP path is flattened.
    if any(t.startswith("fake_") or t.endswith("_fake") for t in tokens):
        return 1
    if any(t.startswith("real_") or t.endswith("_real") for t in tokens):
        return 0

    return None


def append_label_if_confident(final_pred, filename, true_labels, pred_labels):
    """Append labels for metrics only when prediction is REAL/FAKE and filename contains ground truth."""
    if final_pred not in ("REAL", "FAKE"):
        return

    true_label = infer_true_label_from_filename(filename)
    if true_label is None:
        return

    true_labels.append(true_label)
    pred_labels.append(1 if final_pred == "FAKE" else 0)

# --------------------------
# Plotting Functions
# --------------------------
def plot_confusion_matrix(true_labels, pred_labels, save_path):
    """Plot a readable 2x2 confusion matrix image for UI."""

    try:
        # Force 2x2 matrix even if only one class appears in the batch
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

        # 600x400 output for confusion matrix
        plt.figure(figsize=(6, 4))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            square=False,  # allow rectangular canvas
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 14},
            xticklabels=["REAL", "FAKE"],
            yticklabels=["REAL", "FAKE"],
            cbar_kws={"shrink": 0.95}
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix", pad=10)

        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        print(f"Confusion matrix plotting error: {e}")
        return False


def plot_roc_comparison(true_labels, raw_scores, calibrated_scores, save_path):
    """Plot raw-CNN and calibrated full-system ROC curves on the same samples."""
    try:
        raw_fpr, raw_tpr, _ = roc_curve(true_labels, raw_scores)
        raw_auc = roc_auc_score(true_labels, raw_scores)
        calibrated_fpr, calibrated_tpr, _ = roc_curve(true_labels, calibrated_scores)
        calibrated_auc = roc_auc_score(true_labels, calibrated_scores)
        plt.figure(figsize=(6, 4))
        plt.plot(raw_fpr, raw_tpr, color="#2F80ED", linewidth=2.2, label=f"Raw EfficientNet-B0 (AUC = {raw_auc:.4f})")
        plt.plot(calibrated_fpr, calibrated_tpr, color="#D97706", linewidth=2.2, label=f"Calibrated full system (AUC = {calibrated_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="#7A7A7A", linestyle="--", linewidth=1.3, label="Random (AUC = 0.5000)")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Raw and Calibrated ROC Curves")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=120)
        plt.close()
        return True
    except Exception as e:
        print(f"ROC curve plotting error: {e}")
        return False

def plot_result_counts(real_count, fake_count, save_path, uncertain_count=0):
    try:
        plt.figure(figsize=(6, 4))
        categories = ["REAL", "FAKE", "UNCERTAIN"]
        counts = [real_count, fake_count, uncertain_count]
        
        bars = plt.bar(categories, counts, color=["#4A90E2", "#7B68EE", "#F0AD4E"])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.ylabel("Image Count")
        plt.title("Detection Results Summary")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        print(f"Result counts plot error: {e}")
        return False

def plot_confidence_distribution(results, save_path):
    """Plot model-score histogram."""
    try:
        # Keep only non-error rows and normalize tuple length
        cleaned = []
        for r in results:
            if not r:
                continue
            if len(r) >= 2 and r[1] != "ERROR":
                cleaned.append(tuple(r[:4]))

        if not cleaned:
            return False

        df = pd.DataFrame(cleaned, columns=["Filename", "Prediction", "Confidence (%)", "Details"])

        plt.figure(figsize=(8, 4))
        ax = sns.histplot(
            data=df,
            x="Confidence (%)",
            hue="Prediction",
            bins=10,
            palette={"REAL": "#4A90E2", "FAKE": "#7B68EE", "UNCERTAIN": "#F0AD4E"},
            kde=True
        )

        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Image Count")
        ax.set_title("Prediction Confidence Distribution")

        # Make count axis use integer ticks (smart labels)
        try:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        except Exception:
            pass

        # Annotate counts on top of each bar
        for patch in ax.patches:
            try:
                h = patch.get_height()
                if h and h > 0:
                    x = patch.get_x() + patch.get_width() / 2.0
                    ax.text(x, h, f"{int(round(h))}", ha="center", va="bottom", fontsize=9)
            except Exception:
                continue

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    
    except Exception as e:
        print(f"Confidence distribution plot error: {e}")
        return False


def attach_raw_roc_to_batch(metrics, plots, detection_results, batch_time):
    """Attach separate raw and calibrated ROC-AUC metadata and a comparison plot."""
    raw_summary, raw_labels, raw_scores = calculate_raw_roc_auc(detection_results)
    calibrated_summary, calibrated_labels, calibrated_scores = calculate_calibrated_roc_auc(detection_results)
    metrics["raw_roc_auc"] = raw_summary
    metrics["calibrated_roc_auc"] = calibrated_summary
    same_samples = raw_labels == calibrated_labels and len(raw_scores) == len(calibrated_scores)
    if raw_summary["available"] and calibrated_summary["available"] and same_samples:
        roc_plot_path = os.path.join(OUTPUTS_FOLDER, f"roc_comparison_{batch_time}.png")
        if plot_roc_comparison(raw_labels, raw_scores, calibrated_scores, roc_plot_path):
            plots["roc_comparison_curve"] = os.path.basename(roc_plot_path)

# --------------------------
# File Processing Functions
# --------------------------
def process_zip_file(zip_file_stream):
    """Process ZIP files using the loaded EfficientNet-B0 model"""
    loaded_models = {m: ACTIVE_MODELS[m] for m in ACTIVE_MODELS.keys() 
                    if ACTIVE_MODELS[m] is not None}
    if not loaded_models:
        raise Exception("No model loaded. Please load an EfficientNet-B0 model first.")
    
    batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = os.path.join(UPLOADS_FOLDER, f"zip_batch_{batch_time}")
    os.makedirs(batch_folder, exist_ok=True)
    
    detection_results = []
    true_labels = []
    pred_labels = []
    
    extracted_count = safe_extract_zip(zip_file_stream, batch_folder)
    if extracted_count == 0:
        raise Exception("ZIP file contains no safe extractable files.")
    
    image_paths = []
    for root, _, files in os.walk(batch_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                image_paths.append(os.path.join(root, filename))

    if not image_paths:
        shutil.rmtree(batch_folder, ignore_errors=True)
        raise Exception("ZIP file contains no supported image files.")

    for image_index, file_path in enumerate(image_paths):
        original_filename = os.path.relpath(file_path, batch_folder).replace(os.sep, "_")
        filename = make_batch_filename(original_filename, image_index)
        if True:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    detection_results.append((filename, "ERROR", 0.0, "Unable to load image"))
                    continue
                
                success, result = process_image(img, enable_gradcam=(image_index < MAX_GRADCAM_BATCH_IMAGES))
                if not success:
                    detection_results.append((filename, "ERROR", 0.0, result))
                    continue
                
                final_pred = result["final_prediction"]
                final_conf = result["final_confidence"]
                
                # Save original and XAI/evidence images for the batch UI
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, filename), img)

                canny_name = f"canny_{batch_time}_{filename}"
                dft_name = f"dft_{batch_time}_{filename}"
                noise_name = f"noise_{batch_time}_{filename}"
                gradcam_name = f"gradcam_{batch_time}_{filename}"

                if result.get("canny_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, canny_name), result["canny_raw"])
                if result.get("dft_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, dft_name), result["dft_raw"])
                if result.get("noise_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, noise_name), result["noise_raw"])
                if result.get("gradcam_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, gradcam_name), result["gradcam_raw"])

                analysis_payload = {
                    "model": result.get("decision_method", "EfficientNet-B0"),
                    "fake_probability": result.get("fake_probability", 0.0),
                    "raw_fake_probability": result.get("raw_fake_probability", 0.0),
                    "calibration_applied": result.get("calibration_applied", False),
                    "threshold_mode": result.get("threshold_label", get_threshold_config()["label"]),
                    "real_threshold_percent": result.get("real_threshold_percent", int(REAL_THRESHOLD * 100)),
                    "fake_threshold_percent": result.get("fake_threshold_percent", int(FAKE_THRESHOLD * 100)),
                    "xai_summary": result.get("xai_summary", ""),
                    "forensic_scores": result.get("forensic_scores", {}),
                    "evidence": {
                        "gradcam": gradcam_name if result.get("gradcam_raw") is not None else None,
                        "dft": dft_name if result.get("dft_raw") is not None else None,
                        "canny": canny_name if result.get("canny_raw") is not None else None,
                        "noise": noise_name if result.get("noise_raw") is not None else None,
                    }
                }
                detection_results.append((
                    filename,
                    final_pred,
                    final_conf,
                    result.get("xai_explanation", "XAI explanation unavailable for this image."),
                    analysis_payload
                ))
                
                append_label_if_confident(final_pred, original_filename, true_labels, pred_labels)
                    
            except Exception as e:
                detection_results.append((filename, "ERROR", 0.0, f"Error: {str(e)}"))
    
    shutil.rmtree(batch_folder, ignore_errors=True)
    
    real_count = sum(1 for r in detection_results if r[1] == "REAL")
    fake_count = sum(1 for r in detection_results if r[1] == "FAKE")
    uncertain_count = sum(1 for r in detection_results if r[1] == "UNCERTAIN")
    error_count = sum(1 for r in detection_results if r[1] == "ERROR")
    valid_conf = [r[2] for r in detection_results if r[1] != "ERROR"]
    avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0
    
    stats = {
        "total_images": len(detection_results),
        "real_count": real_count,
        "fake_count": fake_count,
        "uncertain_count": uncertain_count,
        "error_count": error_count,
        "avg_confidence": avg_confidence
    }
    
    metrics = calculate_ml_metrics(true_labels, pred_labels)

    plots = {}
    if metrics.get("available") and metrics["confusion_matrix"] != [[0,0],[0,0]]:
        cm_plot_path = os.path.join(OUTPUTS_FOLDER, f"confusion_matrix_{batch_time}.png")
        if plot_confusion_matrix(true_labels, pred_labels, cm_plot_path):
            plots["confusion_matrix"] = os.path.basename(cm_plot_path)
    
    count_plot_path = os.path.join(OUTPUTS_FOLDER, f"result_counts_{batch_time}.png")
    if plot_result_counts(real_count, fake_count, count_plot_path, uncertain_count):
        plots["result_counts"] = os.path.basename(count_plot_path)
    
    conf_plot_path = os.path.join(OUTPUTS_FOLDER, f"confidence_dist_{batch_time}.png")
    if plot_confidence_distribution(detection_results, conf_plot_path):
        plots["confidence_dist"] = os.path.basename(conf_plot_path)
    attach_raw_roc_to_batch(metrics, plots, detection_results, batch_time)
    
    return {
        "results": detection_results, 
        "stats": stats, 
        "metrics": metrics,
        "plots": plots, 
        "timestamp": batch_time
    }

def process_single_or_multiple_images(image_files):
    """Process images using the loaded EfficientNet-B0 model"""
    loaded_models = {m: ACTIVE_MODELS[m] for m in ACTIVE_MODELS.keys() 
                    if ACTIVE_MODELS[m] is not None}
    if not loaded_models:
        raise Exception("No model loaded. Please load an EfficientNet-B0 model first.")
    
    batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    detection_results = []
    true_labels = []
    pred_labels = []
    
    for image_index, file in enumerate(image_files):
        original_filename = file.filename or f"image_{image_index + 1}.jpg"
        filename = make_batch_filename(original_filename, image_index)
        try:
            file_data = file.read()
            img_np = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if img is None:
                detection_results.append((filename, "ERROR", 0.0, "Unable to load image"))
                continue
            
            # Save original image
            original_path = os.path.join(OUTPUTS_FOLDER, filename)
            cv2.imwrite(original_path, img)
            
            success, result = process_image(img, enable_gradcam=(image_index < MAX_GRADCAM_BATCH_IMAGES))
            if not success:
                detection_results.append((filename, "ERROR", 0.0, result))
                continue
            
            final_pred = result["final_prediction"]
            final_conf = result["final_confidence"]
            
            # Save forensic evidence images
            canny_name = f"canny_{batch_time}_{filename}"
            dft_name = f"dft_{batch_time}_{filename}"
            noise_name = f"noise_{batch_time}_{filename}"
            gradcam_name = f"gradcam_{batch_time}_{filename}"
            
            if result.get("canny_raw") is not None:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, canny_name), result["canny_raw"])
            if result.get("dft_raw") is not None:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, dft_name), result["dft_raw"])
            if result.get("noise_raw") is not None:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, noise_name), result["noise_raw"])
            if result.get("gradcam_raw") is not None:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, gradcam_name), result["gradcam_raw"])
            
            analysis_payload = {
                "model": result.get("decision_method", "EfficientNet-B0"),
                "fake_probability": result.get("fake_probability", 0.0),
                "raw_fake_probability": result.get("raw_fake_probability", 0.0),
                "calibration_applied": result.get("calibration_applied", False),
                "threshold_mode": result.get("threshold_label", get_threshold_config()["label"]),
                "real_threshold_percent": result.get("real_threshold_percent", int(REAL_THRESHOLD * 100)),
                "fake_threshold_percent": result.get("fake_threshold_percent", int(FAKE_THRESHOLD * 100)),
                "xai_summary": result.get("xai_summary", ""),
                "forensic_scores": result.get("forensic_scores", {}),
                "evidence": {
                    "gradcam": gradcam_name if result.get("gradcam_raw") is not None else None,
                    "dft": dft_name if result.get("dft_raw") is not None else None,
                    "canny": canny_name if result.get("canny_raw") is not None else None,
                    "noise": noise_name if result.get("noise_raw") is not None else None,
                }
            }
            detection_results.append((
                filename, 
                final_pred, 
                final_conf, 
                result.get("xai_explanation", "XAI explanation unavailable for this image."),
                analysis_payload
            ))
            append_label_if_confident(final_pred, original_filename, true_labels, pred_labels)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            detection_results.append((filename, "ERROR", 0.0, f"Error: {str(e)}"))
    
    real_count = sum(1 for r in detection_results if r[1] == "REAL")
    fake_count = sum(1 for r in detection_results if r[1] == "FAKE")
    uncertain_count = sum(1 for r in detection_results if r[1] == "UNCERTAIN")
    error_count = sum(1 for r in detection_results if r[1] == "ERROR")
    valid_conf = [r[2] for r in detection_results if r[1] != "ERROR"]
    avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0
    
    stats = {
        "total_images": len(detection_results),
        "real_count": real_count,
        "fake_count": fake_count,
        "uncertain_count": uncertain_count,
        "error_count": error_count,
        "avg_confidence": avg_confidence
    }
    
    metrics = calculate_ml_metrics(true_labels, pred_labels)

    plots = {}
    if metrics.get("available") and metrics["confusion_matrix"] != [[0,0],[0,0]]:
        cm_plot_path = os.path.join(OUTPUTS_FOLDER, f"confusion_matrix_{batch_time}.png")
        if plot_confusion_matrix(true_labels, pred_labels, cm_plot_path):
            plots["confusion_matrix"] = os.path.basename(cm_plot_path)
    
    count_plot_path = os.path.join(OUTPUTS_FOLDER, f"result_counts_{batch_time}.png")
    if plot_result_counts(real_count, fake_count, count_plot_path, uncertain_count):
        plots["result_counts"] = os.path.basename(count_plot_path)
    
    conf_plot_path = os.path.join(OUTPUTS_FOLDER, f"confidence_dist_{batch_time}.png")
    if plot_confidence_distribution(detection_results, conf_plot_path):
        plots["confidence_dist"] = os.path.basename(conf_plot_path)
    attach_raw_roc_to_batch(metrics, plots, detection_results, batch_time)
    
    return {
        "results": detection_results, 
        "stats": stats, 
        "metrics": metrics,
        "plots": plots, 
        "timestamp": batch_time
    }

def is_cleanup_target(filename):
    """Return True if file inside outputs/ can safely removed."""
    return filename.lower().endswith(OUTPUT_CLEANUP_EXTENSIONS)


def wipe_outputs_folder(reason="startup"):
    """
    Completely clear temporary generated files from outputs/.
    Runs once on app startup so old files are removed and keep folder clean.
    """
    removed = 0
    try:
        if not os.path.exists(OUTPUTS_FOLDER):
            os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
            return 0

        for filename in os.listdir(OUTPUTS_FOLDER):
            file_path = os.path.join(OUTPUTS_FOLDER, filename)

            if os.path.isfile(file_path) and is_cleanup_target(filename):
                try:
                    os.remove(file_path)
                    removed += 1
                except Exception as e:
                    print(f"Failed to remove output file {filename}: {e}")

        print(f"Output cleanup ({reason}): removed {removed} file(s).")
        return removed

    except Exception as e:
        print(f"Output wipe error: {e}")
        return removed


def cleanup_outputs(max_files=OUTPUT_MAX_FILES, max_age_seconds=OUTPUT_MAX_FILE_AGE_SECONDS, force=False):
    """
    Periodically clean outputs folder.
    Rules:
    1. Delete temporary output files older than max_age_seconds.
    2. If there are still too many files, keep only the newest max_files.
    3. Run at most once every OUTPUT_CLEANUP_INTERVAL_SECONDS unless force=True.
    """
    global LAST_OUTPUT_CLEANUP_TS

    try:
        now = time.time()

        if not force and (now - LAST_OUTPUT_CLEANUP_TS) < OUTPUT_CLEANUP_INTERVAL_SECONDS:
            return 0

        LAST_OUTPUT_CLEANUP_TS = now

        if not os.path.exists(OUTPUTS_FOLDER):
            os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
            return 0

        files = []
        removed = 0

        for filename in os.listdir(OUTPUTS_FOLDER):
            file_path = os.path.join(OUTPUTS_FOLDER, filename)

            if not os.path.isfile(file_path) or not is_cleanup_target(filename):
                continue

            try:
                mtime = os.path.getmtime(file_path)
                age = now - mtime

                if age > max_age_seconds:
                    os.remove(file_path)
                    removed += 1
                else:
                    files.append((filename, mtime))

            except Exception as e:
                print(f"Cleanup check failed for {filename}: {e}")

        # Limit file count by removing oldest generated files.
        if len(files) > max_files:
            files.sort(key=lambda x: x[1])
            extra_count = len(files) - max_files

            for filename, _ in files[:extra_count]:
                try:
                    os.remove(os.path.join(OUTPUTS_FOLDER, filename))
                    removed += 1
                except Exception as e:
                    print(f"Failed to remove old output file {filename}: {e}")

        if removed > 0:
            print(f"Periodic output cleanup: removed {removed} file(s).")

        return removed

    except Exception as e:
        print(f"Cleanup error: {e}")
        return 0


# Startup cleanup: clear previous session output files once when app starts.
# Set FYP2_SKIP_STARTUP_CLEAN=1 if you need to keep outputs.
if os.environ.get("FYP2_SKIP_STARTUP_CLEAN", "0") != "1":
    wipe_outputs_folder("startup")


# --------------------------
# Video Processing (MP4 frame sampling)
# --------------------------
class VideoFrameSampler:
    """
    Video Frame Deepfake Detection

    READ ME:
    - Keep existing image pipeline untouched (process_image, voting, metrics, plots...etc).
    - Add video support in a separate, self-contained block to prevent conflicts.
    - Detect only 20% of all frames by default, with min 5 and max 120 frames.
    """

    def __init__(self, sample_ratio: float = 0.20, min_samples: int = 5, max_samples: int = 120):
        self.sample_ratio = float(sample_ratio)
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)

    def _safe_int(self, v, default=0):
        try:
            return int(v)
        except Exception:
            return int(default)

    def _compute_indices(self, total_frames: int):
        if total_frames <= 0:
            return [0]
        n = max(self.min_samples, int(total_frames * self.sample_ratio))
        n = min(n, self.max_samples, total_frames)
        if n <= 1:
            return [0]
        # Evenly spaced indices across the video
        step = (total_frames - 1) / float(n - 1)
        idxs = [int(round(i * step)) for i in range(n)]
        # De-duplicate while preserving order
        seen = set()
        out = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def analyze(self, video_path: str, display_name: str = None):
        """Return a dict similar to single-image processing, but aggregated across sampled frames."""
        if display_name is None:
            display_name = os.path.basename(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Unable to open video: {display_name}"

        total_frames = self._safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
        indices = self._compute_indices(total_frames)

        frame_results = []  # list of (pred, conf)
        errors = 0
        thumb_frame = None

        for k, idx in enumerate(indices):
            try:
                if total_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    errors += 1
                    continue

                # Keep the first successfully decoded frame as thumbnail
                if thumb_frame is None:
                    thumb_frame = frame.copy()

                success, result = process_image(frame, enable_gradcam=False)
                if not success:
                    errors += 1
                    continue

                frame_results.append((result.get('final_prediction', 'ERROR'), float(result.get('final_confidence', 0.0))))

            except Exception as e:
                errors += 1
                continue

        cap.release()

        if not frame_results:
            return False, f"No frames could be analyzed for {display_name} (errors={errors})."

        # Aggregate one-model frame decisions across sampled frames.
        votes = {'REAL': 0, 'FAKE': 0, 'UNCERTAIN': 0}
        confs = {'REAL': [], 'FAKE': [], 'UNCERTAIN': []}
        for pred, conf in frame_results:
            if pred in votes:
                votes[pred] += 1
                confs[pred].append(conf)

        final_pred = max(votes, key=votes.get)
        # If there is a tie, select the tied class with higher average confidence.
        max_vote = votes[final_pred]
        tied = [k for k, v in votes.items() if v == max_vote]
        if len(tied) > 1:
            final_pred = max(tied, key=lambda k: float(sum(confs[k]) / max(len(confs[k]), 1)))
        final_conf = round(float(sum(confs[final_pred]) / max(len(confs[final_pred]), 1)), 2)

        # Save thumbnail for UI display
        thumb_filename = None
        try:
            if thumb_frame is not None:
                base = os.path.splitext(os.path.basename(display_name))[0]
                stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                thumb_filename = f"video_thumb_{stamp}_{base}.jpg"
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, thumb_filename), thumb_frame)
        except Exception:
            thumb_filename = None

        analysis = (
            f"Video frame sampling: analyzed {len(frame_results)}/{len(indices)} frames "
            f"(sample_ratio={self.sample_ratio:.0%}, errors={errors}). "
            f"Frame votes: {votes}"
        )

        return True, {
            'display_name': display_name,
            'thumbnail': thumb_filename,
            'final_prediction': final_pred,
            'final_confidence': final_conf,
            'votes': votes,
            'frames_analyzed': len(frame_results),
            'frames_planned': len(indices),
            'errors': errors,
            'analysis': analysis,
        }


def process_single_or_multiple_videos(video_files, sample_ratio: float = 0.20):
    """Process one or more uploaded MP4 videos. Returns a batch_data dict (same shape as images)."""
    loaded_models = {m: ACTIVE_MODELS[m] for m in ACTIVE_MODELS.keys() if ACTIVE_MODELS[m] is not None}
    if not loaded_models:
        raise Exception("No model loaded. Please load an EfficientNet-B0 model first.")

    batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sampler = VideoFrameSampler(sample_ratio=sample_ratio)

    detection_results = []
    true_labels = []
    pred_labels = []

    for f in video_files:
        original_filename = f.filename
        filename = sanitize_filename(original_filename, default=f"video_{len(detection_results) + 1}.mp4")
        if not filename:
            continue

        tmp_path = os.path.join(UPLOADS_FOLDER, f"tmp_{batch_time}_{filename}")
        try:
            f.save(tmp_path)
            ok, out = sampler.analyze(tmp_path, display_name=filename)
            if not ok:
                detection_results.append((filename, 'ERROR', 0.0, out))
                continue

            shown_name = out['thumbnail'] if out.get('thumbnail') else filename
            detection_results.append((
                shown_name,
                out['final_prediction'],
                out['final_confidence'],
                f"{out['analysis']} | Source: {filename}"
            ))

            # Optional ground-truth heuristic by filename. UNCERTAIN is excluded from binary metrics.
            append_label_if_confident(out['final_prediction'], filename, true_labels, pred_labels)

        except Exception as e:
            detection_results.append((filename, 'ERROR', 0.0, f"Error: {str(e)}"))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    real_count = sum(1 for r in detection_results if r[1] == 'REAL')
    fake_count = sum(1 for r in detection_results if r[1] == 'FAKE')
    uncertain_count = sum(1 for r in detection_results if r[1] == 'UNCERTAIN')
    error_count = sum(1 for r in detection_results if r[1] == 'ERROR')
    valid_conf = [r[2] for r in detection_results if r[1] != 'ERROR']
    avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0

    stats = {
        'total_images': len(detection_results),
        'real_count': real_count,
        'fake_count': fake_count,
        'uncertain_count': uncertain_count,
        'error_count': error_count,
        'avg_confidence': avg_confidence
    }

    metrics = calculate_ml_metrics(true_labels, pred_labels)

    plots = {}
    if metrics.get("available") and metrics['confusion_matrix'] != [[0,0],[0,0]]:
        cm_plot_path = os.path.join(OUTPUTS_FOLDER, f"confusion_matrix_{batch_time}.png")
        if plot_confusion_matrix(true_labels, pred_labels, cm_plot_path):
            plots['confusion_matrix'] = os.path.basename(cm_plot_path)

    count_plot_path = os.path.join(OUTPUTS_FOLDER, f"result_counts_{batch_time}.png")
    if plot_result_counts(real_count, fake_count, count_plot_path, uncertain_count):
        plots['result_counts'] = os.path.basename(count_plot_path)

    conf_plot_path = os.path.join(OUTPUTS_FOLDER, f"confidence_dist_{batch_time}.png")
    if plot_confidence_distribution(detection_results, conf_plot_path):
        plots['confidence_dist'] = os.path.basename(conf_plot_path)
    attach_raw_roc_to_batch(metrics, plots, detection_results, batch_time)

    return {
        'results': detection_results,
        'stats': stats,
        'metrics': metrics,
        'plots': plots,
        'timestamp': batch_time
    }


def process_mixed_files(files):
    """Process a mixed list of uploads: images + mp4 videos (zip handled elsewhere)."""
    benchmark_wall_start = time.perf_counter()
    benchmark_cpu_start = time.process_time()
    benchmark_monitor = BenchmarkResourceMonitor(SELECTED_DEVICE)
    benchmark_monitor.__enter__()
    image_files = []
    video_files = []
    other_files = []

    for f in files:
        name = (f.filename or '').lower()
        if name.endswith('.mp4'):
            video_files.append(f)
        elif name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) or (hasattr(f, 'mimetype') and str(f.mimetype).startswith('image/')):
            image_files.append(f)
        else:
            other_files.append(f)

    # Process images first (this will consume file streams). Keep it separate.
    combined_results = None
    if image_files:
        combined_results = process_single_or_multiple_images(image_files)
    if video_files:
        video_results = process_single_or_multiple_videos(video_files)
        if combined_results is None:
            combined_results = video_results
        else:
            # Merge video results into image batch
            combined_results['results'].extend(video_results['results'])
            # Recompute stats
            all_results = combined_results['results']
            real_count = sum(1 for r in all_results if r[1] == 'REAL')
            fake_count = sum(1 for r in all_results if r[1] == 'FAKE')
            uncertain_count = sum(1 for r in all_results if r[1] == 'UNCERTAIN')
            error_count = sum(1 for r in all_results if r[1] == 'ERROR')
            valid_conf = [r[2] for r in all_results if r[1] != 'ERROR']
            avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0
            combined_results['stats'] = {
                'total_images': len(all_results),
                'real_count': real_count,
                'fake_count': fake_count,
                'uncertain_count': uncertain_count,
                'error_count': error_count,
                'avg_confidence': avg_confidence,
            }
            # Keep plots/metrics from image batch; mixed ground-truth is unreliable.

    if combined_results is None:
        raise Exception('No supported files found. Please upload images, zip, or mp4 videos.')

    if other_files:
        # Add warnings as ERROR rows (keep UI consistent)
        for f in other_files:
            if f.filename:
                combined_results['results'].append((sanitize_filename(f.filename), 'ERROR', 0.0, 'Unsupported file type'))
        all_results = combined_results['results']
        combined_results['stats']['total_images'] = len(all_results)
        combined_results['stats']['error_count'] = sum(1 for r in all_results if r[1] == 'ERROR')
        combined_results['stats']['uncertain_count'] = sum(1 for r in all_results if r[1] == 'UNCERTAIN')

    benchmark_wall_seconds = time.perf_counter() - benchmark_wall_start
    benchmark_cpu_seconds = time.process_time() - benchmark_cpu_start
    benchmark_monitor.__exit__(None, None, None)
    return attach_uploaded_batch_benchmark(combined_results, benchmark_wall_seconds, benchmark_cpu_seconds, benchmark_monitor)
# --------------------------
# Flask App
# --------------------------
app = Flask(__name__, static_folder=OUTPUTS_FOLDER)
app.secret_key = "deepfake_detector_2026"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

def get_ui_template():
    accept_attr = ",".join(SUPPORTED_MODELS.keys())
    supported_exts_text = ", ".join([ext.lstrip('.') for ext in SUPPORTED_MODELS.keys()])
    
    template = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Deepfake Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: Arial, sans-serif;
            color: #333;
        }
        .app-shell {
            position: relative;
        }
        .detector-layout {
            align-items: flex-start;
        }
        .config-panel,
        .detection-panel {
            transition: flex-basis 0.25s ease, max-width 0.25s ease, opacity 0.2s ease, transform 0.25s ease;
        }
        .config-panel {
            transform: translateX(0);
            overflow: hidden;
        }
        body.config-collapsed .config-panel {
            flex: 0 0 0;
            max-width: 0;
            opacity: 0;
            padding-left: 0;
            padding-right: 0;
            pointer-events: none;
            transform: translateX(-105%);
        }
        body.config-collapsed .detection-panel {
            flex: 0 0 100%;
            max-width: 100%;
        }
        .config-toggle-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            width: auto;
            min-height: 36px;
            margin-bottom: 12px;
            border: 1px solid #d7e4f5;
            background: #ffffff;
            color: #315f93;
            border-radius: 8px;
            padding: 7px 12px;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .config-toggle-btn:hover {
            background: #eef6ff;
            color: #244c79;
        }
        .custom-card {
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #eee;
        }
        .drop-zone {
            border: 2px dashed #4A90E2;
            border-radius: 12px;
            background-color: #f0f7ff;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 20px;
            position: relative;
            z-index: 10;
        }
        .drop-zone:hover, .drop-zone.active {
            background-color: #e6f0ff;
            border-color: #357abd;
        }
        .custom-btn {
            background: linear-gradient(to right, #4A90E2, #5B86E5);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            width: 100%;
            transition: background 0.2s ease;
        }
        .custom-btn:hover {
            background: linear-gradient(to right, #357abd, #4A90E2);
            color: white;
        }
        .small-text {
            color: #6c757d;
            font-size: 13px;
        }
        .gradcam-limit-card {
            border: 1px solid #dceaf8;
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            border-radius: 8px;
            padding: 12px;
        }
        .gradcam-limit-header {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin-bottom: 10px;
        }
        .gradcam-limit-title {
            color: #34495e;
            font-size: 13px;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }
        .gradcam-limit-subtitle {
            color: #6c757d;
            font-size: 12px;
            line-height: 1.35;
            margin-top: 3px;
        }
        .gradcam-limit-form {
            display: grid;
            grid-template-columns: minmax(72px, 96px) minmax(92px, 1fr);
            gap: 8px;
            align-items: center;
            margin-bottom: 8px;
        }
        .gradcam-limit-input {
            height: 36px;
            border-radius: 8px !important;
            border: 1px solid #cddff2;
            text-align: center;
            font-weight: 300;
            color: #25384f;
        }
        .gradcam-limit-input:focus {
            border-color: #4A90E2;
            box-shadow: 0 0 0 0.16rem rgba(74, 144, 226, 0.18);
        }
        .gradcam-apply-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            min-height: 36px;
            border: none;
            border-radius: 8px;
            background: #4A90E2;
            color: #ffffff;
            padding: 7px 12px;
        }
        .gradcam-apply-btn:hover {
            background: #357abd;
            color: #ffffff;
        }
        .gradcam-limit-note {
            display: flex;
            align-items: flex-start;
            gap: 7px;
            color: #6c757d;
            font-size: 12px;
            line-height: 1.35;
        }
        .gradcam-limit-note i {
            color: #4A90E2;
            margin-top: 1px;
        }
        .result-card {
            background-color: #fefeff;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            border-left: 4px solid #7B68EE;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        .metric-badge {
            background-color: #f0efff;
            color: #7B68EE;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
        }
        .plot-img {
            max-width: 100%;
            border-radius: 8px;
            margin-top: 10px;
            border: 1px solid #eee;
        }
        #file-input {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }
        .preview-container {
            display: none;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }
        .preview-image {
            max-width: 100px;
            max-height: 100px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            object-fit: cover;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .results-table th, .results-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .results-table th {
            background-color: #f8f9fa;
            font-weight: 500;
        }
        .results-table th.analysis-column,
        .results-table td.analysis-column {
            min-width: 360px;
            width: 52%;
        }
        .analysis-cell {
            font-size: 0.80rem;
            line-height: 1.4;
        }
        .analysis-text {
            margin-top: 6px;
            white-space: normal;
        }
        .file-count-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 50px;
            height: 50px;
            background-color: #f0f0f0;
            border-radius: 50%;
            font-weight: bold;
        }
        .model-badge {
            font-size: 11px;
            padding: 2px 6px;
            margin-right: 3px;
            margin-bottom: 3px;
        }
        .rgb-model { background-color: #4A90E2; }
        .dft-model { background-color: #FF6B6B; }
        .canny-model { background-color: #51CF66; }
        .card-img-container {
            height: 140px;
            overflow: hidden;
            background: #f0f0f0;
        }
        .evidence-card-note {
            color: #6c757d;
            font-size: 0.70rem;
            line-height: 1.25;
        }
        .visual-evidence-card {
            height: 100%;
            transition: transform 0.2s ease;
        }
        .visual-evidence-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        @media (max-width: 767.98px) {
            body.config-collapsed .config-panel {
                flex: 0 0 100%;
                max-width: 100%;
                max-height: 0;
                margin-bottom: 0;
            }
            body.config-collapsed .detection-panel {
                flex: 0 0 100%;
                max-width: 100%;
            }
        }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dropZone = document.getElementById('drop-zone');
            const fileInput = document.getElementById('file-input');
            const previewContainer = document.getElementById('preview-container');
            const dropZoneDefaultContent = document.getElementById('drop-zone-default');

            previewContainer.style.display = 'none';

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, unhighlight, false);
            });

            dropZone.addEventListener('drop', handleDrop, false);
            fileInput.addEventListener('change', generatePreviews, false);

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            function highlight() {
                dropZone.classList.add('active');
            }

            function unhighlight() {
                dropZone.classList.remove('active');
            }

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    generatePreviews();
                }
            }

            function toggleDropZoneContent(hasFiles) {
                if (hasFiles) {
                    dropZoneDefaultContent.style.display = 'none';
                    dropZone.classList.add('has-files');
                    previewContainer.style.display = 'flex';
                    previewContainer.style.gap = '10px';
                } else {
                    dropZoneDefaultContent.style.display = 'block';
                    dropZone.classList.remove('has-files');
                    previewContainer.style.display = 'none';
                }
            }

            function generatePreviews() {
                previewContainer.innerHTML = '';
                const files = fileInput.files;

                if (!files || files.length === 0) {
                    toggleDropZoneContent(false);
                    return;
                }

                toggleDropZoneContent(true);

                let previewCount = 0;
                for (let i = 0; i < files.length && previewCount < 3; i++) {
                    if (files[i].type.startsWith('image/')) {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            const img = document.createElement('img');
                            img.src = e.target.result;
                            img.className = 'preview-image';
                            img.alt = files[i].name;
                            previewContainer.appendChild(img);
                        };
                        reader.readAsDataURL(files[i]);
                        previewCount++;
                    }
                }

                if (files.length > previewCount) {
                    const remaining = files.length - previewCount;
                    const indicator = document.createElement('div');
                    indicator.className = 'file-count-indicator';
                    indicator.textContent = `+${remaining}`;
                    previewContainer.appendChild(indicator);
                }
            }

            const deviceSelect = document.getElementById('device-select');
            if (deviceSelect) {
                deviceSelect.addEventListener('change', function() {
                    if (this.value === 'cuda' && !('{{ has_cuda }}' === 'True' || '{{ has_cuda }}' === 'true')) {
                        alert('Warning: CUDA (GPU) is not available and will be switching to CPU...');
                        this.value = 'cpu';
                    }
                });
            }

            const configToggle = document.getElementById('config-toggle');
            const configToggleIcon = document.getElementById('config-toggle-icon');
            const configToggleLabel = document.getElementById('config-toggle-label');

            function setConfigCollapsed(collapsed) {
                document.body.classList.toggle('config-collapsed', collapsed);

                if (configToggle) {
                    configToggle.setAttribute('aria-expanded', String(!collapsed));
                    configToggle.title = collapsed ? 'Show configuration' : 'Hide configuration';
                }

                if (configToggleIcon) {
                    configToggleIcon.className = collapsed ? 'bi bi-layout-sidebar-inset' : 'bi bi-layout-sidebar';
                }

                if (configToggleLabel) {
                    configToggleLabel.textContent = collapsed ? 'Show Config' : 'Hide Config';
                }

                try {
                    localStorage.setItem('deepfakeConfigCollapsed', collapsed ? 'true' : 'false');
                } catch (e) {
                    // Ignore storage errors in private browsing or locked-down browsers.
                }
            }

            if (configToggle) {
                let savedCollapsed = false;
                try {
                    savedCollapsed = localStorage.getItem('deepfakeConfigCollapsed') === 'true';
                } catch (e) {
                    savedCollapsed = false;
                }

                setConfigCollapsed(savedCollapsed);
                configToggle.addEventListener('click', function() {
                    setConfigCollapsed(!document.body.classList.contains('config-collapsed'));
                });
            }

            toggleDropZoneContent(fileInput.files.length > 0);
        });
    </script>
</head>
<body>
<div class="container py-4">
    <div class="mb-4">
        <h3>Deepfake Detector</h3>
        <p class="small-text">Check your images or videos for deepfakes...</p>
    </div>
    
    <div class="app-shell">
        <button type="button" id="config-toggle" class="config-toggle-btn" aria-expanded="true" aria-controls="config-panel" title="Hide configuration">
            <i id="config-toggle-icon" class="bi bi-layout-sidebar"></i>
            <span id="config-toggle-label">Hide Config</span>
        </button>
    </div>

    <div class="row g-4 detector-layout">
        <div id="config-panel" class="col-md-4 config-panel">
            <div class="custom-card">
                <h5 class="mb-3">Config</h5>
                
                <div class="mb-3">
                    <label class="form-label small-text"><strong>Computing Using:</strong></label>
                    <form method="post" action="/set_device" class="mb-2">
                        <select name="device" id="device-select" class="form-select form-select-sm" onchange="this.form.submit()">
                            <option value="cpu" {{ 'selected' if selected_device == 'cpu' else '' }}>CPU (Slower, Universal)</option>
                            <option value="cuda" {{ 'selected' if selected_device == 'cuda' else '' }} {{ 'disabled' if not has_cuda else '' }}>GPU (CUDA, Faster) {{ '(Unavailable)' if not has_cuda else '' }}</option>
                        </select>
                    </form>
                </div>

                <div class="mb-3">
                    <label class="form-label small-text"><strong>Threshold Mode:</strong></label>
                    <form method="post" action="/set_threshold_mode" class="mb-2">
                        <select name="threshold_mode" class="form-select form-select-sm" onchange="this.form.submit()">
                            {% for mode_key, mode in threshold_modes.items() %}
                            <option value="{{ mode_key }}" {{ 'selected' if selected_threshold_mode == mode_key else '' }}>
                                {{ mode.label }} (REAL &le; {{ (mode.real * 100)|int }}%, FAKE &ge; {{ (mode.fake * 100)|int }}%)
                            </option>
                            {% endfor %}
                        </select>
                    </form>
                    <div class="small-text">
                        {{ current_threshold_config.description }}
                    </div>
                </div>

                <div class="mb-3 p-2 border rounded bg-light">
                    <form method="post" action="/set_force_binary_mode">
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" role="switch" id="force-binary-mode"
                                   name="force_binary" value="1" onchange="this.form.submit()"
                                   {{ 'checked' if force_binary_comparison else '' }}>
                            <label class="form-check-label small-text" for="force-binary-mode">
                                <strong>Force REAL/FAKE for comparison</strong>
                            </label>
                        </div>
                    </form>
                    <div class="small-text text-muted mt-1">Uses a 0.50 decision line and disables UNCERTAIN. Both ROC-AUC values are unchanged.</div>
                </div>

                <div class="mb-3 gradcam-limit-card">
                    <div class="gradcam-limit-header">
                        <div>
                            <label class="gradcam-limit-title" for="gradcam-limit-input">Batch Grad-CAM Limit</label>
                            <div class="gradcam-limit-subtitle">Show upper batch heatmaps only. The limit show how many does it show.</div>
                        </div>
                    </div>
                    <form method="post" action="/set_gradcam_limit" class="gradcam-limit-form">
                        <input id="gradcam-limit-input" type="number" name="gradcam_limit" class="form-control gradcam-limit-input" min="0" max="500" step="1" value="{{ max_gradcam_batch_images }}">
                        <button type="submit" class="gradcam-apply-btn">
                            <span>Apply</span>
                        </button>
                    </form>
                    <div class="gradcam-limit-note">
                        <i class="bi bi-clock-history"></i>
                        <span>More heatmaps, slower display.</span>
                    </div>
                </div>
                
                <label class="form-label small-text"><strong>Load Detection Model:</strong></label>
                <form method="POST" action="/load_model" enctype="multipart/form-data">
                    <div class="mb-3">
                        <input type="file" name="model_files" class="form-control mb-2" accept="{{ accept_attr }}" multiple required>
                        <p class="small-text">Supported formats: pt, pth, safetensors
                        <br>Supported architectures:
                        <br><span class="badge rgb-model">CNN Model: Custom-Trained EfficientNet-B0</span></p>
                    </div>
                    <button type="submit" class="custom-btn">Load Model</button>
                </form>
                
                <div class="mt-3">
                    <form method="post" action="/clear_models" onsubmit="return confirm('Clear loaded model?')">
                        <button type="submit" class="btn btn-outline-danger btn-sm w-100">
                            <i class="bi bi-trash"></i> Clear Loaded Model
                        </button>
                    </form>
                </div>
                
                {% if success_msg %}
                <div class="alert alert-success mt-3 small" role="alert">
                    {{ success_msg }}
                </div>
                {% endif %}

                {% if info_msg %}
                <div class="alert alert-info mt-3 small" role="alert">
                    {{ info_msg }}
                </div>
                {% endif %}
                
                {% if error_msg %}
                <div class="alert alert-danger mt-3 small" role="alert">
                    {{ error_msg }}
                </div>
                {% endif %}
                
                <div class="mt-3 small-text">
                    <strong>Current device:</strong> {{ selected_device.upper() }}<br>
                    <strong>Threshold Mode:</strong> {{ current_threshold_config.label }} 
                    (REAL &le; {{ (current_threshold_config.real * 100)|int }}%, FAKE &ge; {{ (current_threshold_config.fake * 100)|int }}%)<br>
                    <strong>Forced Comparison:</strong> {{ 'ON (0.50)' if force_binary_comparison else 'OFF' }}<br>
                    <strong>Batch Grad-CAM Limit:</strong> {{ max_gradcam_batch_images }} image(s)<br>
                    <strong>Loaded Model ({{ loaded_count }}/1):</strong> {% if loaded_count == 0 %}None{% else %}
                    <div class="mt-1">
                        {% for model_name, model_file in active_model_names.items() %}
                            {% if model_file %}
                            <span class="badge model-badge rgb-model">CNN Model: {{ model_file }}</span><br>
                            {% endif %}
                        {% endfor %}
                    </div>
                    {% endif %}

                </div>
            </div>
        </div>
        
        <div id="detection-panel" class="col-md-8 detection-panel">
            <div class="custom-card">
                <h5 class="text-center mb-4">Detect Deepfakes Images/Videos</h5>
                
                <form method="post" enctype="multipart/form-data" action="/detect">
                    <div id="drop-zone" class="drop-zone">
                        <div id="drop-zone-default">
                            <i class="bi bi-cloud-upload" style="font-size: 40px; color: #4A90E2;"></i>
                            <div class="mt-3"><strong>Drag & Drop files here</strong></div>
                            <div class="small-text mt-1">Supports: single image, multiple images, ZIP files, MP4 videos</div>
                        </div>
                        
                        <div id="preview-container" class="preview-container"></div>
                        <input type="file" id="file-input" name="files" multiple accept="image/*,video/mp4,.mp4,.zip">
                    </div>
                    
                    <button type="submit" class="custom-btn">Start Detection</button>
                </form>
                
                {% if single_result %}
                <div class="result-card mt-4 border-start border-4 {{ 'border-danger' if single_result.prediction == 'FAKE' else ('border-warning' if single_result.prediction == 'UNCERTAIN' else 'border-success') }}">
                    <h6><i class="bi bi-search me-2"></i>Detection Result</h6>
                    <div class="row">
                        <div class="col-6">
                            <p><strong>Prediction:</strong> <span class="badge {{ 'bg-danger' if single_result.prediction == 'FAKE' else ('bg-warning text-dark' if single_result.prediction == 'UNCERTAIN' else 'bg-success') }}">{{ single_result.prediction }}</span></p>
                        </div>
                        <div class="col-6 text-end">
                            <p><strong>Confidence:</strong> {{ single_result.confidence }}%</p>
                        </div>
                    </div>
                    <p class="small-text mb-1">
                        <strong>Fake Probability:</strong> {{ single_result.fake_probability }}%
                        {% if single_result.calibration_applied %}
                        <span class="badge bg-info text-dark ms-1">Validated calibration</span>
                        <span class="text-muted ms-1">Raw EfficientNet-B0: {{ single_result.raw_fake_probability }}%</span>
                        {% endif %}
                    </p>
                    <p class="small-text mt-2"><strong>XAI Explanation:</strong> {{ single_result.explanation }}</p>
                    {% if single_result.evidence %}
                    <div class="mt-3">
                        <small><strong>Visual XAI Evidence</strong></small>
                        <div class="row row-cols-2 row-cols-md-4 g-2 mt-1">
                            {% if single_result.evidence.gradcam %}
                            <div class="col"><small>Grad-CAM</small><img src="/outputs/{{ single_result.evidence.gradcam }}" class="img-fluid border rounded"></div>
                            {% endif %}
                            {% if single_result.evidence.dft %}
                            <div class="col"><small>DFT</small><img src="/outputs/{{ single_result.evidence.dft }}" class="img-fluid border rounded"></div>
                            {% endif %}
                            {% if single_result.evidence.canny %}
                            <div class="col"><small>Canny</small><img src="/outputs/{{ single_result.evidence.canny }}" class="img-fluid border rounded"></div>
                            {% endif %}
                            {% if single_result.evidence.noise %}
                            <div class="col"><small>Noise Residual</small><img src="/outputs/{{ single_result.evidence.noise }}" class="img-fluid border rounded"></div>
                            {% endif %}
                        </div>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
                
                {% if batch_data %}
                <div class="result-card mt-4">
                    <h6><strong>Batch Results ({{ batch_data.timestamp }})</strong></h6>
                    
                    <div class="d-flex flex-wrap gap-2 mb-3">
                        <span class="metric-badge bg-light text-dark border">Total: {{ batch_data.stats.total_images }}</span>
                        <span class="metric-badge bg-success text-white">Real: {{ batch_data.stats.real_count }}</span>
                        <span class="metric-badge bg-danger text-white">Fake: {{ batch_data.stats.fake_count }}</span>
                        <span class="metric-badge bg-warning text-dark">Uncertain: {{ batch_data.stats.uncertain_count|default(0) }}</span>
                        <span class="metric-badge bg-secondary text-white">Error: {{ batch_data.stats.error_count }}</span>
                        <span class="metric-badge bg-info text-white">Avg. Conf: {{ batch_data.stats.avg_confidence }}%</span>
                    </div>
                    {% if batch_data.metrics %}
                    <div class="mb-3 p-2 bg-light rounded">
                        <small><strong>Performance Metrics:</strong></small>
                        <small class="text-muted ms-1">
                            {{ batch_data.metrics.message }}
                        </small><br>
                        <div class="d-flex justify-content-between mt-1 flex-wrap gap-2">
                            <small>
                                Acc: {{ batch_data.metrics.accuracy }}{% if batch_data.metrics.available %}%{% endif %}
                                <i class="bi bi-info-circle ms-1 text-muted" title="Accuracy is shown only when filenames contain REAL or FAKE ground-truth labels."></i>
                            </small>
                            <small>
                                Prec: {{ batch_data.metrics.precision }}{% if batch_data.metrics.available %}%{% endif %}
                                <i class="bi bi-info-circle ms-1 text-muted" title="Precision uses FAKE as the positive class: TP / (TP + FP). Of all predicted FAKE, how many were truly FAKE."></i>
                            </small>
                            <small>
                                Rec: {{ batch_data.metrics.recall }}{% if batch_data.metrics.available %}%{% endif %}
                                <i class="bi bi-info-circle ms-1 text-muted" title="Recall uses FAKE as the positive class: TP / (TP + FN). Of all true FAKE, how many were detected."></i>
                            </small>
                            <small>
                                F1: {{ batch_data.metrics.f1_score }}{% if batch_data.metrics.available %}%{% endif %}
                                <i class="bi bi-info-circle ms-1 text-muted" title="F1 uses FAKE as the positive class and balances precision and recall."></i>
                            </small>
                            <small>
                                Raw ROC-AUC: {{ batch_data.metrics.raw_roc_auc.value if batch_data.metrics.raw_roc_auc is defined else 'N/A' }}
                                <i class="bi bi-info-circle ms-1 text-muted" title="Uses filename ground truth and only the raw EfficientNet-B0 fake score. Forensic statistics and calibration are excluded."></i>
                            </small>
                            <small>
                                Calibrated ROC-AUC: {{ batch_data.metrics.calibrated_roc_auc.value if batch_data.metrics.calibrated_roc_auc is defined else 'N/A' }}
                                <i class="bi bi-info-circle ms-1 text-muted" title="Uses the final fake probability after validation-trained forensic calibration."></i>
                            </small>
                        </div>
                        {% if batch_data.metrics.raw_roc_auc is defined %}
                        <small class="text-muted d-block mt-1">{{ batch_data.metrics.raw_roc_auc.message }}</small>
                        {% endif %}
                        {% if batch_data.metrics.calibrated_roc_auc is defined %}
                        <small class="text-muted d-block">{{ batch_data.metrics.calibrated_roc_auc.message }}</small>
                        {% endif %}
                    </div>
                    {% endif %}

                    {% if batch_data.benchmark %}
                    <div class="mb-3 p-2 border rounded">
                        <small><strong>Resource and Processing Statistics</strong></small>
                        <small class="text-muted d-block">Measured automatically from this uploaded batch.</small>
                        <div class="d-flex justify-content-between mt-2 flex-wrap gap-2">
                            <small>Total time: {{ batch_data.benchmark.wall_seconds }} s</small>
                            <small>CPU time: {{ batch_data.benchmark.process_cpu_seconds }} s</small>
                            <small>CPU usage: {% if batch_data.benchmark.process_cpu_utilization_percent is defined and batch_data.benchmark.process_cpu_utilization_percent is not none %}{{ batch_data.benchmark.process_cpu_utilization_percent }}%{% else %}N/A{% endif %}</small>
                        </div>
                        <div class="d-flex justify-content-between mt-1 flex-wrap gap-2">
                            <small>Peak RAM: {{ batch_data.benchmark.peak_process_ram_mb if batch_data.benchmark.peak_process_ram_mb is not none else 'N/A' }} MB</small>
                            <small>Peak GPU allocated: {{ batch_data.benchmark.peak_gpu_allocated_mb if batch_data.benchmark.peak_gpu_allocated_mb is not none else 'N/A' }} MB</small>
                        </div>
                    </div>
                    {% endif %}

                    
                    <div class="table-responsive mt-3" style="max-height: 460px; overflow-y: auto;">
                        <table class="results-table table-sm">
                            <thead>
                                <tr>
                                    <th>File Name</th>
                                    <th>Prediction</th>
                                    <th>Conf (%)</th>
                                    <th class="analysis-column">Analysis</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for result in batch_data.results %}
                                <tr class="{{ 'table-danger' if result[1] == 'FAKE' else '' }}">
                                    <td><small>{{ result[0] }}</small></td>
                                    <td><span class="badge {{ 'bg-danger' if result[1] == 'FAKE' else ('bg-warning text-dark' if result[1] == 'UNCERTAIN' else 'bg-success') }}">{{ result[1] }}</span></td>
                                    <td>{{ result[2] }}</td>
                                    <td class="analysis-column">
                                        {% if result|length > 4 and result[4] %}
                                            <div class="analysis-cell">
                                                <div class="d-flex flex-wrap gap-1 align-items-center">
                                                    <span class="badge bg-secondary">{{ 'EfficientNet-B0 + Calibration' if result[4].get('calibration_applied') else 'EfficientNet-B0' }}</span>
                                                    <span class="badge bg-info text-dark">Fake Probability: {{ result[4].get('fake_probability', '-') }}%</span>
                                                    {% if result[4].get('calibration_applied') %}<span class="badge bg-light text-dark border">Raw: {{ result[4].get('raw_fake_probability', '-') }}%</span>{% endif %}
                                                    <span class="badge bg-primary">Mode: {{ result[4].get('threshold_mode', '-') }}</span>
                                                </div>
                                                <div class="analysis-text">
                                                    {{ result[3] }}
                                                </div>
                                            </div>
                                        {% else %}
                                            <small>{{ result[3] }}</small>
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    <!-- Visual Evidence Preview Section -->
                    <div class="mt-4 pt-3 border-top">
                        <h6 class="mb-3 border-bottom pb-2">Visual Evidence Preview</h6>
                        <div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 g-3">
                            {% for result in batch_data.results %}
                            <div class="col">
                                <div class="card h-100 shadow-sm border-{{ 'danger' if result[1] == 'FAKE' else ('warning' if result[1] == 'UNCERTAIN' else 'success') }} visual-evidence-card">
                                    <div class="card-img-container">
                                        <img src="/outputs/{{ result[0] }}" class="card-img-top w-100 h-100" style="object-fit: contain;" alt="{{ result[0] }}">
                                    </div>
                                    <div class="card-body p-2">
                                        <div class="d-flex justify-content-between align-items-center mb-1">
                                            <span class="badge {{ 'bg-danger' if result[1] == 'FAKE' else ('bg-warning text-dark' if result[1] == 'UNCERTAIN' else 'bg-success') }}" style="font-size: 0.65rem;">
                                                {{ result[1] }}
                                            </span>
                                            <small class="text-muted" style="font-size: 0.7rem;">{{ result[2] }}%</small>
                                        </div>

                                        {% if result|length > 4 and result[4] %}
                                        <div style="font-size: 0.72rem; line-height:1.25;">
                                            <div class="d-flex flex-wrap gap-1 align-items-center mb-1">
                                                <span class="badge bg-secondary">{{ 'EfficientNet-B0 + Calibration' if result[4].get('calibration_applied') else 'EfficientNet-B0' }}</span>
                                                <span class="badge bg-info text-dark">Fake: {{ result[4].get('fake_probability', '-') }}%</span>
                                                {% if result[4].get('calibration_applied') %}<span class="badge bg-light text-dark border">Raw: {{ result[4].get('raw_fake_probability', '-') }}%</span>{% endif %}
                                                <span class="badge bg-primary">{{ result[4].get('threshold_mode', '-') }}</span>
                                            </div>
                                            <div class="evidence-card-note mb-1">
                                                Full explanation is shown in the Analysis table.
                                            </div>

                                            {% if result[4].get('evidence') %}
                                            <div class="row row-cols-2 g-1 mt-1">
                                                {% if result[4]['evidence'].get('gradcam') %}
                                                <div class="col"><small>Grad-CAM</small><img src="/outputs/{{ result[4]['evidence']['gradcam'] }}" class="img-fluid border rounded"></div>
                                                {% endif %}
                                                {% if result[4]['evidence'].get('dft') %}
                                                <div class="col"><small>DFT</small><img src="/outputs/{{ result[4]['evidence']['dft'] }}" class="img-fluid border rounded"></div>
                                                {% endif %}
                                                {% if result[4]['evidence'].get('canny') %}
                                                <div class="col"><small>Canny</small><img src="/outputs/{{ result[4]['evidence']['canny'] }}" class="img-fluid border rounded"></div>
                                                {% endif %}
                                                {% if result[4]['evidence'].get('noise') %}
                                                <div class="col"><small>Noise</small><img src="/outputs/{{ result[4]['evidence']['noise'] }}" class="img-fluid border rounded"></div>
                                                {% endif %}
                                            </div>
                                            {% endif %}
                                        </div>
                                        {% else %}
                                        <p class="card-text mb-0" style="font-size: 0.75rem;">
                                            {{ result[3] }}
                                        </p>
                                        {% endif %}
                                    </div>
                                    <div class="card-footer py-1 bg-white border-top-0">
                                        <code class="text-muted" style="font-size: 0.65rem;">{{ result[0] }}</code>
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>



                    {% if batch_data.plots.result_counts or batch_data.plots.confidence_dist or batch_data.plots.confusion_matrix or batch_data.plots.roc_comparison_curve or batch_data.plots.raw_roc_curve %}
                    <div class="mt-4 pt-3 border-top">
                        <h6 class="mb-3">Statistical Analysis Visualization</h6>
                        <div class="row g-2">
                            {% if batch_data.plots.result_counts %}
                            <div class="col-md-4">
                                <small class="d-block text-center mb-1">Result Counts</small>
                                <img src="{{ url_for('outputs_file', filename=batch_data.plots.result_counts) }}" class="img-fluid border rounded shadow-sm">
                            </div>
                            {% endif %}
                            
                            {% if batch_data.plots.confidence_dist %}
                            <div class="col-md-4">
                                <small class="d-block text-center mb-1">Confidence Distribution</small>
                                <img src="{{ url_for('outputs_file', filename=batch_data.plots.confidence_dist) }}" class="img-fluid border rounded shadow-sm">
                            </div>
                            {% endif %}
                            
                            {% if batch_data.plots.confusion_matrix %}
                            <div class="col-md-4">
                                <small class="d-block text-center mb-1">Confusion Matrix</small>
                                <img src="{{ url_for('outputs_file', filename=batch_data.plots.confusion_matrix) }}" class="img-fluid border rounded shadow-sm">
                            </div>
                            {% endif %}

                            {% if batch_data.plots.roc_comparison_curve %}
                            <div class="col-md-6">
                                <small class="d-block text-center mb-1">Raw and Calibrated ROC Curves</small>
                                <img src="{{ url_for('outputs_file', filename=batch_data.plots.roc_comparison_curve) }}" class="img-fluid border rounded shadow-sm">
                            </div>
                            {% elif batch_data.plots.raw_roc_curve %}
                            <div class="col-md-6">
                                <small class="d-block text-center mb-1">Raw EfficientNet-B0 ROC Curve</small>
                                <img src="{{ url_for('outputs_file', filename=batch_data.plots.raw_roc_curve) }}" class="img-fluid border rounded shadow-sm">
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <footer class="mt-5 text-center small-text">
        <small>
        Deepfake Detector | Model Directory: /models | Output Directory: /outputs | Supported Formats: JPG, PNG, ZIP, MP4<br>
         
        </small>
    </footer>
</div>
</body>
</html>
"""
    return template.replace("{{ accept_attr }}", accept_attr).replace("{{ supported_exts_text }}", supported_exts_text)

# =======================
# Flask Routes
# =======================
@app.route('/', methods=['GET'])
def index():
    cleanup_outputs()  # Clean up old files
    
    template = get_ui_template()
    
    success_msg = session.pop('success_msg', None)
    info_msg = session.pop('info_msg', None)
    error_msg = session.pop('error_msg', None)
    batch_key = session.pop('batch_data_key', None)
    result_key = session.pop('result_data_key', None)

    # Backward compatibility for older saved session payloads.
    batch_data = get_cached_result_data(batch_key) if batch_key else session.pop('batch_data', None)
    result_data = get_cached_result_data(result_key) if result_key else session.pop('result_data', None)

    # Count loaded models
    loaded_count = sum(1 for model in ACTIVE_MODELS.values() if model is not None)
    
    return render_template_string(
        template,
        has_cuda=torch.cuda.is_available(),
        selected_device=SELECTED_DEVICE.type,
        active_model_names=ACTIVE_MODEL_NAMES,
        loaded_count=loaded_count,
        threshold_modes=THRESHOLD_MODES,
        selected_threshold_mode=CURRENT_THRESHOLD_MODE,
        current_threshold_config=get_threshold_config(),
        force_binary_comparison=FORCE_BINARY_COMPARISON,
        max_gradcam_batch_images=MAX_GRADCAM_BATCH_IMAGES,
        success_msg=success_msg,
        info_msg=info_msg,
        error_msg=error_msg,
        batch_data=batch_data,
        single_result=result_data
    )

@app.route('/set_device', methods=['POST'])
def set_device():
    global SELECTED_DEVICE, ACTIVE_MODELS
    device = request.form.get('device', 'cpu')

    if device == 'cuda' and torch.cuda.is_available():
        SELECTED_DEVICE = torch.device('cuda')
        target_label = "GPU"
    else:
        SELECTED_DEVICE = torch.device('cpu')
        target_label = "CPU"

    # Move any loaded model to the newly selected device to prevent tensor/model mismatch errors.
    try:
        for name, model in ACTIVE_MODELS.items():
            if model is not None:
                ACTIVE_MODELS[name] = model.to(SELECTED_DEVICE).eval()
        session['success_msg'] = f"Switched to {target_label}"
    except Exception as e:
        session['error_msg'] = f"Device switch failed: {e}"

    return redirect(url_for('index'))

@app.route('/set_threshold_mode', methods=['POST'])
def set_threshold_mode_route():
    mode = request.form.get('threshold_mode', 'standard')
    if set_threshold_mode(mode):
        cfg = get_threshold_config()
        session['success_msg'] = (
            f"Threshold mode set to {cfg['label']} "
            f"(REAL ≤ {int(cfg['real'] * 100)}%, FAKE ≥ {int(cfg['fake'] * 100)}%)."
        )
    else:
        session['error_msg'] = "Invalid threshold mode selected."
    return redirect(url_for('index'))


@app.route('/set_force_binary_mode', methods=['POST'])
def set_force_binary_mode_route():
    global FORCE_BINARY_COMPARISON
    FORCE_BINARY_COMPARISON = request.form.get('force_binary') == '1'
    if FORCE_BINARY_COMPARISON:
        session['success_msg'] = "Forced REAL/FAKE comparison mode enabled at the 0.50 decision line."
    else:
        session['success_msg'] = "Forced comparison mode disabled; UNCERTAIN is available again."
    return redirect(url_for('index'))

@app.route('/set_gradcam_limit', methods=['POST'])
def set_gradcam_limit_route():
    global MAX_GRADCAM_BATCH_IMAGES

    raw_limit = request.form.get('gradcam_limit', '').strip()
    try:
        limit = int(raw_limit)
    except ValueError:
        session['error_msg'] = "Grad-CAM batch limit must be a whole number."
        return redirect(url_for('index'))

    if limit < 0 or limit > 500:
        session['error_msg'] = "Grad-CAM batch limit must be between 0 and 500."
        return redirect(url_for('index'))

    MAX_GRADCAM_BATCH_IMAGES = limit
    if limit == 0:
        session['success_msg'] = "Batch Grad-CAM disabled. DFT, Canny, and noise evidence will still be generated."
    else:
        session['success_msg'] = f"Batch Grad-CAM limit set to the first {limit} image(s)."
    return redirect(url_for('index'))

@app.route('/load_model', methods=['POST'])
def load_model_route():
    # Support multi-select: <input name="model_files" multiple>
    # Backward-compatible: accept legacy single name="model_file"
    files = []
    if 'model_files' in request.files:
        files = request.files.getlist('model_files')
    elif 'model_file' in request.files:
        f = request.files.get('model_file')
        files = [f] if f else []

    # Remove empty placeholders (no selection)
    files = [f for f in files if f and (f.filename or '').strip()]

    if not files:
        session['error_msg'] = "No model selected"
        return redirect(url_for('index'))

    success_msgs = []
    error_msgs = []
    info_msgs = []

    if len(files) > 1:
        info_msgs.append("Multiple model files selected. Loaded the first selected model; choose another file to switch again.")

    model_file = files[0]
    safe_model_name = sanitize_filename(model_file.filename, default="model.pth")
    try:
        temp_path = os.path.join(MODELS_FOLDER, safe_model_name)
        model_file.save(temp_path)

        # Always use auto-detection. If a model is already active, the loader will just replaces it.
        success, msg = load_model_with_auto_detect(temp_path)

        if success:
            success_msgs.append(msg)
        else:
            error_msgs.append(msg)

    except Exception as e:
        error_msgs.append(f"Failed to load model '{safe_model_name}': {str(e)}")

    if success_msgs and not error_msgs:
        session['success_msg'] = " | ".join(success_msgs)
    elif success_msgs and error_msgs:
        session['success_msg'] = " | ".join(success_msgs)
        session['error_msg'] = " | ".join(error_msgs)
    else:
        session['error_msg'] = " | ".join(error_msgs) if error_msgs else "Failed to load model(s)."

    if info_msgs:
        session['info_msg'] = " | ".join(info_msgs)

    return redirect(url_for('index'))

@app.route('/clear_models', methods=['POST'])
def clear_models():
    """Clear the loaded EfficientNet-B0 model if one is loaded."""
    global ACTIVE_MODELS, ACTIVE_MODEL_NAMES

    loaded_count = sum(1 for model in ACTIVE_MODELS.values() if model is not None)
    if loaded_count == 0:
        session['info_msg'] = "No model to clear. Please load an EfficientNet-B0 model first."
        return redirect(url_for('index'))

    release_active_model()
    session['success_msg'] = "Loaded EfficientNet-B0 model cleared successfully."
    return redirect(url_for('index'))

@app.route('/detect', methods=['POST'])
def detect():
    if 'files' in request.files:
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            session['error_msg'] = "No valid files selected"
            return redirect(url_for('index'))
        
        try:
            zip_files = [f for f in files if f.filename.lower().endswith('.zip')]
            if zip_files:
                zip_wall_start = time.perf_counter()
                zip_cpu_start = time.process_time()
                zip_monitor = BenchmarkResourceMonitor(SELECTED_DEVICE)
                zip_monitor.__enter__()
                batch_data = process_zip_file(zip_files[0].stream)
                zip_wall_seconds = time.perf_counter() - zip_wall_start
                zip_cpu_seconds = time.process_time() - zip_cpu_start
                zip_monitor.__exit__(None, None, None)
                batch_data = attach_uploaded_batch_benchmark(batch_data, zip_wall_seconds, zip_cpu_seconds, zip_monitor)
                session['batch_data_key'] = cache_result_data(batch_data)
                return redirect(url_for('index'))
            
            batch_data = process_mixed_files(files)
            session['batch_data_key'] = cache_result_data(batch_data)
            return redirect(url_for('index'))
            
        except Exception as e:
            session['error_msg'] = str(e)
            return redirect(url_for('index'))
    
    if 'image_file' in request.files:
        file = request.files['image_file']
        if file.filename == '':
            session['error_msg'] = "No image selected"
            return redirect(url_for('index'))
        
        try:
            filename = make_batch_filename(file.filename or "single_image.jpg", prefix="single")
            img_np = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if img is None:
                session['error_msg'] = "Failed to read image"
                return redirect(url_for('index'))
            
            success, result = process_image(img)
            
            if not success:
                session['error_msg'] = result
            else:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, filename), img)
                
                canny_name = f"canny_single_{filename}"
                dft_name = f"dft_single_{filename}"
                noise_name = f"noise_single_{filename}"
                gradcam_name = f"gradcam_single_{filename}"
                
                if result.get("canny_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, canny_name), result["canny_raw"])
                if result.get("dft_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, dft_name), result["dft_raw"])
                if result.get("noise_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, noise_name), result["noise_raw"])
                if result.get("gradcam_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, gradcam_name), result["gradcam_raw"])
                
                result_data = {
                    "prediction": result["final_prediction"],
                    "confidence": result["final_confidence"],
                    "explanation": result.get("xai_explanation", "XAI explanation unavailable for this image."),
                    "model": "EfficientNet-B0",
                    "fake_probability": result.get("fake_probability", 0.0),
                    "raw_fake_probability": result.get("raw_fake_probability", 0.0),
                    "calibration_applied": result.get("calibration_applied", False),
                    "decision_method": result.get("decision_method", "EfficientNet-B0"),
                    "forensic_scores": result.get("forensic_scores", {}),
                    "evidence": {
                        "canny": canny_name if result.get("canny_raw") is not None else None,
                        "dft": dft_name if result.get("dft_raw") is not None else None,
                        "noise": noise_name if result.get("noise_raw") is not None else None,
                        "gradcam": gradcam_name if result.get("gradcam_raw") is not None else None,
                    }
                }
                session['result_data_key'] = cache_result_data(result_data)

        except Exception as e:
            session['error_msg'] = str(e)
        
        return redirect(url_for('index'))
    
    session['error_msg'] = "No files uploaded"
    return redirect(url_for('index'))

@app.route('/outputs/<filename>')
def outputs_file(filename):
    return send_from_directory(OUTPUTS_FOLDER, filename)


# --------------------------
# Integrated benchmark mode
# --------------------------
class BenchmarkResourceMonitor:
    def __init__(self, device, interval=0.05):
        self.device = device
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None
        self.peak_rss = 0
        self.peak_gpu_allocated = 0
        self.peak_gpu_reserved = 0

    @staticmethod
    def rss_bytes():
        if os.name != 'nt':
            return 0
        class Counters(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else 0

    def sample(self):
        self.peak_rss = max(self.peak_rss, self.rss_bytes())
        if self.device.type == 'cuda' and torch.cuda.is_available():
            self.peak_gpu_allocated = max(self.peak_gpu_allocated, int(torch.cuda.memory_allocated(self.device)))
            self.peak_gpu_reserved = max(self.peak_gpu_reserved, int(torch.cuda.memory_reserved(self.device)))

    def loop(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        self.sample()


def benchmark_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def benchmark_samples(root, limit_per_class, seed):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    groups = {0: [], 1: []}
    for current_root, _, filenames in os.walk(root):
        parts = {part.lower() for part in os.path.normpath(current_root).split(os.sep)}
        label = 0 if 'real' in parts else 1 if 'fake' in parts else None
        if label is None:
            continue
        groups[label].extend(
            os.path.join(current_root, name) for name in filenames
            if os.path.splitext(name)[1].lower() in extensions
        )
    if not groups[0] or not groups[1]:
        raise ValueError('Benchmark dataset must contain image files below real and fake folders.')
    for paths in groups.values():
        paths.sort(key=str.lower)
    count = min(len(groups[0]), len(groups[1]))
    if limit_per_class is not None and int(limit_per_class) > 0:
        count = min(count, int(limit_per_class))
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        indices = np.arange(len(groups[label]))
        rng.shuffle(indices)
        selected.extend((groups[label][int(index)], label) for index in indices[:count])
    selected.sort(key=lambda item: item[0].lower())
    return selected


def benchmark_nvidia():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=False
        )
        return {'available': result.returncode == 0, 'raw': result.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'available': False, 'error': str(exc)}


def benchmark_write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_process_cpu_percent(cpu_seconds, wall_seconds):
    """Return whole-process CPU use normalized to the machine's logical cores."""
    if not wall_seconds:
        return None
    logical_cores = max(1, int(os.cpu_count() or 1))
    return round(min(100.0, cpu_seconds / (wall_seconds * logical_cores) * 100.0), 4)


def attach_uploaded_batch_benchmark(batch_data, wall_seconds, cpu_seconds, monitor=None):
    """Save and return benchmark statistics for a normal Flask upload batch."""
    labelled = []
    prediction_rows = []
    for result in batch_data.get('results', []):
        if len(result) < 5 or not isinstance(result[4], dict):
            continue
        true_label = infer_true_label_from_filename(result[0])
        fake_probability = result[4].get('fake_probability')
        raw_fake_probability = result[4].get('raw_fake_probability')
        if true_label is None or fake_probability is None or raw_fake_probability is None:
            continue
        probability = float(fake_probability)
        if probability > 1.0:
            probability /= 100.0
        raw_probability = float(raw_fake_probability)
        if raw_probability > 1.0:
            raw_probability /= 100.0
        prediction = int(probability >= 0.5)
        labelled.append((true_label, prediction, probability, raw_probability))
        prediction_rows.append({
            'filename': result[0],
            'true_label': 'FAKE' if true_label else 'REAL',
            'application_prediction': result[1],
            'fake_probability': round(probability, 8),
            'raw_fake_probability': round(raw_probability, 8),
            'binary_prediction_0_5': 'FAKE' if prediction else 'REAL',
        })

    labels = [row[0] for row in labelled]
    predictions = [row[1] for row in labelled]
    probabilities = [row[2] for row in labelled]
    raw_probabilities = [row[3] for row in labelled]
    summary = {
        'source': 'normal Flask upload or drag-and-drop batch',
        'scope': 'resource and processing statistics for the uploaded batch',
        'sample_count_uploaded': len(batch_data.get('results', [])),
        'sample_count_labelled': len(labels),
        'real_count_labelled': sum(label == 0 for label in labels),
        'fake_count_labelled': sum(label == 1 for label in labels),
        'application_uncertain_count': batch_data.get('stats', {}).get('uncertain_count', 0),
        'wall_seconds': round(wall_seconds, 6),
        'process_cpu_seconds': round(cpu_seconds, 6),
        'process_cpu_utilization_percent': normalized_process_cpu_percent(cpu_seconds, wall_seconds),
        'peak_process_ram_mb': round(monitor.peak_rss / 1024 ** 2, 4) if monitor else None,
        'peak_gpu_allocated_mb': round(monitor.peak_gpu_allocated / 1024 ** 2, 4) if monitor else None,
        'peak_gpu_reserved_mb': round(monitor.peak_gpu_reserved / 1024 ** 2, 4) if monitor else None,
        'checkpoint': ACTIVE_MODEL_NAMES.get('efficientnet'),
        'raw_efficientnet_roc_auc': (
            round(float(roc_auc_score(labels, raw_probabilities)), 6)
            if len(set(labels)) == 2 else None
        ),
        'calibrated_system_roc_auc': (
            round(float(roc_auc_score(labels, probabilities)), 6)
            if len(set(labels)) == 2 and ACTIVE_CALIBRATOR_BUNDLE is not None else None
        ),
    }
    timestamp = batch_data.get('timestamp', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(RESOURCES_STATISTIC_FOLDER, exist_ok=True)
    with open(os.path.join(RESOURCES_STATISTIC_FOLDER, f'flask_batch_{timestamp}.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    benchmark_write_csv(os.path.join(RESOURCES_STATISTIC_FOLDER, f'flask_predictions_{timestamp}.csv'), prediction_rows)
    batch_data['benchmark'] = summary
    return batch_data


def run_benchmark():
    parser = argparse.ArgumentParser(description='Benchmark the FYP2 custom EfficientNet-B0 detector.')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-dataset-root', required=True)
    parser.add_argument('--benchmark-checkpoint', default=os.path.join(MODELS_FOLDER, 'efficientnet_b0_best.pth'))
    parser.add_argument('--benchmark-limit-per-class', type=int, default=None)
    parser.add_argument('--benchmark-seed', type=int, default=42)
    parser.add_argument('--benchmark-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    global SELECTED_DEVICE, ACTIVE_MODELS, ACTIVE_MODEL_NAMES
    checkpoint = os.path.abspath(args.benchmark_checkpoint)
    dataset_root = os.path.abspath(args.benchmark_dataset_root)
    if args.benchmark_device == 'cuda' or (args.benchmark_device == 'auto' and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable.')
        SELECTED_DEVICE = torch.device('cuda')
    else:
        SELECTED_DEVICE = torch.device('cpu')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f'Dataset folder does not exist: {dataset_root}')
    os.makedirs(RESOURCES_STATISTIC_FOLDER, exist_ok=True)
    for name in os.listdir(RESOURCES_STATISTIC_FOLDER):
        if name.endswith(('.json', '.csv')):
            os.remove(os.path.join(RESOURCES_STATISTIC_FOLDER, name))
    ACTIVE_MODELS = {'efficientnet': None}
    ACTIVE_MODEL_NAMES = {'efficientnet': None}
    success, message = load_model_with_auto_detect(checkpoint)
    if not success:
        raise RuntimeError(message)
    model = ACTIVE_MODELS['efficientnet']
    samples = benchmark_samples(dataset_root, args.benchmark_limit_per_class, args.benchmark_seed)
    rows, labels, predictions, probabilities, timings, errors = [], [], [], [], [], []
    raw_predictions, raw_probabilities, cnn_timings = [], [], []
    with BenchmarkResourceMonitor(SELECTED_DEVICE) as monitor:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        with torch.inference_mode():
            for image_path, label in samples:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    errors.append({'path': image_path, 'error': 'OpenCV could not decode image'})
                    continue
                start = time.perf_counter()
                tensor = preprocess_efficientnet_image(image)
                if SELECTED_DEVICE.type == 'cuda':
                    torch.cuda.synchronize(SELECTED_DEVICE)
                logit = model(tensor)
                if SELECTED_DEVICE.type == 'cuda':
                    torch.cuda.synchronize(SELECTED_DEVICE)
                cnn_elapsed = (time.perf_counter() - start) * 1000.0
                raw_probability = float(torch.sigmoid(logit).reshape(-1)[0].item())
                forensic_scores = calculate_forensic_scores(image)
                probability, calibration_applied = calibrate_fake_probability(raw_probability, forensic_scores)
                total_elapsed = (time.perf_counter() - start) * 1000.0

                raw_prediction = int(raw_probability >= 0.5)
                prediction = int(probability >= 0.5)
                labels.append(label); predictions.append(prediction); probabilities.append(probability); timings.append(total_elapsed)
                raw_predictions.append(raw_prediction); raw_probabilities.append(raw_probability); cnn_timings.append(cnn_elapsed)
                rows.append({
                    'path': image_path,
                    'true_label': 'FAKE' if label else 'REAL',
                    'raw_fake_probability': round(raw_probability, 8),
                    'fake_probability': round(probability, 8),
                    'raw_prediction': 'FAKE' if raw_prediction else 'REAL',
                    'prediction': 'FAKE' if prediction else 'REAL',
                    'application_state': probability_to_three_state(probability)[0],
                    'calibration_applied': calibration_applied,
                    'cnn_inference_ms': round(cnn_elapsed, 6),
                    'processing_ms': round(total_elapsed, 6),
                    **forensic_scores,
                })
                monitor.sample()
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
    if not labels:
        raise RuntimeError('No images were successfully evaluated.')
    raw_metrics = {
        'accuracy_percent': round(accuracy_score(labels, raw_predictions) * 100, 4),
        'precision_fake_percent': round(precision_score(labels, raw_predictions, zero_division=0) * 100, 4),
        'recall_fake_percent': round(recall_score(labels, raw_predictions, zero_division=0) * 100, 4),
        'f1_fake_percent': round(f1_score(labels, raw_predictions, zero_division=0) * 100, 4),
        'confusion_matrix_real_fake': confusion_matrix(labels, raw_predictions, labels=[0, 1]).tolist(),
        'binary_threshold': 0.5,
        'roc_auc': round(float(roc_auc_score(labels, raw_probabilities)), 6) if len(set(labels)) == 2 else None,
    }
    calibration_active = ACTIVE_CALIBRATOR_BUNDLE is not None
    metrics = {
        'system': 'fyp2_efficientnet_b0_forensic_calibrated' if calibration_active else 'fyp2_custom_efficientnet_b0',
        'model': 'EfficientNet-B0 + validated forensic calibration' if calibration_active else 'EfficientNet-B0',
        'sample_count_requested': len(samples), 'sample_count_evaluated': len(labels),
        'real_count': sum(label == 0 for label in labels), 'fake_count': sum(label == 1 for label in labels), 'errors': len(errors),
        'accuracy_percent': round(accuracy_score(labels, predictions) * 100, 4),
        'precision_fake_percent': round(precision_score(labels, predictions, zero_division=0) * 100, 4),
        'recall_fake_percent': round(recall_score(labels, predictions, zero_division=0) * 100, 4),
        'f1_fake_percent': round(f1_score(labels, predictions, zero_division=0) * 100, 4),
        'roc_auc': round(float(roc_auc_score(labels, probabilities)), 6) if len(set(labels)) == 2 else None,
        'confusion_matrix_real_fake': confusion_matrix(labels, predictions, labels=[0, 1]).tolist(), 'binary_threshold': 0.5,
        'wall_seconds': round(wall_seconds, 6), 'process_cpu_seconds': round(cpu_seconds, 6),
        'process_cpu_utilization_percent': normalized_process_cpu_percent(cpu_seconds, wall_seconds),
        'mean_cnn_inference_ms': round(statistics.mean(cnn_timings), 6),
        'mean_image_processing_ms': round(statistics.mean(timings), 6),
        'median_image_processing_ms': round(statistics.median(timings), 6),
        'p95_image_processing_ms': round(float(np.percentile(timings, 95)), 6),
        'raw_efficientnet_baseline': raw_metrics,
        'calibration_applied': calibration_active,
        'calibrator_method': ACTIVE_CALIBRATOR_BUNDLE.get('selected_method') if calibration_active else None,
        'calibrator': FORENSIC_CALIBRATOR_PATH if calibration_active else None,
        'calibrator_sha256': benchmark_sha256(FORENSIC_CALIBRATOR_PATH) if calibration_active else None,
        'peak_process_ram_mb': round(monitor.peak_rss / 1024 ** 2, 4), 'peak_gpu_allocated_mb': round(monitor.peak_gpu_allocated / 1024 ** 2, 4), 'peak_gpu_reserved_mb': round(monitor.peak_gpu_reserved / 1024 ** 2, 4),
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()), 'checkpoint': checkpoint, 'checkpoint_sha256': benchmark_sha256(checkpoint), 'checkpoint_size_mb': round(os.path.getsize(checkpoint) / 1024 ** 2, 6), 'dataset_root': dataset_root, 'device_used': str(SELECTED_DEVICE), 'torch_version': torch.__version__, 'cuda_available': bool(torch.cuda.is_available()), 'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    manifest = {'created_at_local': datetime.datetime.now().isoformat(timespec='seconds'), 'script': os.path.abspath(__file__), 'python': sys.version, 'platform': platform.platform(), 'checkpoint': checkpoint, 'checkpoint_sha256': metrics['checkpoint_sha256'], 'calibrator': metrics['calibrator'], 'calibrator_sha256': metrics['calibrator_sha256'], 'dataset_root': dataset_root, 'sample_count': len(samples), 'real_count': sum(label == 0 for _, label in samples), 'fake_count': sum(label == 1 for _, label in samples), 'limit_per_class': args.benchmark_limit_per_class, 'seed': args.benchmark_seed, 'device_used': str(SELECTED_DEVICE), 'cuda_available': bool(torch.cuda.is_available()), 'cuda_device': metrics['cuda_device'], 'nvidia_smi': benchmark_nvidia(), 'provenance_note': 'Integrated FYP2 benchmark; one EfficientNet-B0 pass plus a validation-trained forensic calibration layer. The CNN checkpoint is unchanged and Flask is not started.'}
    with open(os.path.join(RESOURCES_STATISTIC_FOLDER, 'benchmark_manifest.json'), 'w', encoding='utf-8') as handle: json.dump(manifest, handle, indent=2)
    with open(os.path.join(RESOURCES_STATISTIC_FOLDER, 'metrics.json'), 'w', encoding='utf-8') as handle: json.dump(metrics, handle, indent=2)
    benchmark_write_csv(os.path.join(RESOURCES_STATISTIC_FOLDER, 'metrics.csv'), [metrics]); benchmark_write_csv(os.path.join(RESOURCES_STATISTIC_FOLDER, 'predictions.csv'), rows)
    if errors:
        with open(os.path.join(RESOURCES_STATISTIC_FOLDER, 'errors.json'), 'w', encoding='utf-8') as handle: json.dump(errors, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        run_benchmark()
    else:
        auto_load_startup_model()
        app.run(host='0.0.0.0', port=8080, debug=False)