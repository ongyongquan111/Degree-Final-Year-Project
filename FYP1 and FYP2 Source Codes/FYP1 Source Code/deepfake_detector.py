# Student Name: Ong Yong Quan
# Student ID: 243UT246XG
# FYP Title: Identifying Deepfake Images
import os
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
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, session
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from safetensors.torch import load_file as load_safetensors
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from torchvision.models import resnet50
import timm
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
# Global Variables
# --------------------------
ACTIVE_MODELS = {
    "xception": None,   # 3-channel Xception
    "resnet": None,     # 3-channel ResNet50
    "vit": None,        # 3-channel ViT
    "dft": None,        # 1-channel DFT model
    "canny": None       # 1-channel Canny model
}
ACTIVE_MODEL_NAMES = {
    "xception": None,
    "resnet": None,
    "vit": None,
    "dft": None,
    "canny": None
}
SELECTED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video processing limits. These keep the existing 20% sampling policy while
# preventing an unusually long video from creating an unbounded workload.
VIDEO_SAMPLE_RATIO = 0.20
VIDEO_MIN_SAMPLES = 5
VIDEO_MAX_SAMPLES = 120

# --------------------------
# Image size configuration
# --------------------------
IMAGE_SIZE = (256, 256)  # Global

# --------------------------
# MODEL_SPECIFIC_INPUT_SIZES
# --------------------------
RESNET_INPUT_SIZE = (256, 256)
XCEPTION_INPUT_SIZE = (299, 299)  # trained with Resize((299,299))
VIT_INPUT_SIZE = (224, 224)       # trained with Resize((224,224))
VIT_MODEL_NAME = 'google/vit-base-patch16-224'

# --------------------------
# Supported Models File Types
# --------------------------
SUPPORTED_MODELS = {
    ".pt": {
        "loader": lambda x: torch.load(x, map_location=SELECTED_DEVICE, weights_only=False),
        "post_processor": None
    },
    ".pth": {
        "loader": lambda x: torch.load(x, map_location=SELECTED_DEVICE, weights_only=False),
        "post_processor": None
    },
    ".safetensors": {
        "loader": lambda x: load_safetensors(x, device=SELECTED_DEVICE.type),
        "post_processor": None
    }
}

# --------------------------
# Image Preprocessing Functions
# --------------------------
def preprocess_rgb_image(img_array, target_size=IMAGE_SIZE, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), apply_normalize=True):
    """
    RGB Preprocessing:

    Match preprocessing on training.
    - ResNet/Xception: Normalize(mean=0.5, std=0.5)
    - ViT: ToTensor only (no Normalize), Resize to 224

    This function handles RGB conversion, resizing and normalization.
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


def preprocess_resnet_image(img_array):
    """ResNet preprocessing: 256x256 + Normalize(0.5,0.5,0.5)."""
    return preprocess_rgb_image(
        img_array,
        target_size=RESNET_INPUT_SIZE,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        apply_normalize=True
    )


def preprocess_xception_image(img_array):
    """Xception preprocessing: 299x299 + Normalize(0.5,0.5,0.5)."""
    return preprocess_rgb_image(
        img_array,
        target_size=XCEPTION_INPUT_SIZE,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        apply_normalize=True
    )


def preprocess_vit_image(img_array):
    """
    ViT preprocessing: 224x224 + ToTensor only.

    Training loop passes raw tensors without processor normalization.
    So inference must do the same.
    """
    return preprocess_rgb_image(
        img_array,
        target_size=VIT_INPUT_SIZE,
        apply_normalize=False
    )

def preprocess_canny_image(img_array, target_size=IMAGE_SIZE):
    """Extract and preprocess Canny edges"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        canny_resized = cv2.resize(canny, target_size)
        canny_normalized = canny_resized / 255.0
        canny_tensor = torch.tensor(canny_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return canny_tensor.to(SELECTED_DEVICE)
    except Exception as e:
        print(f"Canny preprocessing error: {e}")
        return None

def preprocess_dft_image(img_array, target_size=IMAGE_SIZE):
    """Extract and preprocess DFT magnitude."""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        dft_resized = cv2.resize(magnitude_spectrum, target_size)
        dft_normalized = cv2.normalize(dft_resized, None, 0, 1, cv2.NORM_MINMAX)
        dft_tensor = torch.tensor(dft_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return dft_tensor.to(SELECTED_DEVICE)
    except Exception as e:
        print(f"DFT preprocessing error: {e}")
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
def build_resnet50_3channel():
    """ResNet50 with 3-channel RGB input - SIGMOID output"""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)  # 1 output for binary classification
    return model

def build_xception_3channel():
    """
    Xception with 3-channel RGB input - SIGMOID output

    Notes: Matches on jupyter notebook training:
      - timm.create_model('xception', pretrained=True/False, num_classes=1000)
      - replace model.fc with nn.Sequential(Linear->ReLU->Dropout->Linear(1))

    Used pretrained=False here because we going to load checkpoint weights here.
    """
    # Build base Xception model
    model = timm.create_model('xception', pretrained=False, num_classes=1000)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1)
    )
    return model

def build_vit_3channel():
    """
    ViT with 3-channel RGB input - SIGMOID output

    Matches with jupyter notebook training:
      - ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
      - replace classifier with nn.Linear(in_features, 1)

    Note: Training resized inputs to 224x224.
    """
    model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1)
    return model

def normalize_legacy_vit_state_dict(state_dict):
    """Translate older Transformers ViT checkpoint names when needed."""
    if not isinstance(state_dict, dict):
        return state_dict
    replacements = (
        ('vit.encoder.layer.', 'vit.layers.'),
        ('.attention.attention.query.', '.attention.q_proj.'),
        ('.attention.attention.key.', '.attention.k_proj.'),
        ('.attention.attention.value.', '.attention.v_proj.'),
        ('.attention.output.dense.', '.attention.o_proj.'),
        ('.intermediate.dense.', '.mlp.fc1.'),
        ('.output.dense.', '.mlp.fc2.'),
    )
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        normalized[new_key] = value
    return normalized

def build_canny_cnn():
    """Simple CNN for Canny edge classification - SIGMOID output"""
    class CannyCNN(nn.Module):
        def __init__(self):
            super(CannyCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # Updated for 256x256: 256 -> 128 -> 64 -> 32, so 128 * 32 * 32
            self.fc1 = nn.Linear(128 * 32 * 32, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 1)  # 1 output for binary classification
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 128 * 32 * 32)  # Updated for 256x256
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    return CannyCNN()

def build_dft_cnn():
    """Simple CNN for DFT magnitude classification - SIGMOID output."""
    class DFTCNN(nn.Module):
        def __init__(self):
            super(DFTCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # Updated for 256x256: 256 -> 128 -> 64 -> 32, so 128 * 32 * 32
            self.fc1 = nn.Linear(128 * 32 * 32, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 1)  # 1 output for binary classification
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 128 * 32 * 32)  # Updated for 256x256
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    return DFTCNN()

# --------------------------
# Model Type Detection Function
# --------------------------
def detect_model_type_from_state_dict(state_dict, filename=None):
    """Detect model type from state dictionary keys"""
    keys = list(state_dict.keys())
    
    # Check filename for hints first
    filename_hint = None
    if filename:
        filename_lower = filename.lower()
        if 'vit' in filename_lower or 'transformer' in filename_lower:
            filename_hint = 'vit'
        elif 'xception' in filename_lower:
            filename_hint = 'xception'
        elif 'resnet' in filename_lower:
            filename_hint = 'resnet'
        elif 'dft' in filename_lower:
            filename_hint = 'dft'
        elif 'canny' in filename_lower:
            filename_hint = 'canny'
    
    # Check for ViT keys
    vit_keys = ['vit.', 'patch_embed', 'position_embeddings', 'attention.attention']
    if any(any(vk in key.lower() for vk in vit_keys) for key in keys):
        return "vit"
    
    # Check for Xception keys.
    xception_keys = ['blocks.', 'depthwise', 'pointwise', 'block', 'skip', 'separableconv']
    if any(any(xk in key.lower() for xk in xception_keys) for key in keys):
        return "xception"
    
    # Check for ResNet keys
    resnet_keys = ['layer1.', 'layer2.', 'layer3.', 'layer4.', 'downsample', 'bn1', 'bn2', 'bn3']
    if any(any(rk in key.lower() for rk in resnet_keys) for key in keys):
        return "resnet"
    
    # Check for simple CNN architecture (1-channel models)
    # Find conv1.weight to check input channels.
    conv1_keys = [k for k in keys if 'conv1' in k.lower() and 'weight' in k.lower()]
    
    if conv1_keys:
        conv1_key = conv1_keys[0]
        try:
            conv1_shape = state_dict[conv1_key].shape
            if len(conv1_shape) == 4:  # Should be [out_channels, in_channels, height, width]
                in_channels = conv1_shape[1]
                
                # Check if it's a 1-channel model (DFT or Canny)
                if in_channels == 1:
                    # Check for DFT or Canny specific patterns.
                    has_dft_keys = any('dft' in key.lower() for key in keys)
                    has_canny_keys = any('canny' in key.lower() for key in keys)
                    
                    if has_dft_keys:
                        return "dft"
                    elif has_canny_keys:
                        return "canny"
                    else:
                        # Check filename hint
                        if filename_hint in ['dft', 'canny']:
                            return filename_hint
                        
                        # Default based on key structure - simple CNN with few layers
                        if len(keys) < 30:  # Simple CNN has fewer parameters
                            # Check for typical CNN structure
                            conv_count = sum(1 for k in keys if 'conv' in k.lower() and 'weight' in k.lower())
                            fc_count = sum(1 for k in keys if ('fc' in k.lower() or 'linear' in k.lower()) and 'weight' in k.lower())
                            
                            if conv_count >= 2 and fc_count >= 1:
                                # Could be either DFT or Canny, use filename hint
                                if filename_hint:
                                    return filename_hint
                                else:
                                    # Default to DFT if filename doesn't help
                                    return "dft"
                
                # Check for 3-channel models
                elif in_channels == 3:
                    # Check if it's ResNet or Xception
                    if filename_hint in ['resnet', 'xception', 'vit']:
                        return filename_hint
                    
                    # Default to ResNet for 3-channel
                    return "resnet"
        except:
            pass
    
    # Still can't determine? we use filename hint..
    if filename_hint:
        return filename_hint
    
    return "resnet"

def detect_model_type_from_object(obj, filename=None):
    """Detect model type from loaded PyTorch object"""
    # First, check is it a dictionary (state dict).
    if isinstance(obj, dict):
        return detect_model_type_from_state_dict(obj, filename)
    
    # Check is it nn.Module
    if isinstance(obj, nn.Module):
        module_name = obj.__class__.__name__.lower()
        if "vit" in module_name:
            return "vit"
        elif "xception" in module_name:
            return "xception"
        elif "resnet" in module_name:
            return "resnet"
        elif "canny" in module_name:
            return "canny"
        elif "dft" in module_name:
            return "dft"
    
    return None

# --------------------------
# Load Models with Auto-Detection
# --------------------------
def load_model_with_auto_detect(model_file_path):
    """
    Load a model with automatic type detection
    Returns: (success: bool, message: str)
    """
    global ACTIVE_MODELS, ACTIVE_MODEL_NAMES
    
    if not os.path.exists(model_file_path):
        return False, f"File not found: {model_file_path}"
    
    file_ext = os.path.splitext(model_file_path)[1].lower()
    if file_ext not in SUPPORTED_MODELS:
        supported_exts = ", ".join(SUPPORTED_MODELS.keys())
        return False, f"Unsupported file type: {file_ext}. Supported: {supported_exts}"
    
    try:
        print(f"Loading model with auto-detection: {model_file_path}")
        
        # Load the model file
        loader = SUPPORTED_MODELS[file_ext]["loader"]
        loaded_data = loader(model_file_path)
        
        # Get filename for detection hints
        filename = os.path.basename(model_file_path)
        
        # Handle state dict with wrapper prefixes (like "model.")
        if isinstance(loaded_data, dict):
            keys = list(loaded_data.keys())
            
            # Remove any "model." or "module." prefix from state dict keys
            if all(k.startswith("model.") for k in keys):
                loaded_data = {k[6:]: v for k, v in loaded_data.items()}
            elif all(k.startswith("module.") for k in keys):
                loaded_data = {k[7:]: v for k, v in loaded_data.items()}
        
        # Detect model type
        model_type = detect_model_type_from_object(loaded_data, filename)
        
        if model_type is None:
            return False, "Unable to auto-detect model type from the file. Please check is it a valid model file..."
        
        print(f"Auto-detected model type: {model_type}")
        
        # Check if model is already loaded
        if ACTIVE_MODELS[model_type] is not None:
            return False, f"A {model_type.upper()} model is already loaded. Please clear it first or load a different model type."
        
        # Build or use the model based on type
        if model_type == "resnet":
            model = build_resnet50_3channel()
            if isinstance(loaded_data, dict):  # State dict
                try:
                    model.load_state_dict(loaded_data, strict=True)
                except Exception as e:
                    return False, f"Failed to load ResNet state dict (strict=True): {e}"
            else:  # Full model
                model = loaded_data

        elif model_type == "xception":
            model = build_xception_3channel()
            if isinstance(loaded_data, dict):  # State dict
                try:
                    model.load_state_dict(loaded_data, strict=True)
                except Exception as e:
                    return False, f"Failed to load Xception state dict (strict=True): {e}"
            else:  # Full model
                model = loaded_data

        elif model_type == "vit":
            model = build_vit_3channel()
            if isinstance(loaded_data, dict):  # State dict
                # With build_vit_3channel() matching training (from_pretrained + 1-logit head), strict=True should be work
                try:
                    model.load_state_dict(normalize_legacy_vit_state_dict(loaded_data), strict=True)
                    print("Loaded ViT model with strict=True")
                except Exception as e:
                    return False, f"Failed to load ViT state dict (strict=True): {e}"
            else:  # Full model
                model = loaded_data

        elif model_type == "canny":
            model = build_canny_cnn()
            if isinstance(loaded_data, dict):  # State dict
                try:
                    model.load_state_dict(loaded_data, strict=True)
                except Exception as e:
                    return False, f"Failed to load Canny state dict (strict=True): {e}"
            else:  # Full model
                model = loaded_data

        elif model_type == "dft":
            model = build_dft_cnn()
            if isinstance(loaded_data, dict):  # State dict
                try:
                    model.load_state_dict(loaded_data, strict=True)
                except Exception as e:
                    return False, f"Failed to load DFT state dict (strict=True): {e}"
            else:  # Full model
                model = loaded_data
        else:
            return False, f"Unknown model type: {model_type}"
        
        # Move model to device and set to eval mode
        model = model.to(SELECTED_DEVICE).eval()
        
        # Store in global variables
        ACTIVE_MODELS[model_type] = model
        ACTIVE_MODEL_NAMES[model_type] = filename
        
        return True, f"Successfully loaded {model_type.upper()} model: {ACTIVE_MODEL_NAMES[model_type]}"
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Failed to load model: {str(e)}"

# --------------------------
# Inference Functions
# --------------------------
def infer_resnet_xception(model, rgb_tensor):
    """Run inference on ResNet or Xception model - ALWAYS USE SIGMOID"""
    with torch.no_grad():
        try:
            logit = model(rgb_tensor)
            
            # Handle different output dimensions
            if logit.dim() == 2 and logit.shape[1] > 1:
                # If model outputs 2 classes, take the second one as fake probability
                # Some models might output real_score, fake_score?
                if logit.shape[1] == 2:
                    prob_fake = torch.sigmoid(logit[0, 1]).item()
                else:
                    # Multi-class, use softmax and take last class as fake
                    probs = torch.softmax(logit, dim=1)
                    prob_fake = probs[0, -1].item()
            else:
                # Single output, sigmoid
                prob_fake = torch.sigmoid(logit).item()
                
            if prob_fake > 0.5:
                prediction = "FAKE"
                confidence = round(prob_fake * 100, 2)
            else:
                prediction = "REAL"
                confidence = round((1 - prob_fake) * 100, 2)
                
        except Exception as e:
            print(f"Error in ResNet/Xception inference: {e}")
            prediction = "REAL"
            confidence = 50.0
    
    return prediction, confidence

def infer_vit(model, rgb_tensor):
    """Run inference on ViT model - ALWAYS USE SIGMOID"""
    with torch.no_grad():
        try:
            outputs = model(pixel_values=rgb_tensor)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Handle different output dimensions
            if logits.dim() == 2 and logits.shape[1] > 1:
                # If model outputs 2 classes, take the second one as fake prediction
                if logits.shape[1] == 2:
                    prob_fake = torch.sigmoid(logits[0, 1]).item()
                else:
                    # Multi-class, use softmax and take last class as fake
                    probs = torch.softmax(logits, dim=1)
                    prob_fake = probs[0, -1].item()
            else:
                # Single output, use sigmoid
                prob_fake = torch.sigmoid(logits).item()
            
            if prob_fake > 0.5:
                prediction = "FAKE"
                confidence = round(prob_fake * 100, 2)
            else:
                prediction = "REAL"
                confidence = round((1 - prob_fake) * 100, 2)
                
        except Exception as e:
            print(f"Error in ViT inference: {e}")
            prediction = "REAL"
            confidence = 50.0
    
    return prediction, confidence

def infer_cnn_1channel(model, input_tensor):
    """Run inference on 1-channel CNN (Canny or DFT) - ALWAYS USE SIGMOID"""
    with torch.no_grad():
        try:
            logit = model(input_tensor)
            prob_fake = torch.sigmoid(logit).item()
            
            if prob_fake > 0.5:
                prediction = "FAKE"
                confidence = round(prob_fake * 100, 2)
            else:
                prediction = "REAL"
                confidence = round((1 - prob_fake) * 100, 2)
                
        except Exception as e:
            print(f"Error in 1-channel CNN inference: {e}")
            prediction = "REAL"
            confidence = 50.0
    
    return prediction, confidence

# --------------------------
# Ticket-Based Voting Function
# --------------------------
def ticket_based_voting(model_predictions):
    """
    Aggregate model predictions using ticket-based voting
    Each model gets 1 vote, vote for final decision based on majority
    In case of tie, use average confidence to decide...
    """
    if not model_predictions:
        return {
            "votes": {"REAL": 0, "FAKE": 0},
            "final_prediction": "ERROR",
            "final_confidence": 0.0,
            "per_model_results": {},
            "model_count": 0
        }
    
    votes = {"REAL": 0, "FAKE": 0}
    fake_confidences = []
    real_confidences = []
    
    for model_name, (pred, conf) in model_predictions.items():
        votes[pred] += 1
        if pred == "FAKE":
            fake_confidences.append(conf)
        else:
            real_confidences.append(conf)
    
    # Determine final prediction (majority vote)
    if votes["FAKE"] > votes["REAL"]:
        final_pred = "FAKE"
        final_conf = round(np.mean(fake_confidences), 2) if fake_confidences else 0.0
    elif votes["REAL"] > votes["FAKE"]:
        final_pred = "REAL"
        final_conf = round(np.mean(real_confidences), 2) if real_confidences else 0.0
    else:
        # Use average confidence
        avg_fake_conf = np.mean(fake_confidences) if fake_confidences else 0.0
        avg_real_conf = np.mean(real_confidences) if real_confidences else 0.0
        if avg_fake_conf > avg_real_conf:
            final_pred = "FAKE"
            final_conf = avg_fake_conf
        else:
            final_pred = "REAL"
            final_conf = avg_real_conf
    
    return {
        "votes": votes,
        "final_prediction": final_pred,
        "final_confidence": final_conf,
        "per_model_results": model_predictions,
        "model_count": len(model_predictions)
    }

# --------------------------
# Main Processing Function
# --------------------------
def process_image(img_array):
    """
    Process image through all loaded models.
    Returns results from all models and final voting result.
    """
    # Get loaded models
    loaded_models = {m: ACTIVE_MODELS[m] for m in ACTIVE_MODELS.keys() 
                    if ACTIVE_MODELS[m] is not None}
    
    if not loaded_models:
        return False, "No models loaded. Please load at least one model."
    # Preprocess tensors ONLY for the models that are loaded (model-specific sizes & normalization)
    resnet_tensor = preprocess_resnet_image(img_array) if 'resnet' in loaded_models else None
    xception_tensor = preprocess_xception_image(img_array) if 'xception' in loaded_models else None
    vit_tensor = preprocess_vit_image(img_array) if 'vit' in loaded_models else None
    canny_tensor = preprocess_canny_image(img_array) if 'canny' in loaded_models else None
    dft_tensor = preprocess_dft_image(img_array) if 'dft' in loaded_models else None

    # Extract visual evidence
    canny_raw = extract_canny_edges(img_array)
    dft_raw = extract_dft_magnitude(img_array)
    
    # Run inference on all loaded models
    model_predictions = {}
    
    for model_type, model in loaded_models.items():
        try:
            if model_type == 'resnet':
                if resnet_tensor is not None:
                    pred, conf = infer_resnet_xception(model, resnet_tensor)
                    model_predictions[model_type] = (pred, conf)

            elif model_type == 'xception':
                if xception_tensor is not None:
                    pred, conf = infer_resnet_xception(model, xception_tensor)
                    model_predictions[model_type] = (pred, conf)
            
            elif model_type == "vit":
                if vit_tensor is not None:
                    pred, conf = infer_vit(model, vit_tensor)
                    model_predictions[model_type] = (pred, conf)
            
            elif model_type == "canny":
                if canny_tensor is not None:
                    pred, conf = infer_cnn_1channel(model, canny_tensor)
                    model_predictions[model_type] = (pred, conf)
            
            elif model_type == "dft":
                if dft_tensor is not None:
                    pred, conf = infer_cnn_1channel(model, dft_tensor)
                    model_predictions[model_type] = (pred, conf)
                    
        except Exception as e:
            print(f"Error in {model_type} inference: {e}")
            model_predictions[model_type] = ("ERROR", 0.0)
    
    # Apply ticket-based voting
    voting_result = ticket_based_voting(model_predictions)
    voting_result["canny_raw"] = canny_raw
    voting_result["dft_raw"] = dft_raw
    
    return True, voting_result

# --------------------------
# Results Handling
# --------------------------
def calculate_ml_metrics(true_labels, pred_labels):
    if len(true_labels) == 0 or len(pred_labels) == 0:
        return {
            "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0,
            "confusion_matrix": [[0,0],[0,0]]
        }
    
    accuracy = round(accuracy_score(true_labels, pred_labels) * 100, 2)
    precision = round(precision_score(true_labels, pred_labels, zero_division=0) * 100, 2)
    recall = round(recall_score(true_labels, pred_labels, zero_division=0) * 100, 2)
    f1 = round(f1_score(true_labels, pred_labels, zero_division=0) * 100, 2)
    cm = confusion_matrix(true_labels, pred_labels).tolist()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

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

def plot_result_counts(real_count, fake_count, save_path):
    try:
        plt.figure(figsize=(6, 4))
        categories = ["REAL", "FAKE"]
        counts = [real_count, fake_count]
        
        bars = plt.bar(categories, counts, color=["#4A90E2", "#7B68EE"])
        
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
    """Plot confidence histogram."""
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
            palette=["#4A90E2", "#7B68EE"],
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

# --------------------------
# File Processing Functions
# --------------------------
def process_zip_file(zip_file_stream):
    """Process ZIP files using all loaded models"""
    loaded_models = {m: ACTIVE_MODELS[m] for m in ACTIVE_MODELS.keys() 
                    if ACTIVE_MODELS[m] is not None}
    if not loaded_models:
        raise Exception("No models loaded. Please load at least one model first!")
    
    batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = os.path.join(UPLOADS_FOLDER, f"zip_batch_{batch_time}")
    os.makedirs(batch_folder, exist_ok=True)
    
    detection_results = []
    true_labels = []
    pred_labels = []
    
    with zipfile.ZipFile(zip_file_stream) as zip_file:
        zip_file.extractall(batch_folder)
    
    for filename in os.listdir(batch_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            file_path = os.path.join(batch_folder, filename)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    detection_results.append((filename, "ERROR", 0.0, "Unable to load image"))
                    continue
                
                success, result = process_image(img)
                if not success:
                    detection_results.append((filename, "ERROR", 0.0, result))
                    continue
                
                final_pred = result["final_prediction"]
                final_conf = result["final_confidence"]
                
                vote_payload = {
                    "model_count": int(result.get('model_count', 0)),
                    "votes": result.get('votes', {"REAL": 0, "FAKE": 0}),
                    "per_model_results": result.get('per_model_results', {}),
                }
                detection_results.append((
                    filename,
                    final_pred,
                    final_conf,
                    f"Voting ({vote_payload['model_count']} models): {vote_payload['votes']}",
                    vote_payload
                ))
                
                pred_label = 1 if final_pred == "FAKE" else 0
                pred_labels.append(pred_label)
                
                if "REAL" in filename.upper():
                    true_labels.append(0)
                elif "FAKE" in filename.upper():
                    true_labels.append(1)
                    
            except Exception as e:
                detection_results.append((filename, "ERROR", 0.0, f"Error: {str(e)}"))
    
    shutil.rmtree(batch_folder, ignore_errors=True)
    
    real_count = sum(1 for r in detection_results if r[1] == "REAL")
    fake_count = sum(1 for r in detection_results if r[1] == "FAKE")
    error_count = sum(1 for r in detection_results if r[1] == "ERROR")
    valid_conf = [r[2] for r in detection_results if r[1] != "ERROR"]
    avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0
    
    stats = {
        "total_images": len(detection_results),
        "real_count": real_count,
        "fake_count": fake_count,
        "error_count": error_count,
        "avg_confidence": avg_confidence
    }
    
    metrics = None
    if len(true_labels) > 0 and len(true_labels) == len(pred_labels):
        metrics = calculate_ml_metrics(true_labels, pred_labels)
    
    plots = {}
    if metrics and metrics["confusion_matrix"] != [[0,0],[0,0]]:
        cm_plot_path = os.path.join(OUTPUTS_FOLDER, f"confusion_matrix_{batch_time}.png")
        if plot_confusion_matrix(true_labels, pred_labels, cm_plot_path):
            plots["confusion_matrix"] = os.path.basename(cm_plot_path)
    
    count_plot_path = os.path.join(OUTPUTS_FOLDER, f"result_counts_{batch_time}.png")
    if plot_result_counts(real_count, fake_count, count_plot_path):
        plots["result_counts"] = os.path.basename(count_plot_path)
    
    conf_plot_path = os.path.join(OUTPUTS_FOLDER, f"confidence_dist_{batch_time}.png")
    if plot_confidence_distribution(detection_results, conf_plot_path):
        plots["confidence_dist"] = os.path.basename(conf_plot_path)
    
    return {
        "results": detection_results, 
        "stats": stats, 
        "metrics": metrics,
        "plots": plots, 
        "timestamp": batch_time
    }

def process_single_or_multiple_images(image_files):
    """Process images using all loaded models"""
    loaded_models = {m: ACTIVE_MODELS[m] for m in ACTIVE_MODELS.keys() 
                    if ACTIVE_MODELS[m] is not None}
    if not loaded_models:
        raise Exception("No models loaded. Please load at least one model first!")
    
    batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    detection_results = []
    true_labels = []
    pred_labels = []
    
    for file in image_files:
        filename = file.filename
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
            
            success, result = process_image(img)
            if not success:
                detection_results.append((filename, "ERROR", 0.0, result))
                continue
            
            final_pred = result["final_prediction"]
            final_conf = result["final_confidence"]
            
            # Save forensic evidence images
            canny_name = f"canny_{batch_time}_{filename}"
            dft_name = f"dft_{batch_time}_{filename}"
            
            if result.get("canny_raw") is not None:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, canny_name), result["canny_raw"])
            if result.get("dft_raw") is not None:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, dft_name), result["dft_raw"])
            
            detection_results.append((
                filename, 
                final_pred, 
                final_conf, 
                # Rich voting payload for UI badges (while keeping the original string).
                f"Voting ({result['model_count']} models): {result['votes']} | Models: {list(result['per_model_results'].keys())}",
                {
                    "model_count": int(result.get('model_count', 0)),
                    "votes": result.get('votes', {"REAL": 0, "FAKE": 0}),
                    "per_model_results": result.get('per_model_results', {}),
                }
            ))
            
            pred_label = 1 if final_pred == "FAKE" else 0
            pred_labels.append(pred_label)
            
            if "REAL" in filename.upper():
                true_labels.append(0)
            elif "FAKE" in filename.upper():
                true_labels.append(1)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            detection_results.append((filename, "ERROR", 0.0, f"Error: {str(e)}"))
        finally:
            # Do not retain decoded image bytes, tensors, or forensic arrays
            # between batch items. This keeps large normal uploads stable.
            file_data = None
            img_np = None
            img = None
            result = None
            if len(detection_results) % 10 == 0:
                gc.collect()
    
    real_count = sum(1 for r in detection_results if r[1] == "REAL")
    fake_count = sum(1 for r in detection_results if r[1] == "FAKE")
    error_count = sum(1 for r in detection_results if r[1] == "ERROR")
    valid_conf = [r[2] for r in detection_results if r[1] != "ERROR"]
    avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0
    
    stats = {
        "total_images": len(detection_results),
        "real_count": real_count,
        "fake_count": fake_count,
        "error_count": error_count,
        "avg_confidence": avg_confidence
    }
    
    metrics = None
    if len(true_labels) > 0 and len(true_labels) == len(pred_labels):
        metrics = calculate_ml_metrics(true_labels, pred_labels)
    
    plots = {}
    if metrics and metrics["confusion_matrix"] != [[0,0],[0,0]]:
        cm_plot_path = os.path.join(OUTPUTS_FOLDER, f"confusion_matrix_{batch_time}.png")
        if plot_confusion_matrix(true_labels, pred_labels, cm_plot_path):
            plots["confusion_matrix"] = os.path.basename(cm_plot_path)
    
    count_plot_path = os.path.join(OUTPUTS_FOLDER, f"result_counts_{batch_time}.png")
    if plot_result_counts(real_count, fake_count, count_plot_path):
        plots["result_counts"] = os.path.basename(count_plot_path)
    
    conf_plot_path = os.path.join(OUTPUTS_FOLDER, f"confidence_dist_{batch_time}.png")
    if plot_confidence_distribution(detection_results, conf_plot_path):
        plots["confidence_dist"] = os.path.basename(conf_plot_path)
    
    return {
        "results": detection_results, 
        "stats": stats, 
        "metrics": metrics,
        "plots": plots, 
        "timestamp": batch_time
    }

def cleanup_outputs(max_files=100):
    """Clean up old output files to prevent disk space issues"""
    try:
        files = os.listdir(OUTPUTS_FOLDER)
        if len(files) > max_files:
            files_with_time = [(f, os.path.getmtime(os.path.join(OUTPUTS_FOLDER, f))) 
                              for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
            files_with_time.sort(key=lambda x: x[1])
            
            for f, _ in files_with_time[:len(files) - max_files + 20]:
                os.remove(os.path.join(OUTPUTS_FOLDER, f))
    except Exception as e:
        print(f"Cleanup error: {e}")



# --------------------------
# Video Processing (MP4 frame sampling)
# --------------------------
class VideoFrameSampler:
    """
    Video Frame Deepfake Detection

    Read my requirement first:
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

        current_frame_index = -1
        for k, idx in enumerate(indices):
            frame = None
            result = None
            try:
                # Read forward sequentially instead of repeatedly seeking with
                # CAP_PROP_POS_FRAMES. Sequential access is more reliable for
                # compressed MP4 files and avoids decoder seek instability.
                while current_frame_index < idx:
                    if not cap.grab():
                        errors += 1
                        break
                    current_frame_index += 1

                if current_frame_index != idx:
                    continue

                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    errors += 1
                    continue

                # Keep the first successfully decoded frame as thumbnail
                if thumb_frame is None:
                    thumb_frame = frame.copy()

                success, result = process_image(frame)
                if not success:
                    errors += 1
                    continue

                frame_results.append((result.get('final_prediction', 'ERROR'), float(result.get('final_confidence', 0.0))))

            except Exception as e:
                errors += 1
                continue
            finally:
                # process_image returns temporary forensic arrays; only the
                # prediction and confidence are retained for video voting.
                frame = None
                result = None

        cap.release()
        gc.collect()
        if SELECTED_DEVICE.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not frame_results:
            return False, f"No frames could be analyzed for {display_name} (errors={errors})."

        # Aggregate: majority vote across frames
        votes = {'REAL': 0, 'FAKE': 0}
        real_confs = []
        fake_confs = []
        for pred, conf in frame_results:
            if pred in votes:
                votes[pred] += 1
                if pred == 'FAKE':
                    fake_confs.append(conf)
                else:
                    real_confs.append(conf)

        if votes['FAKE'] > votes['REAL']:
            final_pred = 'FAKE'
            final_conf = round(float(sum(fake_confs) / max(len(fake_confs), 1)), 2)
        elif votes['REAL'] > votes['FAKE']:
            final_pred = 'REAL'
            final_conf = round(float(sum(real_confs) / max(len(real_confs), 1)), 2)
        else:
            # tie-break with higher average confidence
            avg_fake = float(sum(fake_confs) / max(len(fake_confs), 1))
            avg_real = float(sum(real_confs) / max(len(real_confs), 1))
            if avg_fake >= avg_real:
                final_pred = 'FAKE'
                final_conf = round(avg_fake, 2)
            else:
                final_pred = 'REAL'
                final_conf = round(avg_real, 2)

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
        raise Exception("No models loaded. Please load at least one model first!")

    batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sampler = VideoFrameSampler(
        sample_ratio=sample_ratio,
        min_samples=VIDEO_MIN_SAMPLES,
        max_samples=VIDEO_MAX_SAMPLES,
    )

    detection_results = []
    true_labels = []
    pred_labels = []

    for f in video_files:
        filename = f.filename
        if not filename:
            continue

        tmp_path = os.path.join(UPLOADS_FOLDER, f"tmp_{batch_time}_{os.path.basename(filename)}")
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

            pred_label = 1 if out['final_prediction'] == 'FAKE' else 0
            pred_labels.append(pred_label)

            # Optional ground-truth heuristic by filename
            if 'REAL' in filename.upper():
                true_labels.append(0)
            elif 'FAKE' in filename.upper():
                true_labels.append(1)

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
    error_count = sum(1 for r in detection_results if r[1] == 'ERROR')
    valid_conf = [r[2] for r in detection_results if r[1] != 'ERROR']
    avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0

    stats = {
        'total_images': len(detection_results),
        'real_count': real_count,
        'fake_count': fake_count,
        'error_count': error_count,
        'avg_confidence': avg_confidence
    }

    metrics = None
    if len(true_labels) > 0 and len(true_labels) == len(pred_labels):
        metrics = calculate_ml_metrics(true_labels, pred_labels)

    plots = {}
    if metrics and metrics['confusion_matrix'] != [[0,0],[0,0]]:
        cm_plot_path = os.path.join(OUTPUTS_FOLDER, f"confusion_matrix_{batch_time}.png")
        if plot_confusion_matrix(true_labels, pred_labels, cm_plot_path):
            plots['confusion_matrix'] = os.path.basename(cm_plot_path)

    count_plot_path = os.path.join(OUTPUTS_FOLDER, f"result_counts_{batch_time}.png")
    if plot_result_counts(real_count, fake_count, count_plot_path):
        plots['result_counts'] = os.path.basename(count_plot_path)

    conf_plot_path = os.path.join(OUTPUTS_FOLDER, f"confidence_dist_{batch_time}.png")
    if plot_confidence_distribution(detection_results, conf_plot_path):
        plots['confidence_dist'] = os.path.basename(conf_plot_path)

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
            error_count = sum(1 for r in all_results if r[1] == 'ERROR')
            valid_conf = [r[2] for r in all_results if r[1] != 'ERROR']
            avg_confidence = round(sum(valid_conf)/len(valid_conf), 2) if valid_conf else 0.0
            combined_results['stats'] = {
                'total_images': len(all_results),
                'real_count': real_count,
                'fake_count': fake_count,
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
                combined_results['results'].append((f.filename, 'ERROR', 0.0, 'Unsupported file type'))
        all_results = combined_results['results']
        combined_results['stats']['total_images'] = len(all_results)
        combined_results['stats']['error_count'] = sum(1 for r in all_results if r[1] == 'ERROR')

    benchmark_wall_seconds = time.perf_counter() - benchmark_wall_start
    benchmark_cpu_seconds = time.process_time() - benchmark_cpu_start
    benchmark_monitor.__exit__(None, None, None)
    return attach_uploaded_batch_benchmark(combined_results, benchmark_wall_seconds, benchmark_cpu_seconds, benchmark_monitor)
# --------------------------
# Flask App
# --------------------------
app = Flask(__name__, static_folder=OUTPUTS_FOLDER)
app.secret_key = "deepfake_detector_2025"
app.config['SESSION_TYPE'] = 'filesystem'

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
        .visual-evidence-card {
            height: 100%;
            transition: transform 0.2s ease;
        }
        .visual-evidence-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
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

            toggleDropZoneContent(fileInput.files.length > 0);
        });
    </script>
</head>
<body>
<div class="container py-4">
    <div class="mb-4">
        <h3>Deepfake Image & Video Frame Detector</h3>
        <p class="small-text">Check your images or videos for deepfakes...</p>
    </div>
    
    <div class="row g-4">
        <div class="col-md-4">
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
                
                <label class="form-label small-text"><strong>Load Detection Model:</strong></label>
                <form method="POST" action="/load_model" enctype="multipart/form-data">
                    <div class="mb-3">
                        <input type="file" name="model_files" class="form-control mb-2" accept="{{ accept_attr }}" multiple required>
                        <p class="small-text">Supported formats: {{ supported_exts_text }}
                        <br>Supported architectures: 
                        <br><span class="badge rgb-model">RGB Models:</span> ResNet-50, Xception, ViT
                        <br><span class="badge dft-model">DFT Model:</span> 1-channel CNN
                        <br><span class="badge canny-model">Canny Model:</span> 1-channel CNN</p>
                    </div>
                    <button type="submit" class="custom-btn">Load Model</button>
                </form>
                
                <div class="mt-3">
                    <form method="post" action="/clear_models" onsubmit="return confirm('Clear all loaded models?')">
                        <button type="submit" class="btn btn-outline-danger btn-sm w-100">
                            <i class="bi bi-trash"></i> Clear All Models
                        </button>
                    </form>
                </div>
                
                {% if success_msg %}
                <div class="alert alert-success mt-3 small" role="alert">
                    {{ success_msg }}
                </div>
                {% endif %}
                
                {% if error_msg %}
                <div class="alert alert-danger mt-3 small" role="alert">
                    {{ error_msg }}
                </div>
                {% endif %}
                
                <div class="mt-3 small-text">
                    <strong>Current device:</strong> {{ selected_device.upper() }}<br>
                    <strong>Loaded Models ({{ loaded_count }}/5):</strong><br>
                    <div class="mt-1">
                        {% for model_name, model_file in active_model_names.items() %}
                            {% if model_file %}
                            <span class="badge model-badge 
                                {% if model_name in ['resnet', 'xception', 'vit'] %}rgb-model
                                {% elif model_name == 'dft' %}dft-model
                                {% elif model_name == 'canny' %}canny-model
                                {% endif %}">
                                {{ model_name|upper }}: {{ model_file }}
                            </span><br>
                            {% endif %}
                        {% endfor %}
                        {% if loaded_count == 0 %}
                        <span class="text-muted">No models loaded</span>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-8">
            <div class="custom-card">
                <h5 class="text-center mb-4">Detect Deepfakes (Images/ZIP)</h5>
                
                <form method="post" enctype="multipart/form-data" action="/detect">
                    <div id="drop-zone" class="drop-zone">
                        <div id="drop-zone-default">
                            <i class="bi bi-cloud-upload" style="font-size: 40px; color: #4A90E2;"></i>
                            <div class="mt-3"><strong>Drag & Drop files here</strong></div>
                            <div class="small-text mt-1">Supports: single image, multiple images, ZIP files, MP4 videos (frame sampling)</div>
                        </div>
                        
                        <div id="preview-container" class="preview-container"></div>
                        <input type="file" id="file-input" name="files" multiple accept="image/*,video/mp4,.mp4,.zip">
                    </div>
                    
                    <button type="submit" class="custom-btn">Start Detection</button>
                </form>
                
                {% if single_result %}
                <div class="result-card mt-4 border-start border-4 {{ 'border-danger' if single_result.prediction == 'FAKE' else 'border-success' }}">
                    <h6><i class="bi bi-search me-2"></i>Detection Result</h6>
                    <div class="row">
                        <div class="col-6">
                            <p><strong>Prediction:</strong> <span class="badge {{ 'bg-danger' if single_result.prediction == 'FAKE' else 'bg-success' }}">{{ single_result.prediction }}</span></p>
                        </div>
                        <div class="col-6 text-end">
                            <p><strong>Confidence:</strong> {{ single_result.confidence }}%</p>
                        </div>
                    </div>
                    <p class="small-text mt-2">{{ single_result.explanation }}</p>
                    {% if single_result.per_model_results %}
                    <div class="mt-3">
                        <small><strong>Individual Model Results ({{ single_result.model_count }} models):</strong></small>
                        <div class="d-flex flex-wrap gap-2 mt-1">
                            {% for model_name, (pred, conf) in single_result.per_model_results.items() %}
                            <span class="badge 
                                {% if model_name in ['resnet', 'xception', 'vit'] %}rgb-model
                                {% elif model_name == 'dft' %}dft-model
                                {% elif model_name == 'canny' %}canny-model
                                {% else %}bg-secondary{% endif %}">
                                {{ model_name|upper }}: {{ pred }} ({{ conf }}%)
                            </span>
                            {% endfor %}
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
                        <span class="metric-badge bg-warning text-dark">Error: {{ batch_data.stats.error_count }}</span>
                        <span class="metric-badge bg-info text-white">Avg. Conf: {{ batch_data.stats.avg_confidence }}%</span>
                    </div>
                    
                    {% if batch_data.metrics %}
                    <div class="mb-3 p-2 bg-light rounded">
                        <small><strong>Performance Metrics:</strong></small><br>
                        <div class="d-flex justify-content-between mt-1 flex-wrap gap-2">
                            <small>
                                Acc: {{ batch_data.metrics.accuracy }}%
                                <i class="bi bi-info-circle ms-1 text-muted" title="Accuracy = (TP + TN) / (TP + TN + FP + FN). Overall correctness on this labeled test batch."></i>
                            </small>
                            <small>
                                Prec: {{ batch_data.metrics.precision }}%
                                <i class="bi bi-info-circle ms-1 text-muted" title="Precision (for FAKE) = TP / (TP + FP). Of all predicted FAKE, how many were truly FAKE."></i>
                            </small>
                            <small>
                                Rec: {{ batch_data.metrics.recall }}%
                                <i class="bi bi-info-circle ms-1 text-muted" title="Recall (for FAKE) = TP / (TP + FN). Of all true FAKE, how many were detected."></i>
                            </small>
                            <small>
                                F1: {{ batch_data.metrics.f1_score }}%
                                <i class="bi bi-info-circle ms-1 text-muted" title="F1 = 2 * (Precision * Recall) / (Precision + Recall). Balances precision & recall."></i>
                            </small>
                        </div>
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
                    
                    <div class="table-responsive mt-3" style="max-height: 300px; overflow-y: auto;">
                        <table class="results-table table-sm">
                            <thead>
                                <tr>
                                    <th>File Name</th>
                                    <th>Prediction</th>
                                    <th>Conf (%)</th>
                                    <th>Forensic Analysis</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for result in batch_data.results %}
                                <tr class="{{ 'table-danger' if result[1] == 'FAKE' else '' }}">
                                    <td><small>{{ result[0] }}</small></td>
                                    <td><span class="badge {{ 'bg-danger' if result[1] == 'FAKE' else 'bg-success' }}">{{ result[1] }}</span></td>
                                    <td>{{ result[2] }}</td>
                                    <td>
                                        {% if result|length > 4 and result[4] and result[4].get('per_model_results') %}
                                            {% set ns = namespace(real=[], fake=[]) %}
                                            {% for model_name, pair in result[4]['per_model_results'].items() %}
                                                {% if pair[0] == 'REAL' %}
                                                    {% set ns.real = ns.real + [model_name] %}
                                                {% else %}
                                                    {% set ns.fake = ns.fake + [model_name] %}
                                                {% endif %}
                                            {% endfor %}
                                            <div style="font-size:0.80rem; line-height:1.25;">
                                                <div><span class="badge bg-secondary">Vote Participate: {{ result[4]['model_count'] }} models</span></div>
                                                <div class="mt-1"><span class="badge bg-success">REAL</span>:
                                                    <span class="text-muted">{{ ns.real|join(', ') if ns.real else '-' }}</span>
                                                </div>
                                                <div class="mt-1"><span class="badge bg-danger">FAKE</span>:
                                                    <span class="text-muted">{{ ns.fake|join(', ') if ns.fake else '-' }}</span>
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

                    <!-- Visual Evidence & Forensic Tags Section -->
                    <div class="mt-4 pt-3 border-top">
                        <h6 class="mb-3 border-bottom pb-2">Visual Evidence & Forensic Tags</h6>
                        <div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 g-3">
                            {% for result in batch_data.results %}
                            <div class="col">
                                <div class="card h-100 shadow-sm border-{{ 'danger' if result[1] == 'FAKE' else 'success' }} visual-evidence-card">
                                    <div class="card-img-container">
                                        <img src="/outputs/{{ result[0] }}" class="card-img-top w-100 h-100" style="object-fit: contain;" alt="{{ result[0] }}">
                                    </div>
                                    <div class="card-body p-2">
                                        <div class="d-flex justify-content-between align-items-center mb-1">
                                            <span class="badge {{ 'bg-danger' if result[1] == 'FAKE' else 'bg-success' }}" style="font-size: 0.65rem;">
                                                {{ result[1] }}
                                            </span>
                                            <small class="text-muted" style="font-size: 0.7rem;">{{ result[2] }}%</small>
                                        </div>
                                        {% if result|length > 4 and result[4] and result[4].get('per_model_results') %}
                                            {% set ns = namespace(real=[], fake=[]) %}
                                            {% for model_name, pair in result[4]['per_model_results'].items() %}
                                                {% if pair[0] == 'REAL' %}
                                                    {% set ns.real = ns.real + [model_name] %}
                                                {% else %}
                                                    {% set ns.fake = ns.fake + [model_name] %}
                                                {% endif %}
                                            {% endfor %}
                                            <div style="font-size: 0.72rem; line-height:1.2;">
                                                <div><span class="badge bg-secondary">Vote Participate: {{ result[4]['model_count'] }} models</span></div>
                                                <div class="mt-1"><span class="badge bg-success">REAL</span>:
                                                    <span class="text-muted">{{ ns.real|join(', ') if ns.real else '-' }}</span>
                                                </div>
                                                <div class="mt-1"><span class="badge bg-danger">FAKE</span>:
                                                    <span class="text-muted">{{ ns.fake|join(', ') if ns.fake else '-' }}</span>
                                                </div>
                                            </div>
                                        {% else %}
                                            <p class="card-text mb-0" style="font-size: 0.75rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
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

                    {% if batch_data.plots.result_counts or batch_data.plots.confidence_dist or batch_data.plots.confusion_matrix %}
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
        Support: 3 RGB models + 2 forensic models (max 5 models)<br>
        <span class="badge rgb-model">RGB Models</span> 
        <span class="badge dft-model">DFT Model</span> 
        <span class="badge canny-model">Canny Model</span>
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
    error_msg = session.pop('error_msg', None)
    batch_data = session.pop('batch_data', None)
    result_data = session.pop('result_data', None)
    
    # Count loaded models
    loaded_count = sum(1 for model in ACTIVE_MODELS.values() if model is not None)
    
    return render_template_string(
        template,
        has_cuda=torch.cuda.is_available(),
        selected_device=SELECTED_DEVICE.type,
        active_model_names=ACTIVE_MODEL_NAMES,
        loaded_count=loaded_count,
        success_msg=success_msg,
        error_msg=error_msg,
        batch_data=batch_data,
        single_result=result_data
    )

@app.route('/set_device', methods=['POST'])
def set_device():
    global SELECTED_DEVICE
    device = request.form.get('device', 'cpu')

    if device == 'cuda' and torch.cuda.is_available():
        SELECTED_DEVICE = torch.device('cuda')
        session['success_msg'] = "Switched to GPU"
    else:
        SELECTED_DEVICE = torch.device('cpu')
        session['success_msg'] = "Switched to CPU"

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

    loaded_count = sum(1 for m in ACTIVE_MODELS.values() if m is not None)
    remaining_slots = max(0, 5 - loaded_count)

    success_msgs = []
    error_msgs = []

    # Load in the order selected, up to remaining slots
    for model_file in files[:remaining_slots]:
        try:
            temp_path = os.path.join(MODELS_FOLDER, model_file.filename)
            model_file.save(temp_path)

            # Always use auto-detection
            success, msg = load_model_with_auto_detect(temp_path)

            if success:
                success_msgs.append(msg)
            else:
                error_msgs.append(msg)

        except Exception as e:
            error_msgs.append(f"Failed to load model '{model_file.filename}': {str(e)}")

    # If user selected more than available slots, warn it
    if len(files) > remaining_slots:
        error_msgs.append(f"Model limit reached (max 5). Loaded only {remaining_slots} file(s).")

    if success_msgs and not error_msgs:
        session['success_msg'] = " | ".join(success_msgs)
    elif success_msgs and error_msgs:
        session['success_msg'] = " | ".join(success_msgs)
        session['error_msg'] = " | ".join(error_msgs)
    else:
        session['error_msg'] = " | ".join(error_msgs) if error_msgs else "Failed to load model(s)."

    return redirect(url_for('index'))

@app.route('/clear_models', methods=['POST'])
def clear_models():
    """Clear all loaded models"""
    global ACTIVE_MODELS, ACTIVE_MODEL_NAMES
    ACTIVE_MODELS = {"xception": None, "resnet": None, "vit": None, "dft": None, "canny": None}
    ACTIVE_MODEL_NAMES = {"xception": None, "resnet": None, "vit": None, "dft": None, "canny": None}
    session['success_msg'] = "All models cleared successfully"
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
                zip_data = process_zip_file(zip_files[0].stream)
                zip_wall_seconds = time.perf_counter() - zip_wall_start
                zip_cpu_seconds = time.process_time() - zip_cpu_start
                zip_monitor.__exit__(None, None, None)
                session['batch_data'] = attach_uploaded_batch_benchmark(zip_data, zip_wall_seconds, zip_cpu_seconds, zip_monitor)
                return redirect(url_for('index'))
            
            session['batch_data'] = process_mixed_files(files)
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
            img_np = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if img is None:
                session['error_msg'] = "Failed to read image"
                return redirect(url_for('index'))
            
            success, result = process_image(img)
            
            if not success:
                session['error_msg'] = result
            else:
                cv2.imwrite(os.path.join(OUTPUTS_FOLDER, file.filename), img)
                
                canny_name = f"canny_single_{file.filename}"
                dft_name = f"dft_single_{file.filename}"
                
                if result.get("canny_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, canny_name), result["canny_raw"])
                if result.get("dft_raw") is not None:
                    cv2.imwrite(os.path.join(OUTPUTS_FOLDER, dft_name), result["dft_raw"])
                
                session['result_data'] = {
                    "prediction": result["final_prediction"],
                    "confidence": result["final_confidence"],
                    "explanation": f"Voting result ({result['model_count']} models): {result['votes']}",
                    "per_model_results": result["per_model_results"],
                    "model_count": result["model_count"]
                }
                
        except Exception as e:
            session['error_msg'] = str(e)
        
        return redirect(url_for('index'))
    
    session['error_msg'] = "No files uploaded"
    return redirect(url_for('index'))

@app.route('/outputs/<filename>')
def outputs_file(filename):
    return send_from_directory(OUTPUTS_FOLDER, filename)


# --------------------------
# Integrated five-model benchmark mode
# --------------------------
class BenchmarkResourceMonitor:
    def __init__(self, device, interval=0.05):
        self.device = device; self.interval = interval; self.stop = threading.Event(); self.thread = None
        self.peak_rss = 0; self.peak_gpu_allocated = 0; self.peak_gpu_reserved = 0

    @staticmethod
    def rss_bytes():
        if os.name != 'nt': return 0
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters(); counters.cb = ctypes.sizeof(Counters)
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
        while not self.stop.is_set(): self.sample(); self.stop.wait(self.interval)

    def __enter__(self):
        self.sample(); self.thread = threading.Thread(target=self.loop, daemon=True); self.thread.start(); return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        if self.thread: self.thread.join(timeout=1.0)
        self.sample()


def benchmark_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def benchmark_samples(root, limit_per_class, seed):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}; groups = {0: [], 1: []}
    for current_root, _, filenames in os.walk(root):
        parts = {part.lower() for part in os.path.normpath(current_root).split(os.sep)}
        label = 0 if 'real' in parts else 1 if 'fake' in parts else None
        if label is not None:
            groups[label].extend(os.path.join(current_root, name) for name in filenames if os.path.splitext(name)[1].lower() in extensions)
    if not groups[0] or not groups[1]: raise ValueError('Benchmark dataset must contain image files below real and fake folders.')
    for paths in groups.values(): paths.sort(key=str.lower)
    count = min(len(groups[0]), len(groups[1]))
    if limit_per_class is not None and int(limit_per_class) > 0:
        count = min(count, int(limit_per_class))
    rng = np.random.default_rng(seed); selected = []
    for label in (0, 1):
        indices = np.arange(len(groups[label])); rng.shuffle(indices); selected.extend((groups[label][int(index)], label) for index in indices[:count])
    selected.sort(key=lambda item: item[0].lower()); return selected


def benchmark_nvidia():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=10, check=False)
        return {'available': result.returncode == 0, 'raw': result.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as exc: return {'available': False, 'error': str(exc)}


def benchmark_write_csv(path, rows):
    if not rows: return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


def normalized_process_cpu_percent(cpu_seconds, wall_seconds):
    """Return whole-process CPU use normalized to the machine's logical cores."""
    if not wall_seconds:
        return None
    logical_cores = max(1, int(os.cpu_count() or 1))
    return round(min(100.0, cpu_seconds / (wall_seconds * logical_cores) * 100.0), 4)


def attach_uploaded_batch_benchmark(batch_data, wall_seconds, cpu_seconds, monitor=None):
    """Save benchmark statistics for a normal FYP1 Flask upload batch."""
    labelled = []; prediction_rows = []
    for result in batch_data.get('results', []):
        if len(result) < 5 or not isinstance(result[4], dict): continue
        tokens = [token for token in re.split(r'[^a-z0-9]+', os.path.basename(str(result[0])).lower()) if token]
        true_label = 1 if 'fake' in tokens else 0 if 'real' in tokens else None
        per_model = result[4].get('per_model_results', {})
        if true_label is None or not per_model: continue
        probabilities = []
        for prediction, confidence in per_model.values():
            score = float(confidence) / 100.0
            probabilities.append(score if prediction == 'FAKE' else 1.0 - score)
        fake_probability = float(np.mean(probabilities))
        binary_prediction = int(fake_probability >= 0.5)
        labelled.append((true_label, binary_prediction, fake_probability))
        prediction_rows.append({'filename': result[0], 'true_label': 'FAKE' if true_label else 'REAL', 'application_prediction': result[1], 'fake_probability_reconstructed': round(fake_probability, 8), 'binary_prediction_0_5': 'FAKE' if binary_prediction else 'REAL'})
    labels = [row[0] for row in labelled]; predictions = [row[1] for row in labelled]; probabilities = [row[2] for row in labelled]; both_classes = len(set(labels)) == 2
    summary = {
        'source': 'normal Flask upload or drag-and-drop batch',
        'scope': 'resource and processing statistics for the uploaded batch',
        'sample_count_uploaded': len(batch_data.get('results', [])),
        'sample_count_labelled': len(labels),
        'wall_seconds': round(wall_seconds, 6),
        'process_cpu_seconds': round(cpu_seconds, 6),
        'process_cpu_utilization_percent': normalized_process_cpu_percent(cpu_seconds, wall_seconds),
        'peak_process_ram_mb': round(monitor.peak_rss / 1024 ** 2, 4) if monitor else None,
        'peak_gpu_allocated_mb': round(monitor.peak_gpu_allocated / 1024 ** 2, 4) if monitor else None,
        'peak_gpu_reserved_mb': round(monitor.peak_gpu_reserved / 1024 ** 2, 4) if monitor else None,
        'loaded_models': ACTIVE_MODEL_NAMES,
    }
    timestamp = batch_data.get('timestamp', datetime.datetime.now().strftime('%Y%m%d_%H%M%S')); os.makedirs(RESOURCES_STATISTIC_FOLDER, exist_ok=True)
    with open(os.path.join(RESOURCES_STATISTIC_FOLDER, f'flask_batch_{timestamp}.json'), 'w', encoding='utf-8') as handle: json.dump(summary, handle, indent=2)
    benchmark_write_csv(os.path.join(RESOURCES_STATISTIC_FOLDER, f'flask_predictions_{timestamp}.csv'), prediction_rows); batch_data['benchmark'] = summary; return batch_data


def fyp1_benchmark_probability(model_type, model, image):
    if model_type == 'resnet': tensor = preprocess_resnet_image(image); output = model(tensor)
    elif model_type == 'xception': tensor = preprocess_xception_image(image); output = model(tensor)
    elif model_type == 'vit':
        tensor = preprocess_vit_image(image); output = model(pixel_values=tensor); output = output.logits if hasattr(output, 'logits') else output
    elif model_type == 'canny': tensor = preprocess_canny_image(image); output = model(tensor)
    else: tensor = preprocess_dft_image(image); output = model(tensor)
    if output.dim() == 2 and output.shape[1] > 1: return float(torch.softmax(output, dim=1)[0, -1].item())
    return float(torch.sigmoid(output).reshape(-1)[0].item())


def fyp1_benchmark_metric(system, labels, probabilities, timings, monitor, checkpoint_paths, wall_seconds, cpu_seconds, errors):
    predictions = [int(probability >= 0.5) for probability in probabilities]
    return {'system': system, 'sample_count_requested': len(labels) + len(errors), 'sample_count_evaluated': len(labels), 'real_count': sum(label == 0 for label in labels), 'fake_count': sum(label == 1 for label in labels), 'errors': len(errors), 'accuracy_percent': round(accuracy_score(labels, predictions) * 100, 4), 'precision_fake_percent': round(precision_score(labels, predictions, zero_division=0) * 100, 4), 'recall_fake_percent': round(recall_score(labels, predictions, zero_division=0) * 100, 4), 'f1_fake_percent': round(f1_score(labels, predictions, zero_division=0) * 100, 4), 'confusion_matrix_real_fake': confusion_matrix(labels, predictions, labels=[0, 1]).tolist(), 'binary_threshold': 0.5, 'wall_seconds': round(wall_seconds, 6), 'process_cpu_seconds': round(cpu_seconds, 6), 'process_cpu_utilization_percent': normalized_process_cpu_percent(cpu_seconds, wall_seconds), 'mean_image_inference_ms': round(statistics.mean(timings), 6), 'median_image_inference_ms': round(statistics.median(timings), 6), 'p95_image_inference_ms': round(float(np.percentile(timings, 95)), 6), 'peak_process_ram_mb': round(monitor.peak_rss / 1024 ** 2, 4), 'peak_gpu_allocated_mb': round(monitor.peak_gpu_allocated / 1024 ** 2, 4), 'peak_gpu_reserved_mb': round(monitor.peak_gpu_reserved / 1024 ** 2, 4), 'parameter_count_total': sum(parameter.numel() for model in ACTIVE_MODELS.values() if model is not None for parameter in model.parameters()), 'checkpoints': checkpoint_paths, 'resource_scope': 'all five FYP1 models resident together'}


def run_benchmark():
    parser = argparse.ArgumentParser(description='Benchmark the FYP1 five-model ensemble.')
    parser.add_argument('--benchmark', action='store_true'); parser.add_argument('--benchmark-dataset-root', required=True); parser.add_argument('--benchmark-model-set', action='append', default=None, choices=['model_set1', 'model_set2']); parser.add_argument('--benchmark-limit-per-class', type=int, default=None); parser.add_argument('--benchmark-seed', type=int, default=42); parser.add_argument('--benchmark-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(); global SELECTED_DEVICE, ACTIVE_MODELS, ACTIVE_MODEL_NAMES
    dataset_root = os.path.abspath(args.benchmark_dataset_root); model_sets = args.benchmark_model_set or ['model_set1', 'model_set2']
    if args.benchmark_device == 'cuda' or (args.benchmark_device == 'auto' and torch.cuda.is_available()):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable.')
        SELECTED_DEVICE = torch.device('cuda')
    else: SELECTED_DEVICE = torch.device('cpu')
    if not os.path.isdir(dataset_root): raise FileNotFoundError(f'Dataset folder does not exist: {dataset_root}')
    samples = benchmark_samples(dataset_root, args.benchmark_limit_per_class, args.benchmark_seed); os.makedirs(RESOURCES_STATISTIC_FOLDER, exist_ok=True)
    for name in os.listdir(RESOURCES_STATISTIC_FOLDER):
        if name.endswith(('.json', '.csv')): os.remove(os.path.join(RESOURCES_STATISTIC_FOLDER, name))
    all_metrics = []; manifest_sets = []
    for model_set in model_sets:
        model_paths = {}; ACTIVE_MODELS = {'xception': None, 'resnet': None, 'vit': None, 'dft': None, 'canny': None}; ACTIVE_MODEL_NAMES = dict.fromkeys(ACTIVE_MODELS)
        set_dir = os.path.join(ROOT_FOLDER, model_set)
        for model_type in ('xception', 'resnet', 'vit', 'dft', 'canny'):
            candidates = sorted([os.path.join(set_dir, name) for name in os.listdir(set_dir) if name.lower().startswith(model_type) and name.lower().endswith(('.pth', '.pt', '.safetensors'))])
            if not candidates: raise FileNotFoundError(f'No {model_type} checkpoint found in {model_set}.')
            model_paths[model_type] = candidates[0]; success, message = load_model_with_auto_detect(candidates[0])
            if not success: raise RuntimeError(f'{model_set}/{model_type}: {message}')
        labels = []; per_model_probs = {name: [] for name in ACTIVE_MODELS}; per_model_times = {name: [] for name in ACTIVE_MODELS}; ensemble_probs = []; ensemble_times = []; rows = []; errors = []
        with BenchmarkResourceMonitor(SELECTED_DEVICE) as monitor:
            wall_start = time.perf_counter(); cpu_start = time.process_time()
            with torch.inference_mode():
                for image_path, label in samples:
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image is None: errors.append({'path': image_path, 'error': 'OpenCV could not decode image'}); continue
                    ensemble_start = time.perf_counter()
                    probabilities = {}
                    for model_type, model in ACTIVE_MODELS.items():
                        start = time.perf_counter(); probability = fyp1_benchmark_probability(model_type, model, image)
                        if SELECTED_DEVICE.type == 'cuda': torch.cuda.synchronize(SELECTED_DEVICE)
                        per_model_times[model_type].append((time.perf_counter() - start) * 1000.0); per_model_probs[model_type].append(probability); probabilities[model_type] = probability
                    mean_probability = float(np.mean(list(probabilities.values()))); votes = sum(probability >= 0.5 for probability in probabilities.values()); ensemble_prediction = int(votes > 2 or (votes == 2 and mean_probability >= 0.5)); ensemble_times.append((time.perf_counter() - ensemble_start) * 1000.0)
                    labels.append(label); ensemble_probs.append(mean_probability); rows.append({'path': image_path, 'true_label': 'FAKE' if label else 'REAL', **{f'{name}_fake_probability': round(value, 8) for name, value in probabilities.items()}, 'ensemble_fake_probability': round(mean_probability, 8), 'ensemble_prediction': 'FAKE' if ensemble_prediction else 'REAL'}); monitor.sample()
            wall_seconds = time.perf_counter() - wall_start; cpu_seconds = time.process_time() - cpu_start
        for model_type in ACTIVE_MODELS: all_metrics.append(fyp1_benchmark_metric(f'{model_set}_{model_type}', labels, per_model_probs[model_type], per_model_times[model_type], monitor, {model_type: model_paths[model_type]}, wall_seconds, cpu_seconds, errors))
        ensemble_metric = fyp1_benchmark_metric(f'{model_set}_five_model_ensemble', labels, ensemble_probs, ensemble_times, monitor, model_paths, wall_seconds, cpu_seconds, errors); ensemble_metric['ensemble_members'] = list(ACTIVE_MODELS.keys()); all_metrics.append(ensemble_metric); benchmark_write_csv(os.path.join(RESOURCES_STATISTIC_FOLDER, f'predictions_{model_set}.csv'), rows); manifest_sets.append({'model_set': model_set, 'checkpoints': {name: {'path': path, 'sha256': benchmark_sha256(path)} for name, path in model_paths.items()}})
    manifest = {'created_at_local': datetime.datetime.now().isoformat(timespec='seconds'), 'script': os.path.abspath(__file__), 'python': sys.version, 'platform': platform.platform(), 'dataset_root': dataset_root, 'sample_count': len(samples), 'real_count': sum(label == 0 for _, label in samples), 'fake_count': sum(label == 1 for _, label in samples), 'limit_per_class': args.benchmark_limit_per_class, 'seed': args.benchmark_seed, 'model_sets': manifest_sets, 'device_used': str(SELECTED_DEVICE), 'cuda_available': bool(torch.cuda.is_available()), 'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'nvidia_smi': benchmark_nvidia(), 'provenance_note': 'Integrated FYP1 benchmark; evaluates the five FYP1 model members together and does not start Flask.'}
    with open(os.path.join(RESOURCES_STATISTIC_FOLDER, 'benchmark_manifest.json'), 'w', encoding='utf-8') as handle: json.dump(manifest, handle, indent=2)
    with open(os.path.join(RESOURCES_STATISTIC_FOLDER, 'metrics.json'), 'w', encoding='utf-8') as handle: json.dump(all_metrics, handle, indent=2)
    benchmark_write_csv(os.path.join(RESOURCES_STATISTIC_FOLDER, 'metrics.csv'), all_metrics); print(json.dumps(all_metrics, indent=2))

if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        run_benchmark()
    else:
        app.run(host='0.0.0.0', port=8080, debug=False)