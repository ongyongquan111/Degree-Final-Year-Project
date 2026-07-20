# Degree Final Year Project

## Project Title: Identifying Deepfake Images

This repository contains the source codes, trained models, dataset links and evaluation records.

FYP1 used five detection models with majority voting, which includes ResNet50, Xception, Vision Transformer, DFT-CNN and Canny-CNN. Updated FYP2 reduces the architecture into one EfficientNet-B0 model together with a lightweight validation-trained forensic fusion classifier.

The detector can analyse single images, multiple images, ZIP batches and MP4 videos through a local Flask interface.

## FYP1 and FYP2 Comparison

| FYP1 Detector | FYP2 Detector |
|---|---|
| Uses five models: ResNet50, Xception, Vision Transformer, DFT-CNN and Canny-CNN. | Uses one EfficientNet-B0 CNN together with a validation-trained HistGradientBoosting forensic fusion classifier. |
| Produces final `REAL` or `FAKE` result using majority voting. | Produces `REAL`, `UNCERTAIN` or `FAKE` result using threshold-based output. |
| Requires five trained model checkpoints to operate the complete ensemble. | Uses one CNN checkpoint and one lightweight fusion file named `efficientnet_b0_forensic_calibrator.joblib`. |
| Supports single images, multiple images, ZIP batches and MP4 videos. | Supports single images, multiple images, ZIP batches and MP4 videos. |
| Provides DFT, Canny and other model outputs used by the ensemble decision. | Provides Grad-CAM, DFT, Canny edge and noise-residual views to support explanation. |
| **How to start:** Download and extract the required trained model files using the link provided in the folder. Double-click `run_detector.bat`, wait for the server to start and open `http://127.0.0.1:8080`. | **How to start:** Double-click `start.bat`. During first run, it creates a virtual environment and installs the required dependencies. Open `http://127.0.0.1:8080` after the server has started. |
| **How to use:** Upload the supported media, wait for all five models to complete processing and review the majority-voting result. | **How to use:** Select CPU or GPU mode, upload the supported media, then review the prediction and available explanation views. |
| [Open FYP1 Source Code](FYP1%20and%20FYP2%20Source%20Codes/FYP1%20Source%20Code) | [Open FYP2 Source Code](FYP1%20and%20FYP2%20Source%20Codes/FYP2%20Source%20Code) |

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- Internet connection during first installation
- Optional NVIDIA GPU for CUDA acceleration

Both detectors run through a localhost Flask server. They can also run using CPU when CUDA is not available.

## Important Note

The prediction and explanation views are provided for screening and research purpose. A `REAL` or `FAKE` output should not be treated as absolute proof that an image is authentic or manipulated.

Detailed guidelines are available in the `README.md` inside each FYP source code folder.
