# FYP1 Deepfake Detector

FYP1 uses five detection models consisting of ResNet50, Xception, Vision Transformer, DFT-CNN and Canny-CNN. Each model produces one vote and the final `REAL` or `FAKE` result is decided using majority voting.

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- Internet connection during first installation
- Required trained model files from the provided Google Drive link
- Optional NVIDIA GPU for CUDA acceleration

## Quick Start

1. Open `Dataset and Trained Model ZIP link` and download the required ZIP file.
2. Extract the required dataset and trained model files according to the provided folder structure.
3. Double-click `run_detector.bat`.
4. During first run, the launcher creates a virtual environment and installs the required dependencies.
5. Wait for the Flask server to start.
6. Open the detector at:

```text
http://127.0.0.1:8080
```

## How to Use

1. Confirm that the required trained models are available.
2. Upload a single image, multiple images, ZIP batch or MP4 video.
3. Wait for the models to complete processing.
4. Review the individual model outputs and final majority-voting result.

## Main Files

```text
deepfake_detector.py                  Main Flask detector
run_detector.bat                      Windows launcher
requirements.txt                      Python dependencies
Train_Dataset.ipynb                   Main training notebook
Train_Gan_Dataset.ipynb               GAN dataset training notebook
Dataset and Trained Model ZIP link    Dataset and trained model download link
resources_statistic_result/           Saved benchmark records
```

## Important Note

The prediction result should be used for screening and research purpose. A `REAL` or `FAKE` result should not be treated as absolute proof that an image is authentic or manipulated.
