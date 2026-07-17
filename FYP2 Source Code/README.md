# FYP2 Deepfake Detector

Deepfake Detector use localhost python Flask server for checking whether an uploaded image or video looks `REAL`, `FAKE`, or `UNCERTAIN`.

The app uses a EfficientNet-B0 model with a lightweight validation-trained forensic calibration layer. It also shows Grad-CAM, frequency view, edge map, and noise residual so the result is easier to inspect instead of only showing a label.

## What It Can Do / Features

- Detect single images, multiple images, ZIP batches, and MP4 videos.
- Auto-load the newest model from the `models/` folder when the app starts.
- Run on CPU, or CUDA GPU when PyTorch detects one.
- Use adjustable threshold modes for `REAL`, `FAKE`, and `UNCERTAIN`.
- Preserve classic force every result into `REAL` or `FAKE` at 0.50 for a controlled comparison. Normal use keeps this option off. (Default is off)
- Show batch summaries, resource statistics, and metrics (only when filenames contain labels like `real` or `fake`)
- Calculate and display both raw EfficientNet-B0 ROC-AUC and calibrated full-system ROC-AUC on the same labelled batch. The raw value excludes forensic calibration, in contrast calibrated value includes it.

## Project Files

```txt
deepfake_detector.py         Main Flask app
start.bat                    Windows launcher
detect_torch_build.py        Helps start.bat choose correct PyTorch
requirements.txt             Python dependencies except PyTorch
Train_Dataset.ipynb          Training notebook
models/                      Put model files here
analysis/                    Validation/calibration scripts and saved results
benchmark_data/              External evaluation dataset folders (calibrator use only)
resources_statistic_result/  Benchmark and uploaded-batch statistics
uploads/                     Temporary uploaded files
outputs/                     Generated result images and plots
```

## Requirements

- Python 3.10 or newer
- Internet connection for the first install
- Optional NVIDIA GPU for CUDA acceleration

The launcher installs PyTorch separately because correct PyTorch build depends whether your machine can use CUDA.

## Quick Start

Double-click:

```bat
start.bat
```

On first run, it will create `venv`, install dependencies, install a matching PyTorch build, and start the server.

Open the app here:

```txt
http://127.0.0.1:8080
```

Put your model file in:

```txt
models/
```

The app supports:

```txt
.pth
.pt
.safetensors
```

If there is already a supported model in `models/`, the app will load it automatically when it starts.

## How to use

1. Start the server.
2. Open `http://127.0.0.1:8080`.
3. Choose CPU or GPU.
4. Check that a model is loaded, or upload a different model.
5. Upload images, a ZIP file, or an MP4 video.
6. Review the prediction results, explanation and visual evidence.

For large batches detects, the config panel has a `Batch Grad-CAM Limit`. Higher values create more heatmaps, but results will take longer to display.

## Model Notes

The base detector expects an EfficientNet-B0 binary classifier with one output logit:

- `0` means real
- `1` means fake

The deployed system uses the raw CNN probability and lightweight forensic calibration, then converts the calibrated fake probability into:

- `REAL`
- `FAKE`
- `UNCERTAIN`

The loader accepts plain state dictionaries and common checkpoint wrappers like `state_dict`, `model_state_dict`, `model_state`, or `net_state_dict`

## Training

`Train_Dataset.ipynb` is the training notebook.

It trains EfficientNet-B0 model using real/fake dataset. Train it on google colab, after training save the checkpoint and copy it into `models/` then restart the detector.

A dataset layout should looks like this:

```txt
dataset/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

## Notes

- The app runs locally on port `8080`.
- Maximum upload size is set to 1024 MB.
- Batch metrics only work when filenames include labels such as `real` or `fake`.
- Raw and calibrated ROC-AUC are only available when the uploaded batch contains both labelled classes; calibrated ROC-AUC also requires calibration to be applied to every labelled sample.

## Credits

Ong Yong Quan - FYP Title: Identifying Deepfake Images