import re
import subprocess

TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"

def detect_nvidia_cuda_version():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", result.stdout)
    if not match:
        return None

    return int(match.group(1)), int(match.group(2))


def select_torch_build():
    cuda_version = detect_nvidia_cuda_version()
    if cuda_version is None:
        return "cpu", "cpu", "none"

    major, minor = cuda_version

    if major > 12 or (major == 12 and minor >= 8):
        return "cuda", "cu128", "12.8"
    if major == 12 and minor >= 6:
        return "cuda", "cu126", "12.6"
    if major == 12 or (major == 11 and minor >= 8):
        return "cuda", "cu118", "11.8"

    return "cpu", "cpu", "none"


build, index, cuda_runtime = select_torch_build()
print("|".join([build, index, cuda_runtime, TORCH_VERSION, TORCHVISION_VERSION]))