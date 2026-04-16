# Stable Diffusion Local Inference

Local text-to-image generation using Stable Diffusion models via the Diffusers library.  
Includes GPU vs CPU performance comparison and multi-model benchmarking on consumer hardware.

## Models Used

| Model | Parameters | Resolution | VRAM |
|-------|-----------|------------|------|
| SD v1-5 | 860M | 512×512 | ~4 GB |
| SDXL 1.0 | 3.5B | 1024×1024 | ~8 GB |
| Juggernaut XL (Ragnarok) | 3.5B | 1024×1024 | ~8 GB |

## Results

### SD v1-5 — CPU (56 threads, 2× Xeon E5-2690 v4)
![cpu result](m_cpu.png)

Generation time: ~600s

### SD v1-5 — GPU (RTX 3060 12GB)
![gpu result](m_gpu.png)

Generation time: ~5s

### SDXL 1.0 — GPU (RTX 3060 12GB)
![sdxl result](movsar_sdxl.png)

Generation time: ~48s

### Juggernaut XL Ragnarok — GPU (RTX 3060 12GB)
![juggernaut result](juggernaut_result.png)

Generation time: ~60s

## Hardware

- CPU: 2× Intel Xeon E5-2690 v4 (28 cores / 56 threads total)
- GPU: NVIDIA GeForce RTX 3060 12GB
- RAM: 64GB DDR4
- OS: Fedora Linux 43

## Setup

```bash
conda create -n sd-env python=3.11 -y
conda activate sd-env

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.31.0 transformers==4.40.0 accelerate==0.30.0 "numpy<2.0" Pillow
```

## Notebooks

| File | Model | Device | Scheduler | Description |
|------|-------|--------|-----------|-------------|
| `hw9_m_gpu.ipynb` | SD v1-5 | GPU | DPM++ | fp16, cudnn.benchmark |
| `hw9_m_cpu.ipynb` | SD v1-5 | CPU | DPM++ | 56 threads, float32 |
| `hw9_movsar_sdxl.ipynb` | SDXL 1.0 | GPU | DPM++ | 1024x1024, fp16 |
| `hw9_juggernaut.ipynb` | Juggernaut XL | GPU | DPM++ | from_single_file, 1024x1024 |
| `M-task9.ipynb` | SD v1-5 | GPU/CPU | DPM++ | auto-detect device |

## GPU vs CPU Comparison

| Metric | GPU (RTX 3060) | CPU (2x Xeon E5-2690 v4) |
|--------|---------------|--------------------------|
| Time (25 steps, 512x512) | ~5s | ~600s |
| Precision | float16 | float32 |
| VRAM / RAM used | ~4 GB VRAM | ~8 GB RAM |
| Speedup | 120x faster | baseline |

## Model Quality Comparison

| Model | Prompt adherence | Detail | Realism | Speed |
|-------|-----------------|--------|---------|-------|
| SD v1-5 | medium | low | low | fast |
| SDXL 1.0 | high | high | high | medium |
| Juggernaut XL | high | very high | very high | medium |

## Key Concepts

- **Scheduler** — algorithm that controls the step-by-step denoising process (DPM++, DDIM)
- **fp16** — half-precision float (16-bit), reduces VRAM usage by 2x vs fp32
- **Seed** — fixed random state for reproducible results across runs
- **CFG scale** — guidance scale: how strictly the model follows the prompt
- **Karras sigmas** — noise schedule that improves image sharpness at fewer steps
- **from_single_file()** — loads model from local .safetensors file instead of HuggingFace

## Prompt Used

```
A sprawling cyberpunk megacity at midnight, rain-slicked streets reflecting
cascades of neon signs in Cyrillic and Japanese, towering brutalist skyscrapers
wrapped in holographic banners, hovercars threading between lit windows,
volumetric fog, ultra-detailed, cinematic 4k, photorealistic render
```

Negative prompt:
```
daytime, sunny, cartoon, anime, low quality, blurry, watermark,
text overlay, deformed architecture, oversaturated
```