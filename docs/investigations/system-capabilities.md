# System Capabilities Report

**Date:** 2026-03-25

---

## Ollama

- **Version:** 0.18.2
- **Downloaded models:** None (empty model list)

## Hardware

| Component | Details |
|-----------|---------|
| **CPU** | Apple M4 Max |
| **RAM** | 128 GB (137,438,953,472 bytes) |
| **GPU** | Apple M4 Max, 40 cores, Metal 3 |
| **Display** | Built-in Liquid Retina XDR (3456x2234) + LG Ultra HD 4K external |

## Disk Space

| Filesystem | Size | Used | Available | Capacity |
|------------|------|------|-----------|----------|
| /dev/disk3s5 | 926 GB | 856 GB | **41 GB free** | 96% used |

**Note:** Disk space is tight. Large model downloads (e.g., 70B quantized models at ~40 GB) would essentially fill the disk. Smaller models (7B-14B) at 4-8 GB each are feasible.

## Python & Package Managers

| Tool | Status |
|------|--------|
| **Python** | 3.14.3 (installed via Homebrew) |
| **pip** | 26.0 (available) |
| **conda** | Not installed |

## ML/AI Python Packages

| Package | Version | Installed? |
|---------|---------|------------|
| torch (PyTorch) | -- | No |
| transformers | -- | No |
| trl | -- | No |
| unsloth | -- | No |
| peft | -- | No |
| datasets | 4.7.0 | Yes |
| accelerate | -- | No |

## Hugging Face CLI

- **Status:** Not installed

---

## Summary

This is a high-end Apple M4 Max machine with 128 GB unified memory and a 40-core GPU -- excellent for running local LLMs via Ollama and for on-device fine-tuning with MLX or similar Apple Silicon-optimized frameworks. However, disk space is constrained at 41 GB free, limiting the size and number of models that can be downloaded. Ollama is installed but has no models pulled yet. The ML/AI Python ecosystem is mostly absent (only `datasets` is installed); PyTorch, transformers, PEFT, TRL, and the Hugging Face CLI would all need to be installed for any fine-tuning workflow.
