#!/usr/bin/env python3
"""
Export the final trained model to Ollama.

Pipeline:
    1. Fuse LoRA adapters into the base model  (mlx_lm.fuse)
    2. Convert fused model to GGUF             (llama.cpp convert_hf_to_gguf.py)
    3. Quantize to Q4_K_M                     (llama-quantize or llama.cpp)
    4. Write Ollama Modelfile
    5. Import into Ollama                      (ollama create)

Usage:
    python scripts/export_to_ollama.py [--adapter-path outputs/grpo/adapters] [--dry-run]
    python scripts/export_to_ollama.py --skip-gguf   # skip if GGUF already done

Prerequisites:
    pip install mlx-lm
    brew install llama.cpp   (or build from source for convert/quantize)
    ollama must be installed and running
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_MLX = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
MODEL_HF = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"

# Priority order for adapter selection
ADAPTER_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "grpo" / "adapters",
    PROJECT_ROOT / "outputs" / "dpo" / "adapters",
    PROJECT_ROOT / "outputs" / "sft" / "adapters",
]

FUSED_PATH = PROJECT_ROOT / "outputs" / "final" / "fused"
GGUF_PATH = PROJECT_ROOT / "outputs" / "final" / "model-q8.gguf"   # q8_0 intermediate (bf16 too large on constrained disks)
GGUF_Q4_PATH = PROJECT_ROOT / "outputs" / "final" / "model-q4_k_m.gguf"
MODELFILE_PATH = PROJECT_ROOT / "outputs" / "final" / "Modelfile"
OLLAMA_MODEL_NAME = "qwen25-tax-3b"

SYSTEM_PROMPT = """\
You are a tax law assistant trained on the Internal Revenue Code (Title 26) \
and Treasury Regulations (26 CFR). You answer questions about US federal tax law \
accurately, cite relevant IRC sections, and note important exceptions and limitations. \
You do not provide personalised tax advice; always recommend consulting a qualified \
tax professional for individual situations.\
"""


# ---------------------------------------------------------------------------
# Step 1: Fuse adapters
# ---------------------------------------------------------------------------

def resolve_adapter(override: str | None) -> Path | None:
    if override:
        p = Path(override)
        if not p.exists():
            print(f"ERROR: Specified adapter path does not exist: {p}")
            sys.exit(1)
        return p
    for candidate in ADAPTER_CANDIDATES:
        if (candidate / "adapter_config.json").exists():
            print(f"Using adapter: {candidate}")
            return candidate
    print(
        "WARNING: No trained adapter found. "
        "Export will fuse the base model without any fine-tuning."
    )
    return None


def resolve_base_model() -> Path:
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        return MODEL_HF
    print(f"ERROR: Base model not found at {MODEL_MLX} or {MODEL_HF}")
    sys.exit(1)


def fuse_adapters(model_path: Path, adapter_path: Path | None, dry_run: bool) -> Path:
    """Merge LoRA adapter weights into the base model via mlx_lm.fuse."""
    FUSED_PATH.mkdir(parents=True, exist_ok=True)

    if adapter_path is None:
        # No adapter — just copy/symlink the base model
        print("No adapter to fuse. Copying base model to fused path ...")
        if not dry_run:
            if FUSED_PATH.exists():
                shutil.rmtree(FUSED_PATH)
            shutil.copytree(str(model_path), str(FUSED_PATH))
        return FUSED_PATH

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", str(model_path),
        "--adapter-path", str(adapter_path),
        "--save-path", str(FUSED_PATH),
        "--dequantize",   # export as bf16 (no quantization) for best GGUF quality
    ]
    print("\nStep 1: Fusing LoRA adapters")
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print("ERROR: mlx_lm.fuse failed.")
            sys.exit(1)
        print(f"Fused model saved to: {FUSED_PATH}")
    return FUSED_PATH


# ---------------------------------------------------------------------------
# Step 2: Convert to GGUF
# ---------------------------------------------------------------------------

def find_llama_cpp_convert() -> Path | None:
    """Find llama.cpp's convert_hf_to_gguf.py script."""
    candidates = [
        # Homebrew (bin/ — newer llama.cpp formula installs here directly)
        Path("/opt/homebrew/bin/convert_hf_to_gguf.py"),
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
        # Homebrew (share/)
        Path("/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
        # Common build locations
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/tmp/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # Try locating via 'llama-quantize' binary path
    quantize = shutil.which("llama-quantize")
    if quantize:
        bin_dir = Path(quantize).parent
        # Check the same bin/ directory first (e.g. /opt/homebrew/bin/)
        for name in ["convert_hf_to_gguf.py", "convert.py"]:
            p = bin_dir / name
            if p.exists():
                return p
        # Then check one level up (legacy build layout)
        parent = bin_dir.parent
        for name in ["convert_hf_to_gguf.py", "convert.py"]:
            p = parent / name
            if p.exists():
                return p

    return None


def convert_to_gguf(fused_path: Path, dry_run: bool) -> Path:
    """Convert the fused HF model to GGUF format."""
    print("\nStep 2: Converting to GGUF")

    converter = find_llama_cpp_convert()
    if converter is None:
        print(
            "WARNING: llama.cpp convert_hf_to_gguf.py not found.\n"
            "Install with: brew install llama.cpp\n"
            "Or build from source: https://github.com/ggerganov/llama.cpp\n"
            "Skipping GGUF conversion — you must convert manually."
        )
        return GGUF_PATH  # placeholder; won't exist

    # If using a local llama.cpp clone (e.g. /tmp/llama.cpp), inject its
    # bundled gguf-py into PYTHONPATH so the convert script finds the right
    # gguf package version (brew's PyPI gguf 0.18.0 may lag behind the binary).
    env = None
    gguf_py = converter.parent / "gguf-py"
    if gguf_py.exists():
        import os
        env = {**os.environ, "PYTHONPATH": str(gguf_py) + ":" + os.environ.get("PYTHONPATH", "")}
        print(f"Using bundled gguf-py from: {gguf_py}")

    cmd = [
        sys.executable, str(converter),
        str(fused_path),
        "--outfile", str(GGUF_PATH),
        "--outtype", "q8_0",   # q8_0 is ~half the size of bf16; llama-quantize cannot re-quantize from q8_0 in brew llama.cpp
    ]
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False, env=env)
        if result.returncode != 0:
            print("ERROR: GGUF conversion failed. Check llama.cpp version compatibility.")
            sys.exit(1)
        print(f"GGUF file saved to: {GGUF_PATH}")
    return GGUF_PATH


def quantize_gguf(gguf_path: Path, dry_run: bool) -> Path:
    """Quantize the GGUF to Q4_K_M for efficient Ollama serving.

    NOTE: llama-quantize (brew llama.cpp) cannot re-quantize from q8_0 input.
    Since GGUF_PATH is now q8_0, this step is skipped and the q8_0 is used
    directly with Ollama. The q8_0 model is ~3.3 GB vs ~2.0 GB for Q4_K_M,
    but has slightly better quality.
    """
    print("\nStep 3: Quantizing to Q4_K_M")

    # If the input is already q8_0 (our default), llama-quantize cannot
    # re-quantize it. Skip quietly and use q8_0 as the Ollama model.
    if gguf_path == GGUF_PATH and "q8" in gguf_path.name:
        print("INFO: Input GGUF is q8_0 — brew llama-quantize cannot re-quantize from q8_0.")
        print("      Using q8_0 directly with Ollama (3.3 GB vs 2.0 GB for Q4_K_M).")
        return gguf_path

    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        print(
            "WARNING: llama-quantize not found.\n"
            "Install with: brew install llama.cpp\n"
            "Skipping quantization — Ollama will use the q8_0 GGUF (larger file)."
        )
        return gguf_path  # fall back to unquantized

    cmd = [quantize_bin, str(gguf_path), str(GGUF_Q4_PATH), "Q4_K_M"]
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("ERROR: Quantization failed. Using unquantized GGUF.")
            return gguf_path
        print(f"Quantized GGUF saved to: {GGUF_Q4_PATH}")
    return GGUF_Q4_PATH


# ---------------------------------------------------------------------------
# Step 3: Write Modelfile
# ---------------------------------------------------------------------------

def write_modelfile(gguf_path: Path, dry_run: bool) -> Path:
    """Write an Ollama Modelfile."""
    print("\nStep 4: Writing Modelfile")
    MODELFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = f"""\
FROM {gguf_path}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
"""
    print(f"Modelfile path: {MODELFILE_PATH}")
    if not dry_run:
        MODELFILE_PATH.write_text(content)
        print("Modelfile written.")
    else:
        print("--- Modelfile contents (dry run) ---")
        print(content)
        print("---")
    return MODELFILE_PATH


# ---------------------------------------------------------------------------
# Step 4: Import to Ollama
# ---------------------------------------------------------------------------

def check_ollama() -> bool:
    ollama = shutil.which("ollama")
    if ollama is None:
        print("WARNING: ollama not found in PATH. Install from https://ollama.com")
        return False
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: ollama is installed but not responding. Is the server running?")
        print("Start with: ollama serve")
        return False
    return True


def import_to_ollama(modelfile_path: Path, model_name: str, dry_run: bool) -> None:
    """Create the Ollama model from the Modelfile."""
    print("\nStep 5: Importing to Ollama")

    if not check_ollama():
        print(f"Skipping Ollama import. Run manually:\n  ollama create {model_name} -f {modelfile_path}")
        return

    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("ERROR: ollama create failed.")
            sys.exit(1)
        print(f"\nModel '{model_name}' imported to Ollama successfully.")
        print(f"Test with: ollama run {model_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model to Ollama")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (default: auto-detect best available)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=OLLAMA_MODEL_NAME,
        help=f"Ollama model name (default: {OLLAMA_MODEL_NAME})",
    )
    parser.add_argument(
        "--skip-fuse",
        action="store_true",
        help="Skip fusion step (use existing fused model at outputs/final/fused)",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip GGUF conversion (use existing GGUF at outputs/final/)",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip Q4_K_M quantization",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    model_name = args.model_name
    base_model = resolve_base_model()
    adapter_path = resolve_adapter(args.adapter_path)

    print("\n" + "=" * 70)
    print("EXPORT TO OLLAMA")
    print(f"  base model:    {base_model}")
    print(f"  adapter:       {adapter_path or 'none'}")
    print(f"  fused output:  {FUSED_PATH}")
    print(f"  gguf output:   {GGUF_Q4_PATH}")
    print(f"  ollama name:   {model_name}")
    print("=" * 70 + "\n")

    # Step 1
    if args.skip_fuse and FUSED_PATH.exists():
        print("Skipping fusion — using existing fused model.")
        fused = FUSED_PATH
    else:
        fused = fuse_adapters(base_model, adapter_path, args.dry_run)

    # Step 2
    if args.skip_gguf and GGUF_PATH.exists():
        print("Skipping GGUF conversion — using existing GGUF.")
        gguf = GGUF_PATH
    else:
        gguf = convert_to_gguf(fused, args.dry_run)

    # Step 3
    if args.skip_quantize:
        final_gguf = gguf
    else:
        final_gguf = quantize_gguf(gguf, args.dry_run)

    # Step 4
    modelfile = write_modelfile(final_gguf, args.dry_run)

    # Step 5
    import_to_ollama(modelfile, model_name, args.dry_run)

    print("\nExport pipeline complete.")
    if not args.dry_run:
        print(f"Test your model: ollama run {model_name}")


if __name__ == "__main__":
    main()
