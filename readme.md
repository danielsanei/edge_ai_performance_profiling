# Hardware-Aware Benchmarking of Generative Transformers on Edge ARM Architectures

---

## Overview

This project benchmarks autoregressive LLM inference on the **Raspberry Pi 5** (Cortex-A76, 8GB LPDDR4X, CPU-only) across a systematic sweep of models, quantization levels, context windows, and thread counts. All computation runs on the four ARM cores with no GPU acceleration. The study simultaneously profiles decoding throughput, generative fidelity (perplexity), and hardware telemetry including PMIC power draw, die temperature, and clock frequency across 202 total configurations.

---

## Models Evaluated

| Model | Params | Q4_K_M | Q8_0 |
|---|---|---|---|
| Llama 3.2 | 1B | 0.7 GB | 1.3 GB |
| Phi-4 Mini | 3.8B | 2.4 GB | 3.9 GB |
| Mistral 7B v0.3 | 7B | 4.1 GB | 7.2 GB |
| Llama 3.1 | 8B | 4.6 GB | (OOM) |

> Llama 3.1 8B Q8 excluded: its 8.0 GB weight file saturates the full 8 GB before any KV cache or OS overhead.

---

## Inference Stack

| Component | Role |
|---|---|
| `llama-cpp-python` | Python binding exposing full llama.cpp API; used for throughput and per-token perplexity in a single pipeline |
| `llama.cpp` | C/C++ inference backend optimized for CPU via BLAS and SIMD intrinsics |
| GGUF | Single-file quantized model format suited to unified memory architectures |
| `vcgencmd` | Raspberry Pi firmware utility for PMIC power rails, die temperature, clock frequency, and throttle status |
| `psutil` | CPU utilization and anonymous RAM consumption |

---

## Experimental Sweep

| Axis | Values |
|---|---|
| Quantization | Q4_K_M, Q8_0 |
| Context window | 512, 2048, 4096 tokens |
| Thread count | 2, 4 |
| Prompts | 5 task types (technical, creative, summarization, logical reasoning, mathematical reasoning) |
| **Total runs** | **202** (of 240 planned; 38 excluded pre-execution or aborted at runtime) |

---

## Key Findings

- **Memory bandwidth is the binding constraint.** Doubling thread count reduces throughput by 19–32% across every model and quantization combination. The LPDDR4X bus, not compute, is the bottleneck. 2 threads strictly dominates 4.
- **Q4 quantization cost is highly model-dependent.** Mistral 7B Q4_K_M matches Q8 perplexity within 10% while delivering 1.68× higher throughput, the best configuration in the study. Phi-4 Mini Q4 degrades perplexity by up to 10× with only 1.40× speedup, making Q4 unsuitable for quality-critical use.
- **The 8 GB ceiling is a hard boundary.** Mistral 7B Q8 at ctx=4096 triggers a 30× throughput collapse to 0.05 TPS as the system falls back to SD card swap, confirmed by simultaneous deviation across six hardware signals: TPS collapsed from 1.54 to 0.05, total inference time ballooned from ~88s to over 43 minutes per prompt, CPU utilization dropped to ~9%, the ARM clock scaled down from 2398 to 1733 MHz, and board power and die temperature fell to 2.97W and ~57°C respectively.
- **Architecture matters more than parameter count.** Mistral 7B Q4 outperforms Llama 3.1 8B Q4 on both speed and fidelity, demonstrating that model architecture, not size alone, determines edge inference performance.

---

## Repository Structure
```
edge_ai_performance_profiling/
├── data/
│   ├── individual/               # Per-run result files
│   ├── old/                      # Previous experiment data
│   ├── plots/                    # Figures used in the paper
│   ├── master_results.csv        # Raw results across all 202 configurations
│   └── benchmark_results.txt     # Human-readable run summaries
├── demo/
│   ├── ollama_demo.py
│   └── ollama_demo_2.py
├── paper/
│   ├── ECE 285 - Final Report.pdf
│   └── ECE 285 - Midterm Report.pdf
├── prompts/
│   └── prompts.json
├── scripts/
│   ├── prototyping/
│   │   ├── ppl.py
│   │   ├── prompt.py
│   │   ├── prompt_2.py
│   │   └── prompt_3.py
│   ├── pipeline.py
│   ├── pipeline2.py
│   └── visualizer.py
├── .gitignore
├── final_report.tex
└── readme.md
```