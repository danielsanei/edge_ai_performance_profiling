# Hardware-Aware Benchmarking of Generative Transformers on Edge ARM

**ECE 285 — Deep Generative Models | UC San Diego | March 2026**

## Overview
Benchmarking study evaluating the performance trade-offs of deploying open-weight LLMs on the Raspberry Pi 5 (Cortex-A76, 8GB RAM, no GPU). Measures how quantization level and inference configuration impact throughput, latency, energy efficiency, and generative fidelity.

## Models
| Model | Params | Q4_K_M | Q8_0 |
|---|---|---|---|
| Llama 3.2 | 1B | 771 MB | 1.3 GB |
| Phi-4 Mini | 3.8B | 2.4 GB | 3.9 GB |
| Mistral 7B v0.3 | 7B | 4.1 GB | 7.2 GB |
| Llama 3.1 | 8B | 4.6 GB | 8.0 GB |

## Stack
- **Ollama** — model serving and management
- **llama.cpp** — CPU-optimized inference backend
- **llama-cpp-python** — PPL computation via per-token log-probabilities
- **Hardware telemetry** — 100ms polling (temp, power, clock, RAM)

## Parameters Swept
- Quantization: Q4_K_M, Q8_0
- Context window: 512, 2048, 4096 tokens
- Thread count: 2, 4
- Prompts: 5 task types × all configs = 202 total runs

## Key Findings
- Q4_K_M is 1.4–1.8× faster than Q8 with minimal quality loss
- Practical RAM ceiling is ~4.6–5 GB model size
- Mistral 7B Q8 at CTX=4096 collapses to 0.05 TPS (swap thrashing)
- 2 threads outperforms 4 — bottleneck is memory bandwidth, not compute
- Llama 3.1 8B Q8 excluded (OOM)

## Author
Daniel Sanei — dsanei@ucsd.edu