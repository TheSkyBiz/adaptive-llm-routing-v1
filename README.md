# Adaptive Routing Between Small and Large Language Models

A system for **cost-efficient and scalable question answering** using adaptive routing between Small Language Models (SLMs) and Large Language Models (LLMs).

---

## Overview

Large Language Models achieve strong performance but are expensive and slow, while Small Language Models are efficient but less reliable.  
This project explores a hybrid system where queries are first handled by an SLM and selectively escalated to an LLM based on confidence signals, enabling a balance between **cost, latency, and accuracy**.

---

## Key Idea

Instead of using an LLM for every query:

- Use an SLM to generate an initial answer  
- Estimate confidence using lightweight signals  
- Escalate to LLM only when necessary  

---

## Architecture

```
SLM → Confidence Signals → Routing Decision → (Accept / Escalate) → LLM
```

---

## Routing Signals

- **Answer Length** — verbosity as an uncertainty proxy  
- **Token Entropy** — generation uncertainty  
- **Log Probability** — model confidence  

---

## Datasets

- SQuAD — extractive question answering  
- SQuAD v2 — includes unanswerable questions  
- HotpotQA — multi-hop reasoning  

---

## Features

- Multiple routing strategies (**v1, v2, v3**)  
- Baseline evaluation for SLM and LLM  
- Cost–accuracy trade-off analysis  
- Calibration evaluation (ECE)  
- Logging and reproducible experiment pipeline  
- Visualization of routing behavior and performance  

---

## Results Summary

- Up to **~70% reduction in LLM usage**  
- Competitive performance with LLM baseline on SQuAD  
- Strong improvements on SQuAD v2 through selective escalation  
- Reduced performance on multi-hop reasoning (HotpotQA), highlighting limitations  

---

## Project Structure

```
models/      → model wrappers (SLM, LLM)
utils/       → prompt formatting and helpers
analysis/    → plotting and visualization scripts
plots/       → generated visualizations
logs/        → experiment outputs (ignored in repo)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Run SLM baseline:

```bash
python run_baseline_slm.py
```

Run LLM baseline:

```bash
python run_baseline_llm.py
```

Run routing pipeline:

```bash
python run_routing_pipeline.py
```

---

## Evaluation

Evaluate SLM:

```bash
python evaluate_slm.py
```

Evaluate LLM:

```bash
python evaluate_llm.py
```

Evaluate routing:

```bash
python evaluate_routing.py
```

---

## Configuration

All experiment settings are controlled via:

```
config.yaml
```

---

## Analysis and Visualization

Scripts in `analysis/` generate:

- Cost vs Accuracy trade-offs  
- Calibration plots (ECE)  
- Routing behavior plots  
- Model comparison visualizations  

---

## Limitations

- Heuristic-based routing (no learned decision model)  
- Weak performance on multi-hop reasoning tasks  
- Calibration issues in both SLM and LLM  
- Limited model scale (1.5B–8B)  

---

## Future Work

- Learned routing (logistic regression)  
- Self-consistency based uncertainty estimation  
- Larger models (14B–70B)  
- Semantic caching for query reuse  
- Speculative decoding for faster inference  

---

## Author

**Aakash Biswas**

---

## Notes

This project was developed as part of an undergraduate thesis focused on efficient LLM systems and is designed to be extended toward research-level contributions.
