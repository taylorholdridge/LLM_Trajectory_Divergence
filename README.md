# TDS: Trajectory Divergence Score

**Research question:** How do linguistic perturbations affect the stability of correct LLM outputs in clinical contexts?

This notebook (`TDS_Code.ipynb`, originally a Google Colab) probes a clinical decision LLM's *internal reasoning stability* under prompt perturbations, not just its final answer. Even when a perturbed prompt produces the same (correct) YES/NO output as the baseline, the model's internal hidden-state trajectory may diverge substantially — this project defines and measures that divergence.

## Dataset

[MedPerturb](https://github.com/abinithago/MedPerturb) — clinical vignettes paired with a gold-standard "self-manage at home?" (YES/NO) label, each available in multiple perturbed forms:

- `baseline` — original clinical context
- `gender_swap` — patient gender swapped
- `colorful_tone` — emotionally embellished phrasing
- `uncertain_tone` — hedged/uncertain phrasing

Only contexts that have **all four** perturbation types are kept, and rows tagged `summary` are dropped.

## Model

`meta-llama/Meta-Llama-3.1-8B-Instruct` (via Hugging Face `transformers`), run with `output_hidden_states=True` so every layer's hidden states are available for analysis. Requires a Hugging Face access token with access to the gated Llama 3.1 model (`login()` / `hf_token`).

## Pipeline

1. **Load & clean** MedPerturb data; keep `context_id`, `perturbation`, `clinical_context`, `gold_standard_manage`; filter to contexts with a complete perturbation set.
2. **Prompt formatting** — each clinical context is wrapped in a fixed physician-style prompt asking for a YES/NO self-management recommendation (mirrors the MedPerturb prompt design).
3. **Forward pass** (`run_model`) — runs the prompt through the model, extracts next-token logits for YES/NO to get a prediction, and returns all hidden states.
4. **Drift metrics**:
   - `compute_layer_drift` — per-layer cosine distance (`1 - cosine_similarity`) between the baseline's and perturbation's *last-token* hidden state.
   - `compute_token_drift` — same cosine-distance comparison but per-token, per-layer (a full drift matrix).
   - `extract_last_token_embeddings` — pulls the last-token embedding at every layer for trajectory/PCA analysis.
5. **Sampling** — a random sample of `SAMPLE_SIZE = 225` context IDs is run through baseline + the three non-`summary` perturbations.
6. **Filtering** — results are kept only where baseline output == perturbed output == gold label, isolating cases where the *final answer stayed correct* so any internal drift can't be explained by an output change.
7. **Trajectory Divergence Score (TDS)** — for each filtered (context, perturbation) pair, the Euclidean distance between baseline and perturbed last-token embeddings is computed per layer, then summarized as:
   - `tds_mean` — average distance across layers
   - `tds_max` — largest single-layer deviation
   - `tds_final` — divergence at the last layer (closest to the decision point)
   - `tds_auc` — area under the layer-wise distance curve (cumulative divergence)

## Visualizations

- Text-length histogram of clinical contexts
- Token-level drift heatmaps (layer × token cosine distance)
- 2D and 3D PCA embedding trajectories (baseline vs. perturbed, per sample)
- Top-10 highest-divergence trajectory plots
- Average trajectory across the full sample (baseline vs. each perturbation type)
- Trajectory divergence line plot across all samples

## Statistical analysis

- One-way ANOVA comparing `tds_mean` across perturbation types
- Pairwise t-tests between perturbation types
- Distribution plots (histogram/KDE/violin) of `tds_auc` by perturbation

## Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
plotly
scipy
torch
transformers
huggingface_hub
```

A Hugging Face token with access to `meta-llama/Meta-Llama-3.1-8B-Instruct`, and GPU access (the notebook was run on Colab), are required to reproduce the model-forward-pass steps.

## Notes

- The `hf_token` in the notebook is a placeholder — swap in your own token before running.
- `SAMPLE_SIZE` and the perturbation types compared can be adjusted in the "Setup of Experiment" section.
