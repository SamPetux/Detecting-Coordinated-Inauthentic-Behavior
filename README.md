# Detecting LLM-Powered Coordinated Inauthentic Behavior via Multi-Signal Synchronization Graphs

Research code for a paper on detecting coordinated groups of accounts whose messages are written by large language
models (LLMs). Such campaigns paraphrase every message, so detectors that rely on copy-paste or lexical similarity
miss them. The project tests whether synchronisation across several signals still reveals the group: who reacts to
which post, when, and with what meaning. Each signal is calibrated against a null model that preserves organic
bursts of activity.

**Status:** work in progress (research design stage). See `docs/` for the analysis, plan and decision log
(in Russian).

## Planned components
- Synchronisation layers between accounts: same target within a time window, semantic similarity within a time
  window, lexical near-duplicates, writing-style similarity.
- Bucket-preserving permutation null model with per-edge significance and false discovery rate control.
- Layer fusion and campaign extraction (Leiden).
- Evaluation:
  - an attacker ladder (copy-paste → templates → LLM paraphrase → LLM personas with several generators →
    desynchronised timing → camouflage), injected into a real background of Russian-language city communities;
  - a counterfactual LLM rewrite of labelled information-operation datasets.

## Data
Raw social-media data are not published: they contain personal data. Only code, aggregate results and, if
released, synthetic campaign texts on fictional topics will be shared.

## Repository layout (planned)
```
cib/        package: data loaders, generators, layers, null model, fusion, baselines, metrics
scripts/    entry points and resumable experiment queues
configs/    exploration and frozen confirmatory configurations
docs/       analysis, plan, sources, task list, decision log, protocol, results
paper/      manuscript sources and figures
```
