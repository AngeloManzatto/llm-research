# Small Language Models: A Developmental Approach to Conversational Intelligence

## Overview

This repository contains a research platform for studying the emergence of conversational abilities in small language models.

Unlike projects focused on building general-purpose assistants or competing with frontier language models, this project investigates how conversational capabilities progressively emerge from a pretrained next-token prediction (NTP) model through carefully designed and measurable training stages.

The emphasis is not on achieving state-of-the-art performance, but on understanding **which abilities emerge, when they emerge, and how they can be objectively measured**.

---

# Research Goal

The primary goal of this project is to study how a small language model evolves from a next-token predictor into a conversational agent through incremental and measurable stages of supervised learning.

Each stage introduces a new capability while preserving the ability to objectively evaluate previously acquired behaviors.

The project follows a scientific methodology in which every training stage is accompanied by deterministic benchmarks that quantify both newly acquired capabilities and remaining failure modes.

---

# Motivation

Recent large language models exhibit remarkable conversational abilities, reasoning skills and alignment behaviors. However, these models are trained using enormous computational resources and complex training pipelines that make it difficult to understand how individual capabilities emerge.

This project takes the opposite approach.

Instead of asking:

> *How can we build the most capable language model?*

we ask:

> *How do conversational abilities emerge in small language models?*

By working with relatively modest models, we hope to observe the developmental process itself rather than treating intelligence as a single final outcome.

---

# Research Philosophy

This project follows four principles.

## 1. Development over performance

The objective is to understand capability acquisition rather than maximizing benchmark scores.

Every training stage should answer a scientific question about the model's development.

---

## 2. Reproducibility

All experiments should be reproducible.

Every model checkpoint must be associated with:

* the training configuration
* the benchmark version
* deterministic evaluation results

---

## 3. Deterministic Evaluation

Whenever possible, evaluation should not depend on humans or external language models.

Benchmarks should be scored using deterministic algorithms based on predefined rules.

Examples include:

* exact match
* keyword matching
* role leakage detection
* repetition detection
* output length
* instruction compliance
* uncertainty detection
* corpus artifact detection

---

## 4. Incremental Complexity

Capabilities are introduced progressively.

Each stage assumes that previous stages have already been successfully learned and verified.

---

# Current Research Roadmap

## Stage 0 — Conversation

Objective:

Teach a pretrained next-token predictor the basic mechanics of conversation.

Capabilities under investigation:

* turn taking
* question answering
* instruction following
* local conversational context
* conversational corrections
* uncertainty
* response formatting

Evaluation:

Deterministic conversation benchmark.

---

## Stage 1 — Conversation Consistency

Study the model's ability to maintain coherent conversations over multiple turns.

Potential topics include:

* reference resolution
* short-term memory
* conversational coherence
* consistency

---

## Stage 2 — Concept Learning

Investigate whether abstract concepts can be introduced through supervised fine-tuning.

Examples include:

* ownership
* time
* causality
* belief
* intention

---

## Stage 3 — Human Values

Study whether models can learn explicit human values and socially desirable behaviors while preserving factual correctness.

---

## Stage 4 — Value Conflicts

Investigate situations where multiple desirable behaviors conflict.

Examples include:

* honesty versus politeness
* privacy versus helpfulness
* safety versus completeness

---

## Stage 5 — Alignment

Study alignment as an emergent consequence of progressively acquired conversational and social capabilities.

---

## Stage 6 — AI Safety

Investigate how developmental training influences robustness, failure modes and resistance to unsafe behaviors.

---

# Benchmark Philosophy

Benchmarks are treated as scientific instruments rather than leaderboards.

Their purpose is to characterize model behavior, not merely assign a score.

Every benchmark should be:

* version controlled
* deterministic
* reproducible
* extensible
* independent of third-party language models

Whenever possible, benchmark evaluation should be fully automated.

---

# Current Model

Current baseline:

* Transformer decoder
* 8 layers
* 8 attention heads
* 768 hidden dimensions
* context length: 1024 tokens
* vocabulary: 32,000 tokens

The initial model has been pretrained using next-token prediction on bilingual English and Portuguese corpora including Wikipedia, CC100, BRWAC and VisionVox.

---

# Repository Structure

```text
benchmarks/
    conversation_level0/

configs/

data/

pipelines/

runs/

src/
    core/
    tasks/
        ntp/
        sft/
            conversation/
```

---

# Long-Term Vision

The long-term objective is not to reproduce the capabilities of frontier language models.

Instead, this repository aims to become an experimental platform for studying the developmental trajectory of small language models, allowing individual conversational, cognitive and alignment-related capabilities to be introduced, measured and analyzed in a controlled and reproducible manner.

---

# Non-Goals

This project does **not** aim to:

* compete with frontier language models;
* maximize public benchmark scores at any cost;
* reproduce proprietary training pipelines;
* build a production-ready conversational assistant.

Instead, the project focuses on understanding the emergence of conversational behavior through controlled experimentation.

---

# Guiding Principle

> **Every new capability introduced into the model must be accompanied by a deterministic benchmark capable of measuring that capability without requiring subjective human or language-model evaluation.**
