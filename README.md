# ActiveScope 🔍

[![Conference](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-lightgrey)](#) 

> **ActiveScope: Actively Seeking and Correcting Perception for MLLMs** <br>
> **ICML 2026**

This is the official repository for the paper **ActiveScope**.

## 📢 Updates & News

> **🚧 Work In Progress **
> 
> Thank you for your interest in our work! We are currently in the process of cleaning up and organizing the codebase. The source code and evaluation scripts will be released here very soon. 
> 

## 📖 Introduction

Multimodal Large Language Models (MLLMs) have demonstrated impressive vision-language understanding, yet they still struggle with fine-grained perception in high-resolution images. Existing training-free methods are often misled by distractors and fail to locate multiple targets.

Our investigation attributes these localization failures to two main causes:
* **Contextual Dominance:** Salient distractors overwhelm target attention and cause inaccurate localization.
* **Semantic Bias:** In multi-object scenarios, global semantics cause the model to fixate on the most salient concept, resulting in incomplete localization.

To address these limitations, we propose **ActiveScope**, a training-free framework that enhances MLLMs by actively seeking and correcting perception. ActiveScope features two core modules:

1. **Semantic Anchor Localization (SAL):** This module utilizes fine-grained semantic anchors to independently localize key targets. This approach effectively mitigates semantic bias.
2. **Interference-Suppressed Refinement (ISR):** This module refines localization by suppressing attention on salient distractions. This mechanism successfully overcomes contextual dominance.

Extensive experiments on high-resolution image understanding benchmarks demonstrate the effectiveness of ActiveScope. 

