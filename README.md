# TRANSFORM-X

**Transfer via Feature Correspondence and Explanations**

This project proposes an interpretable framework for transfer learning across heterogeneous domains (text, vision, and speech) using feature correspondences derived from model explanation techniques like **Random Forest feature importance**, **SHAP**, and **LIME**.

## Overview

Traditional transfer learning assumes aligned feature spaces, which fails in cross-modal or structurally distinct domains. TRANSFORM-X addresses this by:

* Computing attribution-based feature correspondences between source and target domains
* Learning a linear projection via Ridge Regression
* Merging transformed source data with target data for final classification

The framework is modular, lightweight, and model-agnostic.

## Datasets Used

| Dataset   | Modality | Source → Target         | Feature Dim. |
| --------- | -------- | ----------------------- | ------------ |
| XNLI      | Text     | English → French        | 1536         |
| Office-31 | Vision   | Amazon → DSLR           | 2048         |
| CREMA-D   | Audio    | Male → Female (emotion) | \~558        |

## Attribution Methods

* **Random Forest Importance**: Stable global ranking
* **SHAP**: Global, interaction-aware attributions
* **LIME**: Local, instance-level explanations

## Usage

1. **Feature Extraction**: Preprocess domain data and extract embeddings.
2. **Attribution Scoring**: Use RF, SHAP, and LIME to compute feature relevance.
3. **Correspondence Matrix**: Compute cosine similarity between source and target attribution vectors.
4. **Projection Learning**: Train a projection matrix via Ridge Regression.
5. **Classification**: Train a model on combined projected source and raw target data.

## Results

Across all datasets, the framework shows accuracy improvements over baseline transfer and target-only models, particularly in low-resource settings.

## Requirements

* Python 3.8+
* `scikit-learn`, `shap`, `lime`, `librosa`, `torch`, `transformers`

