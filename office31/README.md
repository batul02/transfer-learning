### Project: Explainable Transfer Learning on Office-31 Dataset (ResNet Features)

This repository implements a pipeline for transfer learning using feature-based correspondences between domains in the Office-31 dataset. The process involves extracting image features via ResNet, computing feature importances, aligning domains with Ridge regression, and evaluating cross-domain classification performance using Random Forests.


### File Descriptions

| Filename                        | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `1_dataload.py`                | Loads the Office-31 dataset from domain folders (Amazon, DSLR). |
| `2_feat_extraction.py`         | Extracts features using a pre-trained ResNet model. Outputs `.npy` feature files. |
| `3_corresponding_matrix.py`    | Computes source-target feature importance matrices using Random Forests.    |
| `4_checking_matrix.py`         | Evaluates alignment (mean discrepancy, cosine similarity, KL-divergence).   |
| `5_target_baseline_rf.py`      | Trains and evaluates a Random Forest baseline model on target data only.    |
| `6_tl_source_to_target.py`     | Performs transfer learning: transforms source → target, combines with target train, trains new model. |
| `7_source_basline_rf.py`       | Trains a Random Forest baseline on source domain only (used for reverse transfer). |
| `8_tl_target_to_source.py`     | Transforms target → source and tests using the source-trained model.        |

---

### Completed Workflow

1. **Feature Extraction**: ResNet feature vectors computed for each image.
2. **Baseline Models**: RF models trained on source and target data independently.
3. **Correspondence Estimation**: Random Forest feature importances used to match informative dimensions across domains.
4. **Transformation Matrix**: Ridge regression fits matrix `P` to align domains.
5. **Transfer Learning**:
   - Source → Target: combine transformed source + target train, evaluate on target test.
   - Target → Source: transform target and test using source-trained model.
6. **Alignment Evaluation**: Matrix similarity evaluated using cosine, Pearson correlation, KL-divergence.



### Experiments Covered

#### Done:
1. **Feature Extraction**:
   - Features extracted using pre-trained **ResNet**.
   - Saved as `.npy` and `.npz` files for both domains.

2. **Base Model Training**:
   - Logistic Regression / Random Forest models trained separately on source and target.
   - Best hyperparameters logged.

3. **Correspondence Generation**:
   - Random Forest feature importances used to create source–target pairs.
   - Top correspondences saved.

4. **Transformation Matrix**:
   - Ridge regression fitted to find matrix `P` aligning source features to target.

5. **Transfer Learning**:
   - Transformed source combined with target train split.
   - Trained a classifier using best-found parameters.
   - Evaluated on target test split.

6. **Reverse Transfer**:
   - Transformed target data into source domain.
   - Tested source-trained model directly.

#### Yet To Be Done:
- SHAP-based correspondence generation.
- LIME-based correspondence evaluation.
- Formal comparison of TL vs base accuracy across domains.
- Hyperparameter tuning of RF and Ridge for further performance improvement
