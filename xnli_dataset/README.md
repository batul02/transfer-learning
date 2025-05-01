###  Folder Structure

```
.
├── lime/                                
│   ├── tl_using_lime_rf.py              # Transfer learning using LIME + Random Forest
│   └── tl_using_lime_lr.py              # Transfer learning using LIME + Logistic Regression
│
├── shap/                                
│   ├── tl_using_shap_rf.py              # Transfer learning using SHAP + Random Forest
│   └── tl_using_shap_lr.py              # Transfer learning using SHAP + Logistic Regression
│
├── 1_tl_new_dataset.py                  # Load and encode English + French data
├── 2_tl_new_dataset_train.py            # Train base models on source and target
├── 3_ridge_P_matrix.py                  # Learn transformation matrix via Ridge regression
├── 4_test_tl.py                         # Evaluate transfer model on target test set
├── rf_model_target.py                   # Train Random Forest on target domain
├── tl_reverse.py                        # Transform target → source and test source model
```

---

###  Workflow Summary

1. **Data Preparation (`1_tl_new_dataset.py`):**
   - Load English and French text (e.g., XNLI)
   - Encode premise-hypothesis pairs using SentenceTransformer
   - Save to `encoded_data.npz`

2. **Base Model Training (`2_tl_new_dataset_train.py`):**
   - Train separate classifiers on source and target
   - Save base models like `clf_fr_base.pkl` and `lr_source_model.pkl`

3. **Correspondence Matching:**
   -  `shap/`: Use SHAP explanations for matching source-target samples
   -  `lime/`: Use LIME weights for matching
   - Match via cosine similarity; apply threshold to select best pairs

4. **Transformation Learning (`3_ridge_P_matrix.py`):**
   - Use Ridge regression to learn mapping between source → target features
   - Save transformation matrix `P`

5. **Transfer Learning (`4_test_tl.py`):**
   - Apply transformation
   - Combine transformed source + target train
   - Train new model and evaluate on target test

6. **Reverse Transfer (`tl_reverse.py`):**
   - Transform target data into source space
   - Use source-trained model directly on transformed target

7. **Random Forest Training (`rf_model_target.py`):**
   - Train RF model only on target data (used in correspondence-based methods)
