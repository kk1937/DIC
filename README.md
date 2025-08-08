# DIC Machine Learning Pipeline

This repository contains a comprehensive machine learning pipeline for predicting and evaluating Disseminated Intravascular Coagulation (DIC) using clinical data. The workflow includes data preprocessing, feature selection, sampling, model training, evaluation, and visualization.

## Features
- Data cleaning and preprocessing
- Feature selection (KS test, correlation, LASSO)
- Over/under sampling (SMOTE, RandomUnderSampler)
- Standardization and outlier removal (Z-score, PCA)
- Model training (Random Forest, KNN, Gradient Boosting, XGBoost)
- Model evaluation (cross-validation, bootstrapping, ROC, PR, MCC, calibration, SHAP)
- Visualization of results

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- statsmodels
- matplotlib
- shap
- xgboost

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your data files in the `./DIC/Data/` directory as required by the scripts.
2. Run the main pipeline:
```bash
python DIC.py
```
3. Output files and visualizations will be saved in `./DIC/Data/output/`.

## File Structure
- `DIC.py`: Main pipeline script
- `data_loader.py`: Data loading utilities
- `feature_engineering.py`: Feature engineering functions
- `model_selection.py`: Model selection and training
- `models.py`: Model definitions
- `evaluation.py`: Model evaluation metrics and visualization
- `utils.py`: Utility functions
- `main.py`: Entry point (if used)

## Output
- CSV files with results and metrics
- Plots for ROC, Precision-Recall, MCC, Calibration, PCA, SHAP

## Notes
- Some data files and outputs are required for full functionality. Please refer to the code for expected file names and formats.
- The pipeline is modular and can be adapted for other clinical prediction tasks.

## License
MIT License

## Contact
For questions or collaboration, please open an issue or contact the repository owner.
