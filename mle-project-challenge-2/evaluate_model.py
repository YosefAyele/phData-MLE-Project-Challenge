import json
import pathlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
from sklearn import model_selection, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from create_model import load_data, SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def evaluate_model_performance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:

    y_pred = model.predict(X_test)
    
    metrics_dict = {
        'mae': metrics.mean_absolute_error(y_test, y_pred),
        'mse': metrics.mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        'r2': metrics.r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        'median_ae': metrics.median_absolute_error(y_test, y_pred)
    }
    
    return metrics_dict, y_pred

def cross_validation_analysis(X: pd.DataFrame, y: pd.Series, model: Pipeline, cv_folds: int = 5) -> Dict[str, Any]:

    cv_scores = {}
    
    # Different scoring metrics
    scoring_metrics = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    
    for metric in scoring_metrics:
        scores = model_selection.cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
        cv_scores[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    
    return cv_scores

def analyze_residuals(y_true: pd.Series, y_pred: np.ndarray, save_plots: bool = True):
    residuals = y_true - y_pred
    
    # Create plots directory
    plots_dir = pathlib.Path("model_evaluation_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Residuals vs Predicted
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    
    # 2. Q-Q plot of residuals
    plt.subplot(2, 2, 2)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    # 3. Histogram of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    
    # 4. Predicted vs Actual
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'red', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(plots_dir / "residual_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Residual analysis plots saved to {plots_dir / 'residual_analysis.png'}")
    
    plt.show()

def feature_importance_analysis(model: Pipeline, feature_names: List[str], X_test: pd.DataFrame, y_test: pd.Series):
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)  # Top 15 features
    
    plt.barh(range(len(top_features)), top_features['importance_mean'], 
             xerr=top_features['importance_std'], alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Top 15 Feature Importance (Permutation-based)')
    plt.gca().invert_yaxis()
    
    plots_dir = pathlib.Path("model_evaluation_plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {plots_dir / 'feature_importance.png'}")
    plt.show()
    
    return importance_df

def learning_curve_analysis(X: pd.DataFrame, y: pd.Series, model: Pipeline):

    train_sizes, train_scores, val_scores = model_selection.learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_absolute_error'
    )
    
    # Convert negative MAE to positive
    train_scores = -train_scores
    val_scores = -val_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Absolute Error')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plots_dir = pathlib.Path("model_evaluation_plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
    print(f"Learning curves plot saved to {plots_dir / 'learning_curves.png'}")
    plt.show()

def analyze_prediction_errors(y_true: pd.Series, y_pred: np.ndarray, percentiles: List[int] = [25, 50, 75, 90, 95]):

    errors = np.abs(y_true - y_pred)
    relative_errors = errors / y_true * 100
    
    # Create price bins
    price_percentiles = np.percentile(y_true, percentiles)
    
    print("\n=== Prediction Error Analysis ===")
    print(f"Overall MAE: ${errors.mean():,.0f}")
    print(f"Overall MAPE: {relative_errors.mean():.2f}%")
    print(f"Median AE: ${np.median(errors):,.0f}")
    
    print(f"\nError distribution by price percentiles:")
    for i, p in enumerate(percentiles):
        if i == 0:
            mask = y_true <= price_percentiles[i]
            label = f"Bottom {p}%"
        else:
            mask = (y_true > price_percentiles[i-1]) & (y_true <= price_percentiles[i])
            label = f"{percentiles[i-1]}-{p}%"
        
        if mask.sum() > 0:
            mae_segment = errors[mask].mean()
            mape_segment = relative_errors[mask].mean()
            print(f"  {label}: MAE=${mae_segment:,.0f}, MAPE={mape_segment:.2f}%")

def main():
    """
    Main function to run comprehensive model evaluation.
    """
    print("=== Comprehensive Model Performance Evaluation ===\n")
    
    # Load data
    print("Loading data...")
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Target mean: ${y.mean():,.0f}")
    
    # Load the trained model
    model_path = pathlib.Path("model/model.pkl")
    if not model_path.exists():
        print("Model not found! Please run create_model.py first.")
        return
    
    model = pickle.load(open(model_path, 'rb'))
    
    # Split data (same split as training for consistency)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=42, test_size=0.25
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 1. Basic performance evaluation
    print("\n1. Evaluating model performance on test set...")
    test_metrics, y_pred = evaluate_model_performance(model, X_test, y_test)
    
    print("Test Set Performance:")
    for metric, value in test_metrics.items():
        if metric in ['mae', 'mse', 'rmse', 'median_ae']:
            print(f"  {metric.upper()}: ${value:,.0f}")
        else:
            print(f"  {metric.upper()}: {value:.4f}")
    
    # 2. Cross-validation analysis
    print("\n2. Performing cross-validation analysis...")
    cv_results = cross_validation_analysis(X, y, model)
    
    print("Cross-Validation Results (5-fold):")
    for metric, results in cv_results.items():
        metric_name = metric.replace('neg_', '').replace('_', ' ').upper()
        mean_val = -results['mean'] if 'neg_' in metric else results['mean']
        std_val = results['std']
        if 'error' in metric:
            print(f"  {metric_name}: ${mean_val:,.0f} (±${std_val:,.0f})")
        else:
            print(f"  {metric_name}: {mean_val:.4f} (±{std_val:.4f})")
    
    # 3. Residual analysis
    print("\n3. Analyzing residuals...")
    analyze_residuals(y_test, y_pred)
    
    # 4. Feature importance analysis
    print("\n4. Analyzing feature importance...")
    feature_names = json.load(open("model/model_features.json", 'r'))
    importance_df = feature_importance_analysis(model, feature_names, X_test, y_test)
    
    print("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<25} {row['importance_mean']:.6f}")
    
    # 5. Learning curve analysis
    print("\n5. Generating learning curves...")
    learning_curve_analysis(X, y, model)
    
    # 6. Error analysis by price range
    print("\n6. Analyzing prediction errors by price range...")
    analyze_prediction_errors(y_test, y_pred)
    
    # 7. Save comprehensive evaluation report
    evaluation_report = {
        'dataset_info': {
            'total_samples': len(X),
            'features': len(feature_names),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'target_min': float(y.min()),
            'target_max': float(y.max()),
            'target_mean': float(y.mean()),
            'target_std': float(y.std())
        },
        'test_performance': test_metrics,
        'cross_validation': cv_results,
        'feature_importance': importance_df.head(20).to_dict('records'),
        'model_type': str(type(model.named_steps['kneighborsregressor']).__name__),
        'model_params': model.named_steps['kneighborsregressor'].get_params()
    }
    
    report_path = pathlib.Path("model_evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2, default=str)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Comprehensive evaluation report saved to: {report_path}")
    print("Plots saved to: model_evaluation_plots/")
    
    # Model assessment summary
    print(f"\n=== Model Assessment Summary ===")
    r2_score = test_metrics['r2']
    mape_score = test_metrics['mape']
    
    if r2_score > 0.8:
        r2_assessment = "Excellent"
    elif r2_score > 0.6:
        r2_assessment = "Good"
    elif r2_score > 0.4:
        r2_assessment = "Fair"
    else:
        r2_assessment = "Poor"
    
    if mape_score < 10:
        mape_assessment = "Excellent"
    elif mape_score < 20:
        mape_assessment = "Good"
    elif mape_score < 30:
        mape_assessment = "Fair"
    else:
        mape_assessment = "Poor"
    
    print(f"R² Score: {r2_score:.4f} ({r2_assessment})")
    print(f"MAPE: {mape_score:.2f}% ({mape_assessment})")
    print(f"The model explains {r2_score*100:.1f}% of the variance in house prices.")
    
    if r2_score < 0.7 or mape_score > 15:
        print("\n⚠️  Model Performance Recommendations:")
        print("  - Consider feature engineering (polynomial features, interactions)")
        print("  - Try different algorithms (Random Forest, Gradient Boosting)")
        print("  - Collect more relevant features")
        print("  - Address outliers in the dataset")
        print("  - Consider geographical features (distance to amenities)")

if __name__ == "__main__":
    main()
