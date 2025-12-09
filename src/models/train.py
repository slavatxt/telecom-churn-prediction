import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.preprocessing import DataPreprocessor, load_data, prepare_features


def train_logistic_regression(X_train, y_train, X_val, y_val):
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'Logistic Regression',
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
    
    return model, metrics


def train_decision_tree(X_train, y_train, X_val, y_val):
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'Decision Tree',
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'Random Forest',
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
    
    return model, metrics


def print_metrics(metrics):
    print(f"\n{'='*50}")
    print(f"Model: {metrics['model_name']}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Train churn prediction models')
    parser.add_argument('--train', type=str, default='data/raw/train.csv',
                       help='Path to training data')
    parser.add_argument('--test', type=str, default='data/raw/test.csv',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['logreg', 'tree', 'rf'],
                       help='Models to train: logreg, tree, rf')
    
    args = parser.parse_args()
    
    print("🚀 Starting model training pipeline...")
    print(f"📂 Train data: {args.train}")
    print(f"📂 Test data: {args.test}")
    
    train_df, test_df = load_data(args.train, args.test)
    print(f"✅ Data loaded: Train shape {train_df.shape}, Test shape {test_df.shape if test_df is not None else 'None'}")
    
    preprocessor = DataPreprocessor()
    train_processed, test_processed = preprocessor.fit_transform(train_df, test_df, scale=False)
    print("✅ Data preprocessed")
    
    X, y = prepare_features(train_processed, 'Churn')
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data split: Train {X_train.shape}, Val {X_val.shape}")
    
    models = {}
    all_metrics = []
    
    if 'logreg' in args.models:
        print("\n📊 Training Logistic Regression...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        logreg, logreg_metrics = train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val)
        print_metrics(logreg_metrics)
        models['logreg'] = {'model': logreg, 'scaler': scaler, 'preprocessor': preprocessor}
        all_metrics.append(logreg_metrics)
    
    if 'tree' in args.models:
        print("\n📊 Training Decision Tree...")
        tree, tree_metrics = train_decision_tree(X_train, y_train, X_val, y_val)
        print_metrics(tree_metrics)
        models['tree'] = {'model': tree, 'preprocessor': preprocessor}
        all_metrics.append(tree_metrics)
    
    if 'rf' in args.models:
        print("\n📊 Training Random Forest...")
        rf, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
        print_metrics(rf_metrics)
        models['rf'] = {'model': rf, 'preprocessor': preprocessor}
        all_metrics.append(rf_metrics)
    
    best_model_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['roc_auc'])
    best_model_name = all_metrics[best_model_idx]['model_name']
    best_roc_auc = all_metrics[best_model_idx]['roc_auc']
    
    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model_data in models.items():
        model_path = output_dir / f'{name}_model.pkl'
        joblib.dump(model_data, model_path)
        print(f"💾 Saved {name} model to {model_path}")
    
    best_key = list(models.keys())[best_model_idx]
    best_model_path = output_dir / 'best_model.pkl'
    joblib.dump(models[best_key], best_model_path)
    print(f"💾 Saved best model to {best_model_path}")
    
    print("\n✨ Training completed successfully!")


if __name__ == '__main__':
    main()
