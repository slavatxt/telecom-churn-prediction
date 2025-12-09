import argparse
import joblib
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_model(model_path: str):
    model_data = joblib.load(model_path)
    return model_data


def make_predictions(model_data, test_df: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model_data['preprocessor']
    model = model_data['model']
    
    test_processed = preprocessor.transform(test_df, scale=False)
    
    if 'scaler' in model_data:
        scaler = model_data['scaler']
        test_processed = scaler.transform(test_processed)
        predictions = model.predict(test_processed)
        predictions_proba = model.predict_proba(test_processed)[:, 1]
    else:
        predictions = model.predict(test_processed)
        predictions_proba = model.predict_proba(test_processed)[:, 1]
    
    results = pd.DataFrame({
        'Id': range(len(predictions)),
        'Churn': predictions,
        'Churn_Probability': predictions_proba
    })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, default='models/best_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/raw/test.csv',
                       help='Path to test data')
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                       help='Path to save predictions')
    parser.add_argument('--with-proba', action='store_true',
                       help='Include probability scores in output')
    
    args = parser.parse_args()
    
    print("🔮 Starting prediction pipeline...")
    print(f"📂 Model: {args.model}")
    print(f"📂 Test data: {args.data}")
    
    model_data = load_model(args.model)
    print("✅ Model loaded")
    
    test_df = pd.read_csv(args.data)
    print(f"✅ Test data loaded: {test_df.shape}")
    
    predictions = make_predictions(model_data, test_df)
    print(f"✅ Predictions made")
    
    if not args.with_proba:
        predictions = predictions[['Id', 'Churn']]
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to {output_path}")
    
    print(f"\n📊 Prediction summary:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted churn: {predictions['Churn'].sum()} ({predictions['Churn'].mean()*100:.2f}%)")
    print(f"Predicted no churn: {(predictions['Churn']==0).sum()} ({(predictions['Churn']==0).mean()*100:.2f}%)")
    
    if args.with_proba:
        print(f"\nProbability statistics:")
        print(f"Mean probability: {predictions['Churn_Probability'].mean():.4f}")
        print(f"Median probability: {predictions['Churn_Probability'].median():.4f}")
        print(f"Max probability: {predictions['Churn_Probability'].max():.4f}")
        print(f"Min probability: {predictions['Churn_Probability'].min():.4f}")
    
    print("\n✨ Prediction completed successfully!")


if __name__ == '__main__':
    main()
