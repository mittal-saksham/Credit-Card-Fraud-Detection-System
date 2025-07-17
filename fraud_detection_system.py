
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """A comprehensive fraud detection system using machine learning algorithms
    with SMOTE for handling imbalanced datasets and multiple model options."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_performance = {}

    def load_data(self, filepath):
        """Load and prepare the credit card fraud dataset"""
        try:
            # Load the dataset
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully with {len(df)} transactions")
            print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

            # Separate features and target
            X = df.drop('Class', axis=1)
            y = df['Class']

            self.feature_names = X.columns.tolist()

            return X, y

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None

    def preprocess_data(self, X, y, use_smote=True):
        """Preprocess data with scaling and SMOTE for imbalanced dataset"""

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply SMOTE to handle class imbalance
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE - Training set size: {len(X_train_scaled)}")
            print(f"Class distribution: {np.bincount(y_train)}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train fraud detection model"""

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )

        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        print("Model training completed!")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Store performance metrics
        self.model_performance = {
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"Model Performance:")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return y_pred, y_pred_proba

    def plot_feature_importance(self, top_n=15):
        """Plot feature importance for Random Forest model"""

        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(top_n), 
                       x='importance', y='feature')
            plt.title('Top Feature Importances for Fraud Detection')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()

            return feature_importance
        else:
            print("Feature importance not available for this model type")
            return None

    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fraud Detection Model')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_fraud(self, transaction_data):
        """Predict fraud for new transactions"""

        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")

        # Ensure transaction_data is a DataFrame
        if isinstance(transaction_data, dict):
            transaction_data = pd.DataFrame([transaction_data])

        # Scale the features
        transaction_scaled = self.scaler.transform(transaction_data)

        # Make prediction
        fraud_probability = self.model.predict_proba(transaction_scaled)[:, 1]
        is_fraud = self.model.predict(transaction_scaled)

        results = []
        for i, (prob, pred) in enumerate(zip(fraud_probability, is_fraud)):
            risk_level = self.get_risk_level(prob)
            results.append({
                'transaction_id': i + 1,
                'fraud_probability': prob,
                'is_fraud': bool(pred),
                'risk_level': risk_level,
                'recommendation': self.get_recommendation(prob)
            })

        return results

    def get_risk_level(self, probability):
        """Determine risk level based on fraud probability"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"

    def get_recommendation(self, probability):
        """Get recommendation based on fraud probability"""
        if probability < 0.3:
            return "Approve transaction"
        elif probability < 0.7:
            return "Review transaction manually"
        else:
            return "Block transaction immediately"

    def save_model(self, filepath):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'performance': self.model_performance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_performance = model_data['performance']
        print(f"Model loaded from {filepath}")


def main():
    """Main function to demonstrate the fraud detection system"""

    # === 1. Initialize the system
    fraud_detector = FraudDetectionSystem()

    print("=== STEP 1: Loading Data ===")
    X, y = fraud_detector.load_data("creditcard_2023.csv")
    if X is None:
        print("Failed to load data.")
        return

    print("\n=== STEP 2: Preprocessing Data ===")
    X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(X, y)

    print("\n=== STEP 3: Training Model ===")
    model = fraud_detector.train_model(X_train, y_train, model_type='logistic_regression')

    print("\n=== STEP 4: Evaluating Model ===")
    y_pred, y_pred_proba = fraud_detector.evaluate_model(X_test, y_test)
    fraud_detector.plot_roc_curve(y_test, y_pred_proba)
    fraud_detector.plot_feature_importance()

    print("\n=== STEP 5: Saving Model ===")
    fraud_detector.save_model("fraud_rf_model.pkl")

    print("\n=== STEP 6: Predicting on a New Transaction ===")
    # Take one example transaction from the test set:
    example_txn = pd.DataFrame([X_test[0]], columns=fraud_detector.feature_names)
    pred_result = fraud_detector.predict_fraud(example_txn)
    print("Sample prediction result:", pred_result[0])

    print("\n*** All steps complete! ***")

if __name__ == "__main__":
    main()
