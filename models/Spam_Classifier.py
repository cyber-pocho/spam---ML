from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import pandas as pd
import numpy as np
from models.TF_IDF_Feature_Extractor import Feature_Extractor  # Fixed import name

class BaseLineSpamClassifier: 
    """Base Line Spam Classifier"""
    def __init__(self):
        self.extractor = Feature_Extractor(max_features=1000, ngram_range=(1,2))
        self.model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        self.is_trained = False
        
    def prepare_data(self, df, test_size=0.3, random_state=42):  # Fixed method name
        """Here we'll split the data into training data and validation data sets"""
        labels = df['label'].map({'legitimate': 0, 'spam': 1})  # Fixed: removed comma

        # Splitting in ratio of 70/15/15
        X_train_df, X_temp_df, y_train, y_temp = train_test_split(
            df, labels, test_size=test_size, 
            stratify=labels, random_state=random_state
        )
        X_val_df, X_test_df, y_val, y_test = train_test_split(
            X_temp_df, y_temp, test_size=0.5, 
            stratify=y_temp, random_state=random_state
        )

        print(f"Data split is the following: ")
        print(f"Train: {len(X_train_df)}, Val: {len(X_val_df)}, Test: {len(X_test_df)} ")
        return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test
        
    def train(self, X_train_df, y_train): 
        """This function trains the model"""
        print("Extracting features and training model...")
        X_train = self.extractor.fit_transform(X_train_df)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training is complete.")
        
    def evaluate(self, X_val_df, y_val): 
        """Here we'll evaluate the model"""
        if not self.is_trained: 
            raise ValueError("train() must have failed, check for code. Model must be trained first.")
        
        X_val = self.extractor.transform(X_val_df)
        y_pred = self.model.predict(X_val)
        y_pred_probability = self.model.predict_proba(X_val)[:, 1]

        # Useful metrics
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        # False-Positive rate
        fpr = cm[0,1] / (cm[0,0] + cm[0,1])  # Fixed spacing

        # Printing out the results
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)
        print(f"Precision in catching the correct spam: {precision:.3f}")  # Fixed spacing and spelling
        print(f"Rate of which the spam was recalled: {recall:.3f}")
        print(f"False-Positive rate: {fpr:.3f} (Legit -> Spam)")
        print(f"CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"              Legit  Spam")
        print(f"Actual Legit    {cm[0,0]:3d}   {cm[0,1]:3d}")  # Fixed formatting
        print(f"       Spam     {cm[1,0]:3d}   {cm[1,1]:3d}")  # Fixed formatting

        self._test_thresholds(y_val, y_pred_probability)

        return {
            'precision': precision,
            'recall': recall, 
            'false_positive_rate': fpr, 
            'confusion_matrix': cm  # Fixed spelling
        }
    
    def _test_thresholds(self, y_true, y_pred_probability): 
        """Here we are testing different confidence thresholds"""
        print(f"\nConfidence Thresholds Analysis...")
        print(f"Threshold  Precision  Recall  Spam Predicted")
        print("-" * 45)

        for threshold in [0.5, 0.7, 0.8, 0.9]: 
            y_pred_thresh = (y_pred_probability >= threshold).astype(int)
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)  # Fixed: was PermissionError
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
            spam_count = y_pred_thresh.sum()
            print(f"{threshold:7.1f}     {prec:7.3f}    {rec:5.3f}    {spam_count:4d}")


def main():  # Fixed: moved outside class
    """Execution"""
    # Loading data
    df = pd.read_json('C:/Users/Famil/Desktop/n1/data/main_source.json')
    print(f"Loaded {len(df):,} emails")
    print(f"Distribution: {df['label'].value_counts().to_dict()}")

    classifier = BaseLineSpamClassifier()

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = classifier.prepare_data(df)
    
    # Trains the model
    classifier.train(X_train_df, y_train)
    results = classifier.evaluate(X_val_df, y_val)

    # Success criteria
    print("\n" + "="*50)
    print("SUCCESS CRITERIA CHECK")
    print("="*50)
    
    target_precision = 0.85  # acceptable goal for first time
    target_fpr = 0.15  # we expect a 0.15 score for False Positive recalls
    
    precision_pass = results['precision'] >= target_precision  # Fixed spelling
    fpr_pass = results['false_positive_rate'] <= target_fpr

    print(f"Target Precision ≥ {target_precision}: {'Pass' if precision_pass else 'x'} ({results['precision']:.3f})")
    print(f"Target FPR ≤ {target_fpr}: {'pass' if fpr_pass else 'x'} ({results['false_positive_rate']:.3f})")
    
    if precision_pass and fpr_pass: 
        print(" Baseline model meets Day 2 targets!")
    else: 
        print(" Model needs improvement for Day 3")


if __name__ == "__main__":
    main()