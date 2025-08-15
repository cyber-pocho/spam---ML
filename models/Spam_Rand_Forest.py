from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, precision_score, recall_score
import pandas as pd
import numpy as np
from TF_IDF_Feature_Extractor import Feature_Extractor
import joblib
from datetime import datetime

class SpamRescueRandomForest:
    """RandomForest classifier for spam rescue with high precision focus"""
    
    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor
        self.model = None
        self.best_params = None
        self.feature_importance_ = None 
        self.is_trained = False
    
    def get_param_grid(self):  
        """Parameter grid optimized for precision over recall"""
        return {
            'n_estimators': [100, 200, 300], 
            'max_depth': [10, 15, 20, None],  
            'min_samples_split': [2, 5, 10], 
            'min_samples_leaf': [1, 2, 4], 
            'class_weight': ['balanced', 'balanced_subsample']
        }
    
    def train_with_tuning(self, X_train_df, y_train, cv=3, scoring='precision'):  
        """Training with hyperparameter tuning that focuses on precision"""
        X_train = self.feature_extractor.fit_transform(X_train_df)

        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, 
            self.get_param_grid(),  
            cv=cv, 
            scoring=scoring, 
            n_jobs=-1, 
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_importance_ = self.model.feature_importances_
        self.is_trained = True
        
        return {
            'best_score': grid_search.best_score_,
            'best_params': self.best_params
        }
    
    def train_fast(self, X_train_df, y_train): 
        """Quick training without hyperparameter tuning"""
        X_train = self.feature_extractor.fit_transform(X_train_df)

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        self.feature_importance_ = self.model.feature_importances_
        self.is_trained = True
    
    def predict_with_confidence(self, X_df, threshold=0.8):  
        """Predict with confidence threshold for spam rescue"""
        if not self.is_trained: 
            raise ValueError('Model must be trained first')
            
        X = self.feature_extractor.transform(X_df)
        probabilities = self.model.predict_proba(X)[:, 1]  

        high_confidence_mask = probabilities >= threshold  
        
        return {
            'probabilities': probabilities, 
            'high_confidence_legitimate': high_confidence_mask, 
            'rescue_candidates': X_df[high_confidence_mask] if len(X_df[high_confidence_mask]) > 0 else pd.DataFrame()
        }
    
    def evaluate_precision_focus(self, X_val_df, y_val):  
        """Evaluation focused on precision metrics for spam rescue"""
        if not self.is_trained: 
            raise ValueError("Model must be trained first")
            
        X_val = self.feature_extractor.transform(X_val_df)
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]

        base_metrics = {
            'precision': precision_score(y_val, y_pred), 
            'recall': recall_score(y_val, y_pred)
        }

        threshold_results = []
        for threshold in [0.7, 0.8, 0.85, 0.9, 0.95]: 
            y_pred_thresh = (y_proba >= threshold).astype(int)
            if y_pred_thresh.sum() > 0: 
                prec = precision_score(y_val, y_pred_thresh, zero_division=0)
                rec = recall_score(y_val, y_pred_thresh, zero_division=0)
                threshold_results.append({
                    'threshold': threshold, 
                    'precision': prec, 
                    'recall': rec, 
                    'rescued_count': y_pred_thresh.sum()
                })
        
        return {
            'base_metrics': base_metrics,
            'threshold_analysis': threshold_results,
            'feature_importance': self.get_top_features()
        }
    
    def get_top_features(self, top_n=15): 
        """Get top feature importance for interpretability"""
        if not self.is_trained or self.feature_importance_ is None: 
            return []
            
        try: 
            feature_names = self.feature_extractor.get_feature_names_out()
        except: 
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]
        
        importance_df = pd.DataFrame({  
            'feature': feature_names, 
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n).to_dict('records')
    
    def save_model(self, filepath=None):
        """Save trained model"""
        if not self.is_trained: 
            raise ValueError('No trained model to save')
        
        if filepath is None: 
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"spam_rescue_rf_{timestamp}.joblib"

        joblib.dump({
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'best_params': self.best_params
        }, filepath)

        return filepath
    
    def load_model(self, filepath): 
        """Load trained model"""
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.feature_extractor = loaded['feature_extractor']
        self.best_params = loaded.get('best_params')
        self.is_trained = True


def main():
    """Usage and execution function"""
    from TF_IDF_Feature_Extractor import Feature_Extractor
    from Spam_Classifier import BaseLineSpamClassifier  

    df = pd.read_json(r'C:\Users\Famil\Desktop\n1\utils\main_source.json')
    print(f"Loaded {len(df)} emails")

    # Initialize
    extractor = Feature_Extractor(max_features=1000, ngram_range=(1,2))
    rf_classifier = SpamRescueRandomForest(extractor)
    
    
    baseline = BaseLineSpamClassifier()
    baseline.train(X_train_df, y_train)
    baseline_results = baseline.evaluate(X_val_df, y_val) #train baseline for comparison
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = baseline.prepare_data(df)
    
    print("Training RandomForest...")
    rf_classifier.train_fast(X_train_df, y_train)
    
    print("Evaluating...")
    results = rf_classifier.evaluate_precision_focus(X_val_df, y_val)
    
    print(f"Base Precision: {results['base_metrics']['precision']:.3f}")
    print(f"Base Recall: {results['base_metrics']['recall']:.3f}")

    print(f"Baseline vs RandomForest:")
    print(f"Baseline Precision: {baseline_results['precision']:.3f}")
    print(f"RandomForest Precision: {results['base_metrics']['precision']:.3f}")
    
    print("\nThreshold Analysis:")
    for thresh_result in results['threshold_analysis']:
        print(f"Threshold {thresh_result['threshold']}: "
              f"Precision={thresh_result['precision']:.3f}, "
              f"Rescued={thresh_result['rescued_count']}")


if __name__ == "__main__": 
    main()
