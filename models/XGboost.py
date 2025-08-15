import models.XGboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class SpamRescueValidator_XGboos: 
    """Alternative classifier using XG boost. Precision focus remains"""

    def __init__(self, feature_extractor = None):
        self.feature_extractor = feature_extractor
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.is_trained = False
        pass
    def get_param_grid(self):
        """Parameters optimized for precision in Spam rescue"""
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate':[0.01, 0.1, 0.2],
            'subsample':[0.9, 0.9, 1.0],
            'colsample_bytree':[0.8, 0.9, 1.0],
            'scale_pos_weight':[1, 2, 3]
        }
    def train_fine_tuning(self, X_train_df, y_train, cv=3, scoring = 'precision'): 
        """train with h-parameter tunning focusin on precision"""
        X_train = self.feature_extractor.fit_transform(X_train_df)
        xgb_model = xgb.XGBClassifier(
            objective = 'binary:logistic',
            eval_metric = 'log_loss', 
            random_state = 42, 
            n_jobs = -1, 
            verbosity = 0 # so we don't get any XGboost warnings
        )
        
        grid_search = GridSearchCV(
            xgb_model, 
            self.get_param_grid(), 
            cv = cv, 
            scoring = scoring, 
            n_jobs = -1, 
            verbose = 0
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self. feature_extractor = self.model.feature_importances_

        return {
            'best_score': grid_search.best_score_,
            'best_params': self.best_params
        }
    def train_fast(self, X_train_df, y_train):
        """Quick training with optimized params for spam rescue"""
        X_train = self.feature_extractor.fit_transform(X_train_df)

        #pretunned parameters for spam rescue
        self.model = xgb.XGBClassifier(
            n_estimators = 200, 
            max_depth=6,
            learning_rate=0.1, 
            subsample=0.9,
            colsample_bytree=0.9, 
            scale_pos_weight=2, # for imbalance
            objective='binary:logistic', 
            eval_metric='logloss', 
            random_state=42, 
            n_jobs=-1, 
            verbositiy=0,  
        )

        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
        self.is_trained = True #changing status of training
    def predict_confidence(self, X_df, threshold=0.8): 
        """Predict with confidence theshold"""
        if not self.is_trained:
            raise ValueError("Model is not trained, train first.")
        X = self.feature_extractor.transform(X_df)
        probabilities = self.model.predict_proba(X)[:, 1] # Probabilities of legitimates

        high_confidence_mask = probabilities >= threshold

        return {
            'probabilities': probabilities, 
            'high_confidence_legitimate': high_confidence_mask, 
            'rescue_candidates': X_df[high_confidence_mask] if len(X_df[high_confidence_mask]) > 0 else pd.DataFrame()
        }
    def get_top_features(self, top_n = 15): 
        """ get top features for interpretability"""
        if not self.is_trained or self.feature_importance is None: 
            return []
        try: 
            feature_names = self.feature_extractor.get_feature_names()
        except: 
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        return importance_df.head(top_n).to_dict('records')
    def save_model(self, filepath=None):
        """Save model"""
        if not self.is_trained: 
            raise ValueError("Model is not trained. Train first.")

        if filepath is None: 
            timestamp = datetime.now().strftime("Y%m%d_%H%M%S")
            filepath = f"spam_rescue_xgb_{timestamp}.joblib"
        joblib.dump({
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'best_params': self.best_params
        }, filepath)
        
        return filepath    


