# CORRECTED VERSION - Replace your Spam_Rescue_Validator.py with this

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (confusion_matrix, classification_report, 
                           precision_recall_curve, roc_curve, auc,
                           precision_score, recall_score, f1_score)
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SpamRescueValidator:
    """Comprehensive validation framework for spam rescue models with precision focus"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results_history = []

    def k_fold_validation(self, model, feature_extractor, X_df, y, cv_folds=5):  # Fixed: method name
        """Perform stratified K-fold cross-validation to ensure balanced classes"""
        
        # Stratified k-fold to maintain spam/legitimate ratio in each fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'precision': 'precision', 
            'recall': 'recall',
            'f1': 'f1',
            'accuracy': 'accuracy'
        }
        
        X_features = feature_extractor.transform(X_df)
        cv_results = cross_validate(
            model, X_features, y, 
            cv=skf, 
            scoring=scoring, 
            return_train_score=True,
            n_jobs=-1
        )
        
        cv_summary = {}
        for metric in scoring.keys(): 
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']

            cv_summary[metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'test_scores': test_scores,
                'overfitting_gap': np.mean(train_scores) - np.mean(test_scores)
            }
        return cv_summary
    
    def confusion_matrix_analysis(self, y_true, y_pred, model_name='Model'):  # Fixed: method name
        """Create detailed confusion matrix analysis with spam rescue focus"""
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'true_negatives': int(tn),   # Correctly identified spam
            'false_positives': int(fp),  # LEGIT EMAILS misclassified as spam -- CRITICAL
            'false_negatives': int(fn),  # Spam marked as legitimate -- CRITICAL
            'true_positives': int(tp),   # Correctly identified legitimate emails 
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0, 
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0, 
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0, 
            'false_positive_rate': fp / (tn + fp) if (tn + fp) > 0 else 0, 
            'rescue_opportunity': int(fp)  # Number of legit emails that could be rescued
        } 

        return {
            'confusion_matrix': cm, 
            'metrics': metrics, 
            'model_name': model_name
        }
    
    def threshold_optimization(self, model, feature_extractor, X_df, y_true, 
                               thresholds=None, target_precision=0.95): 
        """Optimize threshold focusing on precision, sacrificing recall if necessary"""
        
        if thresholds is None:
            thresholds = np.arange(0.5, 0.99, 0.05)
        
        X_features = feature_extractor.transform(X_df)
        y_proba = model.predict_proba(X_features)[:, 1]  # Probabilities for legitimate class

        threshold_results = []

        for threshold in thresholds: 
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            # CRITICAL FIX: Skip if NO positive predictions (was inverted logic)
            if y_pred_thresh.sum() == 0: 
                continue
                
            precision = precision_score(y_true, y_pred_thresh, zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)

            rescued_count = y_pred_thresh.sum()

            threshold_results.append({
                'threshold': threshold, 
                'precision': precision, 
                'recall': recall, 
                'f1_score': f1, 
                'rescued_count': rescued_count, 
                'meets_target': precision >= target_precision
            })
            
        # Find best threshold that meets precision target
        valid_thresholds = [r for r in threshold_results if r['meets_target']]
        recommended_threshold = None

        if valid_thresholds:
            # Choose threshold with highest recall among those meeting precision target
            recommended_threshold = max(valid_thresholds, key=lambda x: x['recall'])
            
        return {
            'threshold_analysis': threshold_results,
            'recommended_threshold': recommended_threshold,
            'target_precision': target_precision
        }
    
    def false_positive_analysis(self, model, feature_extractor, X_df, y_true, 
                                threshold=0.8, top_n=10): 
        """Analyze legitimate emails being missed - critical for spam rescue"""
        
        X_features = feature_extractor.transform(X_df)
        y_proba = model.predict_proba(X_features)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Find false positives (legitimate emails classified as spam)
        fp_mask = (y_true == 1) & (y_pred == 0)
        fp_indices = np.where(fp_mask)[0]
        
        if len(fp_indices) == 0:
            return {
                'false_positive_count': 0, 
                'false_positive_rate': 0.0, 
                'sample_false_positives': []
            }
            
        # Get confidence scores for false positives
        fp_probabilities = y_proba[fp_indices]
        fp_emails = X_df.iloc[fp_indices].copy()
        fp_emails['confidence_score'] = fp_probabilities
        
        # Sort by confidence (lowest first - "worst" classifications)
        fp_emails_sorted = fp_emails.sort_values('confidence_score')
        
        sample_fps = []
        for idx, row in fp_emails_sorted.head(top_n).iterrows(): 
            sample_fps.append({
                'subject': row.get('subject', 'N/A')[:100], 
                'body_preview': row.get('body', 'N/A')[:200],
                'confidence_score': row['confidence_score'],
                'sender': row.get('sender', 'N/A') 
            })
            
        fp_rate = len(fp_indices) / len(y_true[y_true == 1])
        
        return {
            'false_positive_count': len(fp_indices),
            'false_positive_rate': fp_rate,
            'sample_false_positives': sample_fps
        }
    
    def compare_models(self, model_results: Dict[str, Dict]): 
        """Compare multiple models side by side"""
        
        comparison = []
        for model_name, results in model_results.items():  # Fixed: .items() not .item()
            # Extract key metrics for comparison
            metrics = results.get('confusion_matrix_analysis', {}).get('metrics', {})  # Fixed: key name
            comparison.append({
                'model': model_name, 
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'false_positive_rate': metrics.get('false_positive_rate', 0), 
                'rescue_opportunity': metrics.get('rescue_opportunity', 0)  # Fixed: spelling
            })

        # Sort by precision first (most important for spam rescue)
        comparison.sort(key=lambda x: x['precision'], reverse=True)
        
        # Generate recommendations
        best_precision = comparison[0] if comparison else None
        best_recall = max(comparison, key=lambda x: x['recall']) if comparison else None

        recommendations = {
            'best_for_precision': best_precision,
            'best_for_recall': best_recall,
            'recommendation': None
        }

        if best_precision and best_precision['precision'] >= 0.95: 
            recommendations['recommendation'] = f"Use {best_precision['model']} - meets precision target"
        elif comparison: 
            recommendations['recommendation'] = f"Consider {comparison[0]['model']} - highest precision but needs improvement"  # Fixed: spacing
        
        return {
            'comparison_table': comparison,
            'recommendations': recommendations
        }
    
    def generate_validation_report(self, model, feature_extractor, X_df, y, 
                                 model_name="Model", save_path=None):
        """Generate comprehensive validation report"""
        
        print(f"Generating validation report for {model_name}...")
        
        # Perform all validation analyses
        cv_results = self.k_fold_validation(model, feature_extractor, X_df, y)  # Fixed: method name
        
        # Get predictions for other analyses
        X_features = feature_extractor.transform(X_df)
        y_pred = model.predict(X_features)
        
        cm_analysis = self.confusion_matrix_analysis(y, y_pred, model_name)  # Fixed: method name
        threshold_analysis = self.threshold_optimization(model, feature_extractor, X_df, y)
        fp_analysis = self.false_positive_analysis(model, feature_extractor, X_df, y)
        
        # Compile full report
        report = {
            'model_name': model_name,
            'cross_validation': cv_results,
            'confusion_matrix_analysis': cm_analysis,
            'threshold_optimization': threshold_analysis,
            'false_positive_analysis': fp_analysis,
            'summary': {
                'precision_cv_mean': cv_results['precision']['test_mean'],
                'precision_cv_std': cv_results['precision']['test_std'],
                'recommended_threshold': threshold_analysis['recommended_threshold'],
                'total_rescue_opportunity': fp_analysis['false_positive_count']
            }
        }
        
        # Store in history
        self.results_history.append(report)
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _print_report_summary(self, report):
        """Print a concise summary of the validation report"""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {report['model_name']}")
        print(f"{'='*60}")
        
        cv = report['cross_validation']
        print(f"Cross-Validation (5-fold):")
        print(f"  Precision: {cv['precision']['test_mean']:.3f} ± {cv['precision']['test_std']:.3f}")
        print(f"  Recall:    {cv['recall']['test_mean']:.3f} ± {cv['recall']['test_std']:.3f}")
        print(f"  F1-Score:  {cv['f1']['test_mean']:.3f} ± {cv['f1']['test_std']:.3f}")
        
        cm_metrics = report['confusion_matrix_analysis']['metrics']
        print(f"\nConfusion Matrix Analysis:")
        print(f"  Precision: {cm_metrics['precision']:.3f}")
        print(f"  Rescue Opportunity: {cm_metrics['rescue_opportunity']} emails")
        print(f"  False Positive Rate: {cm_metrics['false_positive_rate']:.3f}")
        
        if report['threshold_optimization']['recommended_threshold']:
            rec_thresh = report['threshold_optimization']['recommended_threshold']
            print(f"\nRecommended Threshold: {rec_thresh['threshold']:.2f}")
            print(f"  At this threshold: Precision={rec_thresh['precision']:.3f}, Rescued={rec_thresh['rescued_count']}")
        else:
            print(f"\nNo threshold meets precision target of {report['threshold_optimization']['target_precision']}")


def main():
    """Example usage of the validation framework"""
    from models.Spam_Rand_Forest import SpamRescueRandomForest
    from models.TF_IDF_Feature_Extractor import Feature_Extractor
    from Spam_Classifier import BaseLineSpamClassifier
    
    # Load data
    df = pd.read_json(r'C:\Users\Famil\Desktop\n1\utils\main_source.json')    
    
    # Prepare data
    baseline = BaseLineSpamClassifier()
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = baseline.prepare_data(df)
    
    # Initialize models
    extractor = Feature_Extractor(max_features=1000, ngram_range=(1,2))
    rf_model = SpamRescueRandomForest(extractor)
    
    # Train RandomForest
    rf_model.train_fast(X_train_df, y_train)
    
    # Initialize validator
    validator = SpamRescueValidator()
    
    # Generate comprehensive validation report
    rf_report = validator.generate_validation_report(
        rf_model.model, rf_model.feature_extractor, 
        X_val_df, y_val, "RandomForest"
    )
    
    print(f"\nValidation complete. {len(validator.results_history)} models evaluated.")


if __name__ == "__main__":
    main()