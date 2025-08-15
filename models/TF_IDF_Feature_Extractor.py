from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import pandas as pd
import numpy as np
import re
import os
from html.parser import HTMLParser


class MLStripper(HTMLParser):
    """Simple HTML tag stripper"""
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    
    def handle_data(self, d):
        self.fed.append(d)
    
    def get_data(self):
        return ''.join(self.fed)


class Feature_Extractor: 
    """TF-IDF Feature Extractor for Email Classification
        
    Args:
        max_features: Maximum number of features to extract
        ngram_range: Range of n-grams (1,1) = unigrams, (1,2) = uni+bigrams
    """

    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF extractors
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams (1,1) = unigrams, (1,2) = uni+bigrams
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.is_fitted = False  # Initialize fitted flag
        
        # HTML converter for cleaning
        self.html_conv = MLStripper()
        
        # Separate vectorizers for subject and body
        self.subject_tfidf = TfidfVectorizer(
            max_features=max_features//2,  # Split features between subject and body
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',  # Only words starting with letter
            min_df=2,  # Must appear in at least 2 documents
            max_df=0.95  # Remove words that appear in >95% of documents
        )
        
        self.body_tfidf = TfidfVectorizer(
            max_features=max_features//2,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',
            min_df=2,
            max_df=0.95
        )

    def strip_html_tags(self, html_text):
        """Strip HTML tags from text"""
        self.html_conv.reset()
        self.html_conv.fed = []
        self.html_conv.feed(html_text)
        return self.html_conv.get_data()

    def cleantxt(self, txt: str) -> str:
        """Clean and preprocess text data"""
        if not txt or pd.isna(txt):
            return ""

        txt = str(txt)
        
        # Handle HTML content
        if '<html' in txt.lower() or '<div' in txt.lower() or '<p>' in txt.lower():
            txt = self.strip_html_tags(txt)

        # Remove email headers and quoted text
        txt = re.sub(r'^(From:|To:|Subject:|Date:|CC:).*$', '', txt, flags=re.MULTILINE)
        txt = re.sub(r'^>.*$', '', txt, flags=re.MULTILINE)
        txt = re.sub(r'-----Original Message-----.*', '', txt, flags=re.DOTALL)
        
        # Replace patterns with tokens
        txt = re.sub(r'http[s]?://\S+|www\.\S+', ' URL_TOKEN ', txt)
        txt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_TOKEN ', txt)
        txt = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' PHONE_TOKEN ', txt)
        txt = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', ' MONEY_TOKEN ', txt)
        txt = re.sub(r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|\$)', ' MONEY_TOKEN ', txt)

        # Handle repeated punctuation
        txt = re.sub(r'[!]{2,}', ' EXCITEMENT_TOKEN ', txt)
        txt = re.sub(r'[?]{2,}', ' QUESTION_TOKEN ', txt)
        txt = re.sub(r'\.{3,}', '...', txt)
        
        # Clean whitespace and convert to lowercase
        txt = re.sub(r'\s+', ' ', txt).strip()
        txt = txt.lower()
        
        return txt

    def fit_transform(self, df): 
        """
        Fit TF-IDF vectorizers and transform the data
        
        Args:
            df: DataFrame with 'subject' and 'body' columns
            
        Returns:
            Combined TF-IDF feature matrix
        """
        # Clean text data
        subs_clean = df['subject'].apply(self.cleantxt)
        body_clean = df['body'].apply(self.cleantxt)

        # Fit and transform subject and body separately
        subs_tfidf_matrix = self.subject_tfidf.fit_transform(subs_clean)
        print(f"Subject TF-IDF shape: {subs_tfidf_matrix.shape}")

        body_tfidf_matrix = self.body_tfidf.fit_transform(body_clean)  # FIXED: Use body_tfidf
        print(f"Body TF-IDF shape: {body_tfidf_matrix.shape}")

        # Combine features horizontally
        combined_tfidf_matrix = hstack([subs_tfidf_matrix, body_tfidf_matrix])
        self.is_fitted = True
        print(f"Combined TF-IDF shape: {combined_tfidf_matrix.shape}")

        return combined_tfidf_matrix

    def transform(self, df): 
        """
        Transform new data using fitted vectorizers
        
        Args:
            df: DataFrame with 'subject' and 'body' columns
            
        Returns:
            Combined TF-IDF feature matrix
        """
        if not self.is_fitted: 
            raise ValueError("Vectorizers must be fitted first! Call fit_transform() first.")
            
        # Clean text data
        subs_clean = df['subject'].apply(self.cleantxt)
        body_clean = df['body'].apply(self.cleantxt)

        # Transform using fitted vectorizers
        subs_tfidf_matrix = self.subject_tfidf.transform(subs_clean)
        body_tfidf_matrix = self.body_tfidf.transform(body_clean)

        # Combine features horizontally (FIXED: use hstack instead of +)
        combined_tfidf_matrix = hstack([subs_tfidf_matrix, body_tfidf_matrix])
        return combined_tfidf_matrix

    def get_feature_names(self): 
        """
        Get feature names from fitted vectorizers
        
        Returns:
            List of feature names
        """
        if not self.is_fitted: 
            raise ValueError("Vectorizers must be fitted first!")
            
        # Get feature names and add prefixes
        subj_features = [f"subj_{feat}" for feat in self.subject_tfidf.get_feature_names_out()]
        body_features = [f"body_{feat}" for feat in self.body_tfidf.get_feature_names_out()]

        return subj_features + body_features
    
    def get_top_features_by_class(self, tfidf_matrix, labels, top_n=20): 
        """
        Get top TF-IDF features for each class
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            labels: Class labels (0 for legitimate, 1 for spam)
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features for each class
        """
        feature_names = self.get_feature_names()
        
        # Convert sparse matrix to dense if needed
        if hasattr(tfidf_matrix, 'toarray'):
            tfidf_dense = tfidf_matrix.toarray()
        else: 
            tfidf_dense = tfidf_matrix
            
        results = {}

        for label in [0, 1]: 
            mask = labels == label
            if mask.sum() == 0: 
                continue
                
            # Calculate average TF-IDF scores for this class
            class_avg = tfidf_dense[mask].mean(axis=0)
            top_indices = class_avg.argsort()[-top_n:][::-1]
            top_features = [(feature_names[i], class_avg[i]) for i in top_indices]

            class_name = 'spam' if label == 1 else 'legitimate'
            results[class_name] = top_features

            print(f"\nTop {top_n} TF-IDF features for {class_name}:")
            for feat, score in top_features:
                print(f"  {feat}: {score:.4f}")
                
        return results


def test_extractor(df): 
    """Test function for the TF-IDF extractor"""
    print("Testing TF-IDF Feature Extractor...")
    
    # Initialize extractor
    tfidf_extractor = Feature_Extractor(max_features=1000, ngram_range=(1, 2))
    
    # Fit and transform data
    tfidf_features = tfidf_extractor.fit_transform(df)

    # Convert labels to numeric if needed
    if df['label'].dtype == 'object': 
        labels = df['label'].map({'legitimate': 0, 'spam': 1}).values
    else: 
        labels = df['label'].values
    
    # Get top features by class
    top_features = tfidf_extractor.get_top_features_by_class(tfidf_features, labels)

    return tfidf_features, tfidf_extractor


if __name__ == "__main__": 
    # Load data
    df = pd.read_json('C:/Users/Famil/Desktop/n1/data/main_source.json')
    print(f"Loaded {len(df)} emails")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Test the extractor
    tfidf_features, tfidf_extractor = test_extractor(df)

    print(f"\nFeature matrix shape: {tfidf_features.shape}")
    print(f"Feature matrix type: {type(tfidf_features)}")
    print(f"Memory usage: ~{tfidf_features.data.nbytes / 1024**2:.1f} MB")