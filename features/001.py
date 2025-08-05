import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import html2text
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

class Class1:
    """Email feature extractor"""
    def __init__(self):
        self.html_conv = html2text.HTML2Text()
        self.html_conv.ignore_links = True
        self.html_conv.ignore_images = True
        self.vect1 = None

    def cleantxt(self, txt = str) -> str: 
        if not txt or pd.isna(txt): 
            return ""
        txt = str(txt)
        if '<html' in txt.lower() or '<div' in txt.lower() or '<p>' in txt.lower(): 
            txt = self.html_conv.handle(txt)
        #removing some characters and lines that may interfere
        txt = re.sub(r'^(From:|To:|Subject:|Date:|CC:).*$', '', txt, flags=re.MULTILINE)
        txt = re.sub(r'^>.*$', '', txt, flags=re.MULTILINE)  # Remove quoted text
        txt = re.sub(r'-----Original Message-----.*', '', txt, flags=re.DOTALL)

        txt - re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     ' URL_TOKEN ', txt) #replacing URLs with txt holders
        txt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     ' EMAIL_TOKEN ', txt)
        txt = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', 
                     ' PHONE_TOKEN ', txt)
        #this removes excesive punctuation
        txt = re.sub(r'[!]{2,}', '!', txt)
        txt = re.sub(r'[?]{2,}', '?', txt)
        txt = re.sub(r'\.{3,}', '...', txt)
        #removes some whitespace if applied
        txt = re.sub(r'\s+', '', txt)
        txt = txt.lower().str()

        return txt
    
    def basicFs(self, row: pd.Series) -> Dict[str, Any]:
        """this extracts meta data args: rows; returns: {}"""
        Fs = {} #Fs stand for features
        subs = str(row.get('subject', '')) #subjects
        b = str(row.get('body','')) #body
        sndr = str(row.get('sender', '')) #sender
        s_dmn = str(row.get('sender_dmn', '')) #sender domain

        Fs['subject_length'] = row.get('sub_len', len(subs))
        Fs['body_length'] = row.get('body_len', len(b))
        Fs['total_length'] = Fs['subject_length'] + Fs['body_length']

        Fs['subject_word_count'] = len(subs.split()) if subs else 0
        Fs['body_word_count'] = len(b.split()) if b else 0
        Fs['total_word_count'] = Fs['subject_word_count'] + Fs['body_word_count']

        if Fs['total_word_count'] > 0: 
            Fs['avg_word_count'] = Fs['total_length'] / Fs['total_word_count']
        else: 
            Fs['avg_word_count'] = 0
        dmn = s_dmn.lower()
        Fs['sender_domain'] = dmn
        Fs['is_gmail'] = 1 if 'gmail.com' in dmn else 0
        Fs['is_yahoo'] = 1 if 'yahoo.com' in dmn else 0
        Fs['is_outlook'] = 1 if 'outlook.com' in dmn else 0
        Fs['is_free_email'] = Fs['is_gmail'] or Fs['is_yahoo'] or Fs['is_outlook']

        Fs['sus_chars'] = 1 if any(char in dmn for char in ['1', '0', '-']) else 0
        Fs['domain_length'] = len(dmn)

        combtxt = subs + '' + b #combined text
        urlp =r'http[s]?://\S+|www\.\S+|\[.*[Ll]ink.*\]'#url pattern
        Fs['url_count'] = len(re.findall(urlp, combtxt))
        Fs['has_urls'] = 1 if Fs['url_count'] > 0 else 0
        Fs['email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combtxt))
        #urgency words indicator
        urgentwrds = ['urgent', 'immediate', 'asap', 'expire', 'deadline', 
                      'limited time', 'act now', 'don\'t wait', 'hurry', 
                      'rush', 'quickly', 'need' ]
        Fs['urgency_word_count'] = sum(1 for word in urgentwrds
                                       if word in combtxt.lower())
        
        spamwrds = ['free', 'win', 'winner', 'congratulations', 'prize', 'lottery', 
                     'click here', 'buy now', 'limited offer', 'act fast', 'guarantee']
        Fs['spam_word_count'] = sum( 1 for word in spamwrds
                                    if word in combtxt.lower())
        cxkwords = ['help', 'support', 'issue', 'problem', 'question', 'inquiry', 
                           'order', 'payment', 'refund', 'cancel', 'account', 'service',
                           'customer', 'assistance', 'resolve', 'ticket']                #customer key words
        Fs['customer_keyword_count'] = sum(1 for word in cxkwords 
                                           if word in combtxt.lower())
        
        if Fs['total_length'] > 0: 
            Fs['exclamation_ratio'] = combtxt.count('!') / Fs['total_length']
            Fs['question_ratio'] = combtxt.count('?') / Fs['total_length']
            Fs['caps_ratio'] = sum( 1 for c in combtxt if c.isupper()) / Fs['total_length']
            Fs['digit_ratio'] = sum( 1 for c in combtxt if c.isdigit()) / Fs['total_length']
        else: 
            Fs['exclamation_ratio'] = 0 
            Fs['question_ratio'] = 0
            Fs['caps_ratio'] = 0
            Fs['digit_ratio'] = 0
        
        Fs['subject_has_caps'] = 1 if any(c.isupper() for c in subs) else 0
        Fs['subject_all_caps'] = 1 if subs.isupper() and len(subs) > 1 else 0
        Fs['subject_has_urgency'] = 1 if any(word in subs.lower() for word in urgentwrds) else 0

        return Fs

def load(file_path = str) -> pd.DataFrame: 
    """load synth data and explore before hand"""
    import json

    if file_path.endswith('json'): 
        try:
            with open(file_path, 'r') as f: 
                dt = json.load(f) 
            df = pd.DataFrame(dt)
        except: 
            dt = []
            with open(file_path, 'r') as f: 
                for l in f: 
                    dt.append(json.loads(l.strip()))
            df = pd.DataFrame(dt)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else: 
        raise ValueError('unsupported format')
    
    df['label_num'] = df['label'].map({'legitimate': 0, 'spam': 1})

    print(f"shape: {df.shape}")
    print(f" cols: {df.columns.tolist()}")
    print(f"\nlabel distro: {df['label'].value_counts()}")
    print(f" num labels: {df['label_num'].value_counts()}")
    print(f"\nBasic stats: ")
    print(f"avg sub length: {df['sub_len'].mean():.1f}")
    print(f"abg body length: {df['body_len'].mean():.1f}")
    print(f" Unique domains: {df['sender_dmn'].nunique()}")

    return df
    
if __name__ == "__main__":
    extct = Class1()

    sample_data = {
        'subject': 'URGENT: Your Account Will Be Suspended!!!',
        'body': 'Click here http://fake-bank.com to verify your account. Contact support@fake-bank.com immediately!',
        'sender': 'noreply@suspicious-domain.com',
        'label': 1  # 1 for spam, 0 for legitimate
    }

    sample_series = pd.Series(sample_data)
    Fs = extct.basicFs(sample_series)

    print("Sample features extracted:")
    for key, value in Fs.items(): 
        print(f"{key}: {value}")


