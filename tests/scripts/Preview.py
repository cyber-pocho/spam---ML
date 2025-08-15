import random as rd
import pandas as pd
from analysis import analysis
from txt_an import txt_analysis
from val import validation

def preview(df, num_samples = 5): 
    """some manual review for emails"""
    email_le = df[df['label'] == 'legitimate']
    email_sp = df[df['label'] == 'spam']

    print("LEGIT EMAILS")

    for i in range(min(num_samples, len(email_le))):
        sample = email_le.iloc[i]
        print("LEGIT SAMPLE")
        print(f" from {sample['sender']}")
        print(f" subject: {sample ['subject']}")
        print(f" body:")
        print(sample['body'][:300] + "..." if len(sample['body']) > 300 else sample['body'])

        b_fields = ['order_number', 'rma_number', 'po_number', 'invoice_number', 
                          'tracking_number', 'quote_number', 'ticket_number']
        ids = []
        for field in b_fields: 
            if field in sample and pd.notna(sample[field]): 
                ids.append(f"{field}: {sample[field]}")
        if ids: 
            print(f" Business IDs: {','.join(ids)}")
        print("-"*50)
    print("SPAM EMAILS ")

    for i in range(min(num_samples, len(email_sp))): 
        sample = email_sp.iloc[i]
        print(f"from: {sample['sender']}")
        print(f" subject: {sample['subject']}")
        print(f" body: ")
        print(sample['body'][:300] + "..." if len(sample['body'])>300 else sample['body'])
        print("-" * 50) 
def qa_report(df): 
    d_stats = analysis(df)
    t_stats = txt_analysis(df)

    score = 0
    maxscore = 8

    if 0.65 <= d_stats['spam_ratio'] <= 0.75: 
        score +=1
        print("label balance PASS")
    else: 
        print("FAIL")
    
    if d_stats['unique_domains'] >= 10:
        score += 1
        print("domain diversity CHECK")
    else: 
        print("not enough diversity REVIEW")
    if d_stats['has_business_identifiers']: 
        score += 1
        print(" business ids PASS")
    else: 
        print("business ids FAIL")
    if d_stats['duplicate_count'] < len(df) * 0.05: 
        score +=1
        print("duplicate check PASS") 
    else: 
        print("duplicate FAIL")

    if t_stats['business_terms_found'] >= 5:
        score += 1
        print("business terms PASS")
    else: 
        print("business terms FAIL")
    
    if t_stats['spam_indicators_found'] >= 5: 
        print("spam indicators PASS")
    else: 
        print("spam indicators FAIL")
    
    if abs(t_stats['spam_avg_length'] - t_stats['legit_avg_length']) > 100:
        score +=1
        print("length diff PASS")
    else: 
        print(" len diff FAIL")
    
    if t_stats['spam_upper_ratio'] > t_stats['legit_upper_ratio'] * 1.2: 
        score +=1
        print(" style diff PASS")
    else:
        print(" style diff FAIL")
    
    final_score = score / maxscore
    print(f"quality score: {final_score}")

    if final_score >= 0.8: 
        print("good qlty")
    elif final_score >= 60: 
        print("subpar qlty")
    else: 
        print("trash quality")
    
def complete_run(): 
    """Run the complete data quality check pipeline"""
    df  = validation()
    if df is None: 
        return None
    preview(df, num_samples=3)
    qa = qa_report(df)

    if qa >= 80: 
        print("excellent data qlty")
    else: 
        print("let's regenerate data, this is sh*t")
    return df

if __name__ == "__main__":
    df = complete_run()
