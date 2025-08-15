import re
from val import validation
from collections import Counter
import matplotlib.pyplot as plt

def txt_analysis(df): 
    """analysis of txt characteristics on emails"""
    print("===== text analysis =====")

    print("len analysis by category")
    if 'sub_len' in df.columns and 'body_len' in df.columns: 
        len_stats = df.groupby('label')[['sub_len', 'body_len']].agg(['mean', 'median', 'std'])
        print(len_stats.round(1))

        # spamb_avg and legitb_avg. Any meaningful differences?
        spamb_avg = df[df['label'] == 'spam']['body_len'].mean()
        legitb_avg = df[df['label'] == 'legitimate']['body_len'].mean()

        if abs(spamb_avg - legitb_avg)>100: 
            print(f"appropiate length diff: spam avg {spamb_avg:.0f}, legit avg {legitb_avg: .0f}")
        else: 
            print(f"similar length, classification of these emails could be tougher. maybe check for random on gen.py")
    else: 
        print("length cols not found, creating lists....")
        df['sub_len'] = df['subject'].str.len()
        df['body_len'] = df['body'].str.len()
    print("==== subject patterns ====")
    spam_sub = df[df['label'] == 'spam']['subject'].tolist()
    legit_sub = df[df['label'] == 'legitimate']['subject'].tolist()

    def extractor(text_list): 
        allw = [] #all words
        for text in text_list: 
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            allw.extend(words)
        return allw
    
    # For both spam and legit subjects, we extract the words 
    # on each email
    spam_w = extractor(spam_sub)
    legit_w = extractor(legit_sub)

    spam_w_cnt = Counter(spam_w)
    legit_w_cnt = Counter(legit_w)

    print("MOST FOUND WORDS ON SPAM SUBJECT LINES")
    for word, count in spam_w_cnt.most_common(10): 
        print(f" {word}: {count}")

    print("\nMOST COMMON WORDS ON legit EMAILS: ")
    for word, count in legit_w_cnt.most_common(10): 
        print(f" {word}: {count}")
    
    # as we might want to find emails that are actual
    # bussiness emails, we can use the business terms 
    # that might be on said emails
    bssns_trms = ['order', 'invoice', 'rma', 'quote', 
                  'purchase', 'po', 'tracking', 'confirmation', 
                  'warranty']
    print(" BUSSINESS TERMINOLOGY")
    legit_txt = ''.join(legit_sub + df[df['label']=='legitimate']['body'].tolist().lower())
    bssns_trms_cnts = {}
    for term in bssns_trms: 
        count = len(re.findall(r'\b' + term + r'\b', legit_txt))
        if count > 0: 
            bssns_trms_cnts[term] = count

    if bssns_trms_cnts: 
        print('business terms found on legit emails: ')
        for term, count in sorted(bssns_trms_cnts.items(), key = lambda x:x[1], reverse = True): 
            print(f" {term}: {count} times")
        print("aggreable business terminology pressence on data set")
    else: 
        print("No business terms found - may not be company specific enough")
    
    spam_indexs = ['urgent', 'free', 'limited', 'act now', 'click', 'offer', 
                      'deal', 'discount', 'winner', 'congratulations', 'claim']
    spam_txt = ' '.join(spam_sub + df[df['label'] == 'spam']['body'].tolist()).lower()
    spam_indexs_counts = {}
    for indicator in spam_indexs: 
        count = len(re.findall(r'\b' + indicator + r'\b', spam_txt))
        if count > 0: 
            spam_indexs_counts[indicator] = count
    print(f"\n SPAM INDICATORS")
    if spam_indexs_counts: 
        print('spam indicators found')
        for indicator, count in sorted(spam_indexs_counts.item(), key= lambda x:x[1], reverse=True)[:10]:
            print(f" {indicator}: {count} times")
        print("acceptable spam indicators presence")
    else: 
        print("spam may not be obvious enough")
    
    print(f"\n====== CHARACTER PATTERN ANALYSIS=========")
    spam_uprat = sum(sum(c.isupper() for c in sub) / len(sub)
                     for sub in spam_sub if len(sub) > 0 / len(spam_sub))
    legit_uprat = sum(sum(c.isupper() for c in sub)/len(sub) 
                      for sub in legit_sub if len(sub)>0 / len(legit_sub))
    print(f"average uppercase ratio in subjects")
    print(f" Spam: {spam_uprat: .3f}")
    print(f"legits: {legit_uprat}")

    if spam_uprat > legit_uprat * 1.5: 
        print("spam emails have more uppercase letters")
    else: 
        print("no real conclusion drawn from data")
    
    spam_exc = sum(sub.count('!') for sub in spam_sub) / len(spam_sub)
    legit_exc = sum(sub.count('!') for sub in legit_sub) / len(legit_sub)
    print(f" avg !s marks per subject: ")
    print(f" spam: {spam_exc:.2f}")
    print(f" legit: {legit_exc:.2f}")

    return {
        'spam_avg_len': df[df['label'] == 'spam']['body_len'].mean(),
        'legit_avg_len': df[df['label'] == 'legitimate']['body_len'].mean(), 
        'business_terms_found':len(bssns_trms_cnts),
        'spam_indicators_found':len(spam_indexs_counts),
        'spam_upper_ratio':spam_uprat,
        'legit_upper_ratio':legit_uprat
    }

df = validation()
if __name__ == '__main__': 
    txt_analysis(df)