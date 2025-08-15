def analysis(df): 
    """analysis of data spam vs legit emails"""
    print('analyzing data...')

    lb_counts = df['label'].value_counts()
    spam_ratio = lb_counts.get('spam', 0)/len(df)
    legit_ratio = lb_counts.get('legitimate', 0)/len(df)

    print(f"total emails: {len(df)}")
    print(f"Spam emails: {lb_counts.get('spam', 0)} ({spam_ratio:.1%})")
    print(f"Legit emails: {lb_counts.get('legitimate', 0)} ({legit_ratio: .1%})")

    if 0.65 <= spam_ratio <= 0.75:
        print(f"ratio correct")
    else: 
        print(f"spam ratio: ({spam_ratio: .1%}) not as expected. Please check generator.")
    
    #distribution analysis 
    print("===DOMAIN ANALYSIS===")
    if 'sender_dmn' in df.columns: 
        dmn_cts = df['sender_dmn'].value_counts()
        total_dmns = len(dmn_cts)
        print(f"total unique domains: {total_dmns}")

        if total_dmns < 10: 
            print("not enough diversity - unrealistic data")
        else: 
            print("diverse enough")
        print("\ntop sender domains (10):")
        for domain, count in dmn_cts.head(10).items(): 
            pr100 = count/len(df) * 100
            print(f" {domain} : {count} emails ({pr100:.1f}%)")
        top_dmn_pct = dmn_cts.iloc[0] / len(df)
        if top_dmn_pct > 0.3:
            print(f" top domain ({dmn_cts.index[0]}) is {top_dmn_pct: .1%} of all emails")
    else: 
        print(" no sender_dmn column found")
    
    legit_df = df[df['label'] == 'legitimate']
    #bfields = "Business Fields" it's too long to code that
    bfields = ['order_number', 'rma_number', 'po_number', 'invoice_number',
               'tracking_number', 'quote_number', 'ticket_number', 'warranty_number']
    print(f"total legitimate emails: {len(legit_df)}")

    id_found = False
    for field in bfields:
        if field in legit_df.columns: 
            non_null_cnt = legit_df[field].notna().sum()
            if non_null_cnt > 0: 
                pct = non_null_cnt / len(legit_df) * 100
                print(f" {field}: {non_null_cnt} emails ({pct: .1f}%)")
                id_found = True
    if not id_found: 
        print("no business ids found in legit emails. gen likely not working")
    else:
        print("bussiness ids present - goood to go")
    dups = df[df.duplicated(subset = ['subject', 'body'], keep = False)]
    if len(dups) > 0: 
        print(f"\n found {len(dups)} duplicated emails. Not enough variety.")
    else:
        print(f"\n No duplicate emails found.")
    return{
        'total_emails':len(df),
        'spam_ratio': spam_ratio, 
        'unique_domains' : len(df['sender_dmn'].unique()) if 'sender_dmn' in df.columns else 0,
        'has_business_ids': id_found,
        'duplicate_count': len(dups)
    }