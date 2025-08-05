import pandas as pd
import random as rd
import json
from datetime import datetime, timedelta
from faker import Faker
import re

fk = Faker()

class gen_email:
    def __init__(self): 
        self.legit_dmns=[
            'amazon.com', 'paypal.com', 'microsoft.com', 'google.com', 'apple.com',
            'netflix.com', 'spotify.com', 'linkedin.com', 'github.com', 'stripe.com',
            'shopify.com', 'salesforce.com', 'zendesk.com', 'mailchimp.com', 'dropbox.com'
        ]
        self.spam_dmns=[
            'tempmail.com', 'guerrillamail.com', 'lottery-winner.net', 'amazing-deals.biz',
            'pharmacy-online.info', 'work-from-home.org', 'get-rich-quick.net', 'free-money.co',
            'weight-loss-miracle.com', 'fake-rolex.shop', 'crypto-profits.investment'
        ]
        self.common_dmns=[
            'support', 'noreply', 'service', 'team', 'info', 'hello', 'contact',
            'notifications', 'security', 'billing', 'admin', 'help'
        ]
    def leg_emails(self): 
        """generating legitimate emails"""
        type1 = rd.choice([
            'customer_support', 'order_confirmation', 'newsletter', 
            'password_reset', 'account_notification', 'billing', 'welcome'
        ])
        
        if type1 == 'customer_support': 
            return self._generate_customer_support()
        elif type1 == 'order_confirmation': 
            return self._generate_order_confirmation()
        elif type1 == 'newsletter':
            return self._generate_newsletter()
        elif type1 == 'password_reset':
            return self._generate_password_reset()
        elif type1 == 'account_notification':
            return self._generate_account_notification()
        elif type1 == 'billing':
            return self._generate_billing()
        else:
            return self._generate_welcome()
    def spam_emails(self): 
        """gen spam emails"""
        type2 = rd.choice([
            'phishing', 'promotional', 'lottery', 'pharmacy', 
            'work_from_home', 'fake_urgency', 'advance_fee'
        ])
        if type2 == 'phishing':
            return self._generate_phishing()
        elif type2 == 'promotional':
            return self._generate_promotional_spam()
        elif type2 == 'lottery':
            return self._generate_lottery_spam()
        elif type2 == 'pharmacy':
            return self._generate_pharmacy_spam()
        elif type2 == 'work_from_home':
            return self._generate_work_from_home()
        elif type2 == 'fake_urgency':
            return self._generate_fake_urgency()
        else:
            return self._generate_advance_fee()
        
    def _generate_customer_support(self): 
        dmn = rd.choice(self.legit_dmns)
        templates = [
            f"Thank you for contacting {dmn.split('.')[0].title()} support. We have received your inquiry about {{issue}} and will respond within 24 hours. Your ticket number is {{ticket}}. Best regards, Customer Support Team",
            f"Hi there! We've resolved your issue with {{issue}}. Please let us know if you need any further assistance. Your satisfaction is our priority. Support Team at {dmn.split('.')[0].title()}",
            f"We're sorry to hear about the trouble you're experiencing with {{issue}}. Our technical team is looking into this and will provide an update soon. Thank you for your patience."
        ]
        issues = ['account access', 'billing inquiry', 'technical problem', 'feature request', 'refund process']
        template = rd.choice(templates)
        
        return {
            'sender': f"{rd.choice(self.common_dmns)}@{dmn}",
            'subject': f"Re: {rd.choice(issues).title()} - Ticket #{rd.randint(10000, 99999)}",
            'body': template.format(issue=rd.choice(issues), ticket=rd.randint(10000, 99999)),
            'label': 'legitimate'
        }
    def _generate_order_confirmation(self):
        domain = rd.choice(self.legit_dmns)
        order_num = rd.randint(100000, 999999)
        amount = rd.uniform(10, 500)
        
        body = f"""Thank you for your order #{order_num}!
        
Your order has been confirmed and will be processed within 1-2 business days.

Order Summary:
- Total: ${amount:.2f}
- Estimated delivery: {fk.date_between(start_date='+1d', end_date='+7d')}
- Tracking information will be sent to this email

Questions? Contact our support team.

Best regards,
{domain.split('.')[0].title()} Team"""
        
        return {
            'sender': f"orders@{domain}",
            'subject': f"Order Confirmation #{order_num}",
            'body': body,
            'label': 'legitimate'
        }
    def _generate_newsletter(self):
        domain = rd.choice(self.legit_dmns)
        company = domain.split('.')[0].title()
        
        topics = ['product updates', 'industry insights', 'company news', 'tips and tricks']
        topic = rd.choice(topics)
        
        body = f"""Hi there!

Welcome to this week's {company} newsletter featuring {topic}.

This week we're covering:
• New feature releases
• Customer success stories  
• Upcoming events
• Industry trends

Read more on our blog or contact us with questions.

Best,
The {company} Team

Unsubscribe | Update preferences"""
        
        return {
            'sender': f"newsletter@{domain}",
            'subject': f"{company} Weekly: {topic.title()}",
            'body': body,
            'label': 'legitimate'
        }
    ######
    def _generate_password_reset(self):
        domain = rd.choice(self.legit_dmns)
        company = domain.split('.')[0].title()
        
        body = f"""Hi,

We received a request to reset your password for your {company} account.

If you requested this, click the link below to reset your password:
[Reset Password Link]

This link will expire in 24 hours.

If you didn't request this, please ignore this email.

{company} Security Team"""
        
        return {
            'sender': f"security@{domain}",
            'subject': f"Password Reset Request - {company}",
            'body': body,
            'label': 'legitimate'
        }
    

    def _generate_account_notification(self):
        domain = rd.choice(self.legit_dmns)
        company = domain.split('.')[0].title()
        
        notifications = [
            'login from new device', 'subscription renewal', 'profile update',
            'security setting change', 'payment method update'
        ]
        notification = rd.choice(notifications)
        
        body = f"""Hello,

This is a notification about your {company} account regarding: {notification}.

Time: {fk.date_time_between(start_date='-1d', end_date='now')}
Location: {fk.city()}, {fk.country()}

If this wasn't you, please contact our security team immediately.

{company} Account Team"""
        
        return {
            'sender': f"notifications@{domain}",
            'subject': f"Account Activity: {notification.title()}",
            'body': body,
            'label': 'legitimate'
        }
    
    def _generate_billing(self):
        domain = rd.choice(self.legit_dmns)
        company = domain.split('.')[0].title()
        amount = rd.uniform(5, 100)
        
        body = f"""Dear Customer,

Your {company} subscription has been renewed.

Amount charged: ${amount:.2f}
Next billing date: {fk.date_between(start_date='+28d', end_date='+32d')}
Payment method: Card ending in {rd.randint(1000, 9999)}

View your invoice or update billing information in your account.

{company} Billing Team"""
        
        return {
            'sender': f"billing@{domain}",
            'subject': f"Payment Confirmation - {company}",
            'body': body,
            'label': 'legitimate'
        }
    
    def _generate_welcome(self):
        domain = rd.choice(self.legit_dmns)
        company = domain.split('.')[0].title()
        
        body = f"""Welcome to {company}!

We're excited to have you on board. Here's what you can do next:

1. Complete your profile setup
2. Explore our features
3. Join our community
4. Contact support if you need help

Get started: [Getting Started Guide]

Welcome aboard!
The {company} Team"""
        
        return {
            'sender': f"welcome@{domain}",
            'subject': f"Welcome to {company}!",
            'body': body,
            'label': 'legitimate'
        }
    
    def _generate_phishing(self):
        # Fake versions of legitimate domains
        fake_domains = ['arnazon.com', 'paypaI.com', 'microsooft.com', 'goog1e.com']
        domain = rd.choice(fake_domains)
        
        urgent_phrases = [
            'URGENT ACTION REQUIRED', 'Account will be suspended', 'Verify immediately',
            'Security breach detected', 'Click now or lose access'
        ]
        
        body = f"""{rd.choice(urgent_phrases)}

Your account has been compromised. You must verify your information immediately.

Click here to secure your account: [Fake Link]

Warning: Failure to verify within 24 hours will result in permanent account closure.

Security Team"""
        
        return {
            'sender': f"security@{domain}",
            'subject': rd.choice(urgent_phrases) + " - Account Verification",
            'body': body,
            'label': 'spam'
        }
    def _generate_promotional_spam(self):
        domain = rd.choice(self.spam_dmns)
        
        pds = ['weight loss pills', 'miracle cream', 'discount electronics', 'fake designer watches']
        pd = rd.choice(pds)
        dct = rd.randint(50, 90)
        
        body = f"""🔥 AMAZING DEAL ALERT! 🔥

Get {dct}% OFF on {pd}!

LIMITED TIME OFFER - ONLY TODAY!!!

✅ Free shipping worldwide
✅ Money back guarantee  
✅ Thousands of satisfied customers

CLICK NOW: [Spam Link]

Don't miss out! This offer expires in 2 hours!

*This email was sent to millions of recipients*"""
        
        return {
            'sender': f"deals@{domain}",
            'subject': f"🔥 {dct}% OFF {pd.title()} - LIMITED TIME!",
            'body': body,
            'label': 'spam'
        }
    ####
    def _generate_lottery_spam(self):
        domain = rd.choice(self.spam_dmns)
        amount = rd.randint(100000, 10000000)
        
        body = f"""CONGRATULATIONS!!!

You have won ${amount:,} in the International Email Lottery!

Your lucky email was randomly selected from millions of participants.

To claim your prize:
1. Reply with your full name and address
2. Send copy of ID
3. Pay processing fee of $500

Reference Number: {rd.randint(100000, 999999)}

Contact our claims department immediately!

Dr. {fk.last_name()}, Lottery Commissioner"""
        
        return {
            'sender': f"lottery@{domain}",
            'subject': f"YOU WON ${amount:,}!!! CLAIM NOW",
            'body': body,
            'label': 'spam'
        }
    
    def _generate_pharmacy_spam(self):
        domain = rd.choice(self.spam_dmns)
        
        meds = ['Generic Viagra', 'Weight Loss Pills', 'Pain Relief', 'Anti-aging cream']
        med = rd.choice(meds)
        dct = rd.randint(60, 85)
        
        body = f"""Get {med} at {dct}% discount!

No prescription needed!
FDA approved (not really)
Discreet packaging
Worldwide shipping

Order now: [Pharmacy Link]

*Not evaluated by FDA
*Results may vary
*Side effects may include everything"""
        
        return {
            'sender': f"pharmacy@{domain}",
            'subject': f"{med} - {dct}% OFF No Prescription!",
            'body': body,
            'label': 'spam'
        }
    
    def _generate_work_from_home(self):
        domain = rd.choice(self.spam_dmns)
        prfit = rd.randint(1000, 5000)
        
        body = f"""Make ${prfit}/week working from home!

No experience needed!
Work only 2 hours per day!
Guaranteed income!

Join thousands of successful people making money online.

What you'll do:
- Data entry (super easy!)
- Email processing  
- Online surveys

Start today: [Work Link]

Limited spots available!"""
        
        return {
            'sender': f"opportunity@{domain}",
            'subject': f"Make ${prfit}/Week From Home - No Experience!",
            'body': body,
            'label': 'spam'
        }
    def _generate_fake_urgency(self):
        dmn = rd.choice(self.spam_dmns)
        
        urgent_scams = [
            'Your computer is infected with 247 viruses!',
            'Microsoft detected suspicious activity!',
            'Your IP address has been compromised!',
            'FBI investigation requires your assistance!'
        ]
        
        scam = rd.choice(urgent_scams)
        
        body = f"""⚠️ URGENT ALERT ⚠️

{scam}

IMMEDIATE ACTION REQUIRED:
1. Do not turn off your computer
2. Call this number NOW: 1-800-FAKE-NUM
3. Give us remote access to fix the problem

Failure to act within 1 hour will result in:
- Complete data loss
- Identity theft
- Legal consequences

Microsoft Certified Technician Team
(We're definitely not Microsoft)"""
        
        return {
            'sender': f"alert@{dmn}",
            'subject': f"⚠️ URGENT: {scam}",
            'body': body,
            'label': 'spam'
        }
    
    ####
    def _generate_advance_fee(self):
        domain = rd.choice(self.spam_dmns)
        pais = fk.country()
        amount = rd.randint(1000000, 50000000)
        
        body = f"""Dear Friend,

I am writing to you from {pais}. I am a prince/businessman/lawyer with access to ${amount:,} that needs to be transferred out of the country.

I need your help to transfer this money. In return, you will receive 30% (${amount * 0.3:,.0f}).

All I need is:
- Your bank account details
- Copy of your passport
- Small transfer fee of $2,000

This is 100% legal and risk-free.

Contact me immediately for this once-in-a-lifetime opportunity.

Prince {fk.first_name()} {fk.last_name()}"""
        
        return {
            'sender': f"prince@{domain}",
            'subject': f"Urgent Business Proposal - ${amount:,}",
            'body': body,
            'label': 'spam'
        }
    
    #######

def gen_dtset(num_emails = 5000, spam_ratio = 0.7): 
    """complete data set"""
    generator = gen_email()
    emails = []

    num_spam  = int(num_emails*spam_ratio)
    num_legit = num_emails - num_spam

    for i in range(num_spam): 
        if i % 500 == 0: 
            print(f"generated {i} spam emails...")
        emails.append(generator.spam_emails())
    for i in range(num_legit): 
        if i % 200 == 0: 
            print(f"generated {i} legit emails")
        emails.append(generator.leg_emails())
    rd.shuffle(emails)

    for i, email in enumerate(emails): 
        email['id'] = i
        email['timestamp'] = fk.date_time_between(start_date='-30d', end_date='now').isoformat()
        email['sub_len'] = len(email['subject'])
        email['body_len'] = len(email['body'])
        email['sender_dmn'] = email['sender'].split('@')[1]
    return emails

if __name__ == "__main__": 
    print("synthetic data generation...")
    dt_set = gen_dtset(num_emails = 5000, spam_ratio = 0.7)
    df =  pd.DataFrame(dt_set)
    df.to_csv('syn_emails.csv', index = False)
    with open('syn_emails.json', 'w') as f: 
        json.dump(dt_set, f, indent=2)
    
    print(f"total emails: {len(dt_set)}")
    print(f"spam: {len(df[df['label'] == 'spam'])}")
    print(f"legit: {len(df[df['label'] == 'legitimate'])}")
    print(f"spam r/r: {len(df[df['label'] == 'spam']) / len(df):.2%}")

    print("\nexamples ===================================")
    print("LEGITIMATE EMAIL:")
    legit_sample = df[df['label'] == 'legitimate'].iloc[1]
    print(f"From: {legit_sample['sender']}")
    print(f"Subject: {legit_sample['subject']}")
    print(f"Body: {legit_sample['body'][:200]}...")
    print("\nSPAM EMAIL:")
    spam_sample = df[df['label'] == 'spam'].iloc[1]
    print(f"From: {spam_sample['sender']}")
    print(f"Subject: {spam_sample['subject']}")
    print(f"Body: {spam_sample['body'][:200]}...")

    print("\n=== some stats that may be useful ===")
    print("Average subject length by type:")
    print(df.groupby('label')['sub_len'].mean())
    print("\nAverage body length by type:")
    print(df.groupby('label')['body_len'].mean())

    print("\nprogram complete")


