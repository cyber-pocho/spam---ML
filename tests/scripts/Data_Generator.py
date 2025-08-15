import pandas as pd
import random as rd
import json
from datetime import datetime, timedelta
from faker import Faker
import uuid
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
        
        # Company-specific data
        self.company_dmns = [
            'customer.company.com', 'vendor.company.com', 'supplier.company.com',
            'orders.company.com', 'support.company.com', 'logistics.company.com'
        ]
        
        self.products = [
            'Industrial Pump Model X200', 'Hydraulic Valve V-450', 'Control Panel CP-890',
            'Safety Switch SS-123', 'Motor Mount MM-567', 'Pressure Gauge PG-340',
            'Circuit Breaker CB-780', 'Relay Module RM-225', 'Sensor Unit SU-445',
            'Cable Assembly CA-660', 'Junction Box JB-330', 'Power Supply PS-990'
        ]
        
        self.suppliers = [
            'ABC Manufacturing', 'TechParts Inc', 'Industrial Solutions LLC',
            'ProComponents Corp', 'Elite Supply Co', 'MegaParts Industries',
            'Precision Tools Ltd', 'Advanced Systems Inc', 'Quality Parts Co'
        ]
        
        self.customer_companies = [
            'Johnson Industries', 'Miller Manufacturing', 'Davis Corp',
            'Wilson Tech', 'Brown Industries', 'Taylor Systems',
            'Anderson Manufacturing', 'Thompson Corp', 'Garcia Industries'
        ]

    def leg_emails(self): 
        """generating legitimate emails - now company-focused"""
        type1 = rd.choice([
            'customer_order', 'rma_request', 'supplier_invoice', 'shipment_notification',
            'quote_request', 'purchase_order', 'delivery_confirmation', 'technical_inquiry',
            'warranty_claim', 'parts_availability'
        ])
        
        if type1 == 'customer_order': 
            return self._generate_customer_order()
        elif type1 == 'rma_request': 
            return self._generate_rma_request()
        elif type1 == 'supplier_invoice':
            return self._generate_supplier_invoice()
        elif type1 == 'shipment_notification':
            return self._generate_shipment_notification()
        elif type1 == 'quote_request':
            return self._generate_quote_request()
        elif type1 == 'purchase_order':
            return self._generate_purchase_order()
        elif type1 == 'delivery_confirmation':
            return self._generate_delivery_confirmation()
        elif type1 == 'technical_inquiry':
            return self._generate_technical_inquiry()
        elif type1 == 'warranty_claim':
            return self._generate_warranty_claim()
        else:
            return self._generate_parts_availability()

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

    # Company-specific legitimate email generators
    def _generate_customer_order(self):
        company = rd.choice(self.customer_companies)
        order_num = f"ORD-{rd.randint(100000, 999999)}"
        product = rd.choice(self.products)
        quantity = rd.randint(1, 50)
        unit_price = rd.uniform(150, 5000)
        
        body = f"""Dear Sales Team,

We would like to place the following order:

Order Number: {order_num}
Customer: {company}
Product: {product}
Quantity: {quantity} units
Unit Price: ${unit_price:.2f}
Total: ${quantity * unit_price:.2f}

Please confirm availability and expected delivery date.

Shipping Address:
{fk.street_address()}
{fk.city()}, {fk.state_abbr()} {fk.zipcode()}

Contact: {fk.name()}
Phone: {fk.phone_number()}

Best regards,
{company} Procurement Team"""
        
        return {
            'sender': f"orders@{company.lower().replace(' ', '')}.com",
            'subject': f"Purchase Order {order_num} - {product}",
            'body': body,
            'label': 'legitimate',
            'order_number': order_num,
            'product': product,
            'company': company
        }

    def _generate_rma_request(self):
        company = rd.choice(self.customer_companies)
        rma_num = f"RMA-{rd.randint(10000, 99999)}"
        original_order = f"ORD-{rd.randint(100000, 999999)}"
        product = rd.choice(self.products)
        issue = rd.choice(['defective', 'wrong item shipped', 'damaged in transit', 'not as described'])
        
        body = f"""RMA Request

RMA Number: {rma_num}
Original Order: {original_order}
Customer: {company}
Product: {product}
Issue: {issue}

Description:
The product received does not meet specifications. We need to return this item for replacement or refund.

Purchased Date: {fk.date_between(start_date='-90d', end_date='-7d')}
Serial Number: SN{rd.randint(100000, 999999)}

Please provide return shipping instructions.

Contact: {fk.name()}
Email: {fk.email()}
Phone: {fk.phone_number()}

{company} Returns Department"""
        
        return {
            'sender': f"returns@{company.lower().replace(' ', '')}.com",
            'subject': f"RMA Request {rma_num} - {product}",
            'body': body,
            'label': 'legitimate',
            'rma_number': rma_num,
            'order_number': original_order,
            'product': product
        }

    def _generate_supplier_invoice(self):
        supplier = rd.choice(self.suppliers)
        invoice_num = f"INV-{rd.randint(100000, 999999)}"
        po_num = f"PO-{rd.randint(50000, 99999)}"
        products = rd.sample(self.products, rd.randint(1, 3))
        
        total = 0
        line_items = ""
        for product in products:
            qty = rd.randint(1, 20)
            price = rd.uniform(100, 2000)
            line_total = qty * price
            total += line_total
            line_items += f"- {product}: {qty} @ ${price:.2f} = ${line_total:.2f}\n"
        
        body = f"""Invoice

From: {supplier}
Invoice Number: {invoice_num}
PO Number: {po_num}
Date: {datetime.now().strftime('%B %d, %Y')}

Line Items:
{line_items}
Total: ${total:.2f}

Payment Terms: Net 30
Due Date: {(datetime.now() + timedelta(days=30)).strftime('%B %d, %Y')}

Please remit payment to:
{supplier}
{fk.street_address()}
{fk.city()}, {fk.state_abbr()} {fk.zipcode()}

Accounts Receivable
{supplier}"""
        
        return {
            'sender': f"billing@{supplier.lower().replace(' ', '').replace(',', '')}.com",
            'subject': f"Invoice {invoice_num} - PO {po_num}",
            'body': body,
            'label': 'legitimate',
            'invoice_number': invoice_num,
            'po_number': po_num,
            'supplier': supplier
        }

    def _generate_shipment_notification(self):
        supplier = rd.choice(self.suppliers)
        tracking_num = f"1Z{rd.randint(100000000000000, 999999999999999)}"
        po_num = f"PO-{rd.randint(50000, 99999)}"
        product = rd.choice(self.products)
        
        body = f"""Shipment Notification

Your order has been shipped!

PO Number: {po_num}
Product: {product}
Tracking Number: {tracking_num}
Carrier: UPS Ground
Estimated Delivery: {fk.date_between(start_date='+1d', end_date='+5d')}

Ship To:
Your Company Name
{fk.street_address()}
{fk.city()}, {fk.state_abbr()} {fk.zipcode()}

Track your shipment: https://ups.com/track?{tracking_num}

{supplier} Shipping Department"""
        
        return {
            'sender': f"shipping@{supplier.lower().replace(' ', '').replace(',', '')}.com",
            'subject': f"Shipment Confirmation - PO {po_num} - Tracking {tracking_num}",
            'body': body,
            'label': 'legitimate',
            'po_number': po_num,
            'tracking_number': tracking_num,
            'product': product
        }

    def _generate_quote_request(self):
        company = rd.choice(self.customer_companies)
        quote_num = f"QUO-{rd.randint(10000, 99999)}"
        products = rd.sample(self.products, rd.randint(1, 4))
        
        product_list = ""
        for i, product in enumerate(products, 1):
            qty = rd.randint(5, 100)
            product_list += f"{i}. {product} - Quantity: {qty}\n"
        
        body = f"""Quote Request

Company: {company}
Quote Number: {quote_num}
Date: {datetime.now().strftime('%B %d, %Y')}

We are requesting a quote for the following items:

{product_list}

Please include:
- Unit pricing
- Volume discounts
- Lead times
- Shipping costs
- Payment terms

Required delivery date: {fk.date_between(start_date='+30d', end_date='+90d')}

Contact Information:
{fk.name()}
{fk.email()}
{fk.phone_number()}

Thank you for your prompt response.

{company} Purchasing Department"""
        
        return {
            'sender': f"purchasing@{company.lower().replace(' ', '')}.com",
            'subject': f"Quote Request {quote_num} - Multiple Items",
            'body': body,
            'label': 'legitimate',
            'quote_number': quote_num,
            'company': company
        }

    def _generate_purchase_order(self):
        company = rd.choice(self.customer_companies)
        po_num = f"PO-{rd.randint(50000, 99999)}"
        product = rd.choice(self.products)
        quantity = rd.randint(10, 200)
        unit_price = rd.uniform(200, 3000)
        
        body = f"""Purchase Order

TO: Your Company
FROM: {company}

Purchase Order Number: {po_num}
Date: {datetime.now().strftime('%B %d, %Y')}

Item Details:
Product: {product}
Quantity: {quantity}
Unit Price: ${unit_price:.2f}
Total: ${quantity * unit_price:.2f}

Delivery Requirements:
Required Date: {fk.date_between(start_date='+14d', end_date='+45d')}
Delivery Address:
{fk.street_address()}
{fk.city()}, {fk.state_abbr()} {fk.zipcode()}

Terms: Net 30
Special Instructions: Handle with care, notify upon delivery

Authorized by: {fk.name()}
Title: Procurement Manager

{company}"""
        
        return {
            'sender': f"procurement@{company.lower().replace(' ', '')}.com",
            'subject': f"Purchase Order {po_num} - {product}",
            'body': body,
            'label': 'legitimate',
            'po_number': po_num,
            'product': product,
            'company': company
        }

    def _generate_delivery_confirmation(self):
        company = rd.choice(self.customer_companies)
        po_num = f"PO-{rd.randint(50000, 99999)}"
        delivery_num = f"DEL-{rd.randint(10000, 99999)}"
        product = rd.choice(self.products)
        
        body = f"""Delivery Confirmation

Delivery Number: {delivery_num}
PO Number: {po_num}
Customer: {company}

We confirm receipt of the following:

Product: {product}
Quantity Received: {rd.randint(10, 100)}
Condition: Good
Received Date: {datetime.now().strftime('%B %d, %Y')}
Received By: {fk.name()}

All items have been inspected and match the purchase order specifications.

Thank you for the timely delivery.

{company} Receiving Department
Phone: {fk.phone_number()}"""
        
        return {
            'sender': f"receiving@{company.lower().replace(' ', '')}.com",
            'subject': f"Delivery Confirmation {delivery_num} - PO {po_num}",
            'body': body,
            'label': 'legitimate',
            'delivery_number': delivery_num,
            'po_number': po_num,
            'product': product
        }

    def _generate_technical_inquiry(self):
        company = rd.choice(self.customer_companies)
        ticket_num = f"TIC-{rd.randint(10000, 99999)}"
        product = rd.choice(self.products)
        
        issues = [
            'installation specifications', 'compatibility questions', 'technical specifications',
            'maintenance procedures', 'troubleshooting guidance', 'operating parameters'
        ]
        issue = rd.choice(issues)
        
        body = f"""Technical Inquiry

Ticket Number: {ticket_num}
Company: {company}
Product: {product}
Issue: {issue}

Dear Technical Support,

We need assistance with {issue} for the {product}. 

Specific questions:
1. What are the recommended operating conditions?
2. Are there any compatibility issues we should be aware of?
3. What maintenance schedule do you recommend?

Current setup details:
- Installation date: {fk.date_between(start_date='-365d', end_date='-30d')}
- Operating environment: Industrial
- Usage: Daily operation

Please provide technical documentation or guidance.

Contact: {fk.name()}
Title: Technical Engineer
Phone: {fk.phone_number()}
Email: {fk.email()}

{company} Engineering Department"""
        
        return {
            'sender': f"engineering@{company.lower().replace(' ', '')}.com",
            'subject': f"Technical Inquiry {ticket_num} - {product}",
            'body': body,
            'label': 'legitimate',
            'ticket_number': ticket_num,
            'product': product,
            'company': company
        }

    def _generate_warranty_claim(self):
        company = rd.choice(self.customer_companies)
        warranty_num = f"WAR-{rd.randint(10000, 99999)}"
        original_order = f"ORD-{rd.randint(100000, 999999)}"
        product = rd.choice(self.products)
        
        body = f"""Warranty Claim

Warranty Claim Number: {warranty_num}
Original Order: {original_order}
Customer: {company}
Product: {product}
Serial Number: SN{rd.randint(100000, 999999)}

Purchase Date: {fk.date_between(start_date='-730d', end_date='-180d')}
Warranty Period: 2 years

Issue Description:
Product has failed prematurely. We believe this is covered under warranty.

Failure Details:
- Date of failure: {fk.date_between(start_date='-30d', end_date='-1d')}
- Symptoms: Intermittent operation, unusual noise
- Operating conditions: Normal industrial environment

We request either repair or replacement under warranty terms.

Supporting documentation attached.

Contact: {fk.name()}
Phone: {fk.phone_number()}

{company} Maintenance Department"""
        
        return {
            'sender': f"maintenance@{company.lower().replace(' ', '')}.com",
            'subject': f"Warranty Claim {warranty_num} - {product}",
            'body': body,
            'label': 'legitimate',
            'warranty_number': warranty_num,
            'order_number': original_order,
            'product': product
        }

    def _generate_parts_availability(self):
        supplier = rd.choice(self.suppliers)
        inquiry_num = f"INQ-{rd.randint(10000, 99999)}"
        products = rd.sample(self.products, rd.randint(2, 5))
        
        availability_info = ""
        for product in products:
            status = rd.choice(['In Stock', 'Back Order', 'Limited Stock', 'Discontinued'])
            lead_time = rd.choice(['Same Day', '1-2 days', '1 week', '2-3 weeks', 'TBD'])
            availability_info += f"- {product}: {status} - Lead Time: {lead_time}\n"
        
        body = f"""Parts Availability Update

Inquiry Number: {inquiry_num}
Date: {datetime.now().strftime('%B %d, %Y')}

Current availability for requested parts:

{availability_info}

Pricing:
Please contact sales for current pricing and volume discounts.

Notes:
- Prices subject to change without notice
- Lead times may vary based on demand
- Custom configurations available upon request

For orders, please reference inquiry number {inquiry_num}.

{supplier} Parts Department
Phone: {fk.phone_number()}
Email: parts@{supplier.lower().replace(' ', '').replace(',', '')}.com"""
        
        return {
            'sender': f"parts@{supplier.lower().replace(' ', '').replace(',', '')}.com",
            'subject': f"Parts Availability {inquiry_num}",
            'body': body,
            'label': 'legitimate',
            'inquiry_number': inquiry_num,
            'supplier': supplier
        }

    # Keep existing spam generators unchanged
    def _generate_phishing(self):
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

def gen_dtset(num_emails = 5000, spam_ratio = 0.7): 
    """complete data set"""
    generator = gen_email()
    emails = []

    num_spam  = int(num_emails*spam_ratio)
    num_legit = num_emails - num_spam

    print(f"generating {num_spam} spam emails...")
    for i in range(num_spam): 
        if i % 500 == 0: 
            print(f"  generated {i} spam emails...")
        emails.append(generator.spam_emails())

    print(f"generating {num_legit} legitimate emails...")
    for i in range(num_legit):
        if i % 200 == 0:
            print(f"  generated {i} legitimate emails...")
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
    print("company-specific synthetic data generation...")
    dt_set = gen_dtset(num_emails = 5000, spam_ratio = 0.7)
    df = pd.DataFrame(dt_set)
    df.to_csv('company_emails.csv', index = False)
    with open('company_emails.json', 'w') as f: 
        json.dump(dt_set, f, indent=2)
    
    print(f"total emails: {len(dt_set)}")
    print(f"spam: {len(df[df['label'] == 'spam'])}")
    print(f"legit: {len(df[df['label'] == 'legitimate'])}")
    print(f"spam ratio: {len(df[df['label'] == 'spam']) / len(df):.2%}")

    print("\nexamples ===================================")
    print("LEGITIMATE EMAIL:")
    legit_sample = df[df['label'] == 'legitimate'].iloc[0]
    print(f"From: {legit_sample['sender']}")
    print(f"Subject: {legit_sample['subject']}")
    print(f"Body: {legit_sample['body'][:300]}...")
    
    print("\nSPAM EMAIL:")
    spam_sample = df[df['label'] == 'spam'].iloc[0]
    print(f"From: {spam_sample['sender']}")
    print(f"Subject: {spam_sample['subject']}")
    print(f"Body: {spam_sample['body'][:200]}...")

    # Show company-specific data
    legit_with_orders = df[(df['label'] == 'legitimate') & (df.get('order_number', '').str.len() > 0)]
    if len(legit_with_orders) > 0:
        print(f"\nlegitimate emails with order numbers: {len(legit_with_orders)}")
        sample_with_order = legit_with_orders.iloc[0]
        print(f"example - order: {sample_with_order.get('order_number', 'N/A')}")
        print(f"product: {sample_with_order.get('product', 'N/A')}")

    print("\nprogram complete")