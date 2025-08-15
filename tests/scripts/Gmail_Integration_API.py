import os
import base64
import email
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GmailIntegratrion: 
    """Gmail integration for spam rescue"""
    #GMAIL SCOPES - to modify emails and send
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.send'
    ]
    def __init__(self, credentials_path = 'credentials.json', token_path='token.json'):
        """
        Initialize GMAIL API
        Args:
            credentials_path: Path to OAuth2 credentials JSON file
            token_path: Path to store/load authentication token: 
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.setup_logging()
    def setup_logging(self):
        """Setup logging for GMAIL ops"""

        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gmail_spam_rescue.log'),
                logging.StreamHandler()
            ]
        )
        self.logger=logging.getLogger(__name__)
    def authenticate(self):
        """
        Authenticate with GMAIL - API using Oauth2
        Setup Instructions:
        1. Go to Google Cloud Console
        2. Enable Gmail API
        3. Create OAuth2 credentials
        4. Download credentials.json
        """
        creds =  None

        if os.path.exists(self.token_path):
            creds=Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        if not creds or not creds.valid: 
            if creds and creds.expired and creds.refresh_token: 
                creds.refresh_token(Request())
            else: 
                if not os.path.exists(self.credentials_path): 
                    raise FileNotFoundError(
                        f"Creds not found for: {self.credentials_path}\n"
                        "Please download OAuth2 credentials from Google Cloud Console"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json)
        self.service = build('gmail', 'v1', credentials=creds)
        self.logger.info("Gmail API authentication successful")
        return True
    def fetch_spam_emails(self, max_results=100, days_back=7): 
        """
        Fetch emails from spam folder for analysis
        
        Args:
            max_results: Maximum number of emails to fetch
            days_back: How many days back to search
            
        Returns:
            List of email dictionaries with metadata and content
        """
        if not self.service:
            raise ValueError("Gmail service not authenticated. Call authenticate() first.")
        try: 
            start_date = datetime.now() - timedelta(days=days_back)
            date_query = start_date.strftime('%Y/%m/%d')

            #search spam folder
            query = f'in:spam after: {date_query}'
            self.logger.info(f"Fetching spam emails with query: {query}")

            #getting messasges IDs
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults = max_results
            ).execute()
            
            messages = results.get('messages', [])
            self.logger.info(f"Found {len(messages)} spam emails")

            #fetch email details
            emails = []
            for i, message in enumerate(messages): 
                try: 
                    #only 250 units per user per 100 seconds
                    if i % 10 == 0 and i>0: 
                        time.sleep(1) # pauses every 10 requests
                    email_data = self._fetch_email_content(message['id'])
                    if email_data: 
                        email.append(email_data)
                except HttpError as error: 
                    self.logger.error(f" Gmail API error: {error}")
                    continue
            self.logger.info(f"Succesfully fetched {len(emails)} complete emails")
            return emails
        except HttpError as error:
            self.logger.error(f"Gmail API error: {error}")
            raise
    def _fetch_email_content(self, message_id):
        """
        Fetch complete email content and metadata
        
        Args:
            message_id: Gmail message ID
            
        Returns:
            Dictionary with email data
        """
        try:
            #Full message
            message = self.service.users().messages().get(
                userID='me', 
                id=message_id, 
                format='full'
            ).execute()

            #headers
            headers = {h['name']:h['value'] for h in message['payload']['headers']}

            #body content
            body = self._extract_email_body(message['payload'])

            #email data structure compatible with existing format
            email_data = {
                'id':message_id, 
                'subject': headers.get('Subject', 'No Subject'),
                'sender': headers.get('From', 'Unknown'),
                'recipient': headers.get('To', 'Unknown'), 
                'date': headers.get('Date', ''),
                'body': body,
                'labels': message.get('labelIds', []),
                'thread_id': message.get('threadId'),
                'timestamp': datetime.now().isoformat(),
                'label': 'spam'  
            }
        except Exception as e: 
            self.logger.error(f"error fetching email content for {message_id}: {str(e)}")
            return None
    def _extract_email_body(self, payload):
        """Extract email body from GMAIL API payload"""
        body =""

        if 'parts' in payload: 
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain': 
                    data = part['body']['data']
                    body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ig')
                elif part['mimeType'] == 'text/html':
                    # if no plain text, fall back to HTML
                    if not body: 
                        data = part['body']['data']
                        body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        else: 
            # single part email
            if payload['mimeType'] == 'text/plain':
                data = payload['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif payload['mimeType'] == 'text/html':
                data = payload['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return body.strip()
    def move_to_inbox(self, message_ids: List[str]):
        """
        Move rescued emails from spam to inbox
        
        Args:
            message_ids: List of Gmail message IDs to rescue
            
        Returns:
            Dict with success/failure counts
        """
        if not self.service:
            raise ValueError("Gmail service not authenticated")
        success_count=0
        failure_count=0

        for message_id in message_ids: 
            try: 
                #this removes the spam label and add the inbox label
                self.service.users().messages().modify(
                    userID='me',
                    id=message_id,
                    body={
                        'removeLabelIDs':['SPAM'],
                        'addLabelIds':['INBOX']
                    }
                ).execute()
                success_count+=1
                self.logger.info(f"Successfully rescued email: {message_id}")

                #rate limiting
                time.sleep(0.1)
            except HttpError as error:
                failure_count += 1
                self.logger.error(f"Failed to rescue email {message_id}: {error}")
        result = {
            'total_processed': len(message_ids),
            'successful_rescues':success_count, 
            'failed_rescues': failure_count
        }

        self.logger.info(f"Rescue operation complete: {result}")
        return result
    def forward_to_dixa(self, email_data: Dict, dixa_email: str):
        """
        Forward rescued email to Dixa platform
        
        Args:
            email_data: Email data dictionary
            dixa_email: Dixa platform email address
            
        Returns:
            Success status
        """
        try:
            # Create forwarded message
            forward_subject = f"[RESCUED FROM SPAM] {email_data['subject']}"
            forward_body = f"""
This email was automatically rescued from the spam folder by the ML spam rescue system.

Original Details:
- From: {email_data['sender']}
- To: {email_data['recipient']}
- Date: {email_data['date']}
- Subject: {email_data['subject']}

--- Original Message ---
{email_data['body']}
            """
            
            # Create message
            message = {
                'raw': base64.urlsafe_b64encode(
                    f"To: {dixa_email}\r\n"
                    f"Subject: {forward_subject}\r\n"
                    f"\r\n{forward_body}".encode('utf-8')
                ).decode('utf-8')
            }
            
            # Send message
            self.service.users().messages().send(
                userId='me',
                body=message
            ).execute()
            
            self.logger.info(f"Successfully forwarded email to Dixa: {email_data['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to forward email to Dixa: {str(e)}")
            return False
    def create_rescue_report(self, rescued_emails: List[Dict], save_path: str = None):
        """
        Create detailed report of rescued emails
        
        Args:
            rescued_emails: List of rescued email data
            save_path: Optional path to save report
            
        Returns:
            Report dictionary
        """
        if not rescued_emails:
            return {'message': 'No emails were rescued'}
        
        # Analyze rescued emails
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_rescued': len(rescued_emails),
            'summary': {
                'subjects': [email['subject'][:50] + '...' if len(email['subject']) > 50 
                           else email['subject'] for email in rescued_emails[:10]],
                'senders': list(set([email['sender'] for email in rescued_emails]))[:10],
                'date_range': {
                    'earliest': min([email.get('date', '') for email in rescued_emails]),
                    'latest': max([email.get('date', '') for email in rescued_emails])
                }
            }
        }
        
        # Save report if path provided
        if save_path:
            df = pd.DataFrame(rescued_emails)
            df.to_csv(save_path, index=False)
            report['saved_to'] = save_path
        
        self.logger.info(f"Rescue report created: {report['total_rescued']} emails rescued")
        return report

def main():
    """Example usage of Gmail spam rescue system"""
    
    ## Initialize Gmail client
    gmail_client = GmailIntegratrion()
    
    ## Authenticate (will open browser first time)
    gmail_client.authenticate()
    
    ## Fetch recent spam emails
    spam_emails = gmail_client.fetch_spam_emails(max_results=50, days_back=7)
    
    print(f"Fetched {len(spam_emails)} emails from spam folder")
    
    ## Convert to DataFrame for ML processing
    df_spam = pd.DataFrame(spam_emails)
    
    ## Here we would run our ML models to identify rescue candidates
    # rescue_candidates = SpamRescueRandomForest.predict_with_confidence(df_spam, threshold=0.85)
    
    ## Example: rescue emails (replace with our ML logic)
    # rescue_message_ids = ['message_id_1', 'message_id_2']
    # gmail_client.move_to_inbox(rescue_message_ids)
    
    print("Gmail API integration ready for ML pipeline")


if __name__ == "__main__":
    main()


