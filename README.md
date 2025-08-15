Hi everyone!

This README is a quick explainer of this "Spam Rescue" Machine Learning model that I've created. 

This specific model was born out of the necessity of erasing a very simple problem that many people may face; in particular those running a mid-sized email queue. 

Often, our google GMAIL pipeline may discard certain messages or emails to the SPAM box where they are very difficult to retrieve and may be quite cumbersome to process this manually. 

Just finding a "Legitimate" email out of hundreds of spam emails, without an extension on your browser or KPI (which may not be private and may not be trust-worthy), may be cumbersome. 

Our model, a "Spam Rescue bot", is, in essence, a self-learning machine that can, with the correct features that we assign it, label an message Legitimate, resend it to the main Inbox and mark it as "not spam". 

It's important to note that this model learns from the data and has a training based on the information that it's given to the the model. 

PRECISSION OVER ACCURACY:
Given the small sample of data and the binary nature of the model, our team (me and my cat) decided to prioritize precission over accuracy.

What this means is that we'd like for the model not to process too many "false-positives" in our confussion matrix but rather a proportionate amount of "true-positives". 

In the folder "Data", you will find some dumb, synthetic data, that was created with the intention of showing off the model. As of now (August 14, 2025), there's no been no training with real data as the amount of data required for the model to train itself is limited and thus it's not worthy as the precission will always be 100%. 

1. As for how this model will get its data, it is quite simple. We will simply integrate any GMAIL API, via the Google Cloud O2Auth, with a JSON token. 

2. Once the data is loaded and validated, see src/val.py the user must, with their knowledge of the features, assign the corresponding features and weights so the model can start learning.

3. Once the model starts learning, the user may feel free to start using the GMAIL integration and both forwarding the email to either GMAIL, DIXA or whatever API they are using to receive and send messages. 

Note: This project is particularly useful to anyone manually checking their spam inbox and having to manually drage the spam out of the box. It may not be useful to the common/standard personal use, but may be useful to medium sized business who would like to avoid the abhorrent task of processing the manual rescue of their spam. 


For any contacts, email me at: juliandavid.alfonso.gomez@gmail.com


Last Update: August 14, 2025