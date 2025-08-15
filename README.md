# Spam Rescue – Machine Learning Model

Hi everyone!  

This README is a quick overview of the **"Spam Rescue"** Machine Learning model I created.  

This project was born out of a common annoyance—especially for those managing medium-sized email queues: **legitimate emails sometimes get thrown into Gmail's Spam folder**. Once they’re in there, retrieving them can be tedious and frustrating.  

Finding one legitimate message among hundreds of spam entries—without using a potentially privacy-invasive browser extension or third-party KPI tool—can be a real headache.  

---

## What is Spam Rescue?

**Spam Rescue Bot** is a self-learning machine learning model designed to identify legitimate emails in your spam folder, move them back to your inbox, and mark them as “Not Spam.”  

With the right features and training data, the model can automatically rescue valuable messages while minimizing the number of false positives.  

> **Note:** This model learns from the data you provide. Its accuracy and usefulness depend heavily on the quality and relevance of your training set.

---

## Precision Over Accuracy

Given our small dataset and the binary nature of the task, **I (and my cat)** decided to prioritize **precision** over accuracy.  

This means the model is tuned to minimize false positives—ensuring that we don’t mistakenly move actual spam into the inbox—while still catching true positives effectively.  

---

## Current Data

Inside the `Data` folder, you’ll find synthetic sample data created purely to demonstrate the model.  

As of **August 14, 2025**, the model has not been trained on real-world data. With such a small dataset, real-data training would result in artificially perfect (100%) precision, which isn’t meaningful for performance evaluation.  

---

## How It Works

1. **Data Acquisition** – Integrate Gmail API via Google Cloud OAuth, using a JSON token for authentication.  
2. **Data Preparation** – Once emails are loaded and validated (see `src/val.py`), assign features and weights based on domain knowledge so the model can start learning.  
3. **Deployment** – Use the Gmail integration to automatically forward rescued emails to your inbox or another platform (Gmail, Dixa, or any other API-supported service).  

---

## Who Is This For?

Spam Rescue is most useful for:  
- Medium-sized businesses managing a shared inbox  
- Teams that frequently lose important client or vendor emails to spam  
- Anyone who wants to **reduce the time spent manually checking and dragging emails out of the spam folder**

For personal, low-volume email users, the benefit may be minimal.  

---

**Contact:** juliandavid.alfonso.gomez@gmail.com  
**Last Update:** August 14, 2025  
