import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
import faiss
from phe import paillier  

public_key, private_key = paillier.generate_paillier_keypair()

df = pd.read_csv("fraud_data.csv")

def encrypt_transaction(row):
    row["amount"] = public_key.encrypt(float(row["amount"]))  
    return row

df = df.apply(encrypt_transaction, axis=1)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode_text(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embedding = model(**tokens).last_hidden_state.mean(dim=1)  
    return embedding.numpy()

def decrypt_value(encrypted_value):
    return private_key.decrypt(encrypted_value)  

transaction_texts = df.apply(
    lambda row: f"User: {row['user_id']}, Merchant: {row['merchant_id']}, Amount: {decrypt_value(row['amount'])}, Type: {row['transaction_type']}", 
    axis=1
)

embeddings = np.vstack([encode_text(txt) for txt in transaction_texts])

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_similar(query_text, top_k=5):
    query_embedding = encode_text(query_text)
    _, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]

t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

def generate_fraud_analysis(transaction_details, retrieved_cases):
    prompt = f"Analyze this transaction: {transaction_details}.\nSimilar cases:\n"
    fraud_count = 0

    for _, case in retrieved_cases.iterrows():
        label = case['fraud_label']
        prompt += f"- {case['transaction_type']} | Amount: {decrypt_value(case['amount'])} | Label: {label}\n"
        if label == "fraud":
            fraud_count += 1

    fraud_probability = fraud_count / len(retrieved_cases)  

    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    output = t5_model.generate(**inputs, max_length=100)
    
    analysis = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return analysis, fraud_probability

def detect_fraud(user_id, merchant_id, amount, transaction_type):
    query_transaction = f"User: {user_id}, Merchant: {merchant_id}, Amount: {decrypt_value(public_key.encrypt(amount))}, Type: {transaction_type}"
    retrieved_cases = retrieve_similar(query_transaction)
    analysis, fraud_prob = generate_fraud_analysis(query_transaction, retrieved_cases)

    return {
        "analysis": analysis,
        "fraud_probability": fraud_prob
    }

query_result = detect_fraud(102, 5, 500, "Online Payment")
print("Fraud Analysis:", query_result["analysis"])
print("Fraud Probability:", query_result["fraud_probability"])
