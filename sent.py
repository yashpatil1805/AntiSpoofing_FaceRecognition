import requests

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer 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"}

data = {"inputs": "I love using Hugging Face!"}
response = requests.post(API_URL, headers=headers, json=data)
print(response.json())
