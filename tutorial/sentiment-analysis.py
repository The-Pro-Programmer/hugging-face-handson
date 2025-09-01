# Please select appropriate GPU while running this notebook
# Confidence score indicates the model's certainty about its prediction.
# Value of confidence score ranges from 0 to 1, with 1 being the highest certainty.
# Higher confidence scores suggest that the model is more certain about its prediction, while lower scores indicate less certainty.
# Values of sentiment labels can be either POSITIVE or NEGATIVE .

# pip install transformer --user
# pip install torch --user

from transformers import pipeline

print("Loading model and tokenizer...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "I love using Hugging Face Transformers!",
    "This is the worst movie I've ever seen.",
    "The food was okay, nothing special.",
    "Absolutely fantastic experience, would recommend to everyone!"
]

print("Analyzing sentiments...")
results = sentiment_analyzer(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}\nSentiment: {result['label']}, Confidence Score: {result['score']:.4f}\n")