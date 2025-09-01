# pip install transformer --user
# pip install torch --user

from transformers import pipeline

spam_classifier = pipeline("text-classification", model="philschmid/distilbert-base-multilingual-cased-sentiment", return_all_scores=True)

texts = [
    "Congratulations! You've won a free ticket to Bahamas. Click here to claim your prize.",
    "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
    "Limited time offer! Buy one get one free on all items in our store.",
    "Meeting rescheduled to next Monday. Please confirm your availability."
]

label_mapping = {'negative': 'SPAM', 'positive': 'NOT SPAM', 'neutral': 'NOT SPAM'}
results = spam_classifier(texts)
for text, result in zip(texts, results):
    best = max(result, key=lambda x: x['score'])
    label = label_mapping.get(best['label'].lower(), 'UNKNOWN')
    print(f"Text: {text}\nLabel: {label}, Confidence Score: {best['score']:.4f}\n")