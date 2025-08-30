# pip install transformer --user
# pip install torch --user

from transformers import AutoModel, AutoTokenizer

#Download model and tokenizer
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use the model and tokenizer
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print("Output shape: ", outputs.last_hidden_state)