from preprocessing import preprocess_text, tokenize_sentences
from model import build_model, train_model
from summarizer import summarize_article
from keras.preprocessing.text import Tokenizer

# Example Parameters
vocab_size = 20000
embedding_dim = 100
max_length = 50

# Load data (mock data for illustration)
train_data = ...
train_labels = ...
val_data = ...
val_labels = ...
test_article = "Breaking news: A major event has occurred. More details to follow."

# Step 1: Build and Train Model
model = build_model(vocab_size, embedding_dim, max_length)
history = train_model(model, train_data, train_labels, val_data, val_labels)

# Step 2: Tokenizer Setup
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_data)  # Fit tokenizer on the training data

# Step 3: Generate Summary
summary = summarize_article(test_article, tokenizer, model, max_length)
print("Original Article:", test_article)
print("Generated Summary:", summary)
