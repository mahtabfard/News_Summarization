from keras_preprocessing.sequence import pad_sequences
from preprocessing import preprocess_text, tokenize_sentences

def summarize_article(article, tokenizer, model, max_length):
    """Generate a summary of the given article."""
    sentences = tokenize_sentences(article)
    processed_sentences = [preprocess_text(sent) for sent in sentences]
    sequences = tokenizer.texts_to_sequences(processed_sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Predict importance scores
    scores = model.predict(padded_sequences)
    summary = ' '.join([
        sentence for i, sentence in enumerate(sentences) if scores[i] > 0.5
    ])
    return summary
