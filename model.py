import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense

def build_model(vocab_size, embedding_dim, max_length):
    """Construct a Bidirectional LSTM model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dense(1, activation='sigmoid')  # Binary classification for sentence importance
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, train_labels, val_data, val_labels, epochs=10, batch_size=64):
    """Train the BiLSTM model."""
    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_labels)
    )
    return history
