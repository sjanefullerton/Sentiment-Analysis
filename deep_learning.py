import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("new2_clean_noneutral.csv")

# Convert sentiment to binary labels
df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0})

# Prepare features and target
X = df['cleaned_text']
y = df['Sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text data to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to ensure uniform input size
X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=200)
X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=200)

# Check alignment of the data
print(f"X_train_pad shape: {X_train_pad.shape}")
print(f"y_train shape: {y_train.shape}")

# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    Dense(1, activation='sigmoid')
])


# Compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
optimizer = Adam(learning_rate=0.001)  # Experiment with lower learning rates
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Class weights to handle imbalanced classes
class_weights = {0: 1., 1: 10.}

# Train the model
model.fit(X_train_pad, y_train.values, epochs=30, batch_size=64, class_weight=class_weights, validation_data=(X_test_pad, y_test.values))

# Make predictions and evaluate
y_pred = (model.predict(X_test_pad) > 0.3).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)