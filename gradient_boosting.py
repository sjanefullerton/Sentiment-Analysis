import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the balanced dataset
df = pd.read_csv("new2_balanced_dataset.csv")

# Convert the 'Sentiment' column into numerical labels
df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0})

# Data cleaning and text preprocessing
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])  # text_data is a list of text samples
y = df['Sentiment']  # Sentiment labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# XGBoost Classifier with Hyperparameter Tuning
model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=15, min_child_weight=5, gamma=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Print classification report for better insight into the model's performance
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
