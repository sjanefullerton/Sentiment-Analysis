import pandas as pd
import re
import string

# Function to clean the text (convert to lowercase, remove digits, punctuation, emojis, and links)
def clean_text(text):
    # Ensure the input is a string before processing
    if not isinstance(text, str):
        return ''  # Return an empty string if the text is not a valid string (e.g., NaN or float)

    # Convert to lowercase
    text = text.lower()

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = re.sub(rf"[{string.punctuation}]", "", text)

    # Remove links (URLs)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove emojis using a more comprehensive regex
    emoji_pattern = re.compile(
        "["  # Emoticons
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Remove emojis

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Removes non-ASCII characters

    return text

# Load the dataset
file_path = "new2.csv"
comments = pd.read_csv(file_path)

# Check if 'Text' column exists
if 'Text' not in comments.columns:
    print("Error: 'Text' column not found in the dataset.")
else:
    # 1. Remove duplicate rows based on the 'Text' column
    comments = comments.drop_duplicates(subset='Text', keep='first')

    # 2. Remove rows where the 'Text' column contains "I am a bot" (case-insensitive)
    comments = comments[~comments['Text'].str.contains(r'i am a bot', case=False, na=False)]

    # 3. Clean the text in the 'Text' column using the clean_text function
    comments['cleaned_text'] = comments['Text'].apply(clean_text)

    # 4. Check the cleaned text for any remaining unwanted characters
    # If needed, you can print a few rows to verify.
    print("Sample of cleaned text:")
    print(comments[['Text', 'cleaned_text']].head())

# Drop the original 'Text' column and keep 'cleaned_text' and 'Sentiment'
df_cleaned = comments.drop(columns=['Text'])

# Ensure only 'cleaned_text' and 'Sentiment' columns are kept
df_cleaned = df_cleaned[['cleaned_text', 'Sentiment']]

df_cleaned['Sentiment'] = df_cleaned['Sentiment'].str.lower()

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv("new2_clean.csv", index=False)

print(df_cleaned)
print(df_cleaned.describe())
print("Original 'Text' column removed. Processed file saved as 'new_comments.csv'.")


df_binary = df_cleaned[df_cleaned['Sentiment'] != 'neutral']
df_binary = df_binary[df_cleaned['Sentiment'] != '']
df_binary.to_csv("new2_clean_noneutral.csv", index=False)
print(df_binary['Sentiment'].value_counts())
print(df_binary.describe())

# Split the dataset into negative and positive classes
negative_df = df_binary[df_binary['Sentiment'] == 'negative']
positive_df = df_binary[df_binary['Sentiment'] == 'positive']

# Undersample the negative class to match the number of positive samples
negative_undersampled = negative_df.sample(n=len(positive_df), random_state=42)

# Combine the undersampled negative class with the positive class
balanced_df = pd.concat([negative_undersampled, positive_df])

# Check the new distribution
print(balanced_df['Sentiment'].value_counts())

# Save the balanced dataset
balanced_df.to_csv("new2_balanced_dataset.csv", index=False)