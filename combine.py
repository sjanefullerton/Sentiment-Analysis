import pandas as pd
import os
import string
import re

# List of file names
data = [
    "2hot_no_neutral_helene_comments.csv",
    "2new_no_neutral_helene_comments.csv",
    "2top_no_neutral_helene_comments.csv",
    "2hot_no_neutral_helene_posts.csv",
    "2new_no_neutral_helene_posts.csv",
    "2top_no_neutral_helene_posts.csv",
    "2hot_no_neutral_helene_titles.csv",
    "2new_no_neutral_helene_titles.csv",
    "2top_no_neutral_helene_titles.csv",
    "2combined_comments.csv",
    "2combined_posts.csv",
    "2combined_titles.csv",
    "all_combined_comments.csv",
    "all_combined_posts.csv",
    "all_combined_titles.csv",
    "deduplicated_comments.csv",
    "deduplicated_posts.csv",
    "deduplicated_titles.csv",
    "filtered_comments.csv",
    "filtered_helene_comments.csv",
    "filtered_helene_titles.csv",
    "filtered_posts.csv",
    "filtered_titles.csv",
]

# Load and combine datasets, skipping empty or missing files
dataframes = []
for file in data:
    file_path = os.path.join("data", "processed", file)  # Construct file path
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:  # Check existence and non-emptiness
        try:
            df = pd.read_csv(file_path)  # Read the CSV file
            dataframes.append(df)  # Add the DataFrame to the list
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"Skipping empty or missing file: {file_path}")

# Combine if there are valid dataframes
if dataframes:
    combined_comments = pd.concat(dataframes, ignore_index=True)
    combined_comments.to_csv("data/processed/combined.csv", index=False)
    print(combined_comments.head())
else:
    print("No valid files to combine.")



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

file_path = "data/processed/combined.csv"
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

# keep only 'cleaned_text' variable
df_cleaned = comments.drop(columns=['Text', 'Sentiment', 'Type', 'Post URL', 'Created', 'Comment Author', 'Author'])

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv("combined_clean.csv", index=False)