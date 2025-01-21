!pip install openai==1.0.0

import logging
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import openai
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# 1. Initialize the OpenAI Client
# ---------------------------
client = OpenAI(api_key='API_Key')  # You can configure the API key as needed


# Configure logging for error handling
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Load the dataset
df = pd.read_csv('/content/Reddit Sentiment Analysis.csv')

# ---------------------------
# 1. Classical EDA
# ---------------------------

# Check dataset structure
print(df.head())
print(df.info())
print(df.describe())

# Subreddit distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='subreddit', order=df['subreddit'].value_counts().index)
plt.title('Distribution of Comments Across Subreddits')
plt.xticks(rotation=45)
plt.show()

# Comment length distribution
df['comment_length'] = df['comment_body'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df['comment_length'], kde=True, bins=30, color='blue')
plt.title('Distribution of Comment Lengths')
plt.xlabel('Comment Length')
plt.ylabel('Frequency')
plt.show()

# Word count in comments
df['word_count'] = df['comment_body'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], kde=True, bins=30, color='green')
plt.title('Distribution of Word Counts in Comments')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()



# ---------------------------
# 2. Define the Sentiment Analysis Function
# ---------------------------
def get_sentiment(comment):
    """
    Uses OpenAI's GPT model to generate a sentiment score for a comment.
    Compatible with openai>=1.0.0.
    """
    try:
        # Correct API call to ChatCompletion.create with new client
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or you can use gpt-4 if preferred
            messages=[
                {"role": "system", "content": "You are a sentiment analysis tool."},
                {"role": "user", "content": f"Analyze the sentiment of this comment on a scale of -1 (most negative) to 1 (most positive): {comment}"}
            ],
            temperature=0.5
        )

        # Extract sentiment score from the response
        # The response is structured as an object, not a dictionary, so we access attributes
        sentiment_text = response.choices[0].message['content'].strip()

        # Convert the sentiment score to a float
        score = float(sentiment_text)
        return score
    except Exception as e:
        logging.error(f"Error for comment: {comment} | Exception: {e}")
        return None

# ---------------------------
# 3. Batch Processing
# ---------------------------
def process_in_batches(dataframe, batch_size=10):
    """
    Processes comments in batches to optimize API usage and handles errors.
    """
    scores = []
    for i in tqdm(range(0, len(dataframe), batch_size), desc="Processing Batches"):
        batch = dataframe['comment_body'].iloc[i:i+batch_size]
        for comment in batch:
            scores.append(get_sentiment(comment))
    return scores

# 4. Testing the Function
# ---------------------------

# Test with a few comments
test_comments = ["This is amazing!!!", "I hate this!", "It's okay, not great, not bad."]
for test in test_comments:
    print(f"Comment: {test} | Sentiment Score: {get_sentiment(test)}")

# ---------------------------
# 5.Apply to Full Dataset
# ---------------------------

# Apply sentiment scoring using batch processing
df['score'] = process_in_batches(df, batch_size=20)

# ---------------------------
# 6.Error Handling & Fallbacks
# ---------------------------

# Replace Non or NaN scores with a default value (e.g. 0 for neutral)
df['score'] = df['score'].fillna(0)

# ---------------------------
# 7.Save the Results
# ---------------------------

# Save the updated dataset
df.to_csv('Reddit_Sentiment_Analysis_Updated.csv', index=False)
print("Updated dataset saved successfully!")