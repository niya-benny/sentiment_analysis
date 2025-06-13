import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import joblib
import os
import string

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

import nltk

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

#===========================================================
# Process 1: Load the dataset
path = "coronavirus_reddit_clean_comments.csv"
data = pd.read_csv(path)

print(data.head())
print(data.columns)
print(data.shape)

#==============================================
# Process 2: Check for missing values
missing_values = data.isnull().sum()

#======================================================================
# Process 3: Save the missing values summary to a text file for inspection
with open('P3.missing_values_summary.txt', 'w') as f:
    f.write("Missing Values Summary:\n")
    f.write(missing_values.to_string())

print(f"Dataset loaded with shape: {data.shape}")
print(f"Procces 3 saved to: 'P3.missing_values_summary.txt'")

#==================================================================
# Process 4: Check for duplicate rows
duplicate_rows = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# Remove duplicate comments, keep the first occurrence
print(f"Shape before removing duplicate comments: {data.shape}")
data = data.drop_duplicates(subset='comment', keep='first')
print(f"Shape after removing duplicate comments: {data.shape}")

# Save dataset after removing duplicate comments
data.to_csv("P4.dropping_duplicates.csv", index=False)
print("Saved dataset after removing duplicate comments to 'P4.dropping_duplicates.csv'")

#==============================================================
#Process 5: Convert all text in 'comment' column to title case
data = pd.read_csv("P4.dropping_duplicates.csv")
data['comment'] = data['comment'].astype(str).str.title()
data.to_csv("P5.title_case.csv", index=False)
print("Process 5: Saved to 'P5.title_case.csv'")

#========================================================
# Process 6: Replace the missing values
data = pd.read_csv("P5.title_case.csv")
data['comment'] = data['comment'].fillna("No comment provided")
data.to_csv("P6.missing_value_replacement.csv", index=False)
print("Process 6: Saved to 'P6.missing_value_replacement.csv'")

#============================================================
#Process 7:Check if there are any columns with missing values
data = pd.read_csv("P6.missing_value_replacement.csv")
missing_after = data.isnull().sum()
print("Process 7: Missing values after replacement:\n", missing_after)

#======================================================================
#Process 8: Define a function to check and remove punctuation

data = pd.read_csv("P6.missing_value_replacement.csv")

# Function to check for punctuation
def contains_punctuation(text):
    return any(char in string.punctuation for char in str(text))

# Function to remove punctuation
def remove_punctuation(text):
    return str(text).translate(str.maketrans('', '', string.punctuation))

df_punct = data[['comment']].copy()
df_punct['contains_punctuation'] = df_punct['comment'].apply(contains_punctuation)
df_punct['no_punctuation'] = df_punct['comment'].progress_apply(remove_punctuation)

df_punct.to_csv('P8.removed_punctuation.csv', index=False)
print("Process 8: Saved to 'P8.removed_punctuation.csv'")
#===============================================================


#============================================================
#Process 9: Define a function to remove stop words
stop_words = set(stopwords.words('english'))

def safe_remove_stop_words(text):
    try:
        tokens = word_tokenize(str(text))
        filtered = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered)
    except Exception as e:
        print(f"Error processing: {text[:50]} -> {e}")
        return "ERROR"

df_no_stop = df_punct[['no_punctuation']].copy()
df_no_stop['no_stop_words'] = df_no_stop['no_punctuation'].progress_apply(safe_remove_stop_words)
df_no_stop.to_csv('P9.remove_stop_words.csv', index=False)
print("Process 9: Saved to 'P9.remove_stop_words.csv'")
#==============================================================

#===========================================================
#Process 10: Define a function for Lemmatization

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(str(text))
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

df_lemma = df_no_stop[['no_stop_words']].copy()
df_lemma['lemmatized'] = df_lemma['no_stop_words'].progress_apply(lemmatize_text)
df_lemma.to_csv('P10.lemmatized.csv', index=False)
print("Process 10: Saved to 'P10.lemmatized.csv'")
#=====================================================================


#==============================================================
#Process 11: Define a function for Stemmer
stemmer = PorterStemmer()

def stem_text(text):
    tokens = word_tokenize(str(text))
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

df_stem = df_lemma[['lemmatized']].copy()
df_stem['stemmed'] = df_stem['lemmatized'].progress_apply(stem_text)
df_stem.to_csv('P11.stemmed.csv', index=False)
print("Process 11: Saved to 'P11.stemmed.csv'")
#==============================================================


#==========================================================
# Process 12: Function to convert text to lowercase
def to_lowercase(text):
    return str(text).lower()

df_lower = df_stem[['stemmed']].copy()
df_lower['lowercase'] = df_lower['stemmed'].progress_apply(to_lowercase)
df_lower.to_csv('P12.lowercase.csv', index=False)
print("Process 12: Saved to 'P12.lowercase.csv'")
#========================================================================

#=================================================================
# Process 13: Convert all applicable columns to appropriate data types
data = pd.read_csv("P12.lowercase.csv")

def convert_data_types(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].nunique()
            if unique_vals < len(df) * 0.1:
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype('string')
        elif df[col].dtype in ['int64', 'float64']:
            continue  # Already numeric
    return df

data = convert_data_types(data)
data.to_csv("P13.after_type_conversion.csv", index=False)
print("Process 13: Saved dataset after type conversion to 'P13.after_type_conversion.csv'")
#================================================================


#=======================================================
# # Process 14: Visualize data for outliers and anomalies from csv
# data = pd.read_csv("P13.after_type_conversion.csv")
#
# # Optional: Add a column for text length
# data['text_length'] = data['lowercase'].astype(str).apply(len)
#
# # Detect numeric columns
# numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
#
# # Create boxplots for numeric columns to detect outliers
#
# for col in numeric_cols:
#     plt.figure(figsize=(10, 4))
#     sns.boxplot(x=data[col])
#     plt.title(f"Boxplot for {col}")
#     plt.savefig(f"boxplot_{col}.png")
#     plt.show()
#     print(f" Saved boxplot for {col} as 'P14.boxplot_{col}.png'")
#===================================================================

#=======================================================
# Process 14: Visualize data for outliers and anomalies from csv

data = pd.read_csv("P13.after_type_conversion.csv")

# Add a column for text length (optional)
data['text_length'] = data['lowercase'].astype(str).apply(len)

# Detect numeric columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Function to remove outliers using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        original_len = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"Removed outliers from '{col}': {original_len - len(df)} rows removed")
    return df

# Visualize original data
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data[col])
    plt.title(f"Original Boxplot for {col}")
    plt.savefig(f"P14.original_boxplot_{col}.png")
    plt.show()

# Remove outliers
data_cleaned = remove_outliers_iqr(data, numeric_cols)

# Visualize cleaned data
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data_cleaned[col])
    plt.title(f"Boxplot for {col} After Outlier Removal")
    plt.savefig(f"P14.cleaned_boxplot_{col}.png")
    plt.show()

# Optional: Save cleaned data
data_cleaned.to_csv("P14.outlier_removed.csv", index=False)
print("Saved cleaned dataset as 'P14.outlier_removed.csv'")
#=======================================================


#=============================================================
# Process 15 : Assuming the text data is in a column named 'comment' and the labels in 'sentiment'
df = pd.read_csv("P14.outlier_removed.csv")

df = df.rename(columns={"lowercase": "comment"})  # assuming 'lowercase' is the cleaned text

# Save updated version
df.to_csv("P15.sentiment_base.csv", index=False)
print("Process 15: saved to 'P15.sentiment_base.csv'")
#=============================================================


#=========================================================
# Process 16 : Sentiment Analysis - Manual Labeling

df = pd.read_csv("P15.sentiment_base.csv")
df['sentiment'] = ""  # Empty column to be manually filled

# Save to CSV so you can label manually in Excel/Editor
df.head(50).to_csv("P16.manual_label_sample.csv", index=False)
print("Process 16: Saved sample to 'P16.manual_label_sample.csv' for manual sentiment labeling")
#==================================================================


#=====================================================================
# Process 17 : Sentiment Analysis - Auto Labeling using TextBlob
df = pd.read_csv("P15.sentiment_base.csv")

# Define function to get sentiment polarity
def get_sentiment(text):
    try:
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            return 'positive'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    except:
        return 'error'

# Apply function to cleaned comments
df['auto_sentiment'] = df['comment'].progress_apply(get_sentiment)

# Save labeled data
df.to_csv("P17.auto_labeled_sentiment.csv", index=False)
print("Process 17: Saved to 'P17.auto_labeled_sentiment.csv'")

#==============================================================


#========================================================
# Process 18 : Continue with the Sentiment Analysis Pipeline (Text preprocessing function)
df = pd.read_csv("P17.auto_labeled_sentiment.csv")

# # Drop rows with missing values
# df.dropna(subset=['comment', 'auto_sentiment'], inplace=True)

# fill missing values
df['comment'] = df['comment'].fillna("No comment provided")

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtag symbol
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove emojis,punctuation,numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

df['comment'] = df['comment'].apply(clean_text)
print(f"Final dataset dimension: {df.shape} ")

# Encode sentiment labels to integers
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['auto_sentiment'])

# Text preprocessing (TF-IDF)
X = df['comment']
y = df['sentiment_encoded']

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Save the LabelEncoder
label_encoder_filename = 'label_encoder.joblib'
joblib.dump(le, label_encoder_filename)
print(f"Label encoder saved as {label_encoder_filename}")

# Save the TF-IDF vectorizer
vectorizer_filename = 'tfidf_vectorizer.joblib'
joblib.dump(tfidf, vectorizer_filename)
print(f"TF-IDF vectorizer saved as {vectorizer_filename}")

print("Process 18: Encoded sentiment labels and vectorized comments")
#====================================================


#=============================================
# Process 19 : Train and Save the Sentiment Analysis Model

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Save the trained Logistic Regression model
model_filename = 'sentiment_model.joblib'
joblib.dump(lr_model, model_filename)
print(f"Trained model saved as {model_filename}")

print("Process 19: Trained and saved model")
#=================================================


#===================================================================
# Process 20 : Continue with the Sentiment Analysis Pipeline (Train and Evaluate the Model)

y_pred = lr_model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy= accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

#Save report
with open("sentiment_model_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nAccuracy: " + str(accuracy_score(y_test, y_pred)))

#======================================================================



