import nltk
import string
import re

# Download and load the stopwords corpus
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# Load the dataset of offensive and non-offensive texts
# and create a list of tuples containing the text and its label
dataset = [
    ("I hate you so much, you are a terrible person", "offensive"),
    ("I really enjoy spending time with you", "non-offensive"),
    ("I can't believe you would say something like that", "offensive"),
    ("I love spending time with you", "non-offensive"),
]

# Define a function to preprocess the text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    # Join the tokens back into a string
    text = ' '.join(tokens)
    return text

# Preprocess the dataset
dataset = [(preprocess_text(text), label) for text, label in dataset]

# Define a function to extract features from the text
def extract_features(text):
    # Define a dictionary to hold the features
    features = {}
    # Add a feature for each word in the text
    for word in nltk.word_tokenize(text):
        features[word] = True
    return features

# Extract the features from the dataset
featuresets = [(extract_features(text), label) for text, label in dataset]

# Train a Naive Bayes classifier on the featuresets
classifier = nltk.NaiveBayesClassifier.train(featuresets)

# Define a function to classify new texts
def classify_text(text):
    # Preprocess the text and extract its features
    text = preprocess_text(text)
    features = extract_features(text)
    # Use the classifier to classify the text
    label = classifier.classify(features)
    return label

# Test the classifier on some example texts
texts = [
    "I hate you and your family",
    "I really appreciate your help",
    "You are a stupid idiot",
    "I love spending time with you",
]
for text in texts:
    label = classify_text(text)
    print(f"Text: {text}\nLabel: {label}\n")

