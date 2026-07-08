import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("=== STEP 1: DOWNLOADING DICTIONARY DATA ===")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# 0. Datapool
reviews_data = [
    ("This product is amazing and perfect, I love it!", 1),
    ("Absolutely terrible, completely broken and useless.", 0),
    ("I am not happy with this purchase, it does not work.", 0),
    ("Incredible quality, highly recommend to everyone.", 1),
    ("Waste of money, awful experience.", 0),
    ("It is not bad, actually pretty good performance.", 1),
    ("Horrible customer service and very cheap material.", 0),
    ("Super sleek design and works flawlessly.", 1),
    ("Fantastic build, totally worth the price!", 1),
    ("Do not buy this! It broke within two days.", 0),
    ("Extremely satisfied with the fast shipping and quality.", 1),
    ("Worst purchase I have ever made online.", 0),
    ("It's ok, not great but not terrible either.", 1),
    ("Brilliant engineering, works like a charm.", 1),
    ("Defective item, stopped working after an hour.", 0),
    ("Excellent value for money, highly durable.", 1)
]
df = pd.DataFrame(reviews_data, columns=['review', 'sentiment'])

print("\n=== STEP 2: RUNNING NLP PRE-PROCESSING PIPELINE ===")
default_stopwords = set(stopwords.words('english'))
negations = {'not', 'no', 'never', 'neither', 'nor', 'but'}
custom_stopwords = default_stopwords - negations 

def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'): return wordnet.ADJ
    elif nltk_tag.startswith('V'): return wordnet.VERB
    elif nltk_tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

lemmatizer = WordNetLemmatizer()

def clean_text_pipeline(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    meaningful_tokens = [word for word in tokens if word.isalpha() and word not in custom_stopwords]
    pos_tags = nltk.pos_tag(meaningful_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized_tokens)

df['cleaned_review'] = df['review'].apply(clean_text_pipeline)

# Show exact processing transformations
for idx, row in df.head(3).iterrows():
    print(f"Original Review: {row['review']}")
    print(f"Cleaned Tokens:  {row['cleaned_review']}\n")

print("=== STEP 3: TRAINING MACHINE LEARNING CLASSIFIER ===")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("\n=== STEP 4: MODEL EVALUATION METRICS ===")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Matrix Report:\n", classification_report(y_test, y_pred))