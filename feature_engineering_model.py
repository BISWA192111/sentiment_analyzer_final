import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Download necessary nltk resources
import nltk.corpus
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
import nltk
from nltk.stem import WordNetLemmatizer

dataset = pd.read_csv("Combined Data.csv")
dataset["status"] = dataset["statement"].fillna("")

def assign_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

dataset["sentiment"] = dataset["status"].apply(assign_sentiment)

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)
    return text

stopwords = set(nltk.corpus.stopwords.words("english")) - {"not"}
lemmatizer = WordNetLemmatizer()

def process_text(text):
    tokens = nltk.word_tokenize(text)
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return " ".join(processed)

dataset["status"] = dataset["status"].apply(clean_text)
dataset["status"] = dataset["status"].apply(process_text)

dataset["polarity"] = dataset["status"].map(lambda text: TextBlob(text).sentiment.polarity)
dataset["length"] = dataset["status"].astype(str).apply(len)
dataset["word_counts"] = dataset["status"].apply(lambda x: len(str(x).split()))

def show_wordcloud(data, title):
    wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=stopwords).generate(str(data))
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.show()

show_wordcloud(dataset[dataset["sentiment"] == "Positive"]["status"], "Positive Sentiment WordCloud")
show_wordcloud(dataset[dataset["sentiment"] == "Neutral"]["status"], "Neutral Sentiment WordCloud")
show_wordcloud(dataset[dataset["sentiment"] == "Negative"]["status"], "Negative Sentiment WordCloud")

encoder = LabelEncoder()
dataset["sentiment"] = encoder.fit_transform(dataset["sentiment"])

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
X = tfidf.fit_transform(dataset["status"])
y = dataset["sentiment"]

smote = SMOTE(random_state=42)
X_final, y_final = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=42)

models = [
    DecisionTreeClassifier(),
    LogisticRegression(),
    SVC(),
    RandomForestClassifier(),
    BernoulliNB(),
    KNeighborsClassifier()
]

model_names = {
    0: "Decision Tree",
    1: "Logistic Regression",
    2: "SVC",
    3: "Random Forest",
    4: "Naive Bayes",
    5: "K-Neighbors"
}

for i, model in enumerate(models):
    print(f"{model_names[i]} Accuracy: {cross_val_score(model, X, y, cv=10, scoring='accuracy').mean():.4f}")

params = {"C": np.logspace(-4, 4, 10), "penalty": ['l1', 'l2']}
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid=params, scoring="accuracy", cv=10, n_jobs=-1)
grid.fit(X_train, y_train)

best_accuracy = grid.best_score_
best_params = grid.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
print("Best Parameters:", best_params)

clf = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver='liblinear')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, predictions))

conf_matrix = confusion_matrix(y_test, predictions)

def plot_cm(cm, classes, title, normalized=False, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.xlabel("Predicted Label", labelpad=20)
    plt.ylabel("Actual Label", labelpad=20)

plot_cm(conf_matrix, classes=["Positive", "Neutral", "Negative"], title="Confusion Matrix")

print(classification_report(y_test, predictions, target_names=["Positive", "Neutral", "Negative"]))
