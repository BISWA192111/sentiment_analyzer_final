import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load labeled phase data (wide format: before, during, after columns)
csv_path = r'common_synonyms_before_during_after_1000.csv'
df = pd.read_csv(csv_path)

# Melt wide format to long format: each row is (sentence, phase)
phase_cols = [col for col in df.columns if col.lower() in ['before', 'during', 'after']]
df_melted = df.melt(value_vars=phase_cols, var_name='phase', value_name='sentence')
df_melted = df_melted.dropna(subset=['sentence'])
X = df_melted['sentence']
y = df_melted['phase']

# 2. Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
vectorizer = TfidfVectorizer(max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# 4. Evaluate
print("Validation accuracy:", clf.score(X_test_vec, y_test))

# 5. Save model and vectorizer
joblib.dump(clf, 'phase_classifier.pkl')
joblib.dump(vectorizer, 'phase_vectorizer.pkl')
print('Phase classifier and vectorizer saved.') 