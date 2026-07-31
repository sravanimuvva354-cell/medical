import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("dataset/Symptom2Disease.csv")

# 2. Check column names
print(df.columns)

# 3. Remove missing values
df = df.dropna()

# 4. Input and output
X = df["text"]
y = df["label"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Test the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# 9. Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("model.pkl created successfully!")