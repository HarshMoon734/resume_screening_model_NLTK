import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('resume_data.csv')

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)   # Remove URLs
    cleanText = re.sub(r'RT|cc', ' ', cleanText) # Remove RT and cc
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)  # Remove hashtags
    cleanText = re.sub(r'@\S+', '  ', cleanText)   # Remove mentions
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)  # Remove punctuation
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Remove non-ASCII characters
    cleanText = re.sub(r'\s+', ' ', cleanText)  # Remove extra whitespace
    return cleanText.strip()  # Strip leading and trailing whitespace

# Clean the 'Resume' column
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Encode 'Category' labels
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Vectorize the 'Resume' column using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
requredText = tfidf.transform(df['Resume'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(requredText, df['Category'], test_size=0.2, random_state=42)

# Train OneVsRestClassifier with KNeighborsClassifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Predict on the test set
ypred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, ypred)
print(f'Accuracy: {accuracy}')