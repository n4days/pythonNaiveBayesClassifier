import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

# Load dataset
file_path = 'DatasetBNBClean.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert the list of words to a single string
data['Words'] = data['Words'].apply(lambda x: ' '.join(eval(x)))  # Ensure x is converted from a list to a string
X = data['Words']
y = data['Label']

# Split into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into a binary feature matrix using CountVectorizer
vectorizer = CountVectorizer(binary=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Bernoulli Naive Bayes model
bnb = BernoulliNB()
bnb.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = bnb.predict(X_test_vec)

# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bnb.classes_)
plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Classification Report in Bar Chart
labels = bnb.classes_
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

report_df = pd.DataFrame({'Labels': labels, 'Precision': precision, 'Recall': recall, 'F1-Score': fscore})
report_df.set_index('Labels', inplace=True)

report_df.plot(kind='bar', figsize=(8, 5))
plt.title('Classification Report Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()

# Word Frequency Plot
word_counts = np.asarray(X_train_vec.sum(axis=0)).flatten()
words = vectorizer.get_feature_names_out()

word_freq_df = pd.DataFrame({'Word': words, 'Count': word_counts})
word_freq_df = word_freq_df.sort_values(by='Count', ascending=False).head(20)

plt.figure(figsize=(10, 5))
sns.barplot(x='Count', y='Word', data=word_freq_df, palette='viridis')
plt.title('Top 20 Most Frequent Words in Training Data')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

# Save results to an Excel file
X_test_combined = pd.DataFrame({'Original_Text': X_test.reset_index(drop=True), 'Prediction': y_pred})
X_test_combined.to_excel('Combined_Output_Simple.xlsx', index=False)
print("\nCombined data (Original_Text and Prediction) has been saved to 'Combined_Output_Simple.xlsx'.")
