# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
dtraining = pd.read_excel("dataTraining.xlsx")
X_train = dtraining.drop(columns=["author", "classification"])
y_train = dtraining.classification

dtesting = pd.read_excel("dataTesting.xlsx")

# Preprocessing: Convert the sparse matrix to dense
preprocessor = ColumnTransformer([
    ('categoric', OneHotEncoder(), ['text1', 'text2']),
])

# Custom transformer to convert sparse matrix to dense
class ToDenseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.toarray() if hasattr(X, 'toarray') else X

# Create pipeline
pipeline = Pipeline([
    ('prep', preprocessor),
    ('to_dense', ToDenseTransformer()),  # Step to convert to dense
    ('algo', GaussianNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on training data and add to dataframe
dtraining["classificationPredict"] = pipeline.predict(dtraining)

# Predict on test data and add to dataframe
dtesting["classificationPredict"] = pipeline.predict(dtesting)

X_test = dtesting.drop(columns=["author", "classification"])
y_test = dtesting.classification

# Display training and test scores
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Training Score: {train_score}")
print(f"Testing Score: {test_score}")

# Plot confusion matrix using seaborn
y_pred = pipeline.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save predictions to Excel files
with pd.ExcelWriter("HasilKlasifikasiDataTraining.xlsx") as writer:
    dtraining.to_excel(writer, index=False)

with pd.ExcelWriter("HasilKlasifikasiDataTesting.xlsx") as writer:
    dtesting.to_excel(writer, index=False)
