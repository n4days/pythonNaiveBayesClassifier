import pandas as pd
import openpyxl as xls

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe

dtraining = pd.read_excel("dataTraining.xlsx")

X_train = dtraining.drop(columns=["author", "classification"])
y_train = dtraining.classification

dtesting = pd.read_excel("dataTraining.xlsx")
dtesting.drop(columns=["author"])

# Preprocessing: convert the sparse matrix to a dense one using toarray
preprocessor = ColumnTransformer([
    ('categoric', cat_pipe(encoder='onehot'), ['text1', 'text2', 'text3']),
])

from sklearn.naive_bayes import GaussianNB
from sklearn.base import TransformerMixin

# Custom transformer to convert sparse matrix to dense
class ToDenseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.toarray() if hasattr(X, 'toarray') else X

pipeline = Pipeline([
    ('prep', preprocessor),
    ('to_dense', ToDenseTransformer()),  # Step to convert to dense
    ('algo', GaussianNB())
])

pipeline.fit(X_train, y_train)

dtraining["classificationPredict"] = pipeline.predict(dtraining)
print(dtraining)

dtesting["classificationPredict"] = pipeline.predict(dtesting)
print(dtesting)

X_test = dtesting.drop(columns=["author", "classification"])
y_test = dtesting.classification

pipeline.score(X_train, y_train)

pipeline.score(X_test, y_test)

from jcopml.plot import plot_confusion_matrix
plot_confusion_matrix(X_train, y_train, X_test, y_test, pipeline)

excelfile1 = pd.ExcelWriter("KlasifikasiDataTesting.xlsx")
dtesting.to_excel(excelfile1)
excelfile1.close()

excelfile1 = pd.ExcelWriter("KlasifikasiDataTraining.xlsx")
dtraining.to_excel(excelfile1)
excelfile1.close()