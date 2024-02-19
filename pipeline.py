import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

class OutlierHandler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming X is a 2D array
        for col_index in range(X.shape[1]):
            column = X[:, col_index]
            Q1 = np.percentile(column, 25)
            Q3 = np.percentile(column, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (column < lower_bound) | (column > upper_bound)
            median_value = np.median(column)
            X[outliers, col_index] = median_value
        return X
    
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('outlier_handler', OutlierHandler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.select_dtypes(include=['float64', 'int64']).columns),
        ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
    ])


model = make_pipeline(
    preprocessor,
    StandardScaler(with_mean=False),
    PCA(n_components=25),
    SMOTE(random_state=42, sampling_strategy = 0.5),
    LogisticRegression(C=0.1, penalty='l1', random_state=10, class_weight='balanced', solver='saga')
)

model.fit(X_train, y_train_encoded)


y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)