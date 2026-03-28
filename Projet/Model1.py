import warnings
from category_encoders import MEstimateEncoder
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Charger les données

data = pd.read_csv("student_dataset/student_failure/train.csv")
# Afficher les premières lignes du dataset
X = data.drop(["score_examen"], axis=1)
y = data.score_examen
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# separate numerical and categorical features
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns
object_list=list(map(lambda col: X_train[col].nunique(), categorical_features))
d = dict(zip(categorical_features, object_list))
print(d)
# show differnet unique values for each categorical feature
for col in categorical_features:
    print(f"{col}: {X_train[col].unique()}")
