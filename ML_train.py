import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("accident_dataset_ML.csv")

x=df.drop("risk_score",axis=1)
y=df["risk_score"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("R² :", r2_score(y_test, y_pred))

print("\nFeature Importance (Coefficients):")

for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:<25} {coef:.5f}")

joblib.dump(
{
    "model": model,
    "feature_columns": x.columns.tolist()
},
"risk_model.pkl"
)