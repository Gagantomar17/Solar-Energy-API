import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("data/Renewable.csv")
print(df.head())
x = df[["temp","pressure","humidity","wind_speed"]]
y = df["Energy delta[Wh]"]

model = LinearRegression()
model.fit(x,y)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Dummy model created!")