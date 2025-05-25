import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def model_training(data: dict, model_filename: str = 'model.pkl'):
    
    df = pd.DataFrame(data)
    X = df.drop(columns=['temp_celsius'])
    y = df['temp_celsius']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, model_filename)
    
    print(f"✅ Model trained and saved as '{model_filename}'")
