import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras
import pickle
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
housing_csv_path = os.path.join(script_dir, 'Housing.csv')
model_pkl_path = os.path.join(script_dir, 'model.pkl')
preprocessor_pkl_path = os.path.join(script_dir, 'preprocessor.pkl')

data = pd.read_csv(housing_csv_path)

categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
X = data.drop('price', axis=1)
y = data['price']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

sample_house = pd.DataFrame({
    'area': [3000],
    'bedrooms': [4],
    'bathrooms': [3],
    'stories': [2],
    'mainroad': ['yes'],
    'guestroom': ['no'],
    'basement': ['yes'],
    'hotwaterheating': ['no'],
    'airconditioning': ['yes'],
    'parking': [2],
    'prefarea': ['yes'],
    'furnishingstatus': ['semi-furnished']
})

sample_encoded = preprocessor.transform(sample_house)
predicted_price = model.predict(sample_encoded)
print(f"Predicted price for sample house: {predicted_price[0][0]:.2f}")
print(f"Actual price for sample house: {y_test.iloc[0]:.2f}")

with open(model_pkl_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(preprocessor_pkl_path, 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)