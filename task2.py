import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("movies.csv")
print("Dataset Preview:\n", df.head())

df = df.drop(columns=['Movie_ID'], errors='ignore')
df = df.ffill()
df = pd.get_dummies(df, columns=['Genre', 'Director', 'Actors'])

X = df.drop(columns=['Rating'])
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:\nMSE: {mse:.2f}\nR² Score: {r2:.2f}")

new_movie_data = {
    'Genre_Action': [1], 'Genre_Comedy': [0], 'Genre_Drama': [0], 'Genre_Horror': [0], 'Genre_Sci-Fi': [0], 
    'Director_Christopher Nolan': [1], 'Director_Wes Anderson': [0], 'Director_Martin Scorsese': [0], 
    'Actors_Leonardo DiCaprio': [1], 'Actors_Bill Murray': [0], 'Actors_Robert De Niro': [0]
}

new_movie = pd.DataFrame(new_movie_data)

for col in X.columns:
    if col not in new_movie.columns:
        new_movie[col] = 0

new_movie = new_movie[X.columns]  # Ensure same column order as training data

predicted_rating = model.predict(new_movie)
print(f"\nPredicted Rating for New Movie: {predicted_rating[0]:.2f}")



