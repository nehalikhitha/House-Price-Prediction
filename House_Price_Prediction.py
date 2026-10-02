print("House Price Prediction")
import pandas as pd

data = pd.read_csv("house_price.csv")

print(data.head())

X = data[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = data["Price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)

print("R2 Score:", score)

new_house = [[1600, 3, 2, 4]]

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0], "Lakhs")