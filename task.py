import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Dataset.csv")

print(data.head())

print(data.shape)

print(data.info())

print(data.describe())

print(data.isnull().sum())


X_axis = data[["Population"]]
Y_axis = data["Profit"]

# Simple Linear Regression
x_train, x_test, y_train, y_test = train_test_split(X_axis, Y_axis, test_size=0.3, random_state=45)
model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

plt.scatter(x_test, y_test, color="red", label="Actual Value")
plt.plot(x_test, y_predict, color="green", linewidth= 2, label="Predicted Value")
plt.title("Prediction of Profit based on Population")
plt.ylabel("Profit")
plt.xlabel("Population")
plt.legend()
plt.show()


# Simple Linear Regression
mse_simple = mean_squared_error(y_test, y_predict)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_predict)
print(f"Simple Linear Regression:- MSE:{mse_simple}, RMSE:{rmse_simple}, R^2: {r2_simple}")


# Multiple Regression
X_multi = data[['Population', 'Income', 'MarketingSpend']]
y_multi = data['Profit']
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.3, random_state=45)
multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)
y_pred_multi = multi_model.predict(X_test_multi)


plt.figure(figsize=(10, 6))
plt.scatter(y_test_multi, y_pred_multi, color='green', label='Predicted Profits')
plt.plot(y_test_multi, y_test_multi, color='red', linewidth=2, label='Actual Profits')
plt.title('Multiple Linear Regression: Actual vs Predicted Profits')
plt.xlabel('Actual Profits')
plt.ylabel('Predicted Profits')
plt.legend()
plt.show()



# Multiple Linear Regression
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)
print(f"Multiple Linear Regression:- MSE:{mse_multi}, RMSE:{rmse_multi}, R^2:{r2_multi}")

if r2_simple > r2_multi:
    print("Simple Linear Regression performs better than Multiple Linear Regression based on R^2 Score.")
else:
    print("Multiple Linear Regression performs better than Simple Linear Regression based on R^2 Score.")

plt.figure(figsize=(10, 6))
metrics = ['MSE', 'RMSE', 'R²']
values_simple = [mse_simple, rmse_simple, r2_simple]
values_multi = [mse_multi, rmse_multi, r2_multi]
x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars
plt.bar(x - width/2, values_simple, width, label='Simple Linear Regression')
plt.bar(x + width/2, values_multi, width, label='Multiple Linear Regression')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Comparison of Regression Models')
plt.xticks(x, metrics)
plt.legend()
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['Population'], data['Income'], data['MarketingSpend'], c=data['Profit'], cmap='viridis', marker='o')

ax.set_xlabel('Population')
ax.set_ylabel('Income')
ax.set_zlabel('MarketingSpend')
ax.set_title('3D Scatter Plot of Features and Profit')

plt.show()