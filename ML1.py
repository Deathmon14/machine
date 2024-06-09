import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/New folder/newclass/trial/Advertising.csv")

x = dataset[['TV']]
y = dataset['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

slr = LinearRegression()
slr.fit(x_train, y_train)

print("Intercept:", slr.intercept_)
print("Coefficient:", slr.coef_)

y_pred_slr = slr.predict(x_test)

print("Prediction of test set:", y_pred_slr)

slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted Value': y_pred_slr})
slr_diff.head()

plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred_slr, 'red')
plt.show()

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))
print('Root Squared: {:2f}'.format(slr.score(x, y) * 100))
print('Mean Absolute Error:', meanAbErr)
print('Mean squared Error:', meanSqErr)
print('Root Mean Squared Error:', rootMeanSqErr)
