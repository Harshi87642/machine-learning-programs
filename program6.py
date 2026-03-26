import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
 

data_path = "heart.csv"  
df = pd.read_csv(data_path)  

X = df['age'].values.reshape(-1, 1)  
y = df['trestbps'].values  

def locally_weighted_regression(X_train, y_train, tau=0.1): 
    """ 
    Locally Weighted Linear Regression function. 
    X_train: Feature data for training 
    y_train: Target data for training 
    tau: Smoothing parameter, smaller values make the model more sensitive to local variations 
    """ 
    m = len(X_train) 
    weights = np.zeros((m, m))  
 
    
    for i in range(m): 
        diff = X_train - X_train[i]  
        weights[:, i] = np.exp(-np.sum(diff**2, axis=1) / (2 * tau**2))  
    
    X_train = np.hstack((np.ones((m, 1)), X_train))   
    theta = np.linalg.inv(X_train.T @ weights @ X_train) @ (X_train.T @ weights @ y_train) 
 
    return theta 
 

theta = locally_weighted_regression(X, y, tau=0.1) 
print("\nCalculated theta (coefficients):", theta)
def predict(X, theta): 
 """ 
 Predict the target using the trained model coefficients. 
 """ 
 X = np.hstack((np.ones((X.shape[0], 1)), X))  
 return X @ theta 
 
y_pred = predict(X, theta) 

plt.scatter(X, y, color='blue', label="Original data") 
plt.plot(X, y_pred, color='red', label="Locally Weighted Regression") 
plt.xlabel('Age') 
plt.ylabel('Resting Blood Pressure') 
plt.title('Locally Weighted Regression: Age vs Resting Blood Pressure') 
plt.legend() 
plt.show()