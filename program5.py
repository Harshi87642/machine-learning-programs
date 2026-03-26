import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 

np.random.seed(42)  
X = np.random.rand(100, 1) 

y = np.where(X[:50] <= 0.5, 1, 2)  
y = np.concatenate([y, np.where(X[50:] <= 0.5, 1, 2)])  
 
X_train = X[:50]  
y_train = y[:50].ravel()  
X_test = X[50:]  
y_test = y[50:].ravel()  
k_values = [1, 2, 3, 4, 5, 20, 30] 
predictions = {} 
for k in k_values: 
 knn = KNeighborsClassifier(n_neighbors=k)  
 knn.fit(X_train, y_train)  
 y_pred = knn.predict(X_test)  
 predictions[k] = y_pred 
for k, pred in predictions.items(): 
 print(f"\nPredictions for k={k}:") 
 print(pred) 

plt.figure(figsize=(10, 5)) 
for k in k_values: 
 plt.scatter(X_test, predictions[k], label=f'k={k}', alpha=0.6) 
plt.axvline(0.5, color='red', linestyle='--', label='Decision Boundary (x=0.5)') 
plt.xlabel("X values") 
plt.ylabel("Predicted Class") 
plt.title("KNN Classification Results for Different k Values") 
plt.legend() 
plt.show()