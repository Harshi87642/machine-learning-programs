import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns  
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler  
 
data = pd.read_csv("iris.csv")  

print("Dataset Sample:\n", data.head()) 
 
df = pd.DataFrame(data) 

scaler = StandardScaler()   
X_scaled = scaler.fit_transform(df.iloc[:, :-1])  

pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled) 
 
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])   

pca_df['species'] = df['species']  
 
plt.figure(figsize=(8, 6)) 
sns.scatterplot(x='PC1', y='PC2', hue=pca_df['species'], palette='deep', data=pca_df)  
plt.title('PCA of Iris Dataset (4D → 2D)', fontsize=14)   
plt.xlabel('Principal Component 1')  
plt.ylabel('Principal Component 2')  
plt.legend(title='Species') 
plt.grid(True) 
plt.show()  

print("\nExplained Variance Ratio of PCA Components:", pca.explained_variance_ratio_)