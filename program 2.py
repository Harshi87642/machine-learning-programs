import pandas as pd   
import seaborn as sns 
import matplotlib.pyplot as plt  
 
 
data = pd.read_csv("D:\Downloads\california_housing.csv")  
df = pd.DataFrame(data)  
correlation_matrix = df.corr() 
 

plt.figure(figsize=(12, 9)) 
sns.heatmap(correlation_matrix, 
            annot=True,  
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5) 
plt.title("Correlation Matrix Heatmap", fontsize=16)  
plt.show()  
 
pairplot = sns.pairplot(df, plot_kws={'alpha': 0.5})   
 
plt.subplots_adjust(top=0.95)  
pairplot.fig.suptitle("Pair Plot of Features", fontsize=16, y=1.02)  
 
plt.show() 