import pandas as pd  
 
data_path = "heart.csv" 
df = pd.read_csv(data_path) 

print("Dataset Overview:") 
print(df.head())  

attributes = df.columns[:-1]  
target_column = df.columns[-1]  
 
positive_examples = df[df[target_column] == 1]  

hypothesis = list(positive_examples.iloc[0, :-1])   

for i in range(1, len(positive_examples)):  
 for j in range(len(hypothesis)): 
  if positive_examples.iloc[i, j] != hypothesis[j]: 
    hypothesis[j] = '?' 
print("\nFinal Hypothesis (Find-S Algorithm):") 
print(hypothesis) 
