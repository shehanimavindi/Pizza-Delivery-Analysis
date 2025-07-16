import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt 

data =  pd.read_csv(r'D:\DATA SCIENCE(Shehani)\Semester 1\Mathematics for Computing\Vs codes\codes vs\script01\Project 1\pizza_delivery_dataset.csv')

print(data.head())
print(data.info())

data.dropna()

from sklearn.model_selection import train_test_split

X = data.drop('Delivery_Status', axis = 1)
y = data['Delivery_Status']

#Make sure X has only numeric values for histogram and model
X = X.select_dtypes(include=['float64','int64'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

train_data = X_train.join(y_train)
train_data.hist(figsize = (15,8))
plt.show()

import seaborn as sns  
sns.boxplot(x='Delivery_Status', y = 'Distance_km', data=data)
plt.title('Distance vs Delivery Status')
plt.show()

sns.boxplot(x ='Delivery_Status', y= 'Temperature', data=data)
plt.title('Temperature vs Delivery Status')
plt.show()