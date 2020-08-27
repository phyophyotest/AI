import pandas as pd 
from sklearn import linear_model
import matplotlib.pyplot as plt
#to get data frame from csv
dataframe=pd.read_csv("homeprices1.csv")
print(dataframe)

#tranin
model=linear_model.LinearRegression()
model.fit(dataframe[['area']],dataframe.price)

#calculate linear equation 
result1=model.predict([[3300]])
result2=model.predict([[5000]])
print(result1)
print(result2)

#
print(model.score(dataframe[['area']],dataframe.price))

#to calculate formula
result3=model.coef_ * 3300 +model.intercept_
result4=model.coef_ *5000 +model.intercept_
print(result3)
print(result4)

#matplot 
plt.xlabel('Area (sqf)')
plt.ylabel('Price (USD)')
plt.scatter(dataframe.area,dataframe.price,color='red',marker='+')
plt.plot(dataframe.area,dataframe.price,color='blue')
plt.show()







