import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

#get data frame
df=pd.read_csv('homeprices.csv')
print(df)



#fill null value
mb=math.floor(df.bedrooms.median())#get median value
df.bedrooms=df.bedrooms.fillna(mb)#fill null value
print(df)

#train
model=LinearRegression()
model.fit(df[['area','bedrooms','age']],df.price)

#predect
result1=model.predict([[3000,3,40]])
result2=model.predict([[5000,4,5]])
print(result1)
print(result2)

#score
print(model.score(df[['area','bedrooms','age']],df.price))

#calculate formaula
cf=model.coef_
result3=cf[0]*3000+cf[1]*3+cf[2]*40+model.intercept_
result4=cf[0]*5000+cf[1]*4+cf[2]*5+model.intercept_
print(result3)
print(result4)

#plot
plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.area,df.price,color='red',marker='*')
plt.plot(df.area,df.price,color='blue')
plt.show()



