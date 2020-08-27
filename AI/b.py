import pandas as pd
import math
from sklearn.linear_model import LinearRegression


df=pd.read_csv("homeprices.csv")
print(df.head(2))#output two row
print(df.shape)#output row and column
#to fill null value
mb=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(mb)
print(df)
# #train
model=LinearRegression()
model.fit(df[['area','bedrooms','age']],df.price)
# #predict
result1=model.predict([[3000,3,40]])
result2=model.predict([[2500,4,5]])
print(result1)
print(result2)
print(model.score(df[['area','bedrooms','age']],df.price))

#calculate formula

cf=model.coef_
result3=cf[0]*3000+cf[1]*3+cf[2]*40+model.intercept_
result4=cf[0]*2500+cf[1]*4+cf[2]*5+model.intercept_

print(result3)
print(result4)
