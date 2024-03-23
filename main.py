import pandas as pd
from sklearn.linear_model import LinearRegression as LR


df = pd.read_csv("data/train.csv")


'''
First, we will start with a simple linear regression.
'''

X = df[['LotArea']]
y = df['SalePrice']

model = LR()
model.fit(X,y)

print('Coefficient:', model.coef_)
print('Intercept: ', model.intercept_)