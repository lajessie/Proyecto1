"""import numpy as np
import pandas as pd
from pandas import DataFrame


numData=20
x = np.random.rand(numData, 1)
y = np.random.rand(numData, 1)
print(x)
print(y)

dataset = pd.DataFrame({'predicto1': list(x), 'predicto2': list(y)}, columns = ['predicto1', 'predicto2'])

print(dataset.head())"""



from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures


#X is the independent variable (bivariate in this case)


#vector is the dependent data
vector = [109.85, 155.72]

#predict is an independent variable for which we'd like to predict the value
predict= [0.49, 0.18]

#generate a model of polynomial features
poly = PolynomialFeatures(degree=2)

#transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
#X_ = poly.fit_transform(X)

#Funcion que nos regresa el set de la variable dependiente
def FrankeFuction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = np.random.randn(numData,1)
    return term1 + term2 + term3 + term4 + noise

#data
numData = 50

# x & y son las variables predictos
x = np.random.rand(numData, 1)
y = np.random.rand(numData, 1)
z = FrankeFuction(x, y)

dataset = pd.DataFrame({'predicto1': list(x), 'predicto2': list(y), 'outcome' : list(z)}, columns = ['predicto1', 'predicto2', 'outcome'])
#print(dataset.head())

indep = dataset[['predicto1', 'predicto2']]
depen = dataset['outcome']

indep_ = poly.fit_transform(indep)

print(dataset.head())
print(indep_)
