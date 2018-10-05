# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures


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

print(dataset)
indep = dataset[['predicto1', 'predicto2']]
depen = dataset['outcome']

regr = LinearRegression()  #Linear regression object
regr.fit(indep, depen)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


poly5 = PolynomialFeatures(degree=5)
PREDS = poly5.fit_transform(x[:,np.newaxis])

clf5 = LinearRegression()
clf5.fit(PREDS, depen)

"""
print('ypredict: ', ypredict)
plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()

"""