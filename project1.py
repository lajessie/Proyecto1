# Importing various packages
from random import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import pandas as pd

#Funcion que nos regresa el set de la variable dependiente
def FrankeFuction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = np.random.randn(numData,1)  ##Adding some noise 
    return term1 + term2 + term3 + term4 + (0.01*noise)

#data
numData = 500
gradoPoly = 5

# x & y son las variables predictos
x = np.random.rand(numData, 1)
y = np.random.rand(numData, 1)
z = FrankeFuction(x, y)

dataset = pd.DataFrame({'predicto1': list(x), 'predicto2': list(y), 'outcome' : list(z)}, columns = ['predicto1', 'predicto2', 'outcome'])

#print(dataset)
indep = dataset[['predicto1', 'predicto2']]
depen = dataset['outcome']

#Using Linear Regression
poly = PolynomialFeatures(degree=gradoPoly)
indep_= poly.fit_transform(indep)
regrGr = LinearRegression()
regrGr.fit(indep_, depen)
intercept  = regrGr.intercept_
beta = regrGr.coef_
depen_pred = regrGr.predict(indep_)

bias = np.sum( (depen- np.mean(depen_pred))**2 )/numData
print('Multiple regression model using the whole data:')
print('Beta parameters:', beta)
print('Mean Squared Error (MSE): ', metrics.mean_squared_error(depen, depen_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(depen, depen_pred)))
print('Variance: ', np.var(depen_pred))
print('Bias:', bias)

"""
error = np.mean( np.mean((depen - depen_pred)**2) )
bbias = np.mean( (depen - np.mean(depen_pred))**2 )
vvariance = np.mean( np.var(depen_pred) )
print('Error:', error)
print('Bias^2:', bbias)
print('Var:', vvariance)
print('{} >= {} + {} = {}'.format(error, bbias, vvariance, bbias+vvariance))
"""

print('\n\n------Splitting between train and test data --------')
#Evaluating model using cross-validation resample technique
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
indep_train, indep_test, dep_train, dep_test = train_test_split(indep, depen, test_size=0.2)

betaMayor = []
numExp = 10
MSE_ = 1.0
for i in range(0,numExp): #Number of experiments
    indep_trainR, dep_trainR = resample(indep_train, dep_train)
    #20% of the data is the test set (given by test_size=0.2)
    poly = PolynomialFeatures(degree=gradoPoly)
    indep_trainR_ = poly.fit_transform(indep_trainR)
    regrGr = LinearRegression()
    regrGr.fit(indep_trainR_, dep_trainR)
    betaS = regrGr.coef_
    betaMayor = np.concatenate((betaMayor, betaS)) #Getting all the values of B
    #Making predictions
    indep_test_ = poly.fit_transform(indep_test) #using the test data as polynomial 
    dep_pred = regrGr.predict(indep_test_)
    #Calculating MSE, R2, Var and Bias for each experiment
      
    MSE = np.mean( np.mean((depen - dep_pred)**2) )
    R2 = np.sqrt(np.mean( np.mean((depen - dep_pred)**2) ))
    bias = np.mean( (depen - np.mean(dep_pred))**2 )
    variance = np.mean( np.var(dep_pred) )   
   
    if (MSE <= MSE_):
        #Saving the values for the model with less MSE
        Beta_= betaS
        MSE_ = MSE
        R2_= R2
        var_ = variance
        bias_ = bias
        
#Getting a Matrix with all the values from B parameters of each experiment    
BetaMatrix = betaMayor.reshape(numExp, len(betaS))
print('Variances for each beta parameters: ', np.var(BetaMatrix,0) )
print('\nConfidence intervals of the parameters beta: (95%)')
for i in range(0, len(betaS)):
    print("B_", i, " {}".format(np.percentile(BetaMatrix[:,i], [2.5, 97.5])))

print('\n--Selecting the model with less MSE--')
print('Beta parameters:', Beta_)
print('Mean Squared Error (MSE): ', MSE_)
print('Root Mean Squared Error:', R2_)
print('Variance: ',var_)
print('Bias:', bias_)
print('Var + Bias =', var_ + bias_)


