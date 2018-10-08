# Importing various packages
from random import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Funcion que nos regresa el set de la variable dependiente
def FrankeFuction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = np.random.randn(numData,1)  ##Adding some noise 
    return term1 + term2 + term3 + term4 + (0.01*noise)

#data
numData = 100

# x & y are the predictors
x = np.random.rand(numData, 1)
y = np.random.rand(numData, 1)
z = FrankeFuction(x, y)

dataset = pd.DataFrame({'predicto1': list(x), 'predicto2': list(y), 'outcome' : list(z)}, columns = ['predicto1', 'predicto2', 'outcome'])

#print(dataset)
indep = dataset[['predicto1', 'predicto2']]
depen = dataset['outcome']
"""
#OLS
[C,R] = np.meshgrid(x,y)
 
x_ = C.reshape(-1,1)
y_ = R.reshape(-1,1)
[n,m] = C.shape
num_data = n*m
"""
d = pd.DataFrame(columns =['mse', 'r2', 'variance', 'bias' ], index=range(6))
    
#perform a standard least square regression analysis using polynomials in x and y up to ﬁfth order
for i in range(0,6):
    poly = PolynomialFeatures(degree=i)
    data= poly.fit_transform(indep)
    
    from sklearn.cross_validation import train_test_split
    data_train, data_test, z_train, z_test = train_test_split(data, z, test_size=0.2)   
        
    beta_ols = np.linalg.inv(data_train.T @ data_train) @ data_train.T @ z_train
    z_hat= data_test@beta_ols
    
    #Using Sklearn
    regrGr = LinearRegression()
    regrGr.fit(data_train, z_train)
    intercept  = regrGr.intercept_
    beta = regrGr.coef_
    depen_pred = regrGr.predict(data_test)
    """
    mse = np.sum( (z_test - z_hat)**2 )/numData
    R2 = 1 - np.sum( (z_test - z_hat)**2 )/np.sum( (z - np.mean(z_hat))**2 )
    var = np.sum( (z_hat- np.mean(z_hat))**2 )/numData
    bias = np.sum( (z_test - np.mean(z_hat))**2 )/numData
    """
    
    mse = np.mean( np.mean((z_test - z_hat)**2) )
    R2 = np.sqrt(np.mean( np.mean((z_test - z_hat)**2) ))
    bias = np.mean( (z_test - np.mean(z_hat))**2 )
    var = np.mean( np.var(z_hat) )  
    
    #print('Beta parameters:', beta_ols)
    d.loc[i].mse = mse
    d.loc[i].r2 = R2
    d.loc[i].bias = bias
    d.loc[i].variance = var
    
    
print('--OLS analysis using polynomials in x and y up to ﬁfth order--')   
print('Firs column is the polynomial degree') 
print(d[1:6])

#Evaluating again for a 5th order polynomial with different train sets in order to get confidence intervals
from sklearn.utils import resample
betaMayor = []
numExp = 10
MSE_ = 1.0
poly5 = PolynomialFeatures(degree=5)
data= poly5.fit_transform(indep)
for i in range(0,numExp): #Number of experiments
    data_trainR, z_trainR = resample(data_train, z_train)
    #20% of the data is the test set (given by test_size=0.2)
    regr = LinearRegression()
    regr.fit(data_trainR, z_trainR)
    z_pred = regr.predict(data_test)
    #Getting all the values of Betas
    betaS = regr.coef_
    betaMayor.append(betaS)

    #Calculating MSE, R2, Var and Bias for each experiment
    MSE = np.mean( np.mean((z_test - z_pred)**2) )
    R2 = np.sqrt(np.mean( np.mean((z_test - z_pred)**2) ))
    bias = np.mean( (z_test - np.mean(z_pred))**2 )
    variance = np.mean( np.var(z_pred) )   
   
    if (MSE <= MSE_):
        #Saving the values for the model with less MSE
        Beta_= betaS
        MSE_ = MSE
        R2_= R2
        var_ = variance
        bias_ = bias
        
#Getting a Matrix with all the values from B parameters of each experiment    
#BetaMatrix = betaMayor.reshape(numExp, len(betaS))
print('')
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
print('----------------------------------')


#**********---RIDGE REGRESSION--*******
from sklearn.linear_model import Ridge
#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
#Fit the model
MSE_R = 1
#Loop to test each alpha value
for i in range(10):
    ridgereg = Ridge(alpha=alpha_ridge[i], normalize=True)
    indep_train_ = poly.fit_transform(indep_train)
    ridgereg.fit(indep_train_, dep_train)
    indep_test_ = poly.fit_transform(indep_test)
    y_pred = ridgereg.predict(indep_test_)

    intercept  = ridgereg.intercept_
    betaRidge = ridgereg.coef_
    MSE_ridge = np.mean( np.mean((depen - dep_pred)**2) )
    R2_ridge = np.sqrt(np.mean( np.mean((depen - dep_pred)**2) ))
    bias_ridge = np.mean( (depen - np.mean(dep_pred))**2 )
    variance_ridge = np.mean( np.var(dep_pred) )   

    #Saving the values for the model with less MSE
    if (MSE_ridge <= MSE_R):
        alphaR= alpha_ridge[i]
        Beta_R= betaRidge
        MSE_R = MSE_ridge
        R2_R= R2_ridge
        var_R = variance_ridge
        bias_R = bias_ridge
        
print('Ridge Regresion')
print('alpha:', alphaR)
print('Beta parameters:', Beta_R)
print('Mean Squared Error (MSE): ', MSE_R)
print('Root Mean Squared Error:', R2_R)
print('Variance: ',var_R)
print('Bias:', bias_R)
print('Var + Bias =', var_R+ bias_R)



#**********---LASSO REGRESSION--*******
from sklearn.linear_model import Lasso
alphas= np.logspace(-4, -1, 6)
regr = Lasso()
scores = [regr.set_params(alpha = alpha).fit(indep_train_, dep_train).score(indep_test_, dep_test) for alpha in alphas]

best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(indep_train_, dep_train)

indep_test_ = poly.fit_transform(indep_test)
y_pred = regr.predict(indep_test_)
MSE_lasso = np.mean( np.mean((depen - dep_pred)**2) )
R2_lasso = np.sqrt(np.mean( np.mean((depen - dep_pred)**2) ))
bias_lasso = np.mean( (depen - np.mean(dep_pred))**2 )
variance_lasso = np.mean( np.var(dep_pred) )   
    
       
print('Lasso Regresion')
print('alpha:', best_alpha)
print('Beta parameters:', regr.coef_)
print('Mean Squared Error (MSE): ', MSE_lasso)
print('Root Mean Squared Error:', R2_lasso)
print('Variance: ',variance_lasso)
print('Bias:', bias_lasso)
print('Var + Bias =', variance_lasso+ bias_lasso)


""" Plotting the whole data 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, c='r', marker='o')
ax.set_ylabel('x axis')
ax.set_ylabel('y axis')
ax.set_ylabel('z axis')
plt.show()
"""

