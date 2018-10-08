#Not finished

import numpy as np
from scipy.misc import imread
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def surface_plot(surface,title, surface1=None):
    M,N = surface.shape

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [X,Y] = np.meshgrid(ax_cols, ax_rows)

    fig = plt.figure()
    if surface1 is not None:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title('Real data')

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(X,Y,surface1,cmap=cm.viridis,linewidth=0)
        plt.title(title)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def my_polyfit(num_data, poly_deg, x, y):
    if poly_deg == 1:
        data_vec = np.c_[np.ones((num_data,1)), x, y]
    elif poly_deg == 2:
        data_vec = np.c_[np.ones((num_data,1)), x, y, x**2, x*y, y**2]
    elif poly_deg == 3:
        data_vec = np.c_[np.ones((num_data,1)), x, y, x**2, x*y, y**2, \
                            x**3, x**2*y, x*y**2, y**3]
    elif poly_deg == 4:
        data_vec = np.c_[np.ones((num_data,1)), x, y, x**2, x*y, y**2, \
                            x**3, x**2*y, x*y**2, y**3, \
                            x**4, x**3*y, x**2*y**2, x*y**3,y**4]
    elif poly_deg == 5: 
        data_vec = np.c_[np.ones((num_data,1)), x, y, x**2, x*y, y**2, \
                            x**3, x**2*y, x*y**2, y**3, \
                            x**4, x**3*y, x**2*y**2, x*y**3,y**4, \
                            x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4,y**5]
    return data_vec


def predict(rows, cols, beta, poly_deg):
    out = np.zeros((np.size(rows), np.size(cols)))

    for i,y_ in enumerate(rows):
        for j,x_ in enumerate(cols):
            if poly_deg == 1:
                data_vec = np.array([1, x_, y_])
            elif poly_deg == 2:
                data_vec = np.array([1, x_, y_, x_**2, x_*y_, y_**2])
            elif poly_deg == 3:
                data_vec = np.array([1, x_, y_, x_**2, x_*y_, y_**2, \
                                x_**3, x_**2*y_, x_*y_**2, y_**3])
            elif poly_deg == 4:
                data_vec = np.array([1, x_, y_, x_**2, x_*y_, y_**2, \
                                x_**3, x_**2*y_, x_*y_**2, y_**3, \
                                x_**4, x_**3*y_, x_**2*y_**2, x_*y_**3,y_**4])
            elif poly_deg == 5:
                data_vec = np.array([1, x_, y_, x_**2, x_*y_, y_**2, \
                                x_**3, x_**2*y_, x_*y_**2, y_**3, \
                                x_**4, x_**3*y_, x_**2*y_**2, x_*y_**3,y_**4, \
                                x_**5, x_**4*y_, x_**3*y_**2, x_**2*y_**3,x_*y_**4,y_**5])
            out[i,j] = data_vec @ beta
    return out

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
if __name__ == '__main__':
    
    terrain1 = imread('SRTM_data_Norway_1.tif')
    [n,m] = terrain1.shape
    
    datasizex = 100
    datasizey = 50
    poly_deg = 5
    noise = 0.01

    ## Toggle comment to change between arange and uniform distribution
    rows = np.arange(0,1,1/datasizey)
    cols = np.arange(0,1,1/datasizex)
    # rows = sorted(np.random.uniform(0,1,datasizey))
    # cols = sorted(np.random.uniform(0,1,datasizex))

    [C,R] = np.meshgrid(cols,rows)
    
    x = C.reshape(-1,1)
    y = R.reshape(-1,1)
    
    #[n,m] = C.shape
    num_data = datasizex*datasizey
    
    data = my_polyfit(num_data, poly_deg, x, y)

    patch = terrain1[row_start:row_end, col_start:col_end]
     
    z = patch.reshape(-1,1)

    # Adding noise to function
    for i in range(0,len(z)):
        z[i] = z[i] + noise*np.random.normal(0,1)
    
    # Calculating parameters with OLS
   # terrain1 = z.reshape(n,m)
    beta_ols = np.linalg.inv(data.T @ data) @ data.T @ z

    fitted_terrain = predict(rows, cols, beta_ols, poly_deg)
    #surface_plot(terrain1, 'Fitted data', fitted_terrain)

    zmean = np.mean(terrain1)

    mse = np.sum( (fitted_terrain - terrain1)**2 )/num_data
    R2 = 1 - np.sum( (fitted_terrain - terrain1)**2 )/np.sum( (fitted_terrain - np.mean(terrain1))**2 )
    var = np.sum( (fitted_terrain - np.mean(fitted_terrain))**2 )/num_data
    bias = np.sum( (terrain1 - np.mean(fitted_terrain))**2 )/num_data

    print("beta:", beta_ols)
    print("mse: %g\nR2: %g"%(mse, R2))
    print("variance: %g"%var)
    print("bias: %g\n"%bias)
    print("---SCIKIT---")
    print("mse: %g\nR2: %g"%(mean_squared_error(terrain1, fitted_terrain), r2_score(terrain1, fitted_terrain)))
    print("variance: %g"%np.var(fitted_terrain))
    plt.show()
