# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:09:57 2018
@author: Sebastiaan Deetman
"""

import scipy.stats 
import pandas as pd
from scipy.optimize import least_squares
import numpy as np
import math
import matplotlib.pyplot as plt
  
#%% Optimization model  
# https://scipy-cookbook.readthedocs.io/items/robust_regression.html
# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp  
  
def model(x,u):
    return scipy.stats.weibull_min.pdf(u,x[0],0,x[1])

def fun(x, u, y):
    return model(x, u) - y

t_start  = 0
t_finish = 401
t_step   = 1

# original set of weibull parameters  --- CHANGE INPUT HERE
            # shape     # scale     # position  #type
distr1 = [  1.486,	           25.595511,     0,      'Weibull']   	# Nomura (avg. of hotels & restaurants, offices, stores & warehouses, & industry)
distr2 = [  2.74955143,     36.02979315,    0,      'Weibull']         	# Daigo (avg. of non-residential steel & concrete buildings)
distr3 = [  4.49162,	       86.03383,      0,      'Weibull']       	# Kapur (avg of commercial & industrial)

name = 'Global_commercial_avg'

#distr = [distr1,distr2,distr3,distr4]
distr = [distr1,distr2,distr3]

average_alpha = (distr[0][0] + distr[1][0])/2
average_beta = (distr[0][1] + distr[1][1])/2

time = np.arange(t_start,t_finish,t_step)
originals = np.zeros((len(time), len(distr))) 

for year in range(0, len(time)):
    for item in range(0, len(distr)):
        if distr[item][3] == 'Weibull':
            originals[year][item] = scipy.stats.weibull_min.pdf(year, distr[item][0],distr[item][2],distr[item][1]) # Call scipy's Weibull function with Shape parameter, loc=0, Scale parameter, and Age
        elif distr[item][3] == 'Lognormal':
            # mean = math.exp(distr[item][0] + (distr[item][1]**2/2))      
            # according to: https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
            originals[year][item] = scipy.stats.lognorm.pdf(year, distr[item][1], loc=0, scale=math.exp(distr[item][0])) 

average = np.average(originals, axis=1) # could be weighted by adding argument: e.g. weights=[1./4, 3./4]    

u = time
y = average

# set the initial guess (starting point to the average of the given weibull parameters)
x0 = [average_alpha, average_beta]
res2 = least_squares(fun, x0, loss='soft_l1', f_scale=1, args=(u, y), verbose=1)

#%% Calculate R-squared (coefficient of determination), 
# See: https://en.wikipedia.org/wiki/Coefficient_of_determination
# See also: https://stackoverflow.com/questions/20115272/calculate-coefficient-of-determination-r2-and-root-mean-square-error-rmse-fo#20115859

shape = res2['x'][0]
scale = res2['x'][1]

def R2_coef_of_determination(time,average,shape,scale):
    ss_res = np.dot((average - scipy.stats.weibull_min.pdf(time,shape,0,scale)),(average - scipy.stats.weibull_min.pdf(time,shape,0,scale)))
    ymean = np.mean(average)
    ss_tot = np.dot((average-ymean),(average-ymean))
    r2 = 1-ss_res/ss_tot    
    print("R^2 :",  1-ss_res/ss_tot)
    return r2

r2 = R2_coef_of_determination(u,y,shape,scale)

#%% Plot comparison

# Setting up a comparison between Weibull distributions
plotvar = pd.DataFrame(index=time)
plotvar[0] = average
plotvar[1] = scipy.stats.weibull_min.pdf(time,shape,0,scale)

for item in range(0, len(originals[0] -1)):
    plotvar[item + 2] = originals.take(indices=item,axis=1)

# Drawing the plot 
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)

ax = fig.add_subplot(111)
ax.text(55, max(y)+ 0.005, 'Weibull: shape=' + "{0:.3f}".format(shape) + ' scale=' + "{0:.3f}".format(scale), style='italic', fontsize=18,
        bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
ax.plot()

for item in range(0, len(originals[0] -1)):
    plt.plot(plotvar[item + 2][0:120], color='green', linewidth=2, label="original" + str(item))

plt.plot(plotvar[0][0:120], color='red', linewidth=2, label="average")
plt.plot(plotvar[1][0:120], color='black', linewidth=2, label="optimized", linestyle='dashed')

plt.tick_params(axis='both', labelsize=20)
fig.suptitle(name + ' (R2= ' + str(round(r2,5)) + ')', fontsize=28)
plt.xlabel('years', fontsize=25)
plt.ylabel('P', fontsize=25)
plt.legend(loc=1, borderaxespad=0.) #bbox_to_anchor=(1.05, 1)
fig.savefig('C:\\Users\\...' + name + '.png') # SET YOU PATH HERE
