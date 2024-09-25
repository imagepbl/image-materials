# according to https://stackoverflow.com/questions/44878372/how-can-i-apply-weights-in-this-scipy-least-squares-optimization-routine
import numpy as np
from scipy import optimize
import math

guess = [25, 3.3, 0.07]
name = 'services'
alpha_fact = 1.1  # set to 1.1 for sensitivity variant (10% above max of y)

def fit_func(ps, xs):
    ## FIT TO EQUATION OF GOMPERTZ CURVE
    result = []
    for item in xs:
        result.append(ps[0] * math.exp(-ps[1] * math.exp((-ps[2]/1000) * item)) )   
    return result

def err_func(ps, xs, ys):
    ## GET FIT
    ys_trial = fit_func(ps, xs)
    ## GET RESIDUALS
    residuals = [(ys[idx] -  ys_trial[idx])**2 for idx in range(len(ys))]
    return sum(residuals)

# see: https://en.wikipedia.org/wiki/Coefficient_of_determination
def R2_coef_of_determination(xs,ys,ps):     
    ss_res = np.dot((ys - fit_func(ps,xs)),(ys - fit_func(ps,xs)))          # sum of the squares of the residuals
    ymean = np.mean(ys)                                                     # mean
    ss_tot = np.dot((ys-ymean),(ys-ymean))                                  # total sum of the squares
    r2 = 1-ss_res/ss_tot    
    print("R^2 :",  1-ss_res/ss_tot)
    return r2

def get_chi_squared(ps, xs, ys, wts):
    ys_trial = fit_func(ps, xs)    
    resid = [(ys[idx] - ys_trial[idx]) **2 * (1 / wts[idx])**2  for idx in range(len(ys))]   
    ss_res = sum(resid)                            # sum of the weighted squares of the residuals 
    print("X^2 (weighted):",  ss_res)
    return ss_res    

import os

os.chdir("C:\\Users\\Sebastiaan\\surfdrive\\Paper_3\\Python")
csv = np.genfromtxt('files_commercial\\data_' + name + '_PPP.csv', delimiter=",")
csv_excl = np.genfromtxt('files_commercial\\outliers_' + name + '_PPP.csv', delimiter=",")

xdata = csv.transpose()[0]
ydata = csv.transpose()[1]
xexcl = csv_excl.transpose()[0]
yexcl = csv_excl.transpose()[1]
pop   = csv.transpose()[2]      # population in thousands
gdp   = csv.transpose()[3]      # GDP per capita in 2016 dollars (PPP)
sigma_choice = pop if name == 'services' else gdp   # choice for sigma based on population (srevices all) or based on GDP/cap (all 4 sub-categories)

# all ones for non-weighted chi calculation
non_weighted = np.ones(len(xdata))

# calculate mean y value (plain or population weighted)
# y_mean = np.mean(ydata)
y_mean_weighted = sum(sigma_choice * ydata) / sum(sigma_choice)

# Make sigma manually based on population
sig_max = 0.3
sig_min = 0.015
sig_max_value = 200000 if name == 'services' else 40000 
sig_perc = [max(sig_min, sig_max-((sig_max-sig_min)/sig_max_value)*sigma_choice[idx]) for idx in range(len(sigma_choice))]
sig_manual = y_mean_weighted * np.array(sig_perc)

# FIND FITS
# Unweighted
ans1 = optimize.minimize(err_func, x0=guess, args=(xdata, ydata), method='SLSQP', bounds=[(0, (max(ydata)*alpha_fact)),(0.00, 20.0),(0.01,1.00)], options={'maxiter': 10000, 'ftol': 1e-06, 'disp': False, 'eps': 1e-08})
[alpha, beta, gamma] = ans1.x[0], ans1.x[1], ans1.x[2]

# Weighted # L-BFGS-B, Powell, SLSQP
ans2 = optimize.minimize(get_chi_squared, x0=guess, args=(xdata, ydata, sig_manual), method='SLSQP', bounds=[(0.00, (max(ydata)*alpha_fact)),(0.00, 20.0),(0.01,1.00)], options={'maxiter': 10000, 'ftol': 1e-06, 'disp': False, 'eps': 1e-08})
[alpha_sig, beta_sig, gamma_sig] = ans2.x[0], ans2.x[1], ans2.x[2]

#Find R-squared or Goodness of fit parameters
rsquared = R2_coef_of_determination(xdata, ydata,[alpha, beta, gamma]) 
rsquared_sig = R2_coef_of_determination(xdata, ydata,[alpha_sig, beta_sig, gamma_sig]) 

chi_squared = get_chi_squared(xdata, ydata,[alpha_sig, beta_sig, gamma_sig], non_weighted)         # non-Weighted R-squared or Goodnes-of-fit parameter X-squared
chi_squared_sig   = get_chi_squared(xdata, ydata,[alpha_sig, beta_sig, gamma_sig], sig_manual)   # Weighted R-squared or Goodnes-of-fit parameter X-squared

# find the total global square meters according to the data in 2017 
data_sqr_meters = ydata * pop
sqr_meters_total_data = sum(data_sqr_meters)

# Find the resulting total global square meters according to the fitted function
# Wherever the function yields negative numbers, we assume 0 
def current_fit(xs, pop, par):  # par = [alpha, beta]
    optim_sqr_meters = []
    for point in range(0,len(xs)):
        if (fit_func(par, [xs[point]])[0] > 0): 
            optim_sqr_meters.append(fit_func(par, [xs[point]])[0] * pop[point])
        else: 
            optim_sqr_meters.append(0)
    return sum(optim_sqr_meters)

optim_sqr_meters_non = current_fit(xdata, pop, [alpha, beta, gamma])
optim_sqr_meters_sig = current_fit(xdata, pop, [alpha_sig, beta_sig, gamma_sig])

current_fit_non = optim_sqr_meters_non/sqr_meters_total_data
current_fit_sig = optim_sqr_meters_sig/sqr_meters_total_data

print('current fit (non-weighted): ' + str(current_fit_non))
print('current fit (pop-weighted): ' + str(current_fit_sig))
#%% Plot

import pandas as pd
import matplotlib.pyplot as plt

# Setting up a comparison between Weibull distributions
plotvar = pd.DataFrame(index=range(0,int(max(xdata + 1)),1))
plotvar[0] = fit_func([alpha,beta, gamma],range(0,int(max(xdata + 1)),1))
plotvar[1] = fit_func([alpha_sig,beta_sig, gamma_sig],range(0,int(max(xdata + 1)),1))

# Drawing the plot 
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax = fig.add_subplot(111)
ax.text(2000, math.floor(max(ydata)-1), 'a=' + "{0:.3f}".format(alpha) + ' b=' + "{0:.3f}".format(beta) + ' c=' + "{0:.4f}".format(gamma) + ' fit=' + "{0:.3f}".format(current_fit_non) + ' R^2=' + "{0:.3f}".format(rsquared), style='italic', fontsize=16,
        bbox={'facecolor':'black', 'alpha':0.5, 'pad':3})
ax.text(2000, math.floor(max(ydata))+0.5, 'a=' + "{0:.3f}".format(alpha_sig) + ' b=' + "{0:.3f}".format(beta_sig) + ' c=' + "{0:.4f}".format(gamma_sig) + ' fit=' + "{0:.3f}".format(current_fit_sig) + ' R^2=' + "{0:.3f}".format(rsquared_sig), style='italic', fontsize=16,
        bbox={'facecolor':'blue', 'alpha':0.5, 'pad':3})
ax.set_ylim([0,math.ceil(max(ydata)*alpha_fact)])
ax.set_xlim([0,95000])
ax.plot()
plt.scatter(xdata, ydata, s=20,  label="data")
plt.scatter(xexcl, yexcl, s=20, marker="x", label="outliers")
plt.plot(plotvar[0], color='black', linewidth=2, label='fit (Gompertz, unweighted)')
plt.plot(plotvar[1], color='blue', linewidth=2, label='fit (Gompertz, weighted)')
plt.tick_params(axis='both', labelsize=20)
fig.suptitle('Development of per capita ' + name + ' floorspace demand (Gompertz)', fontsize=28)
plt.xlabel('SVA US-$ /cap (2016, PPP) / yr', fontsize=25)
plt.ylabel('m2/cap ' + name + ' floorspace', fontsize=25)
plt.legend(loc=4, borderaxespad=0.) #bbox_to_anchor=(1.05, 1)
fig.savefig(name + '_fit.png')
