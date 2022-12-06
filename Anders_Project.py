import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.signal as sp
from iminuit import Minuit
from iminuit.cost import LeastSquares

#Read pendulum periods

def read_csv(filename):
	dat = pd.read_csv(filename, sep = ',', header = 13, names = ["Time (s)", "voltage (V)"]) 
	return dat

def find_peaks(filename): 

	dat = read_csv(filename)
	peaks = sp.find_peaks(dat["voltage (V)"], height = 4, distance = 100)
	time = []
	for i in peaks[0]: 
		time.append(dat["Time (s)"][i])
	return time


filename = "Anders_L_fp.csv" 
path = "Data/Incline/"   


print(path + filename)

dat = read_csv(path + filename)

print(find_peaks(path + filename))
plt.plot(dat["Time (s)" ], dat["voltage (V)"])

plt.show()

#Fit the function with a polynomial using minuit, here using sample data

def line(x, a, b): 
    return a + x * b

np.random.seed(1)
data_x = np.linspace(0, 1, 10)
data_yerr = 0.1  # could also be an array
data_y = line(data_x, 1, 2) + data_yerr * np.random.randn(len(data_x))


leastSquares = LeastSquares(data_x, data_y, data_yerr, line)

m = Minuit(leastSquares, a=0, b=0)  # starting values for α and β

m.migrad()  # finds minimum of least_squares function
m.hesse()   # accurately computes uncertainties

# draw data and fitted line
plt.errorbar(data_x, data_y, data_yerr, fmt="o", label="data")
plt.plot(data_x, line(data_x, *m.values), label="fit")

fit_info = [
	f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}",]
for p, v, e in zip(m.parameters, m.values, m.errors):
	fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info));
plt.show()