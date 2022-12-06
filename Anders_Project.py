#%%
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.signal as sp
from iminuit import Minuit
from iminuit.cost import LeastSquares

#%%
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

x = np.array(find_peaks(path + filename))

dat = read_csv(path + filename)

print(find_peaks(path + filename))
plt.plot(dat["Time (s)" ], dat["voltage (V)"])

plt.show()

#%%

"_____________PENDULUM_DATA____________"

Pen_length = np.array([1842.02, 1841.97, 1841.83, 1842.2, 1841.4])
Pen_length_err = np.array([0.2, 0.4, 1.2, 0.4, 0.2])

Pen_laser = np.array([18.743, 18.782, 18.834])
Pen_laser_err = np.array([0.01, 0.01, 0.01])

Pen_laser_top = np.array([18.768, 1878, 1878.4, 1877.1, 1877.2])
Pen_laser_top_err = np.array([1, 0.01, 0.5, 0.01, 0.01])

Hook_length = np.array([2.1, 2.1, 2.2, 2.1, 2.1])
Hook_length_err = np.array([0.1, 0.1, 0.2, 0.1, 0.1])

Floor_to_pen = np.array([30.3, 31.8, 33.5, 31.3, 30.2])
Floor_to_pen_err = np.array([0.5, 0.6, 0.6, 0.6, 1])

Pen_size = np.array([3.001, 3.02, 3.009, 3.002, 3.002])
Pen_size_err = np.array([0.001, 0.08, 0.01, 0.001, 0.002])

"_____________INCLINE_PLANE_DATA____________"

Cecilie_inc = np.array([22.6, 39.15, 56.75, 73.3, 90.9])
Cecilie_inc_err = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

Gustav_inc = np.array([22.6, 39.15, 56.8, 73.25, 90.85])
Gustav_inc_err = np.array([0.1, 0.1, 0.05, 0.05, 0.05])

Anders_inc = np.array([22.6, 39.2, 56.85, 73.35, 91.1])
Anders_inc_err = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

Morten_inc = np.array([22.55, 39.15, 56.75, 73.26, 90.9])
Morten_inc_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

Victoria_inc = np.array([22.58, 39.19, 56.81, 73.24, 90.88])
Victoria_inc_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

Angle = np.array([14.3, 14.3, 14.3, 14.1, 14.3])
Angle_err = np.array([0.5, 0.5, 0.2, 0.5, 0.5])

Angle_turn = np.array([13.1, 13.15, 13.1, 13.2, 13.0])
Angle_turn_err = np.array([0.3, 0.5, 0.2, 0.3, 0.2])

New_cute_ball = np.array([1.20, 1.20, 1.20, 1.20, 1.20])
New_cute_ball = np.array([0.01, 0.04, 0.0000001, 0.01, 0.02])

Rail_width = np.array([0.612, 0.628, 0.605, 0.6, 0.630])
Rail_width_err = np.array([0.04, 0.05, 0.03, 0.05, 0.05])


#%%
print(len(find_peaks(path + filename)), len(Anders_inc))


plt.show()
#%%

print(np.array([find_peaks(path + filename)]).size)
print(Anders_inc.size)

#%%
#Fit the function with a polynomial using minuit, here using sample data

def line(x, a, b, c): 
	return a + x * b + c*x**2



#%%

leastSquares = LeastSquares(find_peaks(path + filename), Anders_inc, Anders_inc_err, line)

m = Minuit(leastSquares, a=16, b=56, c=44)  # starting values for α and β

m.migrad()  # finds minimum of least_squares function
m.hesse()   # accurately computes uncertainties

print(Anders_inc)
print(x)
print(Anders_inc_err.size)

#%%

# draw data and fitted line
plt.errorbar(x, Anders_inc, yerr=Anders_inc_err, fmt="o")
plt.plot(x, line(x, *m.values), label="fit")

fit_info = [
	f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(x) - m.nfit}",]
for p, v, e in zip(m.parameters, m.values, m.errors):
	fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info));
plt.show()
# %%
