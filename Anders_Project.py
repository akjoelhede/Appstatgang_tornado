#%%
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.signal as sp
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


#%%
#Very ugly arrays of all data used in the pendulum and incline plane calculations of the graviational acceleration

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
New_cute_ball_err = np.array([0.01, 0.04, 0.0000001, 0.01, 0.02])

Rail_width = np.array([0.612, 0.628, 0.605, 0.6, 0.630])
Rail_width_err = np.array([0.04, 0.05, 0.03, 0.05, 0.05])

#%%

"Define all the functions to calculate the gravitational acceleration and the error on this"

def csc(theta):
	y = 1/np.sin(theta)
	return y 

def cot(theta):
	y = np.cos(theta)/np.sin(theta)
	return y 

def pend_g(L,T):

	g = L*(2*np.pi/T)**2

	return g

def pend_g_err(L,T,L_err,T_err):

	sig_g = ((2*np.pi/T)**2*L_err - L*((8*np.pi**2)/(T**3))*T_err)**2

	return sig_g

def inc_g(a, theta, D_ball, D_rail):

	g = a/(np.sin(theta*np.pi/180))*(1 + 2/5 *(D_ball**2/(D_ball**2-D_rail**2)))

	return g

def egravity_acc (acc, theta, D_ball, d_rail, eacc, etheta, eD_ball, eD_rail):
    rho_a = (1+(2*D_ball**2)/(D_ball**2-d_rail**2))/np.sin(theta*np.pi/180)
    a_contr = rho_a**2 * eacc**2
    a = acc*np.cos(theta*np.pi/180)
    b1 = 2 * (D_ball**2)
    b2 = 5 * ((D_ball**2) - (d_rail**2))
    b = 1 + (b1 / b2)
    c = (np.sin(theta*np.pi/180))**2
    rho_theta = (a) * (b) / (c)
    theta_contr = rho_theta**2 * (etheta*np.pi/180)**2
    rho_D_ball = acc*(4*D_ball/5*(D_ball**2-d_rail**2)-4*D_ball**3/(5*(D_ball**2-d_rail**2)**2)/np.sin(theta*np.pi/180))
    D_ball_contr = rho_D_ball**2 * eD_ball**2
    rho_d_rail = 4*acc*D_ball**2*d_rail/5*np.sin(theta*np.pi/180)*(D_ball**2-d_rail**2)**2
    d_rail_contr = rho_d_rail**2 * eD_rail**2
    eg = np.sqrt(a_contr + theta_contr + D_ball_contr + d_rail_contr)
    return eg


#%%
#Reads the data from a .csv file and puts it into a pandas dataframe
def read_csv(filename):
	dat = pd.read_csv(filename, sep = ',', header = 13, names = ["Time (s)", "voltage (V)"]) 
	return dat

#Used SciPy to find the peaks where the ball passes the each of the individual 5 gates on the rail.
def find_peaks(filename): 

	dat = read_csv(filename)
	peaks = sp.find_peaks(dat["voltage (V)"], height = 4, distance = 100)
	time = []
	for i in peaks[0]: 
		time.append(dat["Time (s)"][i])
	return time

"_______________FIT AND PLOT FOR INCLINE PLANE WITH ONE FILE______________"

#List of filenames used in the incline plane
filename = "Anders_L_fp.csv" 
path = "Data/Incline/"   

print(path + filename)


x = np.array(find_peaks(path + filename))
y = Anders_inc
y_err = Anders_inc_err


#Read and plot the Voltage vs time to for a visualization of the peaks
dat = read_csv(path + filename)
print(find_peaks(path + filename))
plt.plot(dat["Time (s)" ], dat["voltage (V)"])

plt.show()


#%%
#Fit the function with a polynomial using minuit

#Define the function we wanna fit after
def line(x, a, b, c): 
	return c + x * b + (a/2)*x**2

#%%
#We are using leastsquares fit here, this takes x, y, y_error, fit_func
leastSquares = LeastSquares(x, y, y_err, line)

#Makes the fit with minuit using our least square fit and guesses on the parameters.
m = Minuit(leastSquares, a=1, b=1, c=1)  

m.migrad()  # finds minimum of least_squares function
m.hesse()   # accurately computes uncertainties

#%%

# draw data and fitted line
plt.errorbar(x, y, yerr=y_err, fmt="o")
plt.plot(x, line(x, *m.values), label="fit")

#Plot our fit info
fit_info = [
	f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(x) - m.nfit}",]
for p, v, e in zip(m.parameters, m.values, m.errors):
	fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info));
plt.show()

# %%
print(inc_g(m.values[0]/100,np.mean(Angle),np.mean(New_cute_ball)/100, np.mean(Rail_width)/100))

#print(np.sqrt(inc_g_err(m.values[0]/100, np.mean(Angle), np.mean(New_cute_ball)/100, np.mean(Rail_width)/100, 6.669/100, np.mean(Angle_err), np.mean(New_cute_ball_err)/100, np.mean(Rail_width_err))))

incline_g = np.array([])
incline_g_err = np.array([])


#%%

"______________________FULLY ITERATIVE LOOP OVER ALL INCLINE PLANE FILES FILES___________________________"

Incline_name_list_fp = ["Anders_L_fp.csv", "cecil_L_fp.csv", "gustav_L_fp.csv", "Vic_L_fp.csv", "Mort_L_fp.csv"]

Incline_name_list_sp = ["Andersl_L_sp.csv", "cecil_L_sp.csv", "gustav_L_sp.csv", "Vic_L_sp.csv", "Mort_L_sp.csv"]

Incline_length_list = [Anders_inc, Cecilie_inc, Gustav_inc, Victoria_inc, Morten_inc]

Incline_length_list_err = [Anders_inc_err, Cecilie_inc_err, Gustav_inc_err, Victoria_inc_err, Morten_inc_err]


Incline_grav = []
Incline_grav_err = []

for i, j, k in zip(Incline_name_list_fp, Incline_length_list, Incline_length_list_err):

	filename = i
	path = "Data/Incline/"   

	x = np.array(find_peaks(path + filename))
	y = j
	y_err = k

	def line(x, a, b, c): 
		return c + x * b + (a/2)*x**2

	#We are using leastsquares fit here, this takes x, y, y_error, fit_func
	leastSquares = LeastSquares(x, y, y_err, line)

	#Makes the fit with minuit using our least square fit and guesses on the parameters.
	m = Minuit(leastSquares, a=1, b=1, c=1)  

	m.migrad()  # finds minimum of least_squares function
	m.hesse()   # accurately computes uncertainties

	g = inc_g(m.values[0]/100,np.mean(Angle),np.mean(New_cute_ball)/100, np.mean(Rail_width)/100)

	g_err = egravity_acc (m.values[0]/100, np.mean(Angle), np.mean(New_cute_ball)/100, np.mean(Rail_width)/100, 6.6, np.mean(Angle_err), np.mean(New_cute_ball_err)/100, np.mean(Rail_width_err)/100 )/100
	#g_err = np.sqrt(inc_g_err(m.values[0]/100, np.mean(Angle), np.mean(New_cute_ball)/100, np.mean(Rail_width)/100, 6.669/100, np.mean(Angle_err), np.mean(New_cute_ball_err)/100, np.mean(Rail_width_err)))

	Incline_grav.append(g)
	Incline_grav_err.append(g_err)

Incline_grav = np.array(Incline_grav)
Incline_grav_err = np.array(Incline_grav_err)
print(np.mean(Incline_grav),np.mean(Incline_grav_err))

# %%

"_______________FIT AND PLOT FOR PENDULUM WITH ONE FILE______________"

def read_dat(filename):
	dat = pd.read_csv(filename, sep = '\t', names = ["Number (n)","Time (s)"]) 
	return dat

filename = "timer_output_gustav2.dat" 
path = "Data/Pendulum/" 

dat = read_dat(path + filename)

x = dat["Number (n)"]
y = dat["Time (s)"]
y_err = np.zeros_like(y)+1
print(len(x))

plt.plot(dat["Number (n)"],dat["Time (s)"], 'o')
# %%
#Fit the function with a polynomial using minuit

#Define the function we wanna fit after
def line(x, a, b): 
	return a * x + b

#%%
#We are using leastsquares fit here, this takes x, y, y_error, fit_func
leastSquares = LeastSquares(x, y, y_err, line)

#Makes the fit with minuit using our least square fit and guesses on the parameters.
m = Minuit(leastSquares, a=8, b=10)  

m.migrad()  # finds minimum of least_squares function
m.hesse()   # accurately computes uncertainties

#%%

# draw data and fitted line
plt.errorbar(x, y, yerr=y_err, fmt="o")
plt.plot(x, line(x, *m.values), label="fit")

#Plot our fit info
fit_info = [
	f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(x) - m.nfit}",]
for p, v, e in zip(m.parameters, m.values, m.errors):
	fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info));
plt.show()
# %%

pend_g(np.mean(Pen_length)/100,m.values[0])

# %%

"__________________FULLY ITERATIVE LOOP OVER ALL PENDULUM FILES_________________________"

Pend_name_list = ["timer_output_anders.dat","timer_output_cecilie.dat","timer_output_gustav2.dat","timer_output_morten.dat","timer_output_victoria.dat"]

Pendulum_Grav = []
Pendulum_Grav_err = []

for i in Pend_name_list:
	filename = i 
	path = "Data/Pendulum/" 

	dat = read_dat(path + filename)

	x = dat["Number (n)"]
	y = dat["Time (s)"]
	y_err = 0.1

	def line(x, a, b): 
		return a * x + b

	#We are using leastsquares fit here, this takes x, y, y_error, fit_func
	leastSquares = LeastSquares(x, y, y_err, line)

	#Makes the fit with minuit using our least square fit and guesses on the parameters.
	m = Minuit(leastSquares, a=8, b=10)

	m.migrad()  # finds minimum of least_squares function
	m.hesse() 

	P_g = pend_g(np.mean(Pen_length)/100,m.values[0])

	P_g_err = pend_g_err(np.mean(Pen_length)/100,m.values[0],np.mean(Pen_length_err)/100,y_err)

	Pendulum_Grav.append(P_g)
	Pendulum_Grav_err.append(P_g_err)
#%%

Pendulum_Grav = np.array(Pendulum_Grav)
Pendulum_Grav_err = np.array(Pendulum_Grav_err)

print(np.mean(Pendulum_Grav), np.mean(Pendulum_Grav_err))


# %%
