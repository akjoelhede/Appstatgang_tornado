#%%
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.signal as sp
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
sys.path.append('External_Functions')
from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax

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

	sig_g = 4 * np.sqrt((np.pi**4*L_err**2/T**4) + (4*L*2*np.pi**4*T_err**2/T**6)) 

	return sig_g

def inc_g(a, theta, D_ball, D_rail):

	g = a/(np.sin(theta*np.pi/180))*(1 + 2/5 *(D_ball**2/(D_ball**2-D_rail**2)))

	return g

def inc_g_err(a, theta, D_ball, D_rail, a_err, theta_err, D_ball_err, D_rail_err):

	sig_g = (1 + 5/2 * (D_ball**2)/(D_ball**2-D_rail**2))*csc(theta*np.pi/180)*a_err - a * (1 + 5/2 * (D_ball**2)/(D_ball**2-D_rail**2))*cot(theta*np.pi/180)*csc(theta*np.pi/180)*theta_err - a*csc(theta*np.pi/180)*(5*D_rail**2*D_ball)/(D_rail**2-D_ball**2)**2 * D_ball_err + a * csc(theta*np.pi/180) * (5*D_rail*D_ball**2)/(D_rail**2-D_ball**2)**2 * D_rail_err

	return sig_g


#%%
#Reads the data from a .csv file and puts it into a pandas dataframe
def read_csv(filename):
	dat = pd.read_csv(filename, sep = ',', header = 13, names = ["Time (s)", "voltage (V)"]) 
	return dat

#Used SciPy to find the peaks where the ball passes the each of the individual 5 gates on the rail.
def find_peaks(filename): 

	dat = read_csv(filename)
	peaks = sp.find_peaks(dat["voltage (V)"], height = 1, distance = 100)
	time = []
	for i in peaks[0]: 
		time.append(dat["Time (s)"][i])
	return time

#%%
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

	#g_err = np.sqrt(inc_g_err(m.values[0]/100, np.mean(Angle), np.mean(New_cute_ball)/100, np.mean(Rail_width)/100, 6.669/100, np.mean(Angle_err), np.mean(New_cute_ball_err)/100, np.mean(Rail_width_err)))

	Incline_grav.append(g)

Incline_grav = np.array(Incline_grav)
print(np.mean(Incline_grav))
# %%

def ChiDaddy(measure,usikkerhed):
	gennem = np.mean(measure)
	chi_val = np.sum((measure-gennem)**2/usikkerhed**2)
	return chi_val, chi2.sf(chi_val,len(measure)-1)


#%%

"_______________FIT AND PLOT FOR PENDULUM WITH ONE FILE______________"

Pend_name_list = ["timer_output_anders.dat","timer_output_cecilie.dat","timer_output_gustav2.dat","timer_output_morten.dat","timer_output_victoria.dat"]

def read_dat(filename):
	dat = pd.read_csv(filename, sep = '\t', names = ["Number (n)","Time (s)"]) 
	return dat

filename = "timer_output_gustav2.dat" 
path = "Data/Pendulum/" 

dat = read_dat(path + filename)

x = dat["Number (n)"]
y = dat["Time (s)"]
y_err = 0.1

#Fit the function with a polynomial using minuit

#Define the function we wanna fit after
def line(x, a, b): 
	return a * x + b

#We are using leastsquares fit here, this takes x, y, y_error, fit_func
Chi2 = Chi2Regression(line, x, y, y_err)

#Makes the fit with minuit using our least square fit and guesses on the parameters.
m = Minuit(Chi2, a=8, b=10)  

m.migrad()  # finds minimum of least_squares function
m.hesse()   # accurately computes uncertainties


res = y - line(x, *m.values)
print(res)

mu, sigma = 0, np.std(res) # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

#%%
fit_info = [f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(x) - m.nfit}",]
for p, v, e in zip(m.parameters, m.values, m.errors):
	fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr=y_err, fmt="o", color = 'black')
ax.errorbar(x, line(x, *m.values), label="fit", color = 'blue' )
ax.set_ylim(-30,180)
ax.set_xlabel('Measurement number (N)', fontsize = 15)
ax.set_ylabel('Timeß elapsed (S)', fontsize = 15)
ax.set_title('Pendulum', fontsize = 20)
ax.legend(title="\n".join(fit_info));

ax2 = ax.twinx()

ax2.errorbar(x, res, np.std(res), fmt = ".", color = 'red', capsize = 3)
ax2.set_ylim(-0.5,3)
ax2.hlines(np.std(res),0,20, color = 'black', linestyle = '--')
ax2.hlines(0,0,20, color = 'black', linestyle = 'solid')
ax2.hlines(-np.std(res),0,20, color = 'black', linestyle = '--')
ax2.set_ylabel('Residuals (S)', fontsize = 15)
yticks = ax2.yaxis.get_major_ticks()
yticks[3].set_visible(False)
yticks[4].set_visible(False)
yticks[5].set_visible(False)
yticks[6].set_visible(False)
yticks[7].set_visible(False)

counts, bin_edges = np.histogram(res, density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

ax3 = ax.inset_axes([0.67, 0.35, 0.3, 0.3])
ax3.errorbar(bin_centres, counts, 1/np.sqrt(counts), 1/15 , fmt =  '.', color = 'black')
ax3.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
ax3.set_xlim(-0.5, 0.5)
ax3.set_ylim(0,6)
ax3.set_title('Distribution of Residuals', fontsize = 9)
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Time Residuals (S)')


#%%
np.std(res)
#%%


print(pend_g(np.mean(Pen_length)/100,m.values[0]), pend_g_err(np.mean(Pen_length-Hook_length-Floor_to_pen)/100,m.values[0],np.mean(Pen_length_err+Hook_length_err+Floor_to_pen_err)/100,y_err))

print(np.mean(Pen_length-Hook_length-Floor_to_pen)/100, np.mean(Pen_length_err+Hook_length_err+Floor_to_pen_err)/100)

print(np.mean(y_err))

print(np.sqrt(1/len(x)),np.sqrt(1/2))

print(pend_g_err(np.mean(Pen_length-Hook_length-Floor_to_pen)/100,m.values[0],np.sqrt(1/len(x)),np.sqrt(1/len(Pen_length))))

#%%
np.mean(Pen_length-Hook_length-Floor_to_pen)/100
#%%
print(np.pi**4*np.mean(Pen_length_err)**2/m.values[0]**4, (4*np.mean(Pen_length)*2*np.pi**4*m.errors[0]**2/m.values[0]**6))
# %%

"__________________FULLY ITERATIVE LOOP OVER ALL PENDULUM FILES_________________________"

Pend_name_list = ["timer_output_anders.dat","timer_output_cecilie.dat","timer_output_gustav2.dat","timer_output_morten.dat","timer_output_victoria.dat"]

Pend_name_Gustav = ["timer_output_gustav2.dat"]

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
	Chi2 = Chi2Regression(line, x, y, y_err)

	#Makes the fit with minuit using our least square fit and guesses on the parameters.
	m = Minuit(Chi2, a=8, b=10)  

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
print(Pendulum_Grav)

# %%

chi2.sf(16.7,16)

# %%
