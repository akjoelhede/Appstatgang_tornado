import numpy as np
import matplotlib.pyplot as plt


#Read pendulum periods

file_name = ['Anders.dat','Morten.dat','Gustav.dat','Victoria.dat','Cecilie.dat']

def read_data(file_name):
	return dat = np.genfromtext(file_name, delimiter='\t', names = ('n', 't'))


plt.plot(t,n,'.')

plt.show()
