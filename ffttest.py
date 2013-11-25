import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import math

def main():
	
	signal1 = [random.uniform(0,1000) for i in range(1000)]
	signal2 = [random.uniform(0,1000)*i+math.pow(-1,i)*i*i for i in range(1000)]

	f1 = sp.fft( signal1 )
	f2 = sp.fft( signal2 )
	
	f = plt.figure()
	f.add_subplot(2,2,1); plt.plot( signal1 )
	f.add_subplot(2,2,2); plt.plot( signal2 )
	f.add_subplot(2,2,3); plt.plot( f1 )
	f.add_subplot(2,2,4); plt.plot( f2 )
	plt.show()

	return

main()
