# -*- coding: UTF-8 -*-

import re
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class OratUtils:

	@staticmethod
	def im2float( image ):
		I = image.astype( 'float' )
		minI = 0.0
		maxI = 255.0
		I[:,:] = (I[:,:] - minI)/(maxI - minI)
		return I

	@staticmethod
	def im2uint8( image ):

		I = image
		I[I>1] = 1
		I[I<0] = 0
		I *= 255
		I = I.astype('uint8')


		return I

	# Performs case sensitive search for text file tfile with string or character c on default.
	# Argument c can be any regular expression
	@staticmethod
	def stringparser( tfile, c ):

		charcount = [] # Holds the lengths of each line
		charlines = []
		charpos = []
		wordlens = []
		k = 0

		fid = open( tfile, 'r' )

		lines = fid.readlines()
		fid.close()

		
		for string in lines:

			# Each string needs to be decoded to utf-8 as the files are saved in utf-8 format. 
			# Without decoding matching would be done to ascii decoding and that causes the 
			# strings to contain extra characters.
			# Also the newline characters are removed so that the length of the lines are 
			# correct

			s = string.rstrip('\n').decode('utf-8', 'ignore')	
			charindlist = []
			charcount.append(len(s))
			
			for m in re.finditer(c, s):
				charindlist.append(m.start())
			
			if charindlist:	
				charpos.append(charindlist)
				
				tempv1 = [k]*len(charindlist)		# These two temporary values are needed to append right amount of linenumbers and wordlenght numbers
				tempv2 = [len(c)]*len(charindlist)	# into their respective vectors
				
				charlines.append(tempv1)
				wordlens.append(tempv2)

			k += 1


		return charcount, charpos, charlines, wordlens




	@staticmethod
	def hfilter( img, d, h, l, n ):

		# img must be in ndarray format. Inside osearch PIL images are converted to scipy images which are in ndarray format

		# h = height
		# l = length

		warnings.filterwarnings('error')

		A = np.zeros( (h,l), np.float )
		F = np.zeros( (h,l), np.float )

		# print str(A.dtype) == "float64"
		# print F.dtype

		# Ongelma tässä on, että laskut pitäisi tehdä kaikkien muuttujien ollen floatteja mutta osa muuttujista käsitellään intteinä ja siten tuloksetkin välillä pyörisettään nollaksi mikäli tulos on <0.5
		for i in range(h):
			for j in range(l):
				try:
					H = float(h)
					L = float(l)
					I = float(i)
					J = float(j)
					D = float(d)
					N = float(n)
					A[i,j] = float(math.sqrt( math.pow( ( I-H/2 ),2 ) + math.pow( ( J-L/2 ),2 ) )) # Distance from the center of the image
					F[i,j] = float(1/( 1 + math.pow( ( D/( A[i,j] ) ) , (2*N) )))
				except Warning:
					print '***** Warning divide by zero happened in Butterworth filtering *****'


		aL = 0.949
		aH = 1.51

		# print A
		# print F


		F[:,:] = ( F[:,:] * (aH - aL) ) + aL

		im_l = np.log( 1+img )

		im_f = np.fft.fft2( im_l )

		im_nf = im_f * F # Computes element by element multiplication c[0,0] = a[0,0]*b[0,0] etc

		im_n = np.abs( np.fft.ifft2( im_nf ) )

		im_e = np.exp( im_n )

		filteredImage = im_e - 1


		#plt.imshow( np.abs(im_f).astype('uint8'), cmap=cm.Greys_r )
		#plt.show()

		#f = plt.figure()
		#f.add_subplot(1,2,1); plt.imshow( OratUtils.im2uint8(np.abs(im_f)), cmap=cm.Greys_r )
		#f.add_subplot(1,2,2); plt.imshow( OratUtils.im2uint8(np.abs(im_nf)), cmap=cm.Greys_r )
		#plt.imshow( img.astype('uint8'), cmap=cm.Greys_r )
		#plt.show()


		return filteredImage