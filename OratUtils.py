import re
import math
import numpy as np
import warnings

class OratUtils:

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
	def hfilter( img, d, r, c, n ):

		# img must be in ndarray format. Inside osearch PIL images are converted to scipy images which are in ndarray format

		# r = height
		# c = length

		#print img

		warnings.filterwarnings('error')

		A = H = np.zeros( (r,c) )


		for i in range(r):
			for j in range(c):
				try:
					A[i,j] = math.sqrt( math.pow(( (i+1) -r/2),2) + math.pow(( (j+1) -c/2),2) )
					H[i,j] = 1/( 1 + math.pow( d/( A[i,j] ) , 2*n ) )
				except Warning:
					print i
					print j
					print A[i,j]

		aL = 0.949
		aH = 1.51

		H[:,:] = ( H[:,:] * (aH - aL) ) + aL

		im_l = np.log( 1+img )

		im_f = np.fft.fft2( im_l )

		im_nf = im_f * H # Computes element by element multiplication c[0,0] = a[0,0]*b[0,0] etc

		im_n = np.abs( np.fft.ifft2( im_nf ) )

		im_e = np.exp( im_n )

		filteredImage = im_e - 1


		return filteredImage