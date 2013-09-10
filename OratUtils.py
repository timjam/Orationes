# -*- coding: UTF-8 -*-

import re
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.measurements import label
from scipy import ndimage

class OratUtils:








	@staticmethod
	def im2float( image ):
		I = image.astype( 'float' )
		minI = 0.0
		maxI = 255.0
		I[:,:] = (I[:,:] - minI)/(maxI - minI)
		return I








	@staticmethod
	def gray2uint8( image ):

		I = image
		I[I>1] = 1
		I[I<0] = 0
		I *= 255
		I = I.astype('uint8')


		return I








	@staticmethod
	def im2bw( image, t ):

		if str(image.dtype) == "float64" :
			
			bw = image
			bw[ bw <= t ] = 0
			bw[ bw >= t ] = 1
			bw.astype('bool')

		elif str(image.dtype) == "uint8" :

			bw = image
			bw[ bw <= math.floor(t*255) ] = 0
			bw[ bw >= math.floor(t*255) ] = 1
			bw.astype('bool')

		else:
			pass

		return bw








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


		F[:,:] = ( F[:,:] * (aH - aL) ) + aL

		im_l = np.log( 1+img )

		im_f = np.fft.fft2( im_l )

		im_nf = im_f * F # Computes element by element multiplication c[0,0] = a[0,0]*b[0,0] etc

		im_n = np.abs( np.fft.ifft2( im_nf ) )

		im_e = np.exp( im_n )

		filteredImage = im_e - 1

		return filteredImage








	@staticmethod
	def contStretch( image, a, h ):

		if h > 30000:
			# This part was supposed to take average sample from the background
			# The limit is set to 30000 to ensure that at this point this will never
			# happen.
			temp = image[100:200, 600:800]
			i_avg = np.mean(temp)
		else:
			i_avg = np.mean(image)

		resI = image[:,:] + a*( image[:,:] - i_avg )

		resI[resI>255] = 255
		resI[resI<0] = 0

		return resI








	@staticmethod
	def boundingBox( cIm ):

		# Take histogram of the image
		hist, bin_edges = np.histogram( cIm, bins=255, range=(0,255), density=False )

		# Binarize the image and invert it to a complement image
		bwI = OratUtils.im2bw(cIm, 0.95)
		compIm = (bwI[:,:] - 1)**2

		# Calculate connected components from the image

		lArray, nFeat = label(compIm)

		#print lArray
		print nFeat

		sizes = ndimage.sum(compIm, lArray, range(1, nFeat+1) )
		largestArea = sizes.max()
		#[i for i, j in enumerate(sizes) if j == sizes]

		print sizes
		print largestArea
		print lInd

		plt.imshow((compIm*255).astype('uint8'), cmap=cm.Greys_r)
		plt.show()

		return []