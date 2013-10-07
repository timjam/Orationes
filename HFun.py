# -*- coding: UTF-8 -*-

import math
import numpy as np
from scipy.ndimage.measurements import label

class HFun:


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

			return bw

		elif str(image.dtype) == "uint8" :

			bw = image
			bw[ bw <= math.floor(t*255) ] = 0
			bw[ bw >= math.floor(t*255) ] = 1
			bw.astype('bool')

			return bw

		else:
			pass

		return bw




	# Get the coordinates and label numbers
	@staticmethod
	def getCoords( size, ran, lArray ):
		# Get the coordinates of each pixel from each labeled patch
		# First generate an array of zeros with the size of 3x the amount of ones in cI3
		# Then populate the right positions with the coordinates of pixels and also their respective labels
		xyl = np.zeros((3,size))

		c = 0
		for i in range(1, ran+1):
			X,Y = np.where( lArray == i )
			p = len(X)
			
			L = np.array([i]*p)
			xyl[:,c:(c+p)] = np.vstack([X,Y,L])
			c = c+p




	@staticmethod
	def remPatches( sizes, lArray, maxSize ):
		oIdxs = np.where( sizes <= maxSize )[0] + 1
		for i in range(len(oIdxs)):
			lArray[ np.where( lArray == oIdxs[i] ) ] = 0

		bwimage = lArray
		bwimage[ bwimage != 0 ] = 1

		return bwimage


	@staticmethod
	def indices(a, func):
	    return [i for (i, val) in enumerate(a) if func(val)]


	@staticmethod
	def remHighPatches( image, height ):

		# No need to copy this array, because all the changes are made into the right memory array that is the compIm2
		#im = np.copy( image ) # Remember to change image to im if this is enabled

		lArrayTemp, nFeatTemp = label( image )

		for i in range(1,nFeatTemp+1):

			A = np.argwhere( lArrayTemp== i )
			(y1, x1), (y2, x2) = A.min(0), A.max(0)
			
			if( y2-y1 > 70 ):
				lArrayTemp[ lArrayTemp == i ] = 0


		image = lArrayTemp
		image[ image != 0] = 1


		return image