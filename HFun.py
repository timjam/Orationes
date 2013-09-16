# -*- coding: UTF-8 -*-

import math
import numpy as np

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

		elif str(image.dtype) == "uint8" :

			bw = image
			bw[ bw <= math.floor(t*255) ] = 0
			bw[ bw >= math.floor(t*255) ] = 1
			bw.astype('bool')

		else:
			pass

		return bw




	# Get the coordinates and label numbers
	@staticmethod
	def getCoords( inum, lArray ):
		X,Y = np.where( lArray == inum )
		L = np.array([inum]*len(X))

		print X
		print Y
		print L

		xyl = np.vstack([X,Y,L])

		return xyl