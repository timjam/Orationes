# -*- coding: UTF-8 -*-

import math
import numpy as np
from scipy.ndimage.measurements import label
from scipy import ndimage

import timeit

class HFun:

	r""" Contains helper functions that are used in various places"""


	@staticmethod
	def im2float( image ):
		"""
			Changes uint8 type images to float64 images.

			:param image: The image to be converted
			:type image: ndarray( uint8 )
			:returns: ndarray( float64 ) -- I		
		"""

		I = image.astype( 'float' )
		minI = 0.0
		maxI = 255.0
		I[:,:] = (I[:,:] - minI)/(maxI - minI)
		return I




	@staticmethod
	def gray2uint8( image ):
		"""
			Converts grasycale images to uint8 type

			:param image: The grayscale image to converted
			:type image: ndarray
			:returns: ndarray( uint8 ) - I
		"""

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




	@staticmethod
	def remPatches( sizes, lArray, maxSize, nFeat ):
		oIdxs = np.where( sizes <= maxSize )[0] + 1

		idxs = np.zeros(nFeat + 1, np.int64)
		idxs[oIdxs] = 1
		feats = idxs[lArray]

		lArray[ lArray != 0 ] = 1
		bwimage = lArray ^ feats

		return bwimage.astype( np.int64 )


	@staticmethod
	def indices(a, func):
	    return [i for (i, val) in enumerate(a) if func(val)]


	@staticmethod
	def remHighPatches( image, height ):


		lArrayTemp, nFeatTemp = label( image )

		pixs = np.array([], np.int64)

		for i in range(lArrayTemp.shape[1]):

			sizes = ndimage.sum(image[:,i], lArrayTemp[:,i], range(1,nFeatTemp+1))
			p = np.where( sizes >= height )[0]+1
			pixs = np.append( pixs, p )

		mi = np.zeros( nFeatTemp+1, np.int64 )
		mi[pixs] = 1
		mf = mi[lArrayTemp]
		fin = mf^image


		return fin.astype( np.bool )



	@staticmethod
	def diffMat( v ):

		# v is a list of all the starting y-coordinates in a vector

		lenv = len(v)

		yDiff = np.zeros((lenv, lenv), np.int16)
		yDiff[:,:] = v

		for i in range(lenv):
			yDiff[:,i] = yDiff[:,i]-v

			# Sets the diagonal components to -1 so that if the difference between two values is zero, they are detected as values on the same y-coordinate
			yDiff[i,i] = -1

		yDiff[ yDiff == 0 ] = 1 # Sets all the zero values to 1 so the coordinates which difference is zero are included in the matrix. Causes those coordinates to be twice in the matrix as their difference is always
								# 0. Other non-zero differences are > 0 in the upper triangle and < 0 in the lower triangle and therefore calculated only once. ( <0 and >25 values are rendered to 0 and in the final
								# calculation in the main thread only non-zero values are used)

		yDiff[ yDiff > 25 ] = 0
		yDiff[ yDiff < 0  ] = 0
		# !!!!! Erota diagonaalin ulkopuoliset nollat! Diagonaalin yläpuoliset nollat jätetään, alapuoliset poistetaan, jottei samoja indeksejä käsitellä kahteen kertaan

		return yDiff