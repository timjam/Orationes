# -*- coding: UTF-8 -*-

import numpy as np
from scipy import ndimage

def main():

	a = np.array([	[1, 1, 1, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 1, 1, 0, 0],
					[0, 0, 0, 0, 1, 1, 0],
					[1, 0, 0, 0, 1, 0, 0],
					[1, 1, 0, 0, 0, 1, 1]	], np.int64)

	print a
	print

	print a.dtype

	labeled_array, numpatches = ndimage.label(a)

	print labeled_array
	print

	# sizes = ndimage.sum(a,labeled_array,range(1,numpatches+1))

	# print sizes
	# print

	# mp = np.where(sizes == sizes.max())[0]+1 
	# print mp
	# print

	# max_index = np.zeros(numpatches + 1, np.int32)
	# max_index[mp] = 1
	# print max_index
	# print

	# max_feature = max_index[labeled_array]
	# print max_feature
	# print

	# fin = max_feature^a

	# print fin
	# print

	# print fin.dtype
	# print

	# lA, npa = ndimage.label( fin, None, None )

	# print lA
	# print lA.dtype
	# print



	return


main()