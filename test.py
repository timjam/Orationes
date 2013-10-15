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

	print "a"
	print a
	print a.dtype
	print

	labeled_array, numpatches = ndimage.label(a)

	print "labeled_array"
	print labeled_array
	print

	pixs = np.array([], np.int64)

	for i in range(a.shape[0]):

		sizes = ndimage.sum(a[:,i], labeled_array[:,i], range(1,numpatches+1))
		p = np.where( sizes > 3 )[0]+1
		print p
		pixs = np.append( pixs, p )
		print sizes
		print pixs
		print

	mi = np.zeros( numpatches+1, np.int64 )
	mi[pixs] = 1
	mf = mi[labeled_array]
	fin = mf^a

	print fin
	print

	# labArYVals = np.where( labeled_array == 2)[0]
	# print labArYVals.max() - labArYVals.min()

	# # A = np.argwhere( labeled_array == [1,2] )
	# # print A
	# # print A.min(0)
	# # print A.max(0)
	# # print
	# sizes1 = np.array([0,1,2,3,4])[labeled_array]
	# print "sizes1"
	# print sizes1
	# print


	# sizes = ndimage.sum(a,labeled_array,range(1,numpatches+1))

	# print "sizes"
	# print sizes
	# print

	# mp = np.where(sizes < 4)[0]+1 
	# print "mp"
	# print mp
	# print

	# max_index = np.zeros(numpatches + 1, np.int32)
	# print "max_index"
	# print max_index
	# print

	# max_index[mp] = 1
	# print "max_index after max_index[mp]"
	# print max_index
	# print

	# max_feature = max_index[labeled_array] # ndarray advanced indexing!
	# print "max_feature"
	# print max_feature
	# print

	# fin = max_feature^a

	# print "fin"
	# print fin
	# print fin.dtype
	# print

	# lA, npa = ndimage.label( fin, None, None )

	# print "lA"
	# print lA
	# print lA.dtype
	# print



	return


main()