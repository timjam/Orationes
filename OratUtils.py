# -*- coding: UTF-8 -*-

import re
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import peakdet
from scipy.ndimage.measurements import label
from scipy import ndimage, signal
from HFun import HFun

class OratUtils:





	# Performs case sensitive search for text file tfile with string or character c on default.
	# Argument c can be any regular expression
	@staticmethod
	def stringparser( tfile, c ):

		charcount = [] 	# Holds the lengths of each line
		charlines = []	# 
		charpos = []	# The positions of the found characters or the first letters of the found words
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
	def contStretch( im, a, h ):

		image = np.copy(im)

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
	def boundingBox( image ):

		cIm = np.copy(image)

		# Take histogram of the image
		hist, bin_edges = np.histogram( cIm, bins=255, range=(0,255), density=False )

		# Binarize the image and invert it to a complement image
		bwI = HFun.im2bw(cIm, 0.95)
		compIm = (bwI[:,:] - 1)**2

		# Calculate connected components from the image
		lArray, nFeat = label(compIm)

		# Calculate the sizes of each labeled patch
		sizes = ndimage.sum(compIm, lArray, range(1, nFeat+1) )

		# Remove the largest patch from the image
		# It is assumed that the largest patch is always the are that's left outside of the margins
		maxInd = np.where( sizes == sizes.max())[0] + 1 	# Find the index which points to largest patchs
		maxPixs = np.where( lArray == maxInd )				# Find the pixels which have the maxInd as label from labeled image
		lArray[ maxPixs ] = 0								# Set the pixels from the largest patch to zero


		# Remove patches which size is smaller or equal to 50 pixels
		# Make the labeled image with the patches removed as the new complement image and change all the labels to 1 and 0s
		compIm2 = HFun.remPatches( sizes, lArray, 50 )




		# Remove all patches which height spans over 70 pixels
		lArrayTemp, nFeatTemp = label(compIm2)

		for i in range(1,nFeatTemp+1):
			A = np.argwhere( lArrayTemp== i )
			#print A
			(y1, x1), (y2, x2) = A.min(0), A.max(0)
			if( y2-y1 > 70 ):
				lArrayTemp[ lArrayTemp == i ] = 0

		compIm2 = lArrayTemp
		compIm2[ compIm2 != 0] = 1





		# Erode the image with vertical line shaped structure element
		SEe = np.zeros((5,5))
		SEe[:,2] = 1
		SEe.astype('bool')

		cI3 = ndimage.binary_erosion(compIm2, structure=SEe).astype(compIm2.dtype)


		# Dilate the image with horizontal line shaped structure element
		SEd = np.zeros((70,70)) # 60 60 ja 30
		SEd[35,:] = 1
		SEd.astype('bool')

		cI3 = ndimage.binary_dilation(cI3, structure=SEd).astype(cI3.dtype)


		# Label the new morphologically operated image
		lArray2, nFeat2 = label( cI3 )
		sizes2 = ndimage.sum( cI3, lArray2, range(1, nFeat2+1) )

		# Get the coordinates of each pixel from each labeled patch
		# First generate an array of zeros with the size of 3 * the amount of ones in cI3
		# Then populate the right positions with the coordinates of pixels and also their respective labels
		# xyl = HFun.getCoords( np.sum(cI3), nFeat2, lArray2 ) ### Näitä ei välttämättä ees tarvita!

		# Remove the dilated patches which size is smaller than 4000 pixels
		cI4 = HFun.remPatches( sizes2, lArray2, 4000 )

		# Label the latest binary image
		lArray3, nFeat3 = label(cI4)

		BBs = np.zeros((5,nFeat3), dtype=np.int16)

		# Calculate the dimensions of the bounding boxes
		for i in range(1,nFeat3+1):
			B = np.argwhere( lArray3==i )
			(ystart, xstart),(ystop, xstop) = B.min(0), B.max(0) # Rajaa siten, että bounding boxin reunaviiva osuu reunimmaisten pikseleiden päälle. Jos halutaan niiden jäävän myös boxin sisään niin pitää molempiin lisätä 1
			BBs[:,i-1] = [i, xstart, ystart, xstop, ystop]


		# Sort the coordinates and labels by their ystart coordinates
		# Jos ei tehdä tätä sorttia niin saadaan erotusmatriisista yDiff suoraan lähekkäisten bounding boxien labelit
		# sIdxs = np.argsort(BBs[2])
		# BBs = BBs[:,sIdxs]

		yss = BBs[2,:] # Lists all the starting y-coordinates into a vector
		yDiff = np.zeros((len(yss), len(yss)), np.int16)
		yDiff[:,:] = yss

		for i in range(len(yss)):
			yDiff[:,i] = yDiff[:,i]-yss


			yDiff[i,i] = -1

		yDiff[ yDiff == 0 ] = 1

		yDiff[ yDiff > 25 ] = 0
		yDiff[ yDiff < 0  ] = 0
		# !!!!! Erota diagonaalin ulkopuoliset nollat! Diagonaalin yläpuoliset nollat jätetään, alapuoliset poistetaan, jottei samoja indeksejä käsitellä kahteen kertaan

		sameBBs = np.argwhere( yDiff != 0 ) # Eli sisältää tosiaan tarpeeksi lähellä olevien BB:iden indeksit BBs listassa


		for i in range( len(sameBBs[:,1])):
			# sameBBs sisältää pareja, jotka ovat tarpeeksi lähekkäin olevien BB:iden indeksejä BBs arrayssa
			# Haetaan siis näitä indeksejä vastaavat BB:iden tiedot muuttujiin a ja b ja yhdistetään ne
			# a and b are [label, xstart, ystart, xstop, ystop]

			a = BBs[:,sameBBs[i,0]]
			b = BBs[:,sameBBs[i,1]]
			c = np.array([0,0,0,0,0])

			# Make new leftmost BB start coordinate by taking the leftmost x coordinate and highest ( lowest index ) y-coordinate
			c[1] = min( a[1], b[1] )
			c[2] = min( a[2], b[2] )

			# Make new rightmost BB stop coordinate by taking the rightmost x coordinate and lowest ( highest index ) y-coordinate
			c[3] = max( a[3], b[3] )
			c[4] = max( a[4], b[4] )

			# Set the label of the new BBs to the same as the label of a
			c[0] = a[0]

			BBs[:,sameBBs[i,0]] = c
			BBs[:,sameBBs[i,1]] = c

		# Get unique bounding boxes thus removing the possible duplicate BBs
		vals, idx = np.unique( BBs[0,:], return_index=True )
		BBs = BBs[:,idx]
		#print BBs
		# for i in range(len(BBs[2,:])):
		# 	x1 = BBs[1,i]
		# 	x2 = BBs[3,i]
		# 	y1 = BBs[2,i]
		# 	y2 = BBs[4,i]
		# 	cI4[y1:y2,x1] = 1
		# 	cI4[y1:y2,x2] = 1
		# 	cI4[y1,x1:x2] = 1
		# 	cI4[y2,x1:x2] = 1



		# f = plt.figure()
		# f.add_subplot(1,3,1); plt.imshow( (compIm2*255).astype('uint8'), cmap=cm.Greys_r )
		# f.add_subplot(1,3,2); plt.imshow( (cI3*255).astype('uint8'), cmap=cm.Greys_r )
		# f.add_subplot(1,3,3); plt.imshow( (cI4*255).astype('uint8'), cmap=cm.Greys_r )
		# plt.show()

		return BBs





	@staticmethod
	def poormanradon( image, iname, height ):

		img = np.copy(image) #HFun.im2bw(image, 0.95)

		# Check if the imagename contains (2) or not
		# Very bad way to choose the area to find lines from, but at the moment there's no other method to do this. Needs to be improved somehow!
		try:
			if( iname.index('(2)') >= 0 ):
				upLim = 290
				downLim = 2400
		except ValueError:
			upLim = 200
			downLim = 2440 # 2520


		linesums = np.zeros((height,1))

		for i in range(height):
			linesums[i,0] = sum(img[i,:])

		inv = (-1)*linesums

		max_peaks, min_peaks = peakdet.peakdetect( inv, None, lookahead=20, delta=100 )
		mp = np.asarray(max_peaks)[:,0]

		mp = mp[ mp > upLim ]
		mp = mp[ mp < downLim ]

		#for j in range(len(mp)):
		#	img[mp[j]-1:mp[j]+1,:] = 0

		return mp




	@staticmethod
	def processlines( charcount, imlines ):

		#print imlines.shape[0]
		linenum = len(charcount)
		nofound = linenum - imlines.shape[0]


		llines = np.zeros((linenum,2))
		llines[:,1] = 1


		if( nofound < 0 ):
			# Do something
			pass

		else:

			for i in range(nofound):
				m = charcount[ charcount == min(charcount) ]
				llines[m,2] = 0
				charcount[m] += 1000

		#print charcount

		return llines




	@staticmethod
	def padlines( imlines, llines, charlines ):

		# Imlines contains the lines got from the image by radontrasnform
		# Llines contains the information about the lines got from the XML and also it contains the 
		# information of if some of the lines is longer or shorter than the mean length of the lines

		temp = llines[:,1]
		temp[ temp > np.mean(temp) ] = 1
		temp[ temp != 1 ] = 0
		llines[:,1] = temp

		# nlines is the number of lines calculated from the XML file
		# The corrected lines are gathered into the rlines

		#print llines.shape[0]
		#print llines.shape[1]

		nlines = llines.shape[0]
		rlines = np.zeros((nlines, 1))
		idx = 1

		# Jos löydettyjä rivejä on vähemmän kuin xml:ssä rivejä, pitää rivejä
		# tasata ja niiden indeksejä vastaamaan mahdollisimman paljon oikeita
		# rivejä. Oletuksena on, että rivit, joissa on keskimääräistä vähemmän
		# kirjiamia, ei tunnistu poormanradonissa, joten ne jää välistä pois ja ne
		# hylätään kokonaan. Tätä oletusta hyväksi käyttäen kuitenkin korjataan
		# rivien indeksit osoittamaan aina oikeaan riviin.

		if( imlines.shape[0] < nlines ):
			
			for i in range(nlines):
				
				if(llines[i,1] == 1):
					rlines[i,0] = imlines[idx,0]
					idx += 1

				else:
					rlines[i,0] = np.nan

			#print rlines

			# The F is happening here?
			# Some sort of forcing the lines to be something if there are more lines in the xml than what's found from the image
			charlines[ charlines > nlines ] = nlines # rlines.shape[0] --> nlines
			wantedlines = rlines[ charlines ]
		else:

			# Flattens the list of lists (charlines) and takes only the unique values, which are number of the lines which has the matches of the word that's been searched
			# These values/indexes are used to choose only those lines that are needed in the search
			cl = np.unique( np.asarray([item for sublist in charlines for item in sublist]) )

			#print charlines
			#print cl

			wantedlines = imlines[ cl ]


		#print wantedlines

		return wantedlines