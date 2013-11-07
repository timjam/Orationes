# -*- coding: UTF-8 -*-

import re
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import peakdet
import json
from scipy.ndimage.measurements import label
from scipy import ndimage, signal
from scipy.misc import fromimage
from HFun import HFun

import timeit

class OratUtils:

	r""" This class contains only static utility methods that are called directly from the main program 'osearch.py'. """






	@staticmethod
	def stringparser( tfile, c ):
		r"""
			Performs case sensitive search for text file tfile with string or character c (char on default).
			Argument c can be any regular expression

			:param tfile: The name of the cleaned XML file
			:type tfile: string
			:param c: The letter or string that is searched from the tfile
			:type c: string/char/regular expression
			:returns: list -- charcount
			:returns: list of lists -- charpos 
			:returns: list of lists -- charlines
			:returns: list of lists -- wordlens

			* *Charcount* is a list containing the lengths of each line.

				* ``[63, 60, 4, 65, 66, 37, 66, ...]``

			* *Charpos* is a list containing lists including the positions of the found characters or the first letters of the found words.

				* ``[[52], [10, 47, 62], [19, 62], [51], ...]``

			* *Charlines* is a list of lists where the length of each sublist tells the number of hits on that line and the element values representing the line number from the XML file.

				* ``[[3], [4, 4, 4], [6, 6], [7], ...]``

			* *Wordlens* is a list containing lists containing the lengths of the words on each line.

				* ``[[3], [3, 3, 3], [3, 3], [3], ...]``


		"""

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
	def hfilter( image, diameter, height, length, n ):

		r"""
			This function performs homomorphic filtering on grayscale images.

			:param image: 2-dimensional ndarray
			:type image: ndarray
			:param diameter: filter diameter
			:type diameter: int
			:param height: Height of the image
			:type height: int
			:param length: Length of the image
			:type length: int
			:param n: Filter order
			:type n: int
			:returns: ndarray -- homomorphically filtered image

			The image must in ndarray format. In osearch PIL images are converted to scipy images which 
			are in ndarray format. Ndarray format allows easy and fast direct access to the pixel values 
			and this function is written entirely only for the ndarrays.


		"""

		img = np.copy(image)

		warnings.filterwarnings('error')

		F = np.zeros( (height,length), np.float )

		H = float(height)
		L = float(length)
		D = float(diameter)
		N = float(n)
	
		Xd = np.zeros( (H,L), np.float )
		Yd = np.zeros( (L,H), np.float )

		if( L%2 == 0):
			Xd[:,:] = np.power( np.arange(L, dtype=np.float)-(L-1)/2,2 )
		else:
			Xd[:,:] = np.power( np.arange(L, dtype=np.float)-L/2,2 )

		if( H%2 == 0):
			Yd[:,:] = np.power( np.arange(H, dtype=np.float)-(H-1)/2,2 )
		else:
			Yd[:,:] = np.power( np.arange(H, dtype=np.float)-H/2,2 )
		

		F = 1/( 1 + np.power( D/np.power( Xd + np.transpose(Yd), 0.5 ),2*N ) )

		aL = 0.949
		aH = 1.51

		F[:,:] = ( F[:,:] * (aH - aL) ) + aL

		im_l = np.log( 1+img )

		im_f = np.fft.fft2( im_l )	#2.7s

		im_nf = im_f * F # Computes element by element multiplication c[0,0] = a[0,0]*b[0,0] etc

		im_n = np.abs( np.fft.ifft2( im_nf ) ) # 2.85s

		im_e = np.exp( im_n )

		filteredImage = im_e - 1

		return filteredImage




	@staticmethod
	def contStretch( im, a, h ):
		r"""
			Performs contrast stretching for grayscale images. Pixel intensities are set to 
			differ 'a' times the average intensity from the original intensity values. The new 
			intensity values are sliced to stay between [0, 255].

			.. math::
				I_{stretched} = I_{old} + a*( I_{old} - I_{average} )

				I_{new} =
				\left\{
				\begin{array}[l]{ll}
				  0, & I_{stretched} < 0 \\
				  I_{stretched}, & 0 \leq I_{stretched} \leq 255\\
				  255, & I_{stretched} > 255
				\end{array}
				\right.
				

			:param im: The image which contrast is to be stretched
			:type im: ndarray
			:param a: multiplication coefficient
			:type a: int
			:param h: The height of image. Used as partial image average switch
			:type h: int
			:returns: ndarray -- contrast stretched image

			Parameter *h* is a switch which could be used to determine if the average intensity 
			is calculated over the whole image or from a small portion of it. Currently it is 
			defaulted in the code to newer happen. Originally the idea was that if the image 
			is very big, the intensity average would be taken from a small sample. To make the 
			function more generic and also because of the nature of the images in Orationes 
			project, it was decided that the average is always calculated over the whole image.


		"""

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
	def boundingBox( image, debug ):

		r"""
			This functions tries to determine the bounding boxes for each text line.

			:param image: the processed image
			:type image: ndarray
			:param debug: debug switch
			:type debug: bool
			:returns: ndarray -- bboxes

			.. math::
				bboxes_{nxm} = 
				\begin{bmatrix}
					\text{patch label numbers}\\
					\text{starting x-coordinates}\\
					\text{starting y-coordinates}\\
					\text{ending x-coordinates}\\
					\text{ending y-coordinates}
				\end{bmatrix}


			*debug* switch can be used to plot the results of the bounding box 
			founding method and to see whether it is working correctly.

			Process pipeline:

			#. Calculate the histogram from the image
			#. Binarize image with threshold 0.95
			#. Label all the patched in on the binarized image
			#. Calculate the sizes of the patches
			#. Remove unnecessary patches

				#. Remove the largest patch. The largest patch is always the patch consisting of the borders and marginals.
				#. Remove patches which size is smaller or equal to 50 pixels
				#. Remove all the patches which are higher than 70 pixels. This removes the possible remaining marginal patches which weren't connected to the major marginal and border patch.

			#. Perform morpholig operations to clean the image and bind the text lines together

				#. Perform erosion with a cross like structure element

					.. math::
						SEe_{5,5} = 
						\begin{bmatrix}
							0 & 0 & 1 & 0 & 0 \\
							0 & 0 & 1 & 0 & 0 \\
							0 & 1 & 1 & 1 & 0 \\
							0 & 0 & 1 & 0 & 0 \\
							0 & 0 & 1 & 0 & 0
						\end{bmatrix}

				#. Perform dilation with a long vertical line. (needs a 70x70 size structure element)

					.. math::
						SEd_{70,70} = 
						\begin{bmatrix}
							0 & 0 & \dots & 0 & 0 \\
							  & \vdots & & \vdots & \\
							1 & 1 & \dots & 1 & 1 \\
							  & \vdots & & \vdots & \\
							0 & 0 & \dots & 0 & 0
						\end{bmatrix}

			#. Label the morphologically operated image
			#. Remove patches which size is less or equal to 4000 pixels
			#. Label the image again with new labels
			#. Calculate the extreme dimensions of each patch. These values are used as the limiting bounding boxes.
			#. Combine the boxes which are horizontally too close as they are thought to be separate boxes on the same textline.
			#. Return the bounding boxes


		"""

		cIm = np.copy(image)

		# Take histogram of the image
		hist, bin_edges = np.histogram( cIm, bins=255, range=(0,255), density=False ) #0.16s

		# Binarize the image and invert it to a complement image
		bwI = HFun.im2bw(cIm, 0.95)	#
		compIm = (bwI[:,:] - 1)**2 	# 0.12s
		compIm = compIm.astype( np.int64 )

		# Calculate connected components from the image
		lArray, nFeat = label(compIm)	# 0.078s
	
		# Calculate the sizes of each labeled patch
		sizes = ndimage.sum(compIm, lArray, range(1, nFeat+1) )	#0.10s

		# Remove the largest patch from the image
		# It is assumed that the largest patch is always the are that's left outside of the margins
		maxInd = np.where( sizes == sizes.max())[0] + 1 	# Find the index which points to largest patchs
		maxPixs = np.where( lArray == maxInd )				# Find the pixels which have the maxInd as label from labeled image
		lArray[ maxPixs ] = 0								# Set the pixels from the largest patch to zero
		# ^0.047s

		# Remove patches which size is smaller or equal to 50 pixels
		# Make the labeled image with the patches removed as the new complement image and change all the labels to 1 and 0s
		compImtmp = HFun.remPatches( sizes, lArray, 50, nFeat )	# 52.7s with loop

		# Remove all patches which height spans over 70 pixels
		compIm2 = HFun.remHighPatches( compImtmp, 70 )	# 32.7s if remPatches done with lopp

		# Erode the image with vertical line shaped structure element
		SEe = np.zeros((5,5)).astype('bool')
		SEe[:,2] = 1
		SEe[2,1:3] = 1

		cI3 = ndimage.binary_erosion(compIm2, structure=SEe).astype(compIm2.dtype)


		# Dilate the image with horizontal line shaped structure element
		SEd = np.zeros((70,70)).astype('bool') # 60 60 ja 30
		SEd[35,:] = 1

		cI3 = ndimage.binary_dilation(cI3, structure=SEd).astype(cI3.dtype)


		# Label the new morphologically operated image
		lArray2, nFeat2 = label( cI3 )
		sizes2 = ndimage.sum( cI3, lArray2, range(1, nFeat2+1) )


		# Remove the dilated patches which size is smaller than 4000 pixels
		cI4 = HFun.remPatches( sizes2, lArray2, 4000, nFeat2 )

		if( debug ):
			f = plt.figure()
			f.add_subplot(1,3,1); plt.imshow( (compIm2*255).astype( 'uint8' ), cmap=cm.Greys_r )
			f.add_subplot(1,3,2); plt.imshow( (cI3*255).astype( 'uint8' ), cmap=cm.Greys_r )
			f.add_subplot(1,3,3); plt.imshow( (cI4*255).astype( 'uint8' ), cmap=cm.Greys_r )
			plt.show()

		# Label the latest binary image
		lArray3, nFeat3 = label(cI4)

		BBs = np.zeros((5,nFeat3), dtype=np.int16)

		# Calculate the dimensions of the bounding boxes
		for i in range(1,nFeat3+1):
			B = np.argwhere( lArray3==i )
			(ystart, xstart),(ystop, xstop) = B.min(0), B.max(0) # Rajaa siten, että bounding boxin reunaviiva osuu reunimmaisten pikseleiden päälle. Jos halutaan niiden jäävän myös boxin sisään niin pitää molempiin lisätä 1
			BBs[:,i-1] = [i, xstart, ystart, xstop, ystop]


		# Calculate the difference matrix for the starting y-coordinates between all bounding boxes
		yDiff = HFun.diffMat( BBs[2,:] )

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

		if( debug ):

			for i in range( len(BBs[2,:]) ):
				sx = BBs[1,i]
				ex = BBs[3,i]
				sy = BBs[2,i]
				ey = BBs[4,i]
				
				cIm[sy, sx:ex] = 0.5
				cIm[ey, sx:ex] = 0.5
				cIm[sy:ey, sx] = 0.5
				cIm[sy:ey, ex] = 0.5

			plt.imshow( (cIm*255).astype('uint8'), cmap=cm.Greys_r )
			plt.show()


		return BBs





	@staticmethod
	def poormanradon( image, iname, height, debug ):

		r"""
			Performs a naive radon-transform and peak detection on the binarized 
			and contrast stretched image and tries to determine where the text 
			lines are in the image.

			:param image: Image
			:type image: ndarray
			:param iname: Image name
			:type iname: string
			:param height: Image height
			:type height: int
			:param debug: Debug switch
			:type debug: bool
			:returns: ndarray -- Array containing the lines which are found using radon transform

			Calculates the intensity sums over each vertical line. The sums are then inverted and 
			peaks are detected from the inverted data. Data inversion wouldn't be necessary in the 
			python code, but this convetion comes from the Matlab code that was ported to python.

			Before the transform, the image is cleaned so that by using static values ( very bad, 
			should be dynamic, but so far there hasn't been time to do that ) the marginals and 
			everything outside them is erased and turned to white. Because the distance between 
			the camera and the page differs in each image, the marginals aren't always on the 
			same position. This combined with static values causes inaccuracy in the erasing 
			process and might cause inaccuracy when detecting the peaks and the lines.

			In the peak detection, it is assumed that a spike is considered a peak if it's 25 units 
			away from a previous detected peak and also if its value difference is at least 1500 
			to its previous value.

			*upLim* in the source means upper limit in the image coordinates, which increase when 
			going down in the image. That's why *upLim* is small. Respectively the 
			*downLim* means the bottom limit in the image coordinates and that's why it's bigger 
			than the upper limit.

		"""

		img = np.copy(image) #HFun.im2bw(image, 0.95)

		# Check if the imagename contains (2) or not
		# Very bad way to choose the area to find lines from, but at the moment there's no other method to do this. Needs to be improved somehow!
		try:
			if( iname.index('(2)') >= 0 ):
				upLim = 260
				downLim = 2490
				leftLim = 262
				rightLim = 1530
		except ValueError:
			upLim = 320 #200
			downLim = 2428 # 2520 # 2440
			leftLim = 270
			rightLim = 1460

		img[0:upLim, :] = 255
		img[downLim::,:] = 255
		img[:, 0:leftLim] = 255
		img[:, rightLim::]= 255

		linesums = np.zeros((height,1))

		for i in range(height):
			linesums[i,0] = sum(img[i,:])

		inv = (-1)*linesums
		minv = inv.mean()
		#inv[ inv > minv+30000 ] = minv-30000

		max_peaks, min_peaks = peakdet.peakdetect( inv, None, lookahead=25, delta=1500 )
		mp = np.asarray(max_peaks)[:,0]

		mp = mp[ mp > upLim ]
		mp = mp[ mp < downLim ]

		if( debug ):
			#img2 = np.copy(image)
			for j in range(len(mp)):
				img[mp[j]-2:mp[j]+2,:] = 150
			f = plt.figure()
			idata = f.add_subplot(1,2,1); idata.set_autoscaley_on(False); idata.set_ylim( [0, len(inv)] ); idata.plot( inv[::-1], range( len(inv) ) )
			f.add_subplot(1,2,2); plt.imshow(img, cmap=cm.Greys_r)
			plt.show()

		return mp




	@staticmethod
	def processlines( charcount, imlines ):
		r"""
			This functions compares the number of lines found from the image to the 
			number of lines found from the XML file and creates a logical vector 
			telling which lines are probably found and which are not.

			:param charcount: Contains the lengths of each line
			:type charcount: list
			:param imlines: Contains the textlines which are found in 'poormanradon'
			:type imlines: ndarray
			:returns: ndarray -- llines
			:returns: ndarray -- imlines

			*llines* is a n*2 vector, where the llines[:,1] is a logical vector 
			containing the information of the probably found and non-found lines.
			
			.. math::
				[1,1,1,1,0,1,0,1, \dots ]^t

			*imlines* is a ndarray containing the y-coordinates of the textlines 
			found from the image with poormanradon. When padding some of the 
			coordinates are removed ( nofound < 0, not used ), some NAN 
			values are added in between some coordinates ( nofound > 0 ) or 
			it is returned unchanged ( nofound == 0 ).

			We calculate a number 'nofound'
			
			.. math::
				\operatorname{nofound}=\operatorname{lines}_{XML}-\operatorname{lines}_{image}
			
			Naturally there are three cases.

			nofound < 0:
				This means that there are more lines found from the image than there 
				actually are. Currently nothing's done here to compensate this 
				behavior.

			nofound > 0:
				This means there aren't enough lines found from the image. Usually the 
				non-found lines are assumed to be the very short lines. When padding 
				the indices of the lines, the shortest lines are always set to be 
				the non-found lines.

			nofound == 0:
				It is assumed that all the textlines were found correctly and the 
				imlines will be returned unchanged.

		"""

		linenum = len(charcount)
		nofound = linenum - imlines.shape[0]

		llines = np.zeros((linenum,2))
		llines[:,1] = 1


		if( nofound < 0 ):
			# Found more lines from the image than what's found from the XML
			# Do something
			return llines, imlines

		elif( nofound > 0):
			# Found less lines from the image than what's found from the XML
			
			for i in range(nofound):
				m = charcount.index( min(charcount) )
				
				llines[m,1] = 0
				charcount[m] += 1000

				# Imlines needs to be padded as well. Non found lines are set tot NAN
				templines = imlines[0:m]
				templines2 = imlines[m::]
				nimlines = np.append( templines, np.nan )
				imlines = np.append( nimlines, templines2 )

		else:
			# Found equal amount of lines from the image as what's found from the XML
			# Lines are assumed to match
			return llines, imlines

		return llines, imlines




	@staticmethod
	def padlines( imlines, llines, charlines ):

		r""" 
		:param imlines: n*1 size ndarray containing the lines (or rather their y-position) got from the image by radontransform
		:type imlines: ndarray
		:param llines: n*2 size ndarray containing the length information of the lines
		:type llines: ndarray
		:param charlines: list of lists telling the position(s) of searched character(s)/word(s) on each line
		:type charlines: list
		:returns: ndarray -- wantedlines

		Long:
		Llines contains the information about the lines got from the XML and also it contains the 
		information if some of the lines are remarkably shorter than other lines. That means that, if there are some lines that 
		are not found from the image, it is assumed that those non-found lines are the shortest lines according to the XML and 
		character count. Those lines are marked as 0 in the second column in llines.

		Short:
		Llines[:,1] contains only 1s and 0s. 1 meaning a line with enough letters to be recognized by poormanradon (pmr) 
		and 0 meaning a line which is probably undetected by pmr

		
		Behavior:
			Number of lines found from the image using pmr is larger than 
			the number of lines calculated from XML:

				TODO! Currently this case is not handled!

			Number of lines found from the image using pmr is smaller than
			the number of lines calculated from XML:

				Pad the lines according to the information in *llines[:,1]*:

			.. math::
				\stackrel{\mbox{llines}}{ \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 0 \\ 1 \\ 1 \\ 0 \\ 1 \end{bmatrix} }
				\begin{matrix} \searrow \\ \searrow \\ \searrow \\ \searrow \\ ~ \\ \longrightarrow \\ \longrightarrow \\ ~ \\ \nearrow \end{matrix}
				\stackrel{\mbox{imlines}}{ \begin{bmatrix} 100 \\ 200 \\ 300 \\ 400 \\ 600 \\ 700 \\ 900 \end{bmatrix} }
				\begin{matrix} \nearrow \\ \nearrow \\ \nearrow \\ \nearrow \\ ~ \\ \longrightarrow \\ \longrightarrow \\ ~ \\ \searrow \end{matrix}
				\stackrel{\mbox{rlines}}{ \begin{bmatrix} 100 \\ 200 \\ 300 \\ 400 \\ NAN \\ 600 \\ 700 \\ NAN \\ 900 \end{bmatrix} } \\
				\\ ~

			Number of lines found from the image using pmr equals to
			the number of lines calculated from XML:

				Pick unique lines from imlines and return them as lines 
				the interesting lines.
			
		
		"""


		nlines = llines.shape[0]		# nlines is the number of lines calculated from the XML file
		rlines = np.zeros((nlines, 1))	# The corrected lines are gathered into the rlines
		idx = 0
		
		if( imlines.shape[0] < nlines ):

			"""
				Jos löydettyjä rivejä on vähemmän kuin xml:ssä rivejä, pitää rivejä
				tasata ja niiden indeksejä vastaamaan mahdollisimman paljon oikeita
				rivejä. Oletuksena on, että rivit, joissa on keskimääräistä vähemmän
				kirjiamia, ei tunnistu poormanradonissa, joten ne jää välistä pois ja ne
				hylätään kokonaan. Tätä oletusta hyväksi käyttäen kuitenkin korjataan
				rivien indeksit osoittamaan aina oikeaan riviin.


				llines:		imlines:		rlines:
				
				[1 ---------->[100 -------->[100
				 1 ----------> 200 --------> 200
				 1 ----------> 300 --------> 300
				 1 ----------> 400 --------> 400
				 0 	.--------> 600 --------. NAN
				 1 /.--------> 700 --------.'600
				 1 / .-------> 900]-------. '700
				 0  / 					   \ NAN
				 1]/ 						'900]

			"""

			for i in range(nlines):
				
				if(llines[i,1] == 1):
					rlines[i] = imlines[idx]
					idx += 1

				else:
					rlines[i,0] = np.nan

			cl = np.unique( np.asarray([item for sublist in charlines for item in sublist]) )
			to_be_removed = np.where( llines[:,1] == 0)[0] + 1 #: Returns the line (in range [1, nlines]) which isn't detected
			cl = np.delete(cl, np.where( cl == to_be_removed )[0]) # Discards the search from the lines which aren't detected

			wantedlines = rlines[ cl ]

		elif( imlines.shape[0] > nlines ):
			# TODO! 
			# Currently identical to the case where imlines.shape[0] == nlines
			cl = np.unique( np.asarray([item for sublist in charlines for item in sublist]) )
			wantedlines = imlines[ cl ]

		else:

			# Flattens the list of lists (charlines) and takes only the unique values, which are number of the lines which has the matches of the word that's been searched
			# These values/indexes are used to choose only those lines that are needed in the search
			cl = np.unique( np.asarray([item for sublist in charlines for item in sublist]) )

			wantedlines = imlines[ cl ]

		return wantedlines



	@staticmethod
	def findCorr( bboxes, slines, charcount, imlines, debug ):

		"""
			asdasdasd

			:param bboxes:
			:type bboxes: ndarray
			:param slines:
			:type slines: ndarray
			:param charcount:
			:type charcount: list
			:param imlines:
			:type imlines: ndarray
			:param debug:
			:type debug: bool
			:returns: ndarray -- m*n ndarray containing the starting and ending coordinates of hits
		"""

		bbYs = bboxes[2,:]
		rounds = slines.shape[0]

		coords = np.zeros((7,rounds), np.int16)

		for i in range( rounds ):

			if( i==0 or ( slines[i] - slines[i-1] > 100 ) ):
				minlim = slines[i] - 70

			else:
				minlim = slines[i-1]

			cBBYstarts = bbYs[ bbYs > minlim ]
			cBBYstarts = cBBYstarts[ cBBYstarts < slines[i] ]

			cBB = HFun.indices( bbYs, lambda x: ( x > minlim and x < slines[i] ) )


			try:
				temp = bboxes[:,bbYs == bbYs[ cBB[0] ] ] #cBBYstarts before
			except IndexError:
				# Some images fails and goes here for some reason. That needs to be found out and see if it causes other errors
				
				if( debug ):
					print "Failed to find correspondences between bboxes and imlines"
					print cBB
					print bbYs
					print bboxes
				else:
					pass


			coords[0,i] = temp[0]	# Sisältää kyseistä bounding boxia vastaavan patching labelin
			coords[1,i] = temp[1]	# Sisältää kyseisen bounding boxin xstart koordinaatin
			coords[2,i] = temp[2]	# Sisältää kyseisen bounding boxin ystart koordinaatin
			coords[3,i] = temp[3]	# Sisältää kyseisen bounding boxin xstop koordinaatin
			coords[4,i] = temp[4]	# Sisältää kyseisen bounding boxin ystop koordinaatin
			# print np.asarray(charcount)[np.where( bboxes[2,:] == temp[2] )[0] +1]
			# ^ Konvertoi ensin charcount listan numpy arrayksi
			# Sen jälkeen haetaan bboxes numpy arrayn toiselta rivilta kaikki niiden sarakkeiden indeksit, joissa sarakkeen arvon on sama kuin temp listan toiset arvot, koska temp listan toisina arvoina on halutut y koordinaatit
			# Sitten otetaan tästä np.where tuloksesta ensimmäinen alkio, koska se sisältää halutun indeksin ja lisätään siihen sitten yksi. Tämä sen takia, että labelit bboxissa on järjestetty 
			# kasvavaan järjestykseen siten, että label on aina indeksi plus yksi ja sitten kaikki onkin ihan vitun sekavaa ... Kusee koska boksin label ei välttämättä ole sama kuin sitä vastaavan rivin järjnro!


			#coords[5,i] = np.asarray(charcount)[np.where( bboxes[2,:] == temp[2] )[0] +1] # Sisältää kyseisellä rivillä olevien kirjainten lukumäärän
			#print np.where( imlines == slines[i] )
			coords[5,i] = charcount[ np.where( imlines == slines[i] )[0] ]
			coords[6,i] = slines[i]	# Sisältää kyseistä bounding boxia vastaan rivin radonmuunnoksesta saadun keskikohdan y-koordinaatin


		return coords

	@staticmethod
	def packCoordsToJson( slines, origimage, coords, charpos, wordlens, bboxes, debug ):
		"""
			asdasdasd

			:param slines:
			:type slines: ndarray
			:param origimage:
			:type origimage: ndarray
			:param coords:
			:type coords: ndarray
			:param charpos:
			:type charpos: list of lists
			:param wordlens:
			:type wordlens: list of lists
			:param bboxes:
			:type bboxes:
			:param debug:
			:type debug: bool
			:returns: json-string -- JSON packed string
		"""

		rounds = slines.shape[0]

		xx = []
		yy = []

		wl = 20*wordlens[0][0]-10

		if( debug ):
			oI = fromimage( origimage )

		for i in range( rounds ): # rounds


			rightbound = coords[1,i]
			leftbound = coords[3,i]
			ccount = coords[5,i]
			linecenter = coords[6,i]



			for j in range( len(charpos[i])):

				X = charpos[i][j] * ( leftbound - rightbound )/ccount + rightbound
				Y = linecenter

				if( debug ):
					try:
						oI[Y-20, X-10:X+wl] = [0,255,0]
						oI[Y+20, X-10:X+wl] = [0,255,0]
						oI[Y-20:Y+20, X-10] = [0,255,0]
						oI[Y-20:Y+20, X+wl] = [0,255,0]
					except IndexError:

						print rightbound
						print leftbound
						print ccount
						print linecenter
						print charpos[i][j]
						print i
						print j
						print
						print charpos
						print
						print slines
						print

						#Kuvan 70 tapauksessa tässä ccount 37 ja charpos 62 ... eli sijainti suurempi kuin mitä rivillä kirjaimia... eli väärät rivit valikoituu jostain syystä

				xx.append( X )
				yy.append( Y )

		if( debug ):
			plt.imshow( oI )
			plt.show()


		# Encode the list into sensible json package or json-string

		rds = len( bboxes[0,:] )
		bbs0 = bboxes[1,:].tolist()
		bbs1 = bboxes[2,:].tolist()
		bbs2 = bboxes[3,:].tolist()
		bbs3 = bboxes[4,:].tolist()

		startx = (np.asarray(xx)-10).tolist()
		starty = (np.asarray(yy)-20).tolist()
		endx = (np.asarray(xx)+wl).tolist()
		endy = (np.asarray(yy)+20).tolist()


		data = { "bounding boxes" : [ {	"x1" : bbs0[i], 
											"y1" : bbs1[i], 
											"x2" : bbs2[i], 
											"y2" : bbs3[i]} 
											for i in range( rds ) ],
					"hits" : [ {			"x1":startx[j], 
											"y1":starty[j], 
											"x2":endx[j], 
											"y2":endy[j]}
											for j in range( len( startx ) ) ] }
		jsondata = json.dumps( data )

		return jsondata