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
	def hfilter( image, d, h, l, n ):

		img = np.copy(image)

		# img must be in ndarray format. Inside osearch PIL images are converted to scipy images which are in ndarray format
		# h = height
		# l = length

		warnings.filterwarnings('error')

		F = np.zeros( (h,l), np.float )

		H = float(h)
		L = float(l)
		D = float(d)
		N = float(n)
	
		Xd = np.zeros( (h,l), np.float )
		Yd = np.zeros( (l,h), np.float )

		Xd[:,:] = np.power( np.arange(l, dtype=np.float)-L/2,2 )
		Yd[:,:] = np.power( np.arange(h, dtype=np.float)-H/2,2 )
		
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

		img = np.copy(image) #HFun.im2bw(image, 0.95)

		# Check if the imagename contains (2) or not
		# Very bad way to choose the area to find lines from, but at the moment there's no other method to do this. Needs to be improved somehow!
		try:
			if( iname.index('(2)') >= 0 ):
				upLim = 290
				downLim = 2400
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

		linenum = len(charcount)
		nofound = linenum - imlines.shape[0]

		llines = np.zeros((linenum,2))
		llines[:,1] = 1


		if( nofound < 0 ):
			# Found more lines from the image than what's found from the XML
			# Do something
			return llines

		elif( nofound > 0):
			# Found less lines from the image than what's found from the XML
			
			for i in range(nofound):
				m = charcount.index( min(charcount) )
				#m = charcount[ charcount == min(charcount) ]
				llines[m,1] = 0
				charcount[m] += 1000
		else:
			# Found equal amount of lines from the image as what's found from the XML
			# Lines are assumed to match
			return llines

		return llines




	@staticmethod
	def padlines( imlines, llines, charlines ):

		# Imlines contains the lines got from the image by radontrasnform
		# Llines contains the information about the lines got from the XML and also it contains the 
		# information of if some of the lines is longer or shorter than the mean length of the lines

		"""
			llines contains only 1s and 0s. 1 meaning a line with enough letters to be recognized by pmr 
			and 0 meaning a line which is probably undetected by pmr
		"""

		# nlines is the number of lines calculated from the XML file
		# The corrected lines are gathered into the rlines

		nlines = llines.shape[0]
		rlines = np.zeros((nlines, 1))
		idx = 0

		# Jos löydettyjä rivejä on vähemmän kuin xml:ssä rivejä, pitää rivejä
		# tasata ja niiden indeksejä vastaamaan mahdollisimman paljon oikeita
		# rivejä. Oletuksena on, että rivit, joissa on keskimääräistä vähemmän
		# kirjiamia, ei tunnistu poormanradonissa, joten ne jää välistä pois ja ne
		# hylätään kokonaan. Tätä oletusta hyväksi käyttäen kuitenkin korjataan
		# rivien indeksit osoittamaan aina oikeaan riviin.

		

		if( imlines.shape[0] < nlines ):

			"""
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
			to_be_removed = np.where( llines[:,1] == 0)[0] + 1 # Returns the line (in range [1, nlines]) which isn't detected
			cl = np.delete(cl, np.where( cl == to_be_removed )[0]) # Discards the search from the lines which aren't detected

			wantedlines = rlines[ cl ]

			return wantedlines
			
		else:

			# Flattens the list of lists (charlines) and takes only the unique values, which are number of the lines which has the matches of the word that's been searched
			# These values/indexes are used to choose only those lines that are needed in the search
			cl = np.unique( np.asarray([item for sublist in charlines for item in sublist]) )

			wantedlines = imlines[ cl ]

		return wantedlines



	@staticmethod
	def findCorr( bboxes, slines, charcount, imlines ):

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
				pass
				#print cBB
				#print bbYs
				#print bboxes


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
	def packCoordsToJson( slines, origimage, coords, charpos, debug ):

		rounds = slines.shape[0]

		xx = []
		yy = []

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
						oI[Y-20, X-10:X+50] = [0,255,0]
						oI[Y+20, X-10:X+50] = [0,255,0]
						oI[Y-20:Y+20, X-10] = [0,255,0]
						oI[Y-20:Y+20, X+50] = [0,255,0]
					except IndexError:
						print rightbound
						print leftbound
						print ccount
						print linecenter
						print charpos[i][j]
						print

						#Kuvan 70 tapauksessa tässä ccount 37 ja charpos 62 ... eli sijainti suurempi kuin mitä rivillä kirjaimia... eli väärät rivit valikoituu jostain syystä

				xx.append( X )
				yy.append( Y )

		if( debug ):
			plt.imshow( oI )
			plt.show()

		# Show the current result. Only for debug purpose. In final version the cooridnates of matches are returned
		# as a list to the main program that's calling this program
		# Encode the list into sensible json package or json-string
		startx = np.asarray(xx)-10
		starty = np.asarray(yy)-20
		endx = np.asarray(xx)+50
		endy = np.asarray(yy)+20

		data = [{"startx":startx.tolist(), "starty":starty.tolist(), "endx":endx.tolist(), "endy":endy.tolist()}]
		jsondata = json.dumps( data )

		return jsondata