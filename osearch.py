# -*- coding: UTF-8 -*-
# Uses Anaconda 1.6.2 64-bit distribution package with Python 2.7.5
#
# Timo Mätäsaho
# University of Oulu
# 2013

import os
import sys
import Image as Im
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from OratUtils import OratUtils
from HFun import HFun
from scipy.misc import fromimage, toimage, imshow

import numpy as np


def osearch( img, txtf, sw ):

	# Open the image and text file with their absolute paths to ensure that the right files from
	# the right place are opened
	curDir = os.path.dirname( os.path.realpath( __file__ ))
	imagename = curDir + "\\Images\\" + img
	tfile = curDir + "\\Texts\\" + txtf


	# Parse the XML file. The XML file must be formatted to raw text with no XML tags. 
	# Should be done separately outside of this program unless this program is updated
	# to do that as well.
	charcount, charpos, charlines, wordlens = OratUtils.stringparser( tfile, sw )

	if not charpos:
		print "Couldn't find string \'%s\' from page %s" %(sw, img)
		return 0


	# Open the original image and convert it to grayscale
	origimage = Im.open(imagename)
	grayimage = origimage.copy().convert("L") # Conversion from RGB to Grayscale


	# Get the dimensions of the image
	ImLength, ImHeight = origimage.size


	# Conversion from PIL image to scipy image and then from uint8 to float
	tI = fromimage(grayimage) # From PIL to scipy image
	tI = HFun.im2float(tI) # From uint8 to float


	# Filter the image and convert it back to grayscale uint8 image
	filteredIm = OratUtils.hfilter( tI, 620, ImHeight, ImLength, 20 )	
	filteredIm = HFun.gray2uint8(filteredIm) # From float to uint8


	# Stretch the contrast of the image
	cIm = OratUtils.contStretch( filteredIm, 20 , ImHeight )


	# Get the bounding boxes covering each line
	# Put in its own thread?
	bboxes = OratUtils.boundingBox( cIm )

	# Get the positions of lines according to the image and its radon transform
	imlines = OratUtils.poormanradon( cIm, imagename, ImHeight )

	# Get the rightlines according to the XML file and the image
	rlines = OratUtils.processlines( charcount, imlines )

	# Get the lines that are used to search the possible hits
	slines = OratUtils.padlines( imlines, rlines, charlines ) # slines - [a1, a2, ..., an], n == number of unique rlines

	#print slines

	# Find the correspondences between the lines which are used for searching and the bounding boxes

	bbYs = bboxes[2,:]
	rounds = slines.shape[0]

	coords = np.zeros((7,rounds), np.int16)

	#print bboxes

	#print charcount

	for i in range( rounds ):

		if( i==0 or ( slines[i] - slines[i-1] > 100 ) ):
			minlim = slines[i] - 70

		else:
			minlim = slines[i-1]

		cBBYstarts = bbYs[ bbYs > minlim ]
		cBBYstarts = cBBYstarts[ cBBYstarts < slines[i] ]

		#cBB = np.where( (bbYs > minlim) and (bbYs < slines[i]) )
		cBB = HFun.indices( bbYs, lambda x: ( x > minlim and x < slines[i] ) )
		#print cBB
		#print "*****\n"

		temp = bboxes[:,bboxes[2,:] == bbYs[ cBB[0] ] ] #cBBYstarts before

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


	xx = []
	yy = []

	oI = fromimage( origimage )

	print coords

	for i in range( rounds ): # rounds


		rightbound = coords[1,i]
		leftbound = coords[3,i]
		ccount = coords[5,i]
		linecenter = coords[6,i]



		for j in range( len(charpos[i])):

			X = charpos[i][j] * ( leftbound - rightbound )/ccount + rightbound
			Y = linecenter

			try:
				oI[ Y-20, X-10:X+50] = [0, 255, 0]
				oI[ Y+20, X-10:X+50] = [0, 255, 0]
				oI[ Y-20:Y+20, X-10] = [0, 255, 0]
				oI[ Y-20:Y+20, X+50] = [0, 255, 0]
				
			except IndexError:
				pass
				# print rightbound
				# print leftbound
				# print ccount
				# print "Y: " + str(Y)
				# print "X: " + str(X)
				# print "Fail"
				# print
				

			xx.append( X )
			yy.append( Y )



	for i in range(bboxes.shape[1]):
		x1 = bboxes[1,i]
		y1 = bboxes[2,i]
		x2 = bboxes[3,i]
		y2 = bboxes[4,i]

		cIm[y1, x1:x2] = 0
		cIm[y2, x1:x2] = 0
		cIm[y1:y2, x1] = 0
		cIm[y1:y2, x2] = 0


	# Show the current result. Only for debug purpose. In final version the cooridnates of matches are returned
	# as a list to the main program that's calling this program
	# Encode the list into sensible json package or json-string
	f = plt.figure()
	f.add_subplot(1,2,1); plt.imshow( filteredIm, cmap=cm.Greys_r ); plt.title(' Eka ')
	f.add_subplot(1,2,2); plt.imshow( cIm, cmap=cm.Greys_r ); plt.title(' Toka ')
	plt.show()

	plt.imshow( oI )
	plt.show()

	# np.savetxt( 'testres1.txt', charcount, delimiter=",", fmt="%i" )
	# np.savetxt( 'testres2.txt', imlines.astype( np.int16 ), delimiter=",", fmt="%i" )
	# np.savetxt( 'testres3.txt', slines.astype( np.int16 ), delimiter=",", fmt="%i" )
	# np.savetxt( 'testres4.txt', rlines.astype( np.int16 ), delimiter=",", fmt="%i" )
	# np.savetxt( 'testres5.txt', bboxes.astype( np.int16 ), delimiter=",", fmt="%i" )
	

	return







if __name__ == "__main__":

	if len(sys.argv) == 4:
		osearch( sys.argv[1], sys.argv[2], sys.argv[3] )
	else:
		print "\nWrong amount of parameters. Need following arguments: "
		print "Name of the image, name of the correspoding raw text file, word that's being searched\n"
		print "For example use:    python osearch.py \"Lit_Ms_E_41 070r.jpg\" \"070\" \"King\" "

