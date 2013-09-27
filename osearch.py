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
	grayimage = origimage.convert("L") # Conversion from RGB to Grayscale


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
	slines = OratUtils.padlines( imlines, rlines, charlines )

	# Find the correspondences between the lines which are used for searching and the bounding boxes

	bbYs = bboxes[2,:]
	rounds = slines.shape[0]

	coords = np.zeros((7,rounds), np.int16)

	#print bboxes

	for i in range( rounds ):

		if( i==0 or ( slines[i] - slines[i-1] > 100 ) ):
			minlim = slines[i] - 70

		else:
			minlim = slines[i-1]

		cBBYstarts = bbYs[ bbYs > minlim ]
		cBBYstarts = cBBYstarts[ cBBYstarts < slines[i] ]

		temp = bboxes[:,bboxes[2,:] == cBBYstarts]
		#print cBBYstarts
		#coords[:,i] = bboxes[:,bboxes[2,:] == cBBYstarts]
		#print coords[:,i]
		coords[0,i] = temp[0]	# Sisältää kyseistä bounding boxia vastaavan patching labelin
		coords[1,i] = temp[1]	# Sisältää kyseisen bounding boxin xstart koordinaatin
		coords[2,i] = temp[2]	# Sisältää kyseisen bounding boxin ystart koordinaatin
		coords[3,i] = temp[3]	# Sisältää kyseisen bounding boxin xstop koordinaatin
		coords[4,i] = temp[4]	# Sisältää kyseisen bounding boxin ystop koordinaatin
		coords[5,i] = charcount[ np.where( bbYs == temp[2])[0] ]	# Sisältää kyseisellä rivillä olevien kirjainten lukumäärän
		coords[6,i] = cBBYstarts[0]	# Sisältää kyseistä bounding boxia vastaan rivin radonmuunnoksesta saadun keskikohdan y-koordinaatin
		#print np.where( bbYs == temp[2] )
		#print np.where( bbYs == temp[2] )[0]

		#print coords


		#xsta = coords[1,i]
		#ysta = coords[2,i]
		#xsto = coords[3,i]
		#ysto = coords[4,i]

		#cIm[ysta, xsta:xsto] = 0
		#cIm[ysto, xsta:xsto] = 0
		#cIm[ysta:ysto, xsta] = 0
		#cIm[ysta:ysto, xsto] = 0

	#print coords

	#print charpos

	#Y = coords[6,:]
	#X = coords[2,:] + ( coords[3,:] - coords[1,:] )/coords[5,:]

	for i in range( 3 ):

		print charpos[i]
		print coords[2,i]
		print coords[3,i]
		print coords[1,i]
		print coords[5,i]

		X = coords[2,i] + charpos[i] * ( coords[3,i]-coords[1,i] )/coords[5,i]
		print X
		Y  = coords[6,i]
		print Y

	#print Y
	#print X

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


	return







if __name__ == "__main__":

	if len(sys.argv) == 4:
		osearch( sys.argv[1], sys.argv[2], sys.argv[3] )
	else:
		print "\nWrong amount of parameters. Need following arguments: "
		print "Name of the image, name of the correspoding raw text file, word that's being searched\n"
		print "For example use:    python osearch.py \"Lit_Ms_E_41 070r.jpg\" \"070\" \"King\" "

