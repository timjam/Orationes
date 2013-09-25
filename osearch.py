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

	rBBs = np.zeros((5,rounds))

	#print rounds
	#print bbYs
	#print
	#print bboxes
	#print

	#print ""

	for i in range( rounds ):

		if( i==0 or ( slines[i] - slines[i-1] > 100 ) ):
			minlim = slines[i] - 70

		else:
			minlim = slines[i-1]

		#print minlim
		#print slines[i]

		cBBYstarts = bbYs[ bbYs > minlim ]
		cBBYstarts = cBBYstarts[ cBBYstarts < slines[i] ]

		#print cBBYstarts

		rBBs = bboxes[:,bboxes[2,:] == cBBYstarts]
		xsta = rBBs[1]
		ysta = rBBs[2]
		xsto = rBBs[3]
		ysto = rBBs[4]

		#print str(xsta) + "\t" + str(ysta) + "\t" + str(xsto) + "\t" + str(ysto)
		#print str(rBBs[1]) + "\t" + str(rBBs[2]) + "\t" + str(rBBs[3]) + "\t" + str(rBBs[4])
		#print

		cIm[ysta, xsta:xsto] = 0
		cIm[ysto, xsta:xsto] = 0
		cIm[ysta:ysto, xsta] = 0
		cIm[ysta:ysto, xsto] = 0


		#print rBBs
		#print
		#print
		#print '*****************************'
		#print
		#print

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

