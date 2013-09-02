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
from scipy.misc import fromimage, toimage, imshow


def osearch( img, txtf, sw ):

	curDir = os.path.dirname( os.path.realpath( __file__ ))
	imagename = curDir + "\\Images\\" + img
	tfile = curDir + "\\Texts\\" + txtf


	charcount, charpos, charlines, wordlens = OratUtils.stringparser( tfile, sw )

	if not charpos:
		print "Couldn't find string %s from page %s" %(sw, img)
		return

	origimage = Im.open(imagename)
	grayimage = origimage.convert("L") # Conversion from RGB to Grayscale

	#plt.imshow(grayimage, cmap = cm.Greys_r)
	#plt.show()

	ImLength, ImHeight = origimage.size

	# Conversion from PIL image to scipy image and then from uint8 to float
	# tI = fromimage(origimage, 'True') # Should do conversion from PIL to scipy and also from RGB to gray
	tI = fromimage(grayimage) # From PIL to scipy image
	tI = tI.astype('float') # From uint8 to float

	filteredIm = OratUtils.hfilter( tI, 620, ImHeight, ImLength, 20 )

	filteredIm = filteredIm.astype('uint8') # From float to uint8



	plt.imshow(filteredIm, cmap = cm.Greys_r)
	plt.show()

	return





if __name__ == "__main__":

	if len(sys.argv) == 4:
		osearch( sys.argv[1], sys.argv[2], sys.argv[3] )
	else:
		print "\nWrong amount of parameters. Need following arguments: "
		print "Name of the image, name of the correspoding raw text file, word that's being searched\n"
		print "For example use:    python osearch.py \"Lit_Ms_E_41 070r.jpg\" \"070\" \"King\" "

