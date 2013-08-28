# -*- coding: UTF-8 -*-
# Uses Anaconda 1.6.2 64-bit distribution package with Python 2.7.5
#
# Timo Mätäsaho
# University of Oulu
# 2013

import os
import sys
import Image
import matplotlib.pyplot as plt
from OratUtils import OratUtils
from scipy.misc import imread


def osearch( img, txtf, sw ):

	curDir = os.path.dirname( os.path.realpath( __file__ ))
	imagename = curDir + "\\Images\\" + img
	tfile = curDir + "\\Texts\\" + txtf


	charcount, charpos, charlines, wordlens = OratUtils.stringparser( tfile, sw )

	if not charpos:
		print "Couldn't find string %s from page %s" %(sw, img)
		return

	image = imread(imagename)
	im = image.convert("L")

	plt.imshow(im)
	plt.show()

	return





if __name__ == "__main__":

	if len(sys.argv) == 4:
		osearch( sys.argv[1], sys.argv[2], sys.argv[3] )
	else:
		print "\nWrong amount of parameters. Need following arguments: "
		print "Name of the image, name of the correspoding raw text file, word that's being searched\n"
		print "For example use:    python osearch.py \"Lit_Ms_E_41 070r.jpg\" \"070\" \"King\" "

