# -*- coding: UTF-8 -*-
# Uses Anaconda 1.6.2 64-bit distribution package with Python 2.7.5
#
# Timo Mätäsaho
# University of Oulu
# 2013

import os
import sys
import Image as Im
import timeit
from OratUtils import OratUtils
from HFun import HFun
from scipy.misc import fromimage
import numpy as np


def getBoxesAndLines( img ):
	r"""
		Optional directly callable program that can be used to extract the bounding box
		and line location information from an image.

		:param img: The name of the image.
		:type img: string
		:returns: JSON string

		Returns a JSON array containing the possible locations of the text lines and 
		bounding boxes.
	"""
	
	debug = Debug

	# Open the image and text file with their absolute paths to ensure that the right files from
	# the right place are opened
	curDir = os.path.dirname( os.path.realpath( __file__ ))
	imagename = curDir + "\\Images\\" + img


	# Open the original image and convert it to grayscale
	try:
		origimage = Im.open(imagename)
	except IOError:
		print "Errno 2: No such file or directory or cannot identify image file \n%s \nCheck the spelling of the filename or ensure name points to an image file." %(imagename)
		return 2
	except:
		print "Unknown error while trying to open file %s" %(imagename)
		return 9

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
	bboxes = OratUtils.boundingBox( cIm, debug )


	# Get the positions of lines according to the image and its radon transform
	imlines = OratUtils.poormanradon( cIm, imagename, ImHeight, debug )


	jsondata = OratUtils.packBoxesAndLines( bboxes, imlines )

	print jsondata

	# The jsondata may have to be returned instead of just printed out. This depends heavily of the behavior of the calling program
	# In this case we use a PHP site to call this program. Need to consult with Ilkka about how the PHP site will handle this file.

	return 1







if __name__ == "__main__":

	if( len(sys.argv) == 2):

		if( isinstance(sys.argv[1], str) ):
			getBoxesAndLines( sys.argv[1] )
		else:
			print "Wrong argument ERROR: Given argument is not a string"

	else:
		print "Wrong amount of parameters"