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


def osearch( img, switch, txtf, sw ):
	r"""
		The main program that only calls the processing methods from OratUtils and HFun classes.

		:param img: The name of the image.
		:type img: string
		:param txtf: The name of the cleaned XML file.
		:type txtf: string
		:param sw: The word or letter that is searched.
		:type sw: string
		:returns: nothing or JSON string

		The return option have to be chosen between returning the string as a return code or 
		is it just printed out for the calling PHP program. Currently it is being printed.

	"""

	debug = True

	# Open the image and text file with their absolute paths to ensure that the right files from
	# the right place are opened
	curDir = os.path.dirname( os.path.realpath( __file__ ))
	imagename = curDir + "\\Images\\" + img
	tfile = curDir + "\\Texts\\" + txtf


	# Parse the XML file. The XML file must be formatted to raw text with no XML tags. 
	# Should be done separately outside of this program unless this program is updated
	# to do that as well.
	if( switch == "-f" ):
		charcount, charpos, charlines, wordlens = OratUtils.txtfparser( tfile, sw )

	if( switch == "-s" ):
		charcount, charpos, charlines, wordlens = OratUtils.stringparser( txtf, sw )

	if not charpos:
		print "Couldn't find string \'%s\' from page %s" %(sw, img)
		return 0


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

	# Get the rightlines according to the XML file and the image
	rlines, imlines = OratUtils.processlines( charcount, imlines )

	# Get the lines that are used to search the possible hits
	slines = OratUtils.padlines( imlines, rlines, charlines ) # slines - [a1, a2, ..., an], n == number of unique rlines

	# Find the correspondences between the lines which are used for searching and the bounding boxes
	coords = OratUtils.findCorr( bboxes, slines, charcount, imlines, debug )

	jsondata = OratUtils.packCoordsToJson( slines, origimage, coords, charpos, wordlens, bboxes, debug )

	print jsondata

	# The jsondata may have to be returned instead of just printed out. This depends heavily of the behavior of the calling program
	# In this case we use a PHP site to call this program. Need to consult with Ilkka about how the PHP site will handle this file.

	return 1







if __name__ == "__main__":

	if len(sys.argv) == 5:

		if( sys.argv[2] == "-f" ):
			osearch( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] )
		elif( sys.argv[2] == "-s" ):
			osearch( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] )
		else:
			print "Wrong parameters"
			print
			print "Use python osearch.py \"imagename\" -switch \"textfilename or string\" \"word\""
			print
			print "As a switch you can use:"
			print "\t-f \t read the text for this page from a file." 
			print "\t\t With this switch you must use the text file name."
			print
			print "\t-s \t use the given string as the text input." 
			print "\t\t With this switch you must pass a string containing the text as a third parameter."

	else:
		print "\nWrong amount of parameters. Need following arguments: "
		print "Name of the image, name of the correspoding raw text file, word that's being searched\n"
		print "For example use:    python osearch.py \"Lit_Ms_E_41 070r.jpg\" -f \"070\" \"King\" "

