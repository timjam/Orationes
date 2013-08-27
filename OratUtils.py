#import codecs
import re

class OratUtils:

	@staticmethod
	def stringparser( tfile, c ):

		string = []
		fid = open( tfile, 'r' )

		string = fid.readline().decode('utf-8', 'ignore')
		fid.close()

		print
		print len(string)
		print
		print "testi " + string[63] + " testi "
		print

		indlist = []

		for m in re.finditer(c, string):
			indlist.append(m.start())	# m.start() osoittaa jokaisen osuman ekaan indeksiin.
			#print m.start()

		print indlist