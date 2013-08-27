import re

class OratUtils:

	# Performs case sensitive search for text file tfile with string or character c on default.
	# Argument c can be any regular expression
	@staticmethod
	def stringparser( tfile, c ):

		lineindlist = [] # Holds lists of the match positions.
		co = []
		charcount = [] # Holds the lengths of each line

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

			lineindlist.append(charindlist)

		print lineindlist
		print charcount