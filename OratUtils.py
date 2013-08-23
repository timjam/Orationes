#import codecs

class OratUtils:

	@staticmethod
	def stringparser( tfile, c ):

		string = []
		fid = open( tfile, 'r' )

		string = fid.readline().decode('utf-8', 'ignore')
		fid.close()

		print len(string)
		#print string
		for c in string:
			print c