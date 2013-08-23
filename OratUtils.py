#import codecs

class OratUtils:

	@staticmethod
	def stringparser( tfile, c ):

		string = []
		fid = open( tfile, 'r' )

		string = fid.readline().decode('utf-8')
		fid.close()

		print len(string)
		print string

