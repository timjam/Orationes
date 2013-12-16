*************************************************************
*															*
*							README							*
*															*
*					OrationesSearch python					*
*						Timo Mätäsaho						*
*					  Univesrity of Oulu					*
*							2013							*
*															*
*************************************************************

This program was designed to be executed from a PHP site with
PHP exec command. However it can be run from commandline as 
well.



Installation:
	
	Install latest Anaconda package on your computer or 
	server. This program was done using 
	Anaconda 1.6.2 64-bit dist package with Python 2.7.5
	Download Anaconda from 'http://continuum.io/downloads'

	Anaconda has all the needed python libraries to run this 
	program and to write the autodocumentation if needed.



Running the program:
	
	Execute the program from PHP by using "exec".

	Execute the program from command line:
		python osearch.py imagename switch data word

		where:
			imagename 	is the full name of the image that 
						is to be searched

			switch is either -s or -f.
			-f 	means the written data is gathered from 
				the plain text XML file
			-s 	means the data is given as a string 
				(from PHP site)

			data is the name of the plaintext XML file 
			when using -f switch and needs to be the 
			data string when using the -s switch

			word is the word or character that is searched 
			from a page. NOTE that it can also be a 
			regular expression for more detailed searches. 
			However the system might not be stable with 
			complex regular expressions.



File structure:
	The root has all the needed python files.
	All the images used must be in the Images folder.
	All the plaintext XMLs must be in the text folder.



Updating the documentation:

	For correct syntax refer to Sphinx documentation guides 
	or learn from the existing documentation syntax :)

	It's possible to include latex style formating in the 
	documentation comments and get them show correctly 
	formatted on website and on pdf documentation.

	code.rst file is a collection file that tells Sphinx 
	which files should be included in the documentation. 
	
	Compile the web document by using
		make html

	Compile the tex document by using
		make latex

		then use latex to produce a PDF document from 
		the tex file.

		NOTE: We all know that latex can be annoying 
		sometimes, but the make latex command automagically 
		includes all the right formatings and packages to 
		the tex file. So if you encounter problems when 
		compiling the tex file, please first check your 
		latex syntax and ensure that you have installed 
		correct latex packages. For more info refer to 
		Sphinx latex documentation.

		http://sphinx-doc.org/builders.html#sphinx.builders.latex.LaTeXBuilder

	Because the peakdet.py is open source from internet and written by a 
	different author, it might not always follow the Sphinx documentation 
	guidelines. That's why it usually produces lots of warnings and errors 
	when compiling the documentation. If you are annoyed by them OR especially 
	if they break the latex code, just remove the entry from the code.rst file.



The program is quite large and complex and far from complete. Hopefully whoever unfortunate soul 
some day might continue this, finds this README file and the already written 
documentation helpful. Time ran out and parts of the code never got refactored to 
more understandable form. Anyway I hope you enjoy your time with the code!   :D