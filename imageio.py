import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook
import Image

#image_file = cbook.get_sample_data('Lit_Ms_E_41 070r.jpg')
#image = imread('Images\Lit_Ms_E_41 070r.jpg')
image = Image.open('Images\Lit_Ms_E_41 070r.jpg')
print image.load()
#image = image.load()
image[200:220, 200:220] = [0,255,0]

#print image[1400,1000]
#print [image[1400,1000,0]*299/1000 + image[1400,1000,1]*587/1000 + image[1400,1000,2]*114/1000, image[1400,1000,0]*299/1000 + image[1400,1000,1]*587/1000 + image[1400,1000,2]*114/1000, image[1400,1000,0]*299/1000 + image[1400,1000,1]*587/1000 + image[1400,1000,2]*114/1000]

image[:,:] = [(image[:,:,0]*299/1000 + image[:,:,1]*587/1000 + image[:,:,2]*114/1000) for x in range(3)]
#print image[1400,1000]
#print image[1401,1001]
#image[:,:] *= [299/1000, 587/1000, 114/1000]

# L = R * 299/1000 + G * 587/1000 + B * 114/1000

#im = image.convert('LA') # converts image from rgb to gray
plt.imshow(image)
plt.axis('off')
plt.show()



#from matplotlib.pyplot import imshow
#from scipy.misc import imread

#I = imread(r"Lit_Ms_E_41 070r.jpg")
#imshow(I)

#from PIL import Image

#im = Image.open(r"Lit_Ms_E_41 070r.jpg")
#im.show()