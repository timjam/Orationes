import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook

#image_file = cbook.get_sample_data('Lit_Ms_E_41 070r.jpg')
image = imread('Images\Lit_Ms_E_41 070r.jpg')
image[200:220,200:220] = [0,255,0]


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