from PIL import Image
import numpy as np

im = Image.open("Data/test1.jpg")
im_resized = im.resize((600, 600), Image.ANTIALIAS)
im_bw = im_resized.convert('1', dither=Image.NONE)
im_data = np.asarray(im_resized)

print(im_data)

