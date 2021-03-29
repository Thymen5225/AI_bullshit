from PIL import Image, ImageOps
import glob
import numpy as np

# returns: image data (list containing image data - image data in [row1], [row2], etc.
# returns: list with corresponding planes and carriers

# greyscale --- value 0 is black, 255 is white

# x and y are hor and vert pixels

# NOTE order not the same as in data cause this does it alphabetically
def getImageData(x, y):
    image_array = []
    carrier_lst = []
    type_lst = []
    for filename in glob.glob('Data/*.jpg'):
        im = Image.open(filename)

        # image data
        im_gray = ImageOps.grayscale(im.resize((x, y), Image.ANTIALIAS))
        im_data = np.asarray(im_gray)
        image_array.append(im_data)

        # fleet & name data
        carrier = filename.split()[1]
        typ = filename.split()[2][:4]
        carrier_lst.append(carrier)
        type_lst.append(typ)

    print(image_array[0])
    return image_array, carrier_lst, type_lst


getImageData(1280, 720)
