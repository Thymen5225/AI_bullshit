from PIL import Image, ImageOps
import glob
import numpy as np
import random

# returns: image data (list containing image data - image data in [row1], [row2], etc.
# returns: list with corresponding planes and carriers

# greyscale --- value 0 is black, 255 is white

# x and y are hor and vert pixels

# NOTE order not the same as in data cause this does it alphabetically
def getImageData(x, y):
    carriers = ['Lufthansa', 'KLM', 'Qantas', 'Emirates', 'AirFrance', 'Etihad', 'Turkish', 'American', 'Iberia', 'Qatar']
    models = ['B747', 'A380', 'B777', 'B787', 'A340', 'A330', 'B737', 'A320', 'A350', 'E190']

    image_array = []
    carrier_lst = []
    type_lst = []
    for filename in glob.glob('Data/*.jpg'):
        im = Image.open(filename)

        # image data
        im_gray = ImageOps.grayscale(im.resize((x, y), Image.ANTIALIAS))
        im_data = np.asarray(im_gray)
        image_array.append(im_data.flatten())

        # print(im_data.flatten())

        # fleet & name data
        carrier = filename.split()[1]
        for i in range(10):
            if carriers[i] == carrier:
                carrier_index = i
        typ = filename.split()[2][:4]
        for i in range(10):
            if models[i] == typ:
                typ_index = i
        carrier_lst.append(carrier_index)
        type_lst.append(typ_index)

    temp = list(zip(image_array, carrier_lst, type_lst))
    random.shuffle(temp)
    image_array, carrier_lst, type_lst = zip(*temp)

    return image_array, carrier_lst, type_lst


def getTestData(x, y):
    carriers = ['Lufthansa', 'KLM', 'Qantas', 'Emirates', 'AirFrance', 'Etihad', 'Turkish', 'American', 'Iberia',
                'Qatar']
    models = ['B747', 'A380', 'B777', 'B787', 'A340', 'A330', 'B737', 'A320', 'A350', 'E190']

    image_array = []
    carrier_lst = []
    type_lst = []
    for filename in glob.glob('testdata/*.jpg'):
        im = Image.open(filename)

        # image data
        im_gray = ImageOps.grayscale(im.resize((x, y), Image.ANTIALIAS))
        im_data = np.asarray(im_gray)
        image_array.append(im_data.flatten())

        # print(im_data.flatten())

        # fleet & name data
        carrier = filename.split()[2]
        for i in range(10):
            if carriers[i] == carrier:
                carrier_index = i
        typ = filename.split()[3][:4]
        for i in range(10):
            if models[i] == typ:
                typ_index = i
        carrier_lst.append(carrier_index)
        type_lst.append(typ_index)

        print(type_lst, carrier_lst, image_array)

    return image_array, carrier_lst, type_lst

# image_array, carrier_lst, type_lst = getImageData(64, 36)
