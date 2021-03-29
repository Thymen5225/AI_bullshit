from PIL import Image
import numpy as np
import pygame as pg

im = Image.open("Data/test1.jpg")
im_resized = im.resize((600, 600), Image.ANTIALIAS)
im_bw = im_resized.convert('1', dither=Image.NONE)
im_data = np.asarray(im_resized)
def readimg(img):  # Turn image into RGB surface arrays
    imgsurface = pg.image.load("data/" + img, "r")
    arrRed = pg.surfarray.pixels_red(imgsurface)
    arrGreen = pg.surfarray.pixels_green(imgsurface)
    arrBlue = pg.surfarray.pixels_blue(imgsurface)

print(im_data)


imgname =  input("Enter image name: ") # INCLUDE THE .JPG .PNG STUFF!!!!!
arrRed, arrGreen, arrBlue = readimg(imgname)
