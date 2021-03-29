import numpy as np
import pygame as pg

def readimg(img):  # Turn image into RGB surface arrays
    imgsurface = pg.image.load("data/" + img + ".jpg", "r")
    arrRed = pg.surfarray.pixels_red(imgsurface)
    arrGreen = pg.surfarray.pixels_green(imgsurface)
    arrBlue = pg.surfarray.pixels_blue(imgsurface)

    return arrRed, arrGreen, arrBlue

imgname =  input("Enter image name: ")
arrRed, arrGreen, arrBlue = readimg(imgname)