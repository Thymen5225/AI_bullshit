print('thymen was here')
print("Petr as well")
print("I pushed this again 3")


from PIL import Image

im = Image.open("\Data\test.png")
im.save("test-600.png", dpi=(600,600))

