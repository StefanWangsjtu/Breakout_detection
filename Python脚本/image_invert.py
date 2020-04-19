from PIL import Image
import PIL.ImageOps
image = Image.open("C:/Users/ASUS/Desktop/a1.png")
invert_image = PIL.ImageOps.invert(image)
invert_image.save("C:/Users/ASUS/Desktop/a2.png")
