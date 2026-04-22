from PIL import image
import os

in_folder=""

def resize_image(image,size=224):
    image.thumbnail((size,size))
    new_image=Image.new("RGB", (size, size), (0, 0, 0))
    new_img.paste(
        image,
        ((size - image.width) // 2, (size - image.height) // 2)
    )
    return new_img
