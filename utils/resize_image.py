from PIL import Image
import os

in_folder="datasets"
out_folder="data"

def resize_image(image,size=224):
    image.thumbnail((size,size))
    new_image=Image.new("RGB", (size, size), (0, 0, 0))
    new_image.paste(
        image,
        ((size - image.width) // 2, (size - image.height) // 2)
    )
    return new_image


os.makedirs(out_folder,exist_ok=True)

for image_file in os.listdir(in_folder):
    if image_file.endswith(".jpg"):
        path = os.path.join(in_folder,image_file) 
        try:
            image=Image.open(path).convert("RGB")
            image=resize_image(image)
            image.save(os.path.join(out_folder, image_file))
            print("Done")
        except Exception as ex:
            print(ex)