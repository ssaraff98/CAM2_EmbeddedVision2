import os
from PIL import Image

def get_images():
    path = "./Samples"
    image_list = []

    # !!! Do error handling with file extensions !!!
    for file in os.listdir(path):
        image_list.append(Image.open(os.path.join(path, file)))

    return image_list

def resize_image(image, max_length=2400):
    width, height = image.size

    if max(width, height) > max_length:
        ratio = float(max_length) / width if height < width else float(max_length) / height
    else:
        ratio = 1

    resized_width  = int(ratio * width)
    resized_height = int(ratio * height)

    resized_width  = resized_width if resized_width % 32 == 0 else (resized_width // 32 - 1) * 32
    resized_height = resized_height if resized_height % 32 == 0 else (resized_height // 32 - 1) * 32

    image = image.resize((int(max(32, resized_width)), int(max(32, resized_height))))
    #image = image.resize((192, 192))

    ratio_width  = resized_width / width
    ratio_height = resized_height / height

    return image, (ratio_width, ratio_height)
