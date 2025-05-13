#!/usr/bin/env python
from PIL import Image, ImageChops, ImageOps
import matplotlib.pyplot as plt
from skimage import io
import skimage as sk
import scipy.ndimage as ndimage
import numpy as np
import numpy as np
from PIL import Image, ImageChops
import sys
import numpy as np


def remove_gray_background(image_path, output_path, threshold = 50):
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        r, g, b, a = item
        # Check if the pixel is roughly gray
        if abs(r - g) < threshold and abs(r - b) < threshold and abs(g - b) < threshold:
            # Check that the gray is not too dark or too light
            if 100 < r < 250:
                newData.append((255, 255, 255, 0))  # Make gray transparent
                continue
        newData.append(item)

    img.putdata(newData)

    # Crop transparent border
    alpha = img.split()[-1]  # Get alpha channel
    bbox = img.getbbox()
    if bbox:
        img_cropped = img.crop(bbox)
    else:
        img_cropped = img
        print('No Content Found')

    plt.imshow(img_cropped)
    plt.show()
    img_cropped.save(output_path, "PNG")

    return(img_cropped)

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    width, height = im.size
    im = im.crop((10, 10, width - 10, height - 10))
    size = im.size

    top_left_pixel = im.getpixel((0,0))
    bottom_right_pixel = im.getpixel((size[0] - 1, size[1] - 1))
    
    is_top_left_white = top_left_pixel[0] >= 240 and top_left_pixel[1] >= 240 and top_left_pixel[2]
    is_bottom_right_white = bottom_right_pixel[0] >= 240 and bottom_right_pixel[1] >= 240 and bottom_right_pixel[2] >= 240
    
    is_top_left_transparent = top_left_pixel[3] < 20 if len(top_left_pixel) > 3 else False
    is_bottom_right_transparent = bottom_right_pixel[3] < 20 if len(bottom_right_pixel) > 3 else False

    if (is_top_left_white and is_bottom_right_white) or (is_top_left_transparent and is_bottom_right_transparent):
        diff = ImageChops.difference(im, bg)
        offset = round(size[1] / 100 * 4)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop((max(bbox[0] - offset, 0), max(bbox[1] - offset, 0), min(bbox[2] + offset, size[0]), min(bbox[3] + offset, size[1])))
        else:
            return im
    else:
        print('No change because no transparency or no white background')
        return im

def resize_images(img1, img2):
    """
    Resizes two images to have the same dimensions (minimum of both).
    """
    width1, height1 = img1.size
    width2, height2 = img2.size

    min_width = min(width1, width2)
    min_height = min(height1, height2)

    img1_resized = img1.resize((min_width, min_height))
    img2_resized = img2.resize((min_width, min_height))

    return img1_resized, img2_resized


streamtraceIMG = sys.argv[1]
experimentIMG = sys.argv[2]

streamCropped = remove_gray_background(streamtraceIMG, "output.png")
streamCropped = trim(streamCropped)

experCropped = remove_gray_background(experimentIMG, "output.png")
experCropped = trim(experCropped)