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

def remove_gray_background(image_path, output_path, threshold = 100):
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

    # Crop transparent borde
    bbox = img.getbbox()
    if bbox:
        img_cropped = img.crop(bbox)
    else:
        img_cropped = img
        print('No Content Found')

    # plt.imshow(img_cropped)
    # plt.show()

    return(img_cropped)

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    width, height = im.size
    im = im.crop((5, 5, width - 5, height - 5))
    size = im.size

    top_left_pixel = im.getpixel((0,0))
    bottom_right_pixel = im.getpixel((size[0] - 1, size[1] - 1))
    
    tol = 150
    is_top_left_white = top_left_pixel[0] >= tol and top_left_pixel[1] >= tol and top_left_pixel[2]
    is_bottom_right_white = bottom_right_pixel[0] >= tol and bottom_right_pixel[1] >= tol and bottom_right_pixel[2] >= tol
    
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

def overlay_images(backgroundIMG, foregroundIMG, output_path, position=(0, 0), opacity=1.0):
    """
    Overlays a foreground image onto a background image.

    Args:
        background_path (str): Path to the background image.
        foreground_path (str): Path to the foreground image.
        output_path (str): Path to save the overlaid image.
        position (tuple): (x, y) coordinates for the top-left corner of the foreground image.
        opacity (float): Opacity of the foreground image (0.0-1.0).
    """

    foregroundIMG = foregroundIMG.resize((backgroundIMG.width, backgroundIMG.height))

    if opacity < 1.0:
      foregroundIMG.putalpha(int(255 * opacity))

    backgroundIMG.paste(foregroundIMG, position, foregroundIMG)
    plt.imshow(backgroundIMG)
    plt.axis('off')
    plt.show()
    # background.save(output_path)

def imshowpair_subplot_rgb(img1_pil, img2_pil, img_compare_fname, titles=("Image A", "Image B", "Abs Diff")):
    # Convert PIL Images to [0,1] float NumPy arrays
    img1 = np.asarray(img1_pil).astype(np.float32) / 255.0
    img2 = np.asarray(img2_pil).astype(np.float32) / 255.0

    # Compute absolute difference
    diff = np.abs(img1 - img2)
    diff_vis = np.clip(diff / np.max(diff), 0, 1)

    # Create subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes, [img1, img2, diff_vis], titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(img_compare_fname, dpi=300, bbox_inches='tight')
    plt.close()

streamtraceIMG = sys.argv[1]
experimentIMG = sys.argv[2]

streamCropped = remove_gray_background(streamtraceIMG, "output.png")
streamCropped = trim(streamCropped)
streamCropped = trim(streamCropped)

experCropped = remove_gray_background(experimentIMG, "output.png")
experCropped = trim(experCropped)
experCropped = trim(experCropped) # Crop twice to remove border outlet and rescale correctly

streamCropped, experCropped = resize_images(streamCropped, experCropped)

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(streamCropped)
axs[1].imshow(experCropped)
plt.show()

overlay_images(streamCropped, experCropped, 'test.png', position=(0, 0), opacity = 0.25)
overlay_images(experCropped, streamCropped, 'test.png', position=(0, 0), opacity = 0.75)

streamPILIMG = Image.fromarray(streamCropped)
experPILIMG = Image.fromarray(experCropped)

imshowpair_subplot_rgb(streamPILIMG, experPILIMG, 'img_compare_fname.png')