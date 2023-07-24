import numpy as np
from PIL import Image, ImageFilter

import cv2
import numpy as np


def rotate_image(image, angle, pivot, keep_point):
    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(pivot, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Calculate the new position of the fixed point after rotation
    new_point = np.dot(M, np.array([keep_point[0], keep_point[1], 1])).astype(int)

    return Image.fromarray(rotated_image), (new_point[0], new_point[1])


def paste_image(
    main_image,
    image_to_paste,
    paste_point,
    start_defect_point,
    blure,
    height_percent,
    width_percent,
    color_factor,
):
    start_point = start_defect_point[0]
    end_point = start_defect_point[1]

    image_to_paste = image_to_paste.crop(
        (
            start_point[0],
            start_point[1],
            end_point[0],
            end_point[1],
        )
    )

    w = image_to_paste.size[0] * (width_percent / 100)
    h = image_to_paste.size[1] * (height_percent / 100)

    image_to_paste = image_to_paste.resize((int(w), int(h)))

    mask = np.array(image_to_paste)
    mask[mask != 0] = 1
    color_factor = color_factor / 100
    mask = mask * 255 * color_factor
    mask = Image.fromarray(mask.astype(np.uint8)).convert("L")
    mask = mask.filter(ImageFilter.GaussianBlur(blure))

    # Paste the foreground onto the result image at the desired location based on the mask
    paste_point[0] = paste_point[0] - 2
    paste_point[1] = paste_point[1] - 2
    main_image = Image.fromarray(main_image)
    main_image.paste(image_to_paste, (paste_point[0], paste_point[1]), mask=mask)
    return main_image
