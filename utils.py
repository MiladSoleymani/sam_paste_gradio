import numpy as np
from PIL import Image, ImageFilter


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
    right, bottom = (
        image_to_paste.size[0] - start_defect_point[1],
        image_to_paste.size[1] - start_defect_point[0],
    )
    image_to_paste = image_to_paste.crop(
        (start_defect_point[1], start_defect_point[0], right, bottom)
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
    main_image = Image.fromarray(main_image)
    main_image.paste(image_to_paste, (paste_point[0], paste_point[1]), mask=mask)
    return main_image
