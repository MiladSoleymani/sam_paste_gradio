import gradio as gr
import numpy as np

from utils import paste_image, rotate_image
from sam.inference import SamGradioRun


class Points:
    def __init__(self) -> None:
        self.__point_list = []
        self.__label_list = []

    def get_point_list(self):
        return np.array(self.__point_list)

    def get_label_list(self):
        return np.array(self.__label_list)

    def append(self, image_point, image_label):
        self.__point_list.append(image_point)
        self.__label_list.append(image_label)

    def clear_points(self):
        self.__point_list = []
        self.__label_list = []
        print(f"The points list of defect : {self.get_point_list()}")
        print(f"The points list of defect : {self.get_label_list()}")


class BorderPoint:
    def __init__(self) -> None:
        self.__point_list = []

    def get_point_list(self):
        return np.array(self.__point_list)

    def append(self, image_point):
        self.__point_list.append(image_point)

    def clear_points(self):
        self.__point_list = []
        print(f"The points list of main image : {self.get_point_list()}")


class StartEndDefectPoint:
    def __init__(self) -> None:
        self.__start_point = None
        self.__end_point = None

    def get_point(self):
        return [self.__start_point, self.__end_point]

    def set(self, image_point, label_point):
        if label_point == "start":
            self.__start_point = image_point
        elif label_point == "end":
            self.__end_point = image_point

    def clear_points(self):
        self.__start_point = None
        self.__end_point = None
        print(f"The points : {self.get_point()}")


def aug_generator(image_to_paste, start_defect_point, rotation_augmentation):
    angle = np.random.choice(list(range(1, int(rotation_augmentation))))
    image_to_paste = np.array(image_to_paste)
    pivot = (image_to_paste.shape[0] // 2, image_to_paste.shape[1] // 2)
    return rotate_image(image_to_paste, angle, pivot, start_defect_point)


POINT_OBJECT = Points()
BORDER_POINT_OBJECT = BorderPoint()
STARTENDDEFECTPOINT = StartEndDefectPoint()
SAM_RUN_OBJECT = SamGradioRun()


# def run(image, label, evt: gr.SelectData):
def run(label_points, evt: gr.SelectData):
    POINT_OBJECT.append(
        (evt.index[0], evt.index[1]), 1 if label_points == "include" else 0
    )
    print(f"The points list: {POINT_OBJECT.get_point_list()}")
    print(f"The labels list: {POINT_OBJECT.get_label_list()}")


def run2(evt: gr.SelectData):
    BORDER_POINT_OBJECT.append((evt.index[0], evt.index[1]))
    print(f"The points list: {BORDER_POINT_OBJECT.get_point_list()}")


def run3(label_points, evt: gr.SelectData):
    STARTENDDEFECTPOINT.set((evt.index[0], evt.index[1]), label_points)
    print(f"The point : {STARTENDDEFECTPOINT.get_point()}")


def generate_mask_with_sam():
    return SAM_RUN_OBJECT.detect(POINT_OBJECT)


def remove_background():
    return SAM_RUN_OBJECT.remove_background(POINT_OBJECT)


def generate(
    main_image,
    batch_size,
    blure,
    height,
    width,
    color_factor,
    rotation_augmentation,
):
    image_to_paste = SAM_RUN_OBJECT.remove_background(POINT_OBJECT)
    main_image_points = BORDER_POINT_OBJECT.get_point_list()
    start_end_defect_point = STARTENDDEFECTPOINT.get_point()

    results = []
    for _ in range(int(batch_size)):
        image_to_paste, start_end_defect_point[0] = aug_generator(
            image_to_paste, start_end_defect_point[0], rotation_augmentation
        )
        for point in main_image_points:
            print(f"img number = {_}, point = {point}")
            results.append(
                paste_image(
                    main_image=main_image,
                    image_to_paste=image_to_paste,
                    paste_point=point,
                    start_defect_point=start_end_defect_point,
                    blure=blure,
                    height_percent=height,
                    width_percent=width,
                    color_factor=color_factor,
                )
            )
    return results


with gr.Blocks() as demo:
    # SAM part
    input_img1 = gr.Image().style(height=800)
    label_points = gr.Radio(
        choices=["include", "exclude"], value="include", label="Points Label"
    )

    model_name = gr.Radio(
        choices=["vit_h", "vit_b", "vit_l"], value="vit_h", label="Model name"
    )
    model_creator = gr.Button(value="Initialize Model")
    clear_button = gr.Button(value="Clear Points")
    generate_mask = gr.Button(value="Generate Mask")
    output_img1 = gr.Image().style(height=800)
    output_img2 = gr.Image().style(height=800)
    crop_points = gr.Radio(choices=["start", "end"], value="start", label="crop_points")

    input_img1.select(run, [label_points], None)
    clear_button.click(POINT_OBJECT.clear_points, None, None)
    model_creator.click(SAM_RUN_OBJECT.initialize, [input_img1, model_name], None)
    generate_mask.click(generate_mask_with_sam, None, output_img1)
    generate_mask.click(remove_background, None, output_img2)

    # paste part
    output_img2.select(run3, [crop_points], None)
    input_img2 = gr.Image().style(height=800)
    batch_size = gr.Number(label="Batch Size")
    blure = gr.Slider(0, 100, label="blure")
    color_factor = gr.Slider(0, 100, label="color_factor")
    height = gr.Slider(0, 1000, label="height", info="Changing height in percentage")
    width = gr.Slider(0, 1000, label="width", info="Changing width in percentage")
    rotation_augmentation = gr.Slider(
        0, 90, label="rotation augmentation", info="Between 0 to 90"
    )

    output_img3 = gr.Gallery().style(height=800)

    generate_img = gr.Button(value="Generate Image")

    input_img2.select(run2, None, None)
    generate_img.click(
        generate,
        [
            input_img2,
            batch_size,
            blure,
            height,
            width,
            color_factor,
            rotation_augmentation,
        ],
        output_img3,
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
