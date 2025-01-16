# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2025/01/16

import cv2
import numpy as np
import onnxruntime as ort


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


class CartoonizerONNX:
    def __init__(self, onnx_model_path="frozen_cartoonizer.onnx"):
        # Load the ONNX model
        self.session = ort.InferenceSession(onnx_model_path)
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def inference(self, input_image):
        # Resize and crop the image
        image = resize_crop(input_image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: batch_image})
        # Process the output
        output = outputs[0][0]
        output = (output + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output


if __name__ == '__main__':
    model = CartoonizerONNX(onnx_model_path="frozen_cartoonizer.onnx")

    image = cv2.imread("test.jpg")
    # Perform inference using the ONNX model
    onnx_output = model.inference(image)
    cv2.imwrite("test-out-onnx.jpg", onnx_output)
    print("Done!")
