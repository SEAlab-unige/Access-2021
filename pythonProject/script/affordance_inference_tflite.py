import tensorflow as tf
import cv2
import numpy as np
import os
from data_loader import get_image_array

flags = tf.compat.v1.flags
flags.DEFINE_string('path_to_tflite',
                    "/home/pavilion/Desktop/tesi_triennale/models/MobileNetV1_UNET_FP32.tflite",
                    "Path to TFLite file")
flags.DEFINE_string('path_to_images',
                    "/home/pavilion/Desktop/tesi_triennale/images",
                    "Path to images directory")
FLAGS = flags.FLAGS


def run_inference_for_single_image(interpreter, input_details, input_data, output_details):
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # For each pixel get the class with maximum confidence
    prediction = np.argmax(prediction, axis=3)

    return prediction[0]


def main():
    # Assertions
    assert FLAGS.path_to_tflite is not None, "Please, provide path_to_tflite parameter"
    assert os.path.exists(FLAGS.path_to_tflite), "Path to tflite file does not exists"
    assert FLAGS.path_to_images is not None, "Please, provide path_to_images parameter"

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=FLAGS.path_to_tflite)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input and output tensors' details
    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']

    # Load images path
    images_path = os.listdir(FLAGS.path_to_images)

    # Prediction loop
    for index, path in enumerate(images_path):
        # Load image
        input_data = cv2.imread(FLAGS.path_to_images + "/" + path)

        # Pre-processing
        input_data = get_image_array(input_data, input_shape[1], input_shape[2], ordering="channels_last")

        # Inference
        pred = run_inference_for_single_image(interpreter, input_details, input_data, output_details)

        # Visualize prediction (RGB format)
        black = np.zeros((output_shape[1], output_shape[2], 3))
        black[pred == 1, :] = [0, 0, 255]  # blue = grasp
        black[pred == 3, :] = [0, 0, 255]
        black[pred == 2, :] = [0, 255, 0]  # green = no grasp
        cv2.imshow('Prediction', cv2.cvtColor(black.astype(np.uint8), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
