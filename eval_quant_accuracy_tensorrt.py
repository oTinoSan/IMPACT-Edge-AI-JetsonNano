import os
import sys
import numpy as np
import cv2
import glob
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def calculate_volume(dims):
    return np.prod(dims).item()

def preprocess_image(image_path):
    # Adjust the size and scale of the image to match the input of your model
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Example size, adjust to your model's input
    img = img.astype(np.float32)
    img = img / 255.0  # Example normalization, adjust as necessary
    img = img.transpose((2, 0, 1))  # Change from HWC to CHW format if needed
    return img

def main(engine_path, data_path):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # Get input and output binding indices
    input_index = engine.get_binding_index('input_0')  # Adjust name as necessary
    output_index = engine.get_binding_index('output_0')  # Adjust name as necessary

    image_paths = glob.glob(os.path.join(data_path, '*.jpg'))  # Assumes .jpg images, adjust if different
    batch_size = 1  # For simplicity, this example handles one image at a time

    for image_path in image_paths:
        img = preprocess_image(image_path)
        np_img = np.array([img], dtype=np.float32)

        # Allocate memory for inputs and outputs
        d_input = cuda.mem_alloc(1 * np_img.nbytes)
        output_shape = engine.get_binding_shape(output_index)
        output_size = calculate_volume(output_shape) * batch_size * np.dtype(np.float32).itemsize
        d_output = cuda.mem_alloc(output_size)

        # Create numpy buffers for input and output
        outputs = np.empty(output_shape, dtype=np.float32)

        # Transfer input data to device
        cuda.memcpy_htod(d_input, np_img)
        # Execute model
        context.execute(batch_size=batch_size, bindings=[int(d_input), int(d_output)])
        # Transfer predictions back
        cuda.memcpy_dtoh(outputs, d_output)

        # Process outputs (for example, applying softmax)
        print("Processed:", image_path, "Output:", outputs)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_engine> <path_to_data>")
        sys.exit(1)
    engine_path = sys.argv[1]
    data_path = sys.argv[2]
    main(engine_path, data_path)
