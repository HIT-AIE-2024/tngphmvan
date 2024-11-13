from abc import ABC, abstractmethod

import numpy as np


class YOLOv8Interface(ABC):
    @abstractmethod
    def load_yolov8_model(self, model_path: str)->None:
        """
        Load the YOLOv8 model from a specified file path.

        Args:
            model_path (str): Path to the YOLOv8 model file.

        Raises:
            FileNotFoundError: If the specified model path does not exist.
        """
        pass

    @abstractmethod
    def quantize_model_to_int8(self)->None:
        """
        Quantize the loaded YOLOv8 model to INT8 precision using TensorRT.

        Raises:
            RuntimeError: If model has not been loaded before quantization.
        """
        pass

    @abstractmethod
    def inference(self, image: np.ndarray, quantized: bool = False)->tuple:
        """
        Run inference on the input image with an option to use the quantized model.

        Args:
            image (np.ndarray): Input image as a numpy array.
            quantized (bool, optional): If True, use the quantized model for inference. 
                                        Defaults to False.

        Returns:
            tuple: Processed image with detections, raw result list, and confidence scores.

        Raises:
            ValueError: If appropriate model is not available for inference.
        """
        pass
    @abstractmethod
    def display(self, result: list) -> None:
        """
        Display the inference result with bounding boxes on the image.

        Args:
            result (list): result after inference.

        Raises:
            ValueError: If no result is available for display.
        """
        pass
