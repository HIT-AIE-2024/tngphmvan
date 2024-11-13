import time
from matplotlib import pyplot as plt
from ultralytics import YOLO
from configs.config import YOLOv8Interface

class YOLOv8TensorRT(YOLOv8Interface):
    def __init__(self, device: str = 'cpu'):
        """
        Initialize YOLOv8TensorRT with model and quantized engine placeholders.

        Args:
            device (str): The device to run the model on ('cpu', 'cuda', or 'cuda:<index>').
                          Defaults to 'cpu'.
        """
        self.model = None
        self.quantized_engine = None
        self.device = device

    def load_yolov8_model(self, path: str = "configs/yolov8n.pt") -> None:
        """
        Load the YOLOv8 model from a pre-trained model file.

        Args:
            path (str): Specifies the path of the YOLOv8 model.
                        Defaults to 'yolov8n.pt'.

        Raises:
            FileNotFoundError: If the YOLO model file cannot be found.
        """
        start = time.time()
        try:
            self.model = YOLO(path)
            print(f"Load model successfully on {self.device} in {time.time() - start:.2f} seconds")
        except FileNotFoundError as e:
            raise FileNotFoundError("YOLO model file not found.") from e

    def quantize_model_to_int8(self, task: str = "detect") -> None:
        """
        Quantize the YOLOv8 model to INT8 using TensorRT and save the engine.

        Args:
            task (str): Specifies the task type for quantized model inference.
                        Defaults to 'detect'.

        Raises:
            RuntimeError: If model quantization fails or workspace size is insufficient.
        """
        start = time.time()
        try:
            self.model.export(format="engine", device=self.device, dynamic=True, batch=8, workspace=4, int8=True)
            self.quantized_engine = YOLO("../configs/yolov8n.engine", task=task)
            print(f"Quantize model successfully on {self.device} in {time.time() - start:.2f} seconds")
        except RuntimeError as e:
            raise RuntimeError("Model quantization failed. Check TensorRT compatibility and settings.") from e

    def inference(self, image_path: str, quantized: bool = False, quantized_path: str = "configs/yolov8n.engine") -> tuple:
        """
        Run inference on the input image with an option to use the quantized model.

        Args:
            image_path (str): Path to the input image file.
            quantized (bool, optional): If True, use the quantized model for inference.
                                        Defaults to False.

        Returns:
            tuple: Processed image with detections, raw result list, and confidence scores.

        Raises:
            ValueError: If no model is available for inference.
            FileNotFoundError: If the specified image file cannot be found.
        """
        result = None
        if not self.model and not self.quantized_engine:
            raise ValueError("Model not loaded. Please load the YOLOv8 model before inference.")

        try:
            start = time.time()
            if quantized:
                if self.quantized_engine is None and quantized_path:
                    self.quantized_engine = YOLO(quantized_path)
                elif self.quantized_engine is None and not quantized_path:
                    self.quantize_model_to_int8("detect")
                result = self.quantized_engine.predict(image_path)
            else:
                result = self.model.predict(image_path)
            print(f"Inference time on {self.device}: {time.time() - start:.2f} seconds")
            detections = result[0].boxes.data  # Lấy tensor chứa các boxes
            confidence_scores = [float(det[4]) for det in detections]  # Chỉ số 4 là confidence score
            return result, confidence_scores
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file '{image_path}' not found.") from e


    def display(self, result: list) -> None:
        """
        Display the inference result with bounding boxes on the image.
        Args:
            result (list): result after inference
        Raises:
            ValueError: If no result is available for display.
        """
        if result:
            plt.figure(figsize=(10, 10))
            plt.imshow(result[0].plot())
            plt.axis('off')
            plt.title("Result")
            plt.show()
        else:
            raise ValueError("No result available for display. Please run inference first.")
