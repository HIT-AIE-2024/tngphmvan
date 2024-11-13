from feature.yolov8_tensorRT import YOLOv8TensorRT

def main():
    # Initialize YOLOv8TensorRT object
    yolo_trt = YOLOv8TensorRT(device='cuda:0')

    # Paths to model file and image
    model_path = "configs/yolov8n.pt"
    image_path = r"bus.jpg"

    # 1. Load YOLOv8 model
    try:
        yolo_trt.load_yolov8_model(model_path)
    except FileNotFoundError as e:
        print(e)
        return  # End if model cannot be loaded

    # 2. Quantize model to INT8
    try:
        yolo_trt.quantize_model_to_int8()
    except RuntimeError as e:
        print(e)
        return  # End if quantization fails

    # 3. Run inference with the original model
    print("\nRunning inference on original model:")
    try:
        result, confidence_original = yolo_trt.inference(image_path, quantized=False)
        # Display the original model inference result
        yolo_trt.display(result)
    except (ValueError, FileNotFoundError) as e:
        print(e)

    # 4. Run inference with the quantized model
    print("\nRunning inference on quantized model:")
    try:
        result_quantized, confidence_quantized = yolo_trt.inference(image_path, quantized=True)
        # Display the quantized model inference result
        yolo_trt.display(result_quantized)
    except (ValueError, FileNotFoundError) as e:
        print(e)

    # Check for mismatch in detection counts between original and quantized models
    if len(confidence_original) != len(confidence_quantized):
        print(
            f"Warning: Mismatch in number of detections between original ({len(confidence_original)}) and quantized model ({len(confidence_quantized)}).")

    # Calculate confidence drop for matched detections
    matched_confidences = zip(confidence_original, confidence_quantized)
    confidence_drops = [orig - quant for orig, quant in matched_confidences]

    print("\nConfidence drops for matched detections:")
    for i, drop in enumerate(confidence_drops):
        print(f"Detection {i + 1}: {drop:.2f}")

if __name__ == "__main__":
    main()
