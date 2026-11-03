#!/usr/bin/env python3
import sys
import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on a single static image.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="runs/detect/tiago_single_class_yolo26/weights/best.pt",
        help="Path to YOLO model weights (.pt)"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Path to input image file (e.g., screenshot.png)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="test_result_4.jpg",
        help="Path/filename to save the output annotated image."
    )
    args = parser.parse_args()

    # Verify input paths
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return

    print(f"Loading YOLO model: {args.model}...")
    model = YOLO(args.model)

    print(f"Running inference on: {args.image}...")
    # Run prediction
    results = model(args.image, verbose=True)

    # Process and print detection details
    result = results[0]
    boxes = result.boxes
    print("\n--- Detections Found ---")
    if len(boxes) == 0:
        print("No objects detected.")
    else:
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"[{idx+1}] Class: '{cls_name}' (id: {cls_id}) | Conf: {conf:.4f} | Box (xyxy): {[round(x, 1) for x in xyxy]}")

    # Save output annotated image
    output_path = args.output
    result.save(filename=output_path)
    print(f"\nSaved annotated image to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
