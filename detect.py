import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
import os
import argparse  # Used to accept command-line arguments

def recognize_plate(image_path, show_plot=True):
    """
    Detects and recognizes a license plate from a given image path.
    """
    
    # --- 1. LOAD MODELS ---
    try:
        # Load Plate Detector (YOLO)
        plate_detector = YOLO('models/best.pt')
        # Initialize the EasyOCR Reader
        reader = easyocr.Reader(['en'])
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure 'models/best.pt' exists.")
        return

    # --- 2. READ AND PROCESS IMAGE ---
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Could not read image from {image_path}.")
    except Exception as e:
        print(e)
        return

    # --- 3. DETECT THE PLATE (YOLOv8) ---
    results = plate_detector(original_image)
    final_plate_text = "NOT DETECTED"

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        
        # Crop the plate from the original image
        cropped_plate = original_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        print("Plate detected and cropped.")

        # --- 4. EXTRACT TEXT (EasyOCR) ---
        ocr_result = reader.readtext(cropped_plate)
        
        if ocr_result:
            detected_texts = [text for (bbox, text, prob) in ocr_result]
            final_plate_text = " ".join(detected_texts).upper()
            print(f"EasyOCR Result: {final_plate_text}")
        else:
            final_plate_text = "NO TEXT FOUND"

        # --- 5. DRAW RESULTS ---
        cv2.rectangle(original_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
        cv2.putText(original_image, final_plate_text, (xyxy[0], xyxy[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    # --- 6. SAVE AND SHOW FINAL IMAGE ---
    output_path = "result.png"
    cv2.imwrite(output_path, original_image)
    print(f"Result saved to: {output_path}")

    # Show the final image (optional)
    if show_plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Final Prediction: {final_plate_text}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # This allows you to run the script from the command line
    parser = argparse.ArgumentParser(description='Automatic Number Plate Recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()
    
    recognize_plate(args.image, show_plot=False) # Don't show plot in command-line mode
