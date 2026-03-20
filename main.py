import cv2
from ultralytics import YOLO

# Constants
MODEL_NAME = 'best.pt'
CONFIDENCE_THRESHOLD = 0.3 # Lowered slightly for testing with phone screens

def calculate_target_center(x1, y1, x2, y2):
    """Calculates the center of the bounding box (target)."""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def main():
    print("[INFO] Loading custom YOLO model...")
    model = YOLO(MODEL_NAME)
    
    # Print the dictionary of classes the model actually learned!
    # This will show us if Pigeon is 0 or 1.
    print(f"[INFO] Classes inside this model: {model.names}")
    
    print("[INFO] Connecting to the camera...")
    cap = cv2.VideoCapture(0)
    
    # Force higher resolution for better detection of small objects
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print("[ERROR] Failed to open the camera!")
        return

    print("[INFO] System active. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        # Object detection
        # IMPORTANT: We removed the 'classes' filter so it shows everything it knows.
        # We also added imgsz=640 to match our training configuration.
        results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)
        
        annotated_frame = results[0].plot()

        cv2.drawMarker(annotated_frame, (frame_center_x, frame_center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Process detected targets
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Extract the specific class ID and its text name from the model
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            target_x, target_y = calculate_target_center(x1, y1, x2, y2)
            offset_x = target_x - frame_center_x
            offset_y = target_y - frame_center_y
            
            cv2.circle(annotated_frame, (target_x, target_y), 5, (0, 0, 255), -1)
            
            # Print the class name along with the coordinates
            print(f"Target: {class_name} (ID: {class_id}) | X:{target_x} Y:{target_y} | dx={offset_x}, dy={offset_y}")

        cv2.imshow("Pigeon Turret - Custom Model", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")

if __name__ == "__main__":
    main()