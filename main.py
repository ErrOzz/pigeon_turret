import cv2
from ultralytics import YOLO

# Constants
MODEL_NAME = 'yolov8n.pt'
BIRD_CLASS_ID = 14 # Bird ID in the COCO dataset
CONFIDENCE_THRESHOLD = 0.5 # Confidence threshold (50%)

def calculate_target_center(x1, y1, x2, y2):
    """Calculates the center of the bounding box (target)."""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def main():
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_NAME)
    
    print("[INFO] Connecting to the camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Failed to open the camera!")
        return

    print("[INFO] System active. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Get frame dimensions (to calculate deviation from the center)
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        # Object detection
        # Set classes=[BIRD_CLASS_ID] to detect only birds, ignoring other COCO objects
        results = model(frame, classes=[BIRD_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Plot bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Draw the center of the frame (the turret's crosshair)
        cv2.drawMarker(annotated_frame, (frame_center_x, frame_center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Process detected targets
        for box in results[0].boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Find the center point of the target
            target_x, target_y = calculate_target_center(x1, y1, x2, y2)
            
            # Calculate the deviation of the target from the center of the frame
            offset_x = target_x - frame_center_x
            offset_y = target_y - frame_center_y
            
            # Draw a red dot directly on the target's center
            cv2.circle(annotated_frame, (target_x, target_y), 5, (0, 0, 255), -1)
            
            # Output tracking information (future placeholder for UART commands to Arduino)
            print(f"Target acquired! X:{target_x} Y:{target_y} | Deviation: dx={offset_x}, dy={offset_y}")

        # Show the resulting frame in a window
        cv2.imshow("Turret Vision PoC", annotated_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")

if __name__ == "__main__":
    main()