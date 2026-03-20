import cv2
import os
from ultralytics import YOLO

# --- Configuration ---
VIDEO_PATH = 'pigeons_video.mp4'     # Change this for each new video
OUTPUT_DIR = 'dataset/train/pigeon'  # Keep this the same to accumulate pigeon images
BIRD_CLASS_ID = 14                   # COCO class ID for 'bird'
CONFIDENCE_THRESHOLD = 0.5           # Minimum confidence to consider it a bird
FRAME_SKIP = 45                      # Process every 45th frame
PADDING = 15                         # Extra pixels around the bird

def create_directory(path):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")

def get_padded_crop(frame, x1, y1, x2, y2, padding):
    """Crops the image with padding, keeping it within frame boundaries."""
    height, width = frame.shape[:2]
    crop_y1 = max(0, int(y1) - padding)
    crop_y2 = min(height, int(y2) + padding)
    crop_x1 = max(0, int(x1) - padding)
    crop_x2 = min(width, int(x2) + padding)
    
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]

def get_starting_index(directory):
    """Counts existing .jpg files to continue numbering without overwriting."""
    if not os.path.exists(directory):
        return 0
    existing_files =[f for f in os.listdir(directory) if f.endswith('.jpg')]
    return len(existing_files)

def main():
    print("[INFO] Loading base YOLO model...")
    model = YOLO('yolov8n.pt')
    
    create_directory(OUTPUT_DIR)
    
    # Check how many images are already in the folder
    saved_images_count = get_starting_index(OUTPUT_DIR)
    print(f"[INFO] Found {saved_images_count} existing images. Resuming numbering.")
    
    print(f"[INFO] Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {VIDEO_PATH}. Check the file path.")
        return

    frame_count = 0

    print("[INFO] Starting extraction. Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Reached the end of the video.")
            break

        frame_count += 1

        # Skip frames to ensure variety in the dataset
        if frame_count % FRAME_SKIP != 0:
            continue

        # Detect birds in the current frame
        results = model(frame, classes=[BIRD_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        display_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            cropped_bird = get_padded_crop(frame, x1, y1, x2, y2, PADDING)
            
            if cropped_bird.size == 0 or cropped_bird.shape[0] < 20 or cropped_bird.shape[1] < 20:
                continue

            # Save the crop with the continuously incrementing index
            filename = os.path.join(OUTPUT_DIR, f"bird_crop_{saved_images_count:05d}.jpg")
            cv2.imwrite(filename, cropped_bird)
            saved_images_count += 1

            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Show the processing feed
        display_resized = cv2.resize(display_frame, (1024, 576))
        cv2.imshow("Dataset Generator", display_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Process interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Extraction complete! Total images in '{OUTPUT_DIR}' is now {saved_images_count}.")

if __name__ == "__main__":
    main()