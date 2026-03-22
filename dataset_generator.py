import cv2
import os
from ultralytics import YOLO

# --- Configuration ---
VIDEO_PATH = 'pigeons_video.mp4'     # Path to the current video
OUTPUT_DIR = 'dataset/train/pigeon'  # Target folder ('pigeon' or 'other_birds')
BIRD_CLASS_ID = 14                   # COCO class ID for 'bird'
CONFIDENCE_THRESHOLD = 0.5           # Minimum confidence for detection
FRAME_SKIP = 15                      # Process every 15th frame
PADDING = 15                         # Extra pixels around the bird
MIN_CROP_SIZE = 180                  # Ignore crops smaller than MIN_CROP_SIZE pixels
MAX_IMAGES_PER_VIDEO = 300           # Stop after extracting this many images from the CURRENT video

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
    """
    Finds the highest index among existing files to continue numbering safely.
    """
    if not os.path.exists(directory):
        return 0
        
    existing_files =[f for f in os.listdir(directory) if f.endswith('.jpg')]
    if not existing_files:
        return 0
        
    max_index = -1
    for filename in existing_files:
        try:
            # Extract number from format 'bird_crop_XXXXX.jpg'
            index_str = filename.split('_')[-1].split('.')[0]
            index = int(index_str)
            if index > max_index:
                max_index = index
        except (ValueError, IndexError):
            continue
            
    return max_index + 1

def main():
    print("[INFO] Loading base YOLO model...")
    model = YOLO('yolov8n.pt')
    
    create_directory(OUTPUT_DIR)
    
    # 1. Global index for safe file naming
    global_index = get_starting_index(OUTPUT_DIR)
    print(f"[INFO] Next file will be named: bird_crop_{global_index:05d}.jpg")
    
    print(f"[INFO] Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {VIDEO_PATH}.")
        return

    frame_count = 0
    # 2. Session counter for the current video limit
    session_count = 0

    print(f"[INFO] Starting extraction. Goal: {MAX_IMAGES_PER_VIDEO} images. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Reached the end of the video.")
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        results = model(frame, classes=[BIRD_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
        display_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            cropped_bird = get_padded_crop(frame, x1, y1, x2, y2, PADDING)
            
            # Filter out tiny or invalid crops
            if cropped_bird.size == 0 or cropped_bird.shape[0] < MIN_CROP_SIZE or cropped_bird.shape[1] < MIN_CROP_SIZE:
                continue

            # Save the file using the global index
            filename = os.path.join(OUTPUT_DIR, f"bird_crop_{global_index:05d}.jpg")
            cv2.imwrite(filename, cropped_bird)
            
            # Increment both counters
            global_index += 1
            session_count += 1

            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Check if we reached the limit for THIS video
            if session_count >= MAX_IMAGES_PER_VIDEO:
                break
        
        # If the limit was reached inside the FOR loop, break the WHILE loop too
        if session_count >= MAX_IMAGES_PER_VIDEO:
            print(f"[INFO] Reached the limit of {MAX_IMAGES_PER_VIDEO} images for this run.")
            break

        display_resized = cv2.resize(display_frame, (1024, 576))
        cv2.imshow("Dataset Generator", display_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Process interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Extraction complete! Saved {session_count} images in this session.")

if __name__ == "__main__":
    main()