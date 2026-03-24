import os

# Configuration: Target directory to clean up
DIRECTORY = 'dataset/train/pigeon'

def rename_files_sequentially():
  print(f"[INFO] Scanning directory: {DIRECTORY}")
  
  if not os.path.exists(DIRECTORY):
    print("[ERROR] Directory does not exist.")
    return

  # Get all .jpg files and sort them alphabetically to preserve original chronological order
  files =[f for f in os.listdir(DIRECTORY) if f.endswith('.jpg')]
  files.sort()
  
  if not files:
    print("[INFO] No images found in the directory.")
    return

  print(f"[INFO] Found {len(files)} files. Starting safe rename process...")

  # Step 1: Rename all files to a temporary name to avoid overwriting conflicts
  temp_files =[]
  for i, filename in enumerate(files):
    old_path = os.path.join(DIRECTORY, filename)
    temp_name = f"temp_{i:05d}.jpg"
    temp_path = os.path.join(DIRECTORY, temp_name)
    
    os.rename(old_path, temp_path)
    temp_files.append(temp_name)

  # Step 2: Rename from temporary names to the final clean sequential names
  for i, temp_name in enumerate(temp_files):
    temp_path = os.path.join(DIRECTORY, temp_name)
    final_name = f"bird_crop_{i:05d}.jpg"
    final_path = os.path.join(DIRECTORY, final_name)
    
    os.rename(temp_path, final_path)

  print(f"[INFO] Successfully renamed {len(files)} files without gaps.")

if __name__ == "__main__":
  rename_files_sequentially()