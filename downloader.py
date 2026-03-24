import yt_dlp
import os

# Configuration
OUTPUT_FILENAME = 'video_to_process.mp4'

def download_youtube_video(url):
    """
    Downloads the best quality video stream (up to 1080p) from YouTube.
    Audio is intentionally skipped to speed up the download process.
    """
    print(f"[INFO] Preparing to download from: {url}")
    
    # Remove the old video file if it exists to prevent conflicts
    if os.path.exists(OUTPUT_FILENAME):
        print(f"[INFO] Removing old video file: {OUTPUT_FILENAME}")
        os.remove(OUTPUT_FILENAME)

    # yt-dlp configuration options
    ydl_opts = {
        # 'bestvideo' selects only the video stream without audio.
        # '[height<=1080]' limits the resolution to 1080p or lower.
        # '[ext=mp4]' forces the mp4 format which plays nicely with OpenCV.
        'format': 'bestvideo[height<=1080][ext=mp4]/best[height<=1080][ext=mp4]/best',
        'outtmpl': OUTPUT_FILENAME,
        'noplaylist': True, # Only download the single video, not the whole playlist
    }

    print("[INFO] Starting download (video only)...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"\n[INFO] Success! Video saved as '{OUTPUT_FILENAME}'.")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")

def main():
    print("=== YouTube Fast Downloader for CV ===")
    url = input("Paste the YouTube video URL here: ").strip()
    
    if not url:
        print("[ERROR] No URL provided. Exiting.")
        return
        
    download_youtube_video(url)

if __name__ == "__main__":
    main()