import yt_dlp
import os
import sys

# Force utf-8 stdout to avoid windows charmap crash
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = "test_videos"
os.makedirs(OUT_DIR, exist_ok=True)

test_cases = {
    "accident": 'ytsearch1:"CCTV car crash intersection dashcam shorts"',
    "weapon": 'ytsearch1:"robbery convenience store knife CCTV short"',
    "fall": 'ytsearch1:"CCTV person fainting falling to ground short"',
    "panic": 'ytsearch1:"crowd running away panic CCTV short"'
}

for name, query in test_cases.items():
    dest_path = f"{OUT_DIR}/{name}.mp4"
    if os.path.exists(dest_path):
        print(f"[OK] {name}.mp4 already exists. Skipping download.")
        continue
        
    print(f"[DOWNLOAD] Fetching {name} scenario...")
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': f'{OUT_DIR}/{name}.%(ext)s',
        'match_filter': yt_dlp.utils.match_filter_func("duration < 600"),
        'noplaylist': True,
        'quiet': False
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        print(f"[SUCCESS] Downloaded {name}.\n")
    except Exception as e:
        print(f"[ERROR] Failed to download {name}: {e}\n")

print("[COMPLETE] Test Video Fetching Done.")
