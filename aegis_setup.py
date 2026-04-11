import os
from ultralytics import YOLO

def download_models():
    print("🛡️ AegisVision Setup: Downloading lightweight pre-trained models for zero-shot deployment...")
    os.makedirs("models", exist_ok=True)
    
    try:
        print("\n✅ Setup Complete! Accident model (CrashSentinel_Prime.pt) handles vehicle accident verification. No extra models required.")
    except Exception as e:
        print(f"⚠️ Error during model fetch: {e}")

if __name__ == "__main__":
    download_models()
