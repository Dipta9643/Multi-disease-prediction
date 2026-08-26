import os
import shutil

src = r"C:\Users\saiki\.gemini\antigravity\brain\ebea4c61-1124-4858-85df-787cc275f6b0\health_vector_1782066370775.png"
dst_dir = r"s:\projects\main\static\images"

try:
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dst_dir, "health_vector.png"))
    print("Image copied successfully to static/images/health_vector.png")
except Exception as e:
    print(f"Error copying image: {e}")
