import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

def save_image_from_url(url, filename, folder='images'):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        path = os.path.join(folder, filename)
        img.save(path)
        return path  # Optionally return saved path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None
    
df = pd.read_csv('albums_genres_cleaned.csv')  #
for idx, row in df.iterrows():
    url = row['image_url']
    filename = f'image_{idx}.jpg'  # or use row['title'] or some unique ID
    save_image_from_url(url, filename)
