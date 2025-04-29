### This function converts an image to HSV color space and detects skin pixels using a predefined skincolor range. ####

import os
import csv
import cv2
import numpy as np
import pandas as pd

def detect_skin_percentage(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin pixels
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    
    # Calculate the number of skin pixels
    num_skin_pixels = np.sum(skin_mask > 0)
    
    # Calculate the total number of pixels
    total_pixels = image.shape[0] * image.shape[1]
    
    # Calculate the percentage of skin pixels
    skin_percentage = (num_skin_pixels / total_pixels) * 100
    
    return skin_percentage

# Example for one image:
# image_path = 'images/image_833.jpg'
# percentage = detect_skin_percentage(image_path)
# print(f"Percentage of skin in the image: {percentage:.2f}%")

# Folder path
image_folder = "data/album_images"
output_csv = "data/new/skin_percentages.csv"

# Collect results
results = []

# Loop over all image files
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg')):
        image_path = os.path.join(image_folder, filename)
        try:
            percent = detect_skin_percentage(image_path)
            if percent is not None:
                results.append({
                    "image": filename,
                    "skin_percentage": round(percent, 2)
                })
            else:
                results.append({
                    "image": filename,
                    "skin_percentage": "unreadable"
                })
        except Exception as e:
            results.append({
                "image": filename,
                "skin_percentage": f"error: {e}"
            })

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["image", "skin_percentage"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Skin percentage results saved to {output_csv}")

### Does not do well detecting skin with greyscale images 


###### Combine with main dataset and save to albums_colors_nudity_skin.csv

# Load main dataset and new nudity one hot encode dataset
df_main = pd.read_csv("data/new/albums_colors_nudity.csv")
df_skin = pd.read_csv("data/new/skin_percentages.csv")

# Strip .jpg from filenames
df_skin['image'] = df_skin['image'].str.replace('.jpg', '', case=False, regex=False)

print(len(df_main))
print(len(df_skin))

# Merge datasets 
df_combined = pd.merge(df_main, df_skin, left_on='album_id', right_on='image', how='left')

# Drop the 'image' column from the merged dataset
df_combined.drop(columns=['image'], inplace=True)

# Save the combined dataset
df_combined.to_csv("data/new/albums_colors_nudity_skin.csv", index=False)

print("Merged dataset saved")
