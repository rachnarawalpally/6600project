##### Classifying images' nudity using a pre-trained model ####
##### From https://github.com/notAI-tech/NudeNet #####


# all_labels = [
#     "FEMALE_GENITALIA_COVERED",
#     "FACE_FEMALE",
#     "BUTTOCKS_EXPOSED",
#     "FEMALE_BREAST_EXPOSED",
#     "FEMALE_GENITALIA_EXPOSED",
#     "MALE_BREAST_EXPOSED",
#     "ANUS_EXPOSED",
#     "FEET_EXPOSED",
#     "BELLY_COVERED",
#     "FEET_COVERED",
#     "ARMPITS_COVERED",
#     "ARMPITS_EXPOSED",
#     "FACE_MALE",
#     "BELLY_EXPOSED",
#     "MALE_GENITALIA_EXPOSED",
#     "ANUS_COVERED",
#     "FEMALE_BREAST_COVERED",
#     "BUTTOCKS_COVERED",
# ] 



### Example with one image
# from nudenet import NudeDetector
# detector = NudeDetector()
# # the 320n model included with the package will be used

# # images with nudity:
# # 1433
# detections = detector.detect('data/album_images/OAhcn27hMCDkSc6mTYhlH5.jpg') # Returns list of detections
# print(detections)



###### Looping through all the images adn saving into nudity_results.csv
import os
import csv
from nudenet import NudeDetector
import pandas as pd
from tqdm import tqdm

# Initialize detector
detector = NudeDetector()

# Folder with album images
image_folder = "data/album_images"
output_csv = "data/new/nudity_results.csv"

# Prepare results list
results = []

# Get list of image files
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg'))
]

# Loop with progress bar
for filename in tqdm(image_files, desc="Running NudeDetector"):
    image_path = os.path.join(image_folder, filename)

    try:
        detections = detector.detect(image_path)

        if detections:
            for det in detections:
                results.append({
                    "image": filename,
                    "label": det.get("class", "unknown"),
                    "confidence": round(det.get("score", 0.0), 4),
                    "box": det.get("box")
                })
        else:
            results.append({
                "image": filename,
                "label": "none_detected",
                "confidence": 0,
                "box": None
            })
    except Exception as e:
        results.append({
            "image": filename,
            "label": "error",
            "confidence": "N/A",
            "box": str(e)
        })

# Save results to CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["image", "label", "confidence", "box"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nDetection results saved to: {output_csv}")




####### Load nudity_results CSV to one hot encode labels
df = pd.read_csv("data/new/nudity_results.csv")
print(df.columns)

# Strip .jpg from filenames
df['image'] = df['image'].str.replace('.jpg', '', case=False, regex=False)

# Create 'face' column
df['face'] = df['label'].str.contains('face', case=False, na=False).astype(int)

# Define all detailed body-part labels to one-hot encode
detailed_labels = [
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

# Create one-hot columns for each detailed label
for label in detailed_labels:
    df[label] = df['label'].str.contains(label, case=False, na=False).astype(int)

# Group by image, take the max of each one-hot column
agg_columns = ['face'] + detailed_labels
nudity_onehot_df = df.groupby('image')[agg_columns].max().reset_index()

# Save cleaned one-hot encoded result
nudity_onehot_df.to_csv("data/new/nudity_onehot_encode.csv", index=False)

print("Saved final one-hot nudity indicators")




###### Combine with main dataset and save to albums_colors_nudity.csv

# Load main dataset and new nudity one hot encode dataset
df_main = pd.read_csv("data/new/albums_with_colors.csv")
df_nudity = pd.read_csv("data/new/nudity_onehot_encode.csv")

# Convert columns names to lowercae
df_nudity.columns = df_nudity.columns.str.lower()

print(len(df_main))
print(len(df_nudity))

# Merge datasets 
df_combined = pd.merge(df_main, df_nudity, left_on='album_id', right_on='image', how='left')

# Drop the 'image' column from the merged dataset
df_combined.drop(columns=['image'], inplace=True)

# Save the combined dataset
df_combined.to_csv("data/new/albums_colors_nudity.csv", index=False)

print("Merged dataset saved")



