import csv
import requests
from collections import Counter
import pandas as pd

import csv
import requests

### Script to get the gender of each artist from Wikidata
### Utilized Chatgpt for this code

def get_gender_or_group_from_wikidata(name):
    try:
        # Search Wikidata
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': name
        }
        res = requests.get(search_url, params=params).json()

        if not res['search']:
            return "Not found"

        qid = res['search'][0]['id']  # Get first matched Q-ID

        # Load full entity data
        entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        entity_data = requests.get(entity_url).json()
        claims = entity_data['entities'][qid]['claims']

        # Check if "instance of" (P31) is a band/musical group
        instance_claims = claims.get('P31', [])
        for instance in instance_claims:
            val_id = instance['mainsnak']['datavalue']['value']['id']
            if val_id in ['Q215380', 'Q2088357', 'Q588244']:
                return "band"

        # Check gender (P21)
        gender_claim = claims.get('P21')
        if not gender_claim:
            return "Unknown"

        gender_id = gender_claim[0]['mainsnak']['datavalue']['value']['id']

        # Look up gender label
        label_url = f"https://www.wikidata.org/wiki/Special:EntityData/{gender_id}.json"
        gender_data = requests.get(label_url).json()
        gender_label = gender_data['entities'][gender_id]['labels']['en']['value']

        return gender_label

    except Exception as e:
        return f"Error: {e}"


# ---- CSV I/O ---- #
input_csv = "albums_genres_grouped.csv"
output_csv = "names_with_gender.csv"

results = []

with open(input_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        name = row['artist_name']
        gender = get_gender_or_group_from_wikidata(name)
        results.append({'name': name, 'gender': gender})
        print(f"{name} → {gender}")

with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['name', 'gender'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n Saved gender and band info to {output_csv}")


#Find counts for each category
df = pd.read_csv("gender_clean.csv")
gender_counts = df['gender'].value_counts()

print("\nGender Counts:")
print(gender_counts)

###### Combine with main dataset and save to albums_colors_nudity_skin_gender.csv

# Load main dataset and new nudity one hot encode dataset
df_main = pd.read_csv("data/new/albums_colors_nudity_skin.csv")
df_gender = pd.read_csv("data/new/artist_genders_more_women.csv")

# Merge datasets 
df_combined = pd.merge(df_main, df_gender, left_on='artist_name', right_on='name', how='left')

# Drop the 'image' column from the merged dataset
df_combined.drop(columns=['name'], inplace=True)

# Save the combined dataset
df_combined.to_csv("data/new/albums_colors_nudity_skin_gender.csv", index=False)

print("Merged dataset saved")


