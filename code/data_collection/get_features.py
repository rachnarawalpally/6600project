

from transformers import CLIPProcessor, CLIPModel
import torch

from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import os
import time
import math
import gc 

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory if it doesn't exist
OUTPUT_DIR = 'album_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving results to directory: {OUTPUT_DIR}")

def analyze_album_cover_with_clip(image_url, categories=None, device=device, retry_count=3):
    """
    Uses OpenAI's CLIP model for zero-shot classification of album covers
    With retry mechanism for robustness
    """
    if categories is None:
        # Default categories for album covers
        categories = [
            # People
            "one person", "group photo", "no people",
            
            # Photographic styles
            "portrait", "close-up shot", "full body shot",
            
            # Scene types
            "nature scene", "city scene", "indoor scene", "outdoor scene",
            "concert scene", "beach scene",
            
            # Visual styles
            "black and white photography", "color photography", "abstract", "realistic photo", "text heavy design",
            
            # Mood/atmosphere  
            "dark vibe", "bright vibe", "happy vibe", "sad vibe", "angry vibe",
            
            # Specific elements
            "musical instruments", "album name visible"]

    for attempt in range(retry_count):
        try:
            # Load model and processor
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

            # Move model to GPU if available
            model = model.to(device)

            # Download and process the image
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Prepare text inputs
            text_inputs = processor(
                text=categories,
                images=None,
                return_tensors="pt",
                padding=True
            )

            # Move inputs to GPU if available
            text_inputs = {key: val.to(device) for key, val in text_inputs.items()}

            # Prepare image inputs
            image_inputs = processor(
                text=None,
                images=image,
                return_tensors="pt",
                padding=True
            )

            # Move image inputs to GPU if available
            image_inputs = {key: val.to(device) for key, val in image_inputs.items()}

            # Get model outputs
            with torch.no_grad():
                outputs = model(**{
                    "input_ids": text_inputs["input_ids"],
                    "attention_mask": text_inputs["attention_mask"],
                    "pixel_values": image_inputs["pixel_values"]
                })

            # Calculate similarity scores
            logits_per_image = outputs.logits_per_image
            probs = torch.nn.functional.softmax(logits_per_image, dim=1)[0].cpu().tolist()

            # Clean up to free memory
            del model, processor, text_inputs, image_inputs, outputs, logits_per_image
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            # Create results dictionary with all categories and scores
            results = [{"category": category, "score": float(score)} for category, score in zip(categories, probs)]

            # Sort by score in descending order
            results.sort(key=lambda x: x["score"], reverse=True)

            # Group results by type
            grouped_results = {
                "all_scores": results,
                "top_matches": results[:10],  # Top 10 overall matches
                
                # Group by category types
                "people": [r for r in results if any(term in r["category"].lower() for term in 
                          ["person", "people", "band", "group"])],
                
                "scene": [r for r in results if any(term in r["category"].lower() for term in 
                         ["scene", "indoor", "outdoor", "studio", "nature", "city"])],
                
                "style": [r for r in results if any(term in r["category"].lower() for term in 
                         ["style", "design", "art", "graphic", "minimalist", "abstract", "photo"])],
                
                "mood": [r for r in results if any(term in r["category"].lower() for term in 
                        ["mood", "vibe", "feeling", "lighting", "contrast"])]}

            return {"success": True, "results": grouped_results}

        except Exception as e:
            if attempt < retry_count - 1:
                print(f"Error analyzing image, retrying ({attempt+1}/{retry_count}): {str(e)}")
                time.sleep(1)  # Wait a bit before retrying
                continue
            else:
                return {"success": False, "error": str(e)}

def get_file_path(filename):
    """Helper function to get file path in the output directory"""
    return os.path.join(OUTPUT_DIR, filename)

def process_batch(df_batch, batch_num, output_file, custom_categories=None, normalize=True, resume_from=0, save_every=25):
    """
    Process a batch of album covers and save the results
    With resume capability for local execution
    """
    print(f"\n====== Processing Batch {batch_num} ======")
    print(f"Processing {len(df_batch)} albums in this batch")
    
    # Get the filename for batch output
    batch_output_file = get_file_path(f"{output_file}_batch{batch_num}.csv")
    temp_file = get_file_path(f"{output_file}_batch{batch_num}_temp.csv")
    
    # Check if the batch is already completed
    if os.path.exists(batch_output_file):
        print(f"Batch {batch_num} already completed! File exists: {batch_output_file}")
        try:
            return pd.read_csv(batch_output_file)
        except:
            print("Could not read existing file. Will reprocess batch.")
    
    # Check for temp file to resume
    all_features = []
    if resume_from > 0 and os.path.exists(temp_file):
        try:
            existing_features = pd.read_csv(temp_file)
            all_features = existing_features.to_dict('records')
            print(f"Loaded {len(all_features)} previously processed albums from {temp_file}")
        except Exception as e:
            print(f"Could not load existing results: {e}")
            resume_from = 0
    
    if resume_from > 0:
        print(f"Resuming from album {resume_from}")

    # Start timing
    start_time = time.time()

    # Process each album cover in this batch, starting from resume point
    for i, (idx, row) in enumerate(tqdm(list(df_batch.iterrows())[resume_from:], 
                                       total=len(df_batch)-resume_from, 
                                       desc=f"Batch {batch_num}")):
        album_id = row.get('album_id', idx)
        album_name = row.get('album_name', f"Album #{album_id}")
        image_url = row.get('image_url')

        if not image_url or pd.isna(image_url):
            print(f"Skipping album {album_id}: No image URL")
            continue

        # Analyze the album cover
        analysis = analyze_album_cover_with_clip(image_url, custom_categories, device=device)

        if not analysis.get("success", False):
            print(f"Error analyzing {album_name}: {analysis.get('error', 'Unknown error')}")
            continue

        # Create a base feature dictionary with album ID
        features = {
            "album_id": album_id,
            "album_name": album_name
        }

        # Add genre if available in the original dataset
        if 'genre' in row:
            features['genre'] = row['genre']

        # Add all category scores as features
        for category_info in analysis['results']['all_scores']:
            # Clean the category name to use as a column name
            category_name = category_info['category'].replace(' ', '_').replace('-', '_').lower()
            category_name = ''.join(c if c.isalnum() or c == '_' else '' for c in category_name)
            features[f'clip_{category_name}'] = category_info['score']

        all_features.append(features)

        # Save intermediate results
        if (i + 1) % save_every == 0:
            albums_processed = resume_from + i + 1
            print(f"Batch {batch_num}: Processed {albums_processed}/{len(df_batch)} albums...")
            temp_df = pd.DataFrame(all_features)
            temp_df.to_csv(temp_file, index=False)
            print(f"Saved checkpoint to: {temp_file}")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Create DataFrame from all feature dictionaries
    features_df = pd.DataFrame(all_features)

    # Calculate processing time
    elapsed_time = time.time() - start_time
    albums_per_second = (len(df_batch) - resume_from) / elapsed_time if elapsed_time > 0 else 0

    # Check if we have features
    if len(features_df) == 0:
        print(f"Batch {batch_num}: No features were extracted!")
        return features_df

    # Normalize features if requested
    if normalize:
        print(f"Batch {batch_num}: Normalizing features...")
        # Identify columns to normalize (all clip_* columns)
        feature_cols = [col for col in features_df.columns if col.startswith('clip_')]

        if feature_cols:
            # Create a scaler
            scaler = MinMaxScaler()

            # Fit and transform
            features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])

    # Save to CSV with batch number in filename
    features_df.to_csv(batch_output_file, index=False)
    
    # Remove temporary file if final file is saved successfully
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print(f"Removed temporary file {temp_file}")
        except:
            pass
    
    # Generate statistics
    num_albums = len(features_df)
    num_features = len([col for col in features_df.columns if col.startswith('clip_')])
    
    print(f"Batch {batch_num} completed successfully:")
    print(f"- {num_albums} albums processed")
    print(f"- {num_features} CLIP features extracted per album")
    print(f"- Processing time: {elapsed_time:.2f} seconds ({albums_per_second:.2f} albums/second)")
    print(f"- Saved to {batch_output_file}")
    
    return features_df

def create_clip_features_dataset_in_batches(csv_file, output_file="album_clip_features", custom_categories=None, 
                                         sample_size=None, normalize=True, num_batches=2, 
                                         specific_batch=None, resume_from=0, save_every=25):
    """
    Creates a dataset with CLIP features for all categories for each album cover
    Processing is done in separate batches for better handling in local environment
    """
    # Load the dataframe
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} albums from {csv_file}")
    
    # Take a sample if specified
    if sample_size and sample_size < len(df):
        df = df.head(sample_size)
        print(f"Using sample of {sample_size} albums")
    
    # Calculate total number of albums and batch size
    total_albums = len(df)
    batch_size = math.ceil(total_albums / num_batches)
    print(f"Splitting {total_albums} albums into {num_batches} batches of approximately {batch_size} albums each")
    
    # List to store batch dataframes
    batch_dfs = []
    
    # Determine which batches to process
    batches_to_process = [specific_batch] if specific_batch else range(1, num_batches + 1)
    
    # Process each batch
    for batch_num in batches_to_process:
        if batch_num < 1 or batch_num > num_batches:
            print(f"Invalid batch number: {batch_num}. Must be between 1 and {num_batches}")
            continue
            
        # Calculate start and end indices for this batch
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(batch_num * batch_size, total_albums)
        
        # Get the slice of dataframe for this batch
        df_batch = df.iloc[start_idx:end_idx].copy()
        df_batch = df_batch.reset_index(drop=True)  # Reset index for clarity
        
        # Process this batch
        batch_df = process_batch(
            df_batch, 
            batch_num=batch_num, 
            output_file=output_file,
            custom_categories=custom_categories,
            normalize=normalize,
            resume_from=resume_from,
            save_every=save_every
        )
        
        # Add to list of batch dataframes
        batch_dfs.append(batch_df)
    
    # Only combine batches if we processed all of them
    if specific_batch is None and len(batch_dfs) == num_batches:
        print("\nAll batches completed! Creating combined file...")
        combined_df = pd.concat(batch_dfs, ignore_index=True)
        combined_file = get_file_path(f"{output_file}_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"Combined file saved to {combined_file}")
    
    return batch_dfs

def combine_batch_files(output_file="album_clip_features", num_batches=2):
    """
    Combines previously processed batch files into one combined file
    """
    all_dfs = []
    
    for batch_num in range(1, num_batches + 1):
        batch_file = get_file_path(f"{output_file}_batch{batch_num}.csv")
        if os.path.exists(batch_file):
            try:
                df = pd.read_csv(batch_file)
                print(f"Loaded batch {batch_num}: {len(df)} albums")
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading batch {batch_num}: {e}")
    
    if not all_dfs:
        print("No batch files found to combine!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined file
    combined_file = get_file_path(f"{output_file}_combined.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined {len(all_dfs)} batches into {combined_file}: {len(combined_df)} total albums")
    
    return combined_df

# Main execution function with easy-to-configure options
def main(
    csv_file='albums_genres_cleaned.csv',  # Path to CSV file with album data
    output_file="album_clip_features",     # Base name for output files
    sample_size=None,                      # Number of albums to process (None for all)
    num_batches=2,                         # Number of batches to split into
    specific_batch=None,                   # Only process this batch (1 or 2), None for all
    normalize=True,                        # Normalize features (recommended)
    save_every=25                          # Save intermediate results every N albums
):
    """Main function - configure parameters here and run this file"""
    
    # Check if any batches are already completed
    batch1_file = get_file_path(f"{output_file}_batch1.csv")
    batch2_file = get_file_path(f"{output_file}_batch2.csv")
    
    # Check if we need to combine batches
    if os.path.exists(batch1_file) and os.path.exists(batch2_file):
        print("Both batches are complete! Combining them...")
        combine_batch_files(output_file, num_batches)
        return
    
    # If batch 1 is already complete but batch 2 isn't, process only batch 2
    if os.path.exists(batch1_file) and not os.path.exists(batch2_file) and specific_batch is None:
        print(f"Batch 1 already completed! File exists: {batch1_file}")
        print("Processing only batch 2...")
        
        # Find where to resume batch 2
        temp2_file = get_file_path(f"{output_file}_batch2_temp.csv")
        resume_from = 0
        
        if os.path.exists(temp2_file):
            try:
                temp_df = pd.read_csv(temp2_file)
                resume_from = len(temp_df)
                print(f"Will resume batch 2 from album {resume_from}")
            except:
                print("Could not determine resume point. Starting batch 2 from beginning.")
        
        # Process only batch 2
        create_clip_features_dataset_in_batches(
            csv_file,
            output_file=output_file,
            normalize=normalize,
            num_batches=num_batches,
            specific_batch=2,
            resume_from=resume_from,
            save_every=save_every
        )
    else:
        # Process requested batches
        if specific_batch:
            # Find where to resume the specific batch
            temp_file = get_file_path(f"{output_file}_batch{specific_batch}_temp.csv")
            resume_from = 0
            
            if os.path.exists(temp_file):
                try:
                    temp_df = pd.read_csv(temp_file)
                    resume_from = len(temp_df)
                    print(f"Will resume batch {specific_batch} from album {resume_from}")
                except:
                    print(f"Could not determine resume point. Starting batch {specific_batch} from beginning.")
            
            batch_message = f"batch {specific_batch}"
        else:
            resume_from = 0
            batch_message = "all batches"
            
        print(f"Processing {batch_message}...")
        create_clip_features_dataset_in_batches(
            csv_file,
            output_file=output_file,
            sample_size=sample_size,
            normalize=normalize,
            num_batches=num_batches,
            specific_batch=specific_batch,
            resume_from=resume_from,
            save_every=save_every
        )

# This will run when you execute the script
if __name__ == "__main__":
    # CONFIGURE YOUR SETTINGS HERE
    main(
        csv_file='../data/new/albums_final.csv',  # Path to your CSV file
        output_file="../album_clip_features",     # Base output filename
        sample_size=None,                      # Set to None to process all albums
        num_batches=2,                         # Number of batches
        specific_batch=None,                   # Process specific batch (1 or 2), None for all
        normalize=False,                        # Normalize features
        save_every=100                          # Save checkpoint frequency
    )