### Face the Music: Can an Album’s Cover Art Predict its Popularity?

## Group Members
Member-1: Jessica Joy \
Member-2: Rachna Rawalpally \
Member-3: Samyu Vakkalanka \
Member-4: Marie Vaughan

## Purpose 
The purpose of this project was to develop a neural network to predict album popularity based on cover art features. Specifically, we examine elements such as imagery, text, composition, color, nudity, and visible skin. Our goal is to uncover patterns that can help emerging artists make more strategic design choices to enhance audience engagement and visibility in an increasingly competitive and saturated music landscape.

The primary data for this project comes from the Spotify API, with additional information gathered from Wikipedia and Last.fm APIs to extract album genre. We used various Python libraries and pretrained models to extract features from the album covers, such as nudity detection, imagery analysis, and composition.

## Setup Instructions 
This project was primarily created for graduate school purposes, and the code is mainly for demonstration. However, the code can be used and reproduced if necessary. Below are the setup steps to get the project up and running:

1. **Cloning this Repository to your local machine**
2. **Installing the Required packages**
3. **Obtaining the run data/new:**
   - `get_albums.ipynb` (need a Spotify API)
   - `get_albumsgenre.ipynb` (need Wikipedia and Last.fm API)
   - `save_images.ipynb` (convert URL images to JPG images)
4. **Collect more features from the album cover** (`code/data_collection`)
5. **To perform modeling**, use `code/models`:
   - a. Run CNN model - `untrained_cnn.ipynb`
   - b. Run ANN model - `ann_popularity_ensemble_final.ipynb`

## Project Structure 
- /data/: Contains files for obtaining the initial setup data from Spotify and other APIs, along with exploratory data analysis (EDA)
    - /old/: Contains the initial setup data for the past five years
    - /new/: Contains files for gathering more album data from Spotify  
- /code/: Includes all the code for extracting features and running models
    - /data_collection/: Code for obtaining features from album covers
    - /models/: Code for running models (CNN and ANN)
- /README: project overview

## Abstract 
Album covers serve as a critical visual interface between artists and listeners, shaping first impressions and setting the emotional tone for the music. In this study, we developed an artificial neural network to investigate whether artistic styles, colors, and thematic elements of album cover art can predict an album’s popularity. We examined questions such as whether female artists display more sexualized imagery compared to male artists, whether sexualized covers attract more listeners, and how composition correlates with album success. Our findings reveal distinct visual patterns associated with popularity across artist genders and genres. While stylistic differences emerged, such as pop albums favoring vibrant imagery and rap/hip-hop albums tending toward darker, more complex designs, the existing popularity of the artist proved to be a major factor in predicting album success.