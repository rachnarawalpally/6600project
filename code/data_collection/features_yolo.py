import pandas as pd
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
import easyocr
from collections import Counter
from tqdm import tqdm
from ultralytics import YOLO
import tempfile
import os

class AlbumCoverFeatureExtractor:
    def __init__(self):
        # Initialize YOLO model 
        self.model = YOLO('yolov8n.pt') 
        
        # Initialize OCR
        self.reader = easyocr.Reader(['en'])
        
        # Get COCO class names from YOLO
        self.coco_classes = list(self.model.names.values())
    
    def process_dataframe(self, df, album_id_col='album_id', url_column='image_url'):
        """Process all images in DataFrame and extract features"""
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing album covers"):
            try:
                features = self.extract_all_features(row[url_column])
                # Add original row data
                features['album_id'] = row[album_id_col]
                features['image_url'] = row[url_column]
                results.append(features)
            except Exception as e:
                print(f"Error processing {row[url_column]}: {str(e)}")
                # Add empty features in case of error
                empty_features = self._get_empty_features()
                empty_features['album_id'] = row[album_id_col]
                empty_features['image_url'] = row[url_column]
                results.append(empty_features)
        
        return pd.DataFrame(results)
    
    def extract_all_features(self, image_url):
        """Extract all features from a single image URL"""
        # Download and process image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_np = np.array(image)
        
        features = {}
        
        # Save temporarily for YOLO processing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Extract YOLO object detection features
            features.update(self._extract_yolo_features(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Extract color features
        features.update(self._extract_color_features(image_np))
        
        # Extract text features
        features.update(self._extract_text_features(image_np))
        
        # Extract composition features
        features.update(self._extract_composition_features(image_np))
        
        # Extract basic image properties
        features['width'] = image.width
        features['height'] = image.height
        features['aspect_ratio'] = image.width / image.height
        
        return features
    
    def _extract_yolo_features(self, image_path):
        """Extract features using YOLO object detection"""
        features = {}
        
        # Get YOLO predictions
        results = self.model(image_path, verbose=False) 
        
        # Initialize counts
        key_classes = ['person', 'car', 'bicycle', 'dog', 'cat', 'chair', 'book', 'bottle']
        for cls_name in key_classes:
            features[f'count_{cls_name}'] = 0
            features[f'max_confidence_{cls_name}'] = 0.0
        
        # Process detections
        all_confidences = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = self.model.names[class_id]
                
                if class_name in key_classes:
                    features[f'count_{class_name}'] += 1
                    features[f'max_confidence_{class_name}'] = max(
                        features[f'max_confidence_{class_name}'], confidence)
                
                all_confidences.append(confidence)
        
        # General object detection features
        features['total_object_count'] = len(all_confidences)
        features['avg_detection_confidence'] = np.mean(all_confidences) if all_confidences else 0.0
        features['has_people'] = features.get('count_person', 0) > 0
        
        return features
    
    def _extract_color_features(self, image_np):
        """Extract color-based features"""
        features = {}
        
        # Convert to different color spaces
        rgb_img = image_np
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Brightness and contrast
        features['avg_brightness'] = float(np.mean(gray))
        features['brightness_std'] = float(np.std(gray))
        features['is_dark'] = features['avg_brightness'] < 85
        features['is_bright'] = features['avg_brightness'] > 170
        
        # Color palette extraction
        pixels = rgb_img.reshape(-1, 3)
        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        color_counts = Counter(kmeans.labels_)
        
        # Store dominant colors and their percentages
        for i, color in enumerate(colors):
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
            percentage = float(color_counts[i] / len(pixels) * 100)
            features[f'color_{i+1}_hex'] = hex_color
            features[f'color_{i+1}_percentage'] = percentage
        
        # Saturation and intensity
        features['avg_saturation'] = float(np.mean(hsv_img[:,:,1]))
        features['is_vibrant'] = features['avg_saturation'] > 100
        features['is_muted'] = features['avg_saturation'] < 50
        
        # Color temperature
        avg_hue = np.mean(hsv_img[:,:,0])
        features['is_warm'] = (avg_hue < 30 or avg_hue > 150)
        features['is_cool'] = not features['is_warm']
        
        return features
    
    def _extract_text_features(self, image_np):
        """Extract text-based features"""
        features = {}
        
        # detect text
        text_results = self.reader.readtext(image_np)
        
        features['text_count'] = len(text_results)
        features['has_text'] = len(text_results) > 0
        
        # Extract text content
        text_content = [result[1] for result in text_results]
        features['text_content'] = ' | '.join(text_content)
        
        # Check for parental advisory
        parental_patterns = ['parental', 'advisory', 'explicit', 'content']
        features['has_parental_advisory'] = any(
            any(pattern.lower() in text.lower() for pattern in parental_patterns)
            for text in text_content)
        
        # Calculate text coverage
        if text_results:
            text_areas = []
            for result in text_results:
                bbox = result[0]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                text_areas.append(area)
            
            total_area = image_np.shape[0] * image_np.shape[1]
            features['text_coverage_ratio'] = float(sum(text_areas) / total_area)
        else:
            features['text_coverage_ratio'] = 0.0
        
        return features
    
    def _extract_composition_features(self, image_np):
        """Extract composition-based features"""
        features = {}
        
        # Edge detection for texture and complexity
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Edge density as complexity measure
        edge_density = np.sum(edges > 0) / (image_np.shape[0] * image_np.shape[1])
        features['edge_density'] = float(edge_density)
        features['is_complex'] = edge_density > 0.1
        features['is_minimal'] = edge_density < 0.05
        
        # Symmetry analysis
        height, width = image_np.shape[:2]
        left_half = image_np[:, :width//2]
        right_half = image_np[:, width//2:]
        right_half_flipped = np.flip(right_half, axis=1)
        
        if left_half.shape == right_half_flipped.shape:
            symmetry_score = np.mean(np.abs(left_half - right_half_flipped))
            features['horizontal_symmetry'] = float(1 - (symmetry_score / 255))
            features['is_symmetric'] = features['horizontal_symmetry'] > 0.8
        else:
            features['horizontal_symmetry'] = 0.0
            features['is_symmetric'] = False
        
        # Visual entropy (complexity measure)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.ravel() / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
        features['visual_entropy'] = float(entropy)
        
        return features
    
    def _get_empty_features(self):
        """Return empty feature dictionary for error cases"""
        features = {}
        
        # Basic properties
        for prop in ['width', 'height', 'aspect_ratio', 'avg_brightness', 'brightness_std',
                     'avg_saturation', 'text_count', 'edge_density', 'horizontal_symmetry',
                     'visual_entropy', 'total_object_count', 'avg_detection_confidence',
                     'text_coverage_ratio']:
            features[prop] = 0.0
        
        # Boolean properties
        for prop in ['is_dark', 'is_bright', 'is_vibrant', 'is_muted', 'is_warm', 
                     'is_cool', 'has_text', 'has_parental_advisory', 'is_complex',
                     'is_minimal', 'is_symmetric', 'has_people']:
            features[prop] = False
        
        # String properties
        features['text_content'] = ''
        
        # Color properties
        for i in range(1, 6):
            features[f'color_{i}_hex'] = '#000000'
            features[f'color_{i}_percentage'] = 0.0
        
        # Object detection counts
        key_classes = ['person', 'car', 'bicycle', 'dog', 'cat', 'chair', 'book', 'bottle']
        for cls_name in key_classes:
            features[f'count_{cls_name}'] = 0
            features[f'max_confidence_{cls_name}'] = 0.0
        
        return features

# Usage
if __name__ == "__main__":
    df = pd.read_csv('../data/new/albums_final.csv')
    
    # Initialize feature extractor
    extractor = AlbumCoverFeatureExtractor()
    
    # Process all images
    result_df = extractor.process_dataframe(df, album_id_col='album_id', url_column='image_url')
    
    # Save results
    result_df.to_csv('album_features_complete.csv', index=False)
    
    print(f"Processed {len(result_df)} albums")
    print(f"Extracted {len(result_df.columns)} features")
    print("\nSample features:")
    print(result_df.columns.tolist())
    print("\nFirst few rows:")
    print(result_df.head())
    
    # Check for errors
    error_count = result_df[result_df['width'] == 0].shape[0]
    if error_count > 0:
        print(f"\nWarning: {error_count} images failed to process properly")

