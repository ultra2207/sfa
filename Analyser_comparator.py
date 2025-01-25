import os
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

class HeroLeaderboardAnalyzer:
    def __init__(self, heroes_folder, leaderboard_folders):
        self.heroes = self._load_hero_images(heroes_folder)
        self.leaderboard_images = self._load_all_leaderboard_images(leaderboard_folders)
    
    def _load_hero_images(self, heroes_folder):
        hero_images = {}
        for filename in os.listdir(heroes_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                hero_name = os.path.splitext(filename)[0]
                img_path = os.path.join(heroes_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                hero_images[hero_name] = img
        return hero_images
    
    def _load_all_leaderboard_images(self, leaderboard_folders):
        all_leaderboard_images = []
        for folder in leaderboard_folders:
            folder_images = [
                (folder, cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE))
                for filename in os.listdir(folder)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            all_leaderboard_images.append(folder_images)
        return all_leaderboard_images
    
    def detect_heroes_in_leaderboards(self, confidence_threshold=0.75, proximity_threshold=10):
        all_folder_detections = []
        
        for folder_index, folder_images in enumerate(tqdm(self.leaderboard_images, desc="Processing Leaderboard Folders")):
            folder_detections = []
            
            for img_filename, leaderboard_img in tqdm(folder_images, desc=f"Processing Images in Folder {folder_index+1}"):
                detected_heroes_in_image = []
                occupied_locations = []
                
                for hero_name, hero_template in self.heroes.items():
                    # Skip if template is larger than leaderboard image
                    if hero_template.shape[0] > leaderboard_img.shape[0] or \
                       hero_template.shape[1] > leaderboard_img.shape[1]:
                        continue
                    
                    # Template matching
                    result = cv2.matchTemplate(leaderboard_img, hero_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= confidence_threshold)
                    
                    for y, x in zip(locations[0], locations[1]):
                        # Check if this location is too close to previously detected heroes
                        is_too_close = any(
                            abs(y - prev_y) <= proximity_threshold and 
                            abs(x - prev_x) <= proximity_threshold 
                            for prev_hero, prev_x, prev_y in occupied_locations
                        )
                        
                        if not is_too_close:
                            detected_heroes_in_image.append((hero_name, x, y, img_filename))
                            occupied_locations.append((hero_name, x, y))
                
                folder_detections.extend(detected_heroes_in_image)
            
            all_folder_detections.append(folder_detections)
        
        return all_folder_detections
    
    def analyze_and_visualize(self):
        # Uncomment for manual adjustments if needed
        # manual_adjustments = {
        #     'itu': 2,
        #     'kotl': 2,
        #     'widow': 1,
        #     'lord_gideon': 2,
        #     'xiang_tzu': 1
        # }
        
        # Detect heroes in all leaderboard folders
        all_detections = self.detect_heroes_in_leaderboards()
        
        # Process hero counts for each folder
        folder_hero_counts = []
        for folder_detections in all_detections:
            hero_counts = Counter()
            
            # Group heroes by image and sort by y-coordinate
            image_heroes = {}
            for hero_name, x, y, img_filename in folder_detections:
                if img_filename not in image_heroes:
                    image_heroes[img_filename] = []
                image_heroes[img_filename].append((hero_name, x, y))
            
            # Count heroes
            for heroes in image_heroes.values():
                for hero_name, _, _ in heroes:
                    hero_counts[hero_name] += 1
            
            folder_hero_counts.append(hero_counts)
        
        # Prepare visualizations
        self._visualize_individual_folders(folder_hero_counts)
        self._visualize_hero_changes(folder_hero_counts)
    
    def _visualize_individual_folders(self, folder_hero_counts):
        # Create individual folder visualizations
        for i, hero_counts in enumerate(folder_hero_counts, 1):
            # Convert to DataFrame
            hero_data = pd.DataFrame([
                {'Hero': hero, 'Frequency': count} 
                for hero, count in sorted(hero_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ])
            
            fig = px.bar(
                hero_data, 
                x='Hero', 
                y='Frequency', 
                title=f'Top 10 Heroes in Leaderboard {i}',
                labels={'Hero': 'Hero', 'Frequency': 'Frequency'},
            )
            
            fig.update_layout(
                xaxis_title='Hero',
                yaxis_title='Frequency',
                xaxis_tickangle=-45,
                height=600,
                width=1200,
            )
            
            fig.show()
            
            # Print hero frequencies for this folder
            print(f"\nHero Frequencies for Leaderboard {i}:")
            for hero, count in hero_counts.most_common(10):
                print(f"{hero}: {count}")
    
    def _visualize_hero_changes(self, folder_hero_counts):
        # Compare hero frequencies across folders
        if len(folder_hero_counts) < 2:
            print("Need at least two leaderboard folders to compare changes")
            return
        
        # Combine hero counts from all folders
        combined_hero_counts = {}
        for hero in set().union(*folder_hero_counts):
            hero_counts = [folder_counts.get(hero, 0) for folder_counts in folder_hero_counts]
            # Calculate the change from first to last folder
            change = hero_counts[-1] - hero_counts[0]
            combined_hero_counts[hero] = {
                'initial_count': hero_counts[0],
                'final_count': hero_counts[-1],
                'change': change
            }
        
        # Prepare data for visualization
        hero_changes = [
            {
                'Hero': hero, 
                'Initial Count': data['initial_count'],
                'Final Count': data['final_count'],
                'Change': data['change']
            } 
            for hero, data in combined_hero_counts.items()
        ]
        
        # Sort by change (descending)
        hero_changes_sorted = sorted(hero_changes, key=lambda x: x['Change'], reverse=True)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add bars for initial and final counts
        fig.add_trace(go.Bar(
            x=[h['Hero'] for h in hero_changes_sorted],
            y=[h['Initial Count'] for h in hero_changes_sorted],
            name='Initial Count',
            marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=[h['Hero'] for h in hero_changes_sorted],
            y=[h['Final Count'] for h in hero_changes_sorted],
            name='Final Count',
            marker_color='red'
        ))
        
        # Customize layout
        fig.update_layout(
            title='Hero Frequency Changes Across Leaderboards',
            xaxis_title='Hero',
            yaxis_title='Frequency',
            barmode='group',
            xaxis_tickangle=-45,
            height=600,
            width=1200,
        )
        
        # Show the plot
        fig.show()
        
        # Print detailed changes
        print("\nHero Frequency Changes:")
        for hero_change in hero_changes_sorted:
            print(f"{hero_change['Hero']}: {hero_change['Initial Count']} -> {hero_change['Final Count']} (Change: {hero_change['Change']})")

# Usage
analyzer = HeroLeaderboardAnalyzer(
    heroes_folder='heroes', 
    leaderboard_folders=['leaderboard1', 'leaderboard2']
)
analyzer.analyze_and_visualize()