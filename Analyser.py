import os
import cv2
import numpy as np
import plotly.express as px
from collections import Counter, defaultdict
from tqdm import tqdm

class HeroLeaderboardAnalyzer:
    def __init__(self, heroes_folder, leaderboard_folder):
        self.heroes = self._load_hero_images(heroes_folder)
        self.leaderboard_images = self._load_leaderboard_images(leaderboard_folder)
    
    def _load_hero_images(self, heroes_folder):
        hero_images = {}
        for filename in os.listdir(heroes_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                hero_name = os.path.splitext(filename)[0]
                img_path = os.path.join(heroes_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                hero_images[hero_name] = img
        return hero_images
    
    def _load_leaderboard_images(self, leaderboard_folder):
        return [
            cv2.imread(os.path.join(leaderboard_folder, filename), cv2.IMREAD_GRAYSCALE)
            for filename in os.listdir(leaderboard_folder)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    
    def detect_heroes_in_leaderboard(self, confidence_threshold=0.75, proximity_threshold=10):
        all_detected_heroes = []
        
        for img_index, leaderboard_img in enumerate(tqdm(self.leaderboard_images, desc="Processing Leaderboard Images")):
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
                        detected_heroes_in_image.append((hero_name, x, y, img_index))
                        occupied_locations.append((hero_name, x, y))
            
            all_detected_heroes.extend(detected_heroes_in_image)
        
        return all_detected_heroes
    
    def analyze_and_visualize(self):
        
        # Detect heroes
        detected_heroes = self.detect_heroes_in_leaderboard()
        
        # Update hero counts with manual adjustments
        hero_counts = Counter()
        #for hero, count in manual_adjustments.items():
        #    hero_counts[hero] = count
        
        # Group heroes by image and sort by y-coordinate
        image_heroes = {}
        for hero_name, x, y, img_index in detected_heroes:
            if img_index not in image_heroes:
                image_heroes[img_index] = []
            image_heroes[img_index].append((hero_name, x, y))
        
        # Track hero pairings
        hero_pairings = defaultdict(Counter)
        
        for img_index, heroes in image_heroes.items():
            # Sort heroes by y-coordinate
            heroes.sort(key=lambda x: x[1])
            
            # Update hero counts
            for hero_name, _, _ in heroes:
                hero_counts[hero_name] += 1
            
            # Find hero pairings
            for i in range(len(heroes)):
                for j in range(i+1, len(heroes)):
                    hero1, hero2 = heroes[i][0], heroes[j][0]
                    if hero1 != hero2:
                        hero_pairings[hero1][hero2] += 1
                        hero_pairings[hero2][hero1] += 1
        
        # Prepare data for Plotly bar chart
        hero_data = [
            {'Hero': hero, 'Frequency': count} 
            for hero, count in sorted(hero_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Create interactive Plotly bar chart
        fig = px.bar(
            hero_data, 
            x='Hero', 
            y='Frequency', 
            title='Hero Frequencies',
            labels={'Hero': 'Hero', 'Frequency': 'Frequency'},
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title='Hero',
            yaxis_title='Frequency',
            xaxis_tickangle=-45,
            height=600,
            width=1200,
        )
        
        # Show the plot
        fig.show()
        
        # Print hero frequencies
        print("\nHero Frequencies:")
        for hero, count in hero_counts.most_common():
            print(f"{hero}: {count}")
        
        # Print top 5 hero pairings for each hero
        print("\nTop 5 Hero Pairings:")
        for hero in hero_pairings:
            top_pairs = hero_pairings[hero].most_common(5)
            print(f"\n{hero}:")
            for paired_hero, pair_count in top_pairs:
                print(f"  {paired_hero}: {pair_count}")

# Usage
analyzer = HeroLeaderboardAnalyzer('heroes', 'leaderboard')
analyzer.analyze_and_visualize()