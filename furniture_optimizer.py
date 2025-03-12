import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from typing import List, Dict, Tuple
import json

class FurnitureItem:
    def __init__(self, name: str, width: float, height: float, min_wall_distance: float = 0, 
                 rotatable: bool = True, category: str = "other"):
        self.name = name
        self.width = width
        self.height = height
        self.min_wall_distance = min_wall_distance
        self.rotatable = rotatable
        self.category = category
        self.x = 0
        self.y = 0
        self.rotated = False
        self.color = self._get_color_for_category()
    
    def _get_color_for_category(self):
        color_map = {
            "seating": "#8B4513",  # brown
            "table": "#D2B48C",    # tan
            "storage": "#2F4F4F",  # dark slate
            "bed": "#4682B4",      # steel blue
            "other": "#778899"     # light slate
        }
        return color_map.get(self.category, "#778899")
    
    def get_dimensions(self):
        if self.rotated:
            return self.height, self.width
        return self.width, self.height
    
    def __repr__(self):
        dims = self.get_dimensions()
        return f"{self.name} ({dims[0]}x{dims[1]}) at ({self.x},{self.y})"

class Room:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.furniture = []
        
    def add_furniture(self, furniture: FurnitureItem):
        self.furniture.append(furniture)
        
    def is_valid_placement(self, item: FurnitureItem, x: float, y: float, rotated: bool) -> bool:
        # Set temporary position and rotation
        orig_x, orig_y, orig_rot = item.x, item.y, item.rotated
        item.x, item.y, item.rotated = x, y, rotated
        
        # Get current dimensions based on rotation
        width, height = item.get_dimensions()
        
        # Check if the item is within room boundaries with wall distance constraint
        in_bounds = (x + width <= self.width and 
                    y + height <= self.height and 
                    x >= 0 and 
                    y >= 0)
        
        # Check collisions with other furniture
        no_collision = True
        if in_bounds:
            for other in self.furniture:
                if other == item:
                    continue
                    
                other_width, other_height = other.get_dimensions()
                
                # Simple collision detection using rectangles
                if (x < other.x + other_width and
                    x + width > other.x and
                    y < other.y + other_height and
                    y + height > other.y):
                    no_collision = False
                    break
        
        # Restore original position and rotation
        item.x, item.y, item.rotated = orig_x, orig_y, orig_rot
        
        return in_bounds and no_collision

    def visualize(self, title="Room Layout"):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw room boundary
        rect = patches.Rectangle((0, 0), self.width, self.height, linewidth=2, 
                                edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # Draw furniture
        for item in self.furniture:
            width, height = item.get_dimensions()
            rect = patches.Rectangle((item.x, item.y), width, height, 
                                    linewidth=1, edgecolor='black', 
                                    facecolor=item.color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add text label in the center of the furniture
            ax.text(item.x + width/2, item.y + height/2, item.name,
                    horizontalalignment='center', verticalalignment='center')
        
        # Set axis limits and labels
        ax.set_xlim(-1, self.width + 1)
        ax.set_ylim(-1, self.height + 1)
        ax.set_xlabel('Width (feet)')
        ax.set_ylabel('Height (feet)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a scale to show dimensions
        scale_length = min(self.width, self.height) / 5
        scale_x = self.width * 0.05
        scale_y = self.height * 0.05
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 'k-', linewidth=2)
        ax.text(scale_x + scale_length/2, scale_y - 0.3, f'{scale_length} ft', 
                horizontalalignment='center')
        
        plt.tight_layout()
        return fig

class FurnitureOptimizer:
    def __init__(self, room: Room, iterations: int = 1000):
        self.room = room
        self.iterations = iterations
        
    def optimize_layout(self, strategy="simulated_annealing"):
        if strategy == "random_search":
            return self._random_search()
        elif strategy == "simulated_annealing":
            return self._simulated_annealing()
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    def _random_search(self):
        best_score = -float('inf')
        best_layout = None
        
        for _ in range(self.iterations):
            # Reset furniture positions
            for item in self.room.furniture:
                self._randomize_furniture_position(item)
            
            # Evaluate layout
            score = self._evaluate_layout()
            
            if score > best_score:
                best_score = score
                best_layout = self._save_current_layout()
        
        # Restore best layout
        self._restore_layout(best_layout)
        return best_score
    
    def _simulated_annealing(self):
        # Initial temperature and cooling rate
        temp = 10.0
        cooling_rate = 0.95
        
        # Initialize furniture with random positions
        for item in self.room.furniture:
            self._randomize_furniture_position(item)
        
        current_score = self._evaluate_layout()
        best_score = current_score
        best_layout = self._save_current_layout()
        
        for i in range(self.iterations):
            # Pick a random piece of furniture
            item = random.choice(self.room.furniture)
            
            # Save current state
            old_x, old_y, old_rotated = item.x, item.y, item.rotated
            
            # Try a new random position
            self._randomize_furniture_position(item)
            
            # Evaluate new layout
            new_score = self._evaluate_layout()
            
            # Determine if we should accept the new layout
            if new_score > current_score:
                # Always accept better solutions
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_layout = self._save_current_layout()
            else:
                # Accept worse solutions with a probability that decreases with temperature
                delta = new_score - current_score
                acceptance_probability = np.exp(delta / temp)
                
                if random.random() > acceptance_probability:
                    # Reject the new position, revert to old position
                    item.x, item.y, item.rotated = old_x, old_y, old_rotated
                else:
                    current_score = new_score
            
            # Cool down the temperature
            temp *= cooling_rate
            
            # If temperature is very low, we can stop
            if temp < 0.01:
                break
        
        # Restore best layout found
        self._restore_layout(best_layout)
        return best_score
    
    def _randomize_furniture_position(self, item: FurnitureItem):
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            # Randomly decide if the item should be rotated
            rotated = random.choice([True, False]) if item.rotatable else False
            
            # Get dimensions based on rotation
            width, height = (item.height, item.width) if rotated else (item.width, item.height)
            
            # Generate random position within room bounds
            x = random.uniform(0, self.room.width - width)
            y = random.uniform(0, self.room.height - height)
            
            # Check if the placement is valid
            if self.room.is_valid_placement(item, x, y, rotated):
                item.x = x
                item.y = y
                item.rotated = rotated
                return True
            
            attempts += 1
        
        return False
    
    def _evaluate_layout(self):
        """
        Evaluate the current furniture layout based on multiple criteria.
        Returns a score where higher is better.
        """
        score = 0
        
        # 1. Use of space - reward layouts that use more of the available space
        used_area = sum(item.get_dimensions()[0] * item.get_dimensions()[1] for item in self.room.furniture)
        total_area = self.room.width * self.room.height
        space_utilization = used_area / total_area
        score += 5 * space_utilization  # Weight factor
        
        # 2. Furniture distance from walls - reward furniture that respects minimum wall distances
        for item in self.room.furniture:
            width, height = item.get_dimensions()
            
            # Distance to walls
            dist_to_left = item.x
            dist_to_right = self.room.width - (item.x + width)
            dist_to_bottom = item.y
            dist_to_top = self.room.height - (item.y + height)
            
            min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
            
            # If minimum distance is respected, add to score
            if min_dist >= item.min_wall_distance:
                score += 2
            else:
                score -= 5  # Penalty for not respecting minimum distance
        
        # 3. Furniture spacing - reward layouts where furniture isn't too crowded
        for i, item1 in enumerate(self.room.furniture):
            for j, item2 in enumerate(self.room.furniture):
                if i >= j:
                    continue
                
                # Calculate center points
                item1_width, item1_height = item1.get_dimensions()
                item2_width, item2_height = item2.get_dimensions()
                
                item1_center = (item1.x + item1_width/2, item1.y + item1_height/2)
                item2_center = (item2.x + item2_width/2, item2.y + item2_height/2)
                
                # Calculate distance between centers
                dist = np.sqrt((item1_center[0] - item2_center[0])**2 + 
                              (item1_center[1] - item2_center[1])**2)
                
                # Reward reasonable spacing
                if dist > 2:  # Minimum desired spacing
                    score += 1
        
        # 4. Preferential placement for certain furniture types
        for item in self.room.furniture:
            # Seating near walls is often preferred
            if item.category == "seating":
                width, height = item.get_dimensions()
                if (item.x < 1 or item.y < 1 or 
                    item.x + width > self.room.width - 1 or 
                    item.y + height > self.room.height - 1):
                    score += 2
            
            # Tables in the center are often preferred
            if item.category == "table":
                width, height = item.get_dimensions()
                center_x = item.x + width/2
                center_y = item.y + height/2
                room_center_x = self.room.width/2
                room_center_y = self.room.height/2
                
                # Distance to center
                dist_to_center = np.sqrt((center_x - room_center_x)**2 + 
                                        (center_y - room_center_y)**2)
                
                # Reward if closer to center
                if dist_to_center < self.room.width/4:
                    score += 3
        
        return score
    
    def _save_current_layout(self):
        """
        Save the current furniture layout configuration.
        """
        layout = []
        for item in self.room.furniture:
            layout.append({
                "name": item.name,
                "x": item.x,
                "y": item.y,
                "rotated": item.rotated
            })
        return layout
    
    def _restore_layout(self, layout):
        """
        Restore a saved furniture layout configuration.
        """
        for i, config in enumerate(layout):
            item = self.room.furniture[i]
            item.x = config["x"]
            item.y = config["y"]
            item.rotated = config["rotated"]

# Sample furniture data generator
def generate_furniture_dataset():
    """
    Generate a sample dataset of common furniture items with dimensions.
    Dimensions are in feet.
    """
    furniture_types = [
        # name, width, height, min_wall_distance, rotatable, category
        ("Bed (Twin)", 3.25, 6.5, 0, True, "bed"),
        ("Bed (Full)", 4.5, 6.5, 0, True, "bed"),
        ("Bed (Queen)", 5, 6.67, 0, True, "bed"),
        ("Bed (King)", 6.33, 6.67, 0, True, "bed"),
        ("Nightstand", 1.5, 1.5, 0, False, "storage"),
        ("Dresser (Small)", 3, 1.67, 0, True, "storage"),
        ("Dresser (Large)", 5, 1.67, 0, True, "storage"),
        ("Desk", 4, 2, 0, True, "table"),
        ("Desk Chair", 2, 2, 0, True, "seating"),
        ("Dining Table (Small)", 3, 3, 0, False, "table"),
        ("Dining Table (Medium)", 4, 4, 0, False, "table"),
        ("Dining Chair", 1.5, 1.5, 0, True, "seating"),
        ("Sofa (Loveseat)", 5, 3, 0, True, "seating"),
        ("Sofa (3-Seater)", 7, 3, 0, True, "seating"),
        ("Coffee Table", 3.5, 1.75, 0, True, "table"),
        ("Side Table", 1.5, 1.5, 0, False, "table"),
        ("Bookshelf (Tall)", 3, 1, 0, False, "storage"),
        ("Bookshelf (Wide)", 4, 1, 0, True, "storage"),
        ("Armchair", 2.5, 2.5, 0, False, "seating"),
        ("TV Stand", 5, 1.5, 0, True, "storage"),
        ("Wardrobe", 4, 2, 0, False, "storage"),
        ("Floor Lamp", 1, 1, 0, False, "other"),
        ("Plant (Small)", 1, 1, 0, False, "other"),
        ("Plant (Large)", 2, 2, 0, False, "other")
    ]
    
    return furniture_types

# Generate sample room configurations with furniture
def generate_room_scenarios(num_scenarios=10):
    furniture_dataset = generate_furniture_dataset()
    scenarios = []
    
    # Common room sizes (width, height) in feet
    room_sizes = [
        (10, 10),  # Small square room
        (12, 10),  # Small rectangular room
        (14, 12),  # Medium room
        (16, 14),  # Larger room
        (20, 15)   # Very large room
    ]
    
    for i in range(num_scenarios):
        # Select a random room size
        width, height = random.choice(room_sizes)
        
        # Randomly adjust dimensions slightly
        width = width + random.uniform(-1, 1)
        height = height + random.uniform(-1, 1)
        
        # Create a room
        room = {
            "width": round(width, 2),
            "height": round(height, 2),
            "furniture": []
        }
        
        # Determine room type and appropriate furniture
        room_type = random.choice(["bedroom", "living_room", "office", "studio"])
        
        if room_type == "bedroom":
            # Essential bedroom furniture
            bed_size = random.choice(["Twin", "Full", "Queen", "King"])
            furniture_names = [f"Bed ({bed_size})", "Nightstand"]
            
            # Optional additional furniture
            optional = ["Dresser (Small)", "Dresser (Large)", "Desk", "Desk Chair", 
                      "Bookshelf (Tall)", "Floor Lamp", "Plant (Small)"]
            num_optional = random.randint(1, min(3, len(optional)))
            furniture_names.extend(random.sample(optional, num_optional))
            
        elif room_type == "living_room":
            # Essential living room furniture
            sofa_type = random.choice(["Loveseat", "3-Seater"])
            furniture_names = [f"Sofa ({sofa_type})", "Coffee Table"]
            
            # Optional additional furniture
            optional = ["Side Table", "TV Stand", "Bookshelf (Wide)", 
                      "Armchair", "Floor Lamp", "Plant (Large)", "Plant (Small)"]
            num_optional = random.randint(2, min(4, len(optional)))
            furniture_names.extend(random.sample(optional, num_optional))
            
        elif room_type == "office":
            # Essential office furniture
            furniture_names = ["Desk", "Desk Chair"]
            
            # Optional additional furniture
            optional = ["Bookshelf (Tall)", "Bookshelf (Wide)", "Side Table", 
                      "Floor Lamp", "Plant (Small)", "Armchair"]
            num_optional = random.randint(1, min(3, len(optional)))
            furniture_names.extend(random.sample(optional, num_optional))
            
        else:  # studio
            # Mix of essential furniture for a studio apartment
            bed_size = random.choice(["Twin", "Full"])
            furniture_names = [f"Bed ({bed_size})", "Desk", "Desk Chair", 
                              "Dining Table (Small)", "Dining Chair"]
            
            # Optional additional furniture
            optional = ["Bookshelf (Tall)", "TV Stand", "Floor Lamp", 
                      "Plant (Small)", "Dresser (Small)"]
            num_optional = random.randint(1, min(3, len(optional)))
            furniture_names.extend(random.sample(optional, num_optional))
        
        # Add selected furniture to the room
        for name in furniture_names:
            # Find furniture data
            furniture_data = next((f for f in furniture_dataset if f[0] == name), None)
            if furniture_data:
                # Add furniture item to room
                furniture_item = {
                    "name": furniture_data[0],
                    "width": furniture_data[1],
                    "height": furniture_data[2],
                    "min_wall_distance": furniture_data[3],
                    "rotatable": furniture_data[4],
                    "category": furniture_data[5]
                }
                room["furniture"].append(furniture_item)
        
        # Add room to scenarios
        scenarios.append({
            "room_type": room_type,
            "room": room
        })
    
    return scenarios

# Function to create and optimize a room layout
def arrange_furniture(room_width, room_height, furniture_list, iterations=1000, strategy="simulated_annealing"):
    """
    Arrange furniture in a room using an optimization algorithm.
    
    Args:
        room_width (float): Width of the room in feet
        room_height (float): Height of the room in feet
        furniture_list (list): List of furniture items with properties
        iterations (int): Number of optimization iterations
        strategy (str): Optimization strategy ("random_search" or "simulated_annealing")
        
    Returns:
        tuple: (Room object, score)
    """
    # Create room
    room = Room(room_width, room_height)
    
    # Add furniture to room
    for furniture_data in furniture_list:
        item = FurnitureItem(
            name=furniture_data["name"],
            width=furniture_data["width"],
            height=furniture_data["height"],
            min_wall_distance=furniture_data.get("min_wall_distance", 0),
            rotatable=furniture_data.get("rotatable", True),
            category=furniture_data.get("category", "other")
        )
        room.add_furniture(item)
    
    # Optimize furniture arrangement
    optimizer = FurnitureOptimizer(room, iterations=iterations)
    score = optimizer.optimize_layout(strategy=strategy)
    
    return room, score

# Function to save a dataset for future use
def save_dataset(scenarios, filename="furniture_scenarios.json"):
    with open(filename, "w") as f:
        json.dump(scenarios, f, indent=2)

# Function to load a dataset
def load_dataset(filename="furniture_scenarios.json"):
    with open(filename, "r") as f:
        return json.load(f)

# Demo function
def demo():
    # Generate scenarios
    scenarios = generate_room_scenarios(5)
    
    # Save scenarios to file
    save_dataset(scenarios)
    
    # Choose a random scenario to demonstrate
    scenario = random.choice(scenarios)
    room_data = scenario["room"]
    
    print(f"Room type: {scenario['room_type']}")
    print(f"Room dimensions: {room_data['width']} x {room_data['height']} feet")
    print(f"Furniture items: {len(room_data['furniture'])}")
    
    # Arrange furniture
    room, score = arrange_furniture(
        room_data["width"], 
        room_data["height"], 
        room_data["furniture"],
        iterations=500
    )
    
    # Display results
    print(f"Optimization score: {score:.2f}")
    print("\nFurniture arrangement:")
    for item in room.furniture:
        width, height = item.get_dimensions()
        rotation = "rotated" if item.rotated else "normal"
        print(f"- {item.name}: Position ({item.x:.2f}, {item.y:.2f}), Size: {width:.2f}x{height:.2f}, Orientation: {rotation}")
    
    # Visualize
    fig = room.visualize(f"{scenario['room_type'].title()} Layout")
    plt.show()
    
    return room, fig

if __name__ == "__main__":
    demo()