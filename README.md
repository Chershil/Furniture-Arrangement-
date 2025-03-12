# Furniture Arrangement Optimizer

This project provides tools to optimize furniture placement in rooms using simulated annealing and other optimization algorithms. The application helps users design efficient room layouts based on furniture dimensions and room constraints.

## Features

- Arrange furniture items optimally within a room
- Visualize furniture layouts with color-coded categories
- Support for multiple optimization strategies (simulated annealing and random search)
- Extensive furniture catalog with common household items
- Multiple interfaces (Web UI with Streamlit, API with Flask, and command-line functionality)

## Setup and Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation Steps

1. Clone the repository or download the source code

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install numpy matplotlib pandas streamlit flask typing
   ```

## Running the Application

### Streamlit Web Interface

The Streamlit interface provides a user-friendly web application for furniture arrangement.

1. Make sure the furniture_optimizer.py module is accessible in your Python path
2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

### Flask API

The Flask API allows integration with other applications or services.

1. Start the Flask server:
   ```bash
   python flask_api.py
   ```
2. The API will be available at http://localhost:5000
3. Use the following endpoints:
   - `POST /api/optimize` - Optimize furniture arrangement
   - `POST /api/visualize` - Generate visualization for furniture arrangement
   - `GET /api/furniture/catalog` - Get furniture catalog
   - `GET /api/health` - Health check

### Command-line Demo

For a quick demonstration of the optimizer functionality:

```bash
python furniture_optimizer.py
```

This will generate and visualize a random room scenario.

## Using the Applications

### Streamlit Web Interface

1. Configure your room dimensions in the sidebar
2. Add furniture items from the catalog
3. Adjust optimization settings (iterations and strategy)
4. Click "Optimize Furniture Arrangement" to see results
5. Download your arrangement as JSON if desired

### Flask API Example

```python
import requests
import json
import matplotlib.pyplot as plt
import base64
import io

# Define your room and furniture
data = {
    "room": {
        "width": 12.0,
        "height": 10.0
    },
    "furniture": [
        {
            "name": "Bed (Full)",
            "width": 4.5,
            "height": 6.5,
            "min_wall_distance": 0,
            "rotatable": True,
            "category": "bed"
        },
        {
            "name": "Desk",
            "width": 4.0,
            "height": 2.0,
            "min_wall_distance": 0,
            "rotatable": True,
            "category": "table"
        }
    ],
    "iterations": 500,
    "strategy": "simulated_annealing"
}

# Optimize furniture arrangement
response = requests.post('http://localhost:5000/api/optimize', json=data)
result = response.json()

# Get visualization
vis_response = requests.post('http://localhost:5000/api/visualize', json=result)
vis_result = vis_response.json()

# Display image (if using in a script with matplotlib)
img_data = base64.b64decode(vis_result['image'])
img = plt.imread(io.BytesIO(img_data))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## Project Structure

- `furniture_optimizer.py` - Core optimization and visualization logic
- `streamlit_app.py` - Streamlit web interface
- `flask_api.py` - Flask API server

## How It Works

The optimizer uses simulated annealing (or random search) to iteratively improve furniture placement. It considers factors like:

- Space utilization
- Furniture spacing
- Wall clearance requirements
- Preferential placement for certain furniture types (e.g., seating near walls, tables in the center)

## Extending the Project

You can extend this project by:

1. Adding new furniture items to the catalog in the `generate_furniture_dataset()` function
2. Implementing additional optimization strategies in the `FurnitureOptimizer` class
3. Enhancing the evaluation function to consider more factors for better arrangement
4. Adding more room templates or pre-defined scenarios
