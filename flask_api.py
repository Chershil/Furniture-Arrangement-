from flask import Flask, request, jsonify, send_file
import json
import io
import base64
import matplotlib.pyplot as plt
from furniture_optimizer import (
    Room, FurnitureItem, FurnitureOptimizer, 
    arrange_furniture, generate_furniture_dataset
)

app = Flask(__name__)

@app.route('/api/optimize', methods=['POST'])
def optimize_furniture():
    """
    API endpoint to optimize furniture arrangement
    
    Expected JSON format:
    {
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
                "rotatable": true,
                "category": "bed"
            },
            ...
        ],
        "iterations": 500,
        "strategy": "simulated_annealing"
    }
    """
    try:
        # Parse the request data
        data = request.json
        
        if not data or 'room' not in data or 'furniture' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
            
        # Extract parameters
        room_width = data['room']['width']
        room_height = data['room']['height']
        furniture_list = data['furniture']
        iterations = data.get('iterations', 500)
        strategy = data.get('strategy', 'simulated_annealing')
        
        # Run the optimization
        room, score = arrange_furniture(
            room_width, 
            room_height, 
            furniture_list,
            iterations=iterations,
            strategy=strategy
        )
        
        # Prepare the response
        result = {
            'score': score,
            'room': {
                'width': room.width,
                'height': room.height
            },
            'furniture': []
        }
        
        # Add furniture details
        for item in room.furniture:
            width, height = item.get_dimensions()
            result['furniture'].append({
                'name': item.name,
                'original_width': item.width,
                'original_height': item.height,
                'width': width,
                'height': height,
                'x': item.x,
                'y': item.y,
                'rotated': item.rotated,
                'category': item.category
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize', methods=['POST'])
def visualize_furniture():
    """
    API endpoint to visualize furniture arrangement
    
    Uses the same JSON format as the optimize endpoint
    """
    try:
        # Parse the request data
        data = request.json
        
        if not data or 'room' not in data or 'furniture' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
            
        # Create room
        room = Room(data['room']['width'], data['room']['height'])
        
        # Add furniture with positions
        for furniture_data in data['furniture']:
            item = FurnitureItem(
                name=furniture_data['name'],
                width=furniture_data.get('original_width', furniture_data['width']),
                height=furniture_data.get('original_height', furniture_data['height']),
                min_wall_distance=furniture_data.get('min_wall_distance', 0),
                rotatable=furniture_data.get('rotatable', True),
                category=furniture_data.get('category', 'other')
            )
            item.x = furniture_data['x']
            item.y = furniture_data['y']
            item.rotated = furniture_data['rotated']
            room.add_furniture(item)
        
        # Generate visualization
        fig = room.visualize("Room Layout")
        
        # Convert plot to image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close(fig)
        
        # Convert to base64 for API response
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        
        return jsonify({'image': img_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/furniture/catalog', methods=['GET'])
def get_furniture_catalog():
    """
    API endpoint to get the furniture catalog
    """
    try:
        # Generate furniture catalog
        furniture_catalog = generate_furniture_dataset()
        
        # Prepare response
        result = []
        for item in furniture_catalog:
            result.append({
                'name': item[0],
                'width': item[1],
                'height': item[2],
                'min_wall_distance': item[3],
                'rotatable': item[4],
                'category': item[5]
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({'status': 'OK'})

@app.route('/')
def index():
    """
    Basic documentation page
    """
    return """
    <html>
        <head>
            <title>Furniture Arrangement API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
                pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>Furniture Arrangement API</h1>
            <p>Welcome to the Furniture Arrangement API. This API helps optimize furniture placement in rooms.</p>
            
            <h2>Endpoints:</h2>
            <ul>
                <li><code>POST /api/optimize</code> - Optimize furniture arrangement</li>
                <li><code>POST /api/visualize</code> - Generate visualization for furniture arrangement</li>
                <li><code>GET /api/furniture/catalog</code> - Get furniture catalog</li>
                <li><code>GET /api/health</code> - Health check</li>
            </ul>
            
            <h2>Example API Usage:</h2>
            <pre>
curl -X POST http://localhost:5000/api/optimize \\
  -H "Content-Type: application/json" \\
  -d '{
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
            "rotatable": true,
            "category": "bed"
        },
        {
            "name": "Desk",
            "width": 4.0,
            "height": 2.0,
            "min_wall_distance": 0,
            "rotatable": true,
            "category": "table"
        }
    ],
    "iterations": 500,
    "strategy": "simulated_annealing"
}'
            </pre>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)