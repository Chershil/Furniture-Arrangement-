import streamlit as st
import matplotlib.pyplot as plt
import json
import pandas as pd
from furniture_optimizer import (
    Room, FurnitureItem, FurnitureOptimizer, 
    arrange_furniture, generate_furniture_dataset
)

st.set_page_config(
    page_title="Furniture Arrangement Optimizer",
    page_icon="🪑",
    layout="wide"
)

# Title and description
st.title("🪑 Furniture Arrangement Optimizer")
st.markdown("""
This app helps you optimize furniture placement in small rooms. 
Simply enter your room dimensions and select furniture items to include.
""")

# Initialize session state for furniture list if not exists
if 'furniture_list' not in st.session_state:
    st.session_state.furniture_list = []

# Load furniture dataset
furniture_catalog = generate_furniture_dataset()
furniture_df = pd.DataFrame(furniture_catalog, 
                           columns=["Name", "Width", "Height", "Min Wall Distance", "Rotatable", "Category"])

# Sidebar for room configuration
st.sidebar.header("Room Configuration")
col1, col2 = st.sidebar.columns(2)
room_width = col1.number_input("Width (feet)", min_value=5.0, max_value=30.0, value=12.0, step=0.5)
room_height = col2.number_input("Height (feet)", min_value=5.0, max_value=30.0, value=10.0, step=0.5)

# Options for optimization
st.sidebar.header("Optimization Settings")
iterations = st.sidebar.slider("Iterations", min_value=100, max_value=2000, value=500, step=100)
strategy = st.sidebar.selectbox("Strategy", ["simulated_annealing", "random_search"], index=0)

# Furniture selection
st.sidebar.header("Add Furniture")
categories = ["All"] + sorted(list(set(furniture_df["Category"])))
selected_category = st.sidebar.selectbox("Filter by Category", categories)

# Filter furniture by category
if selected_category != "All":
    filtered_furniture = furniture_df[furniture_df["Category"] == selected_category]
else:
    filtered_furniture = furniture_df

# Display available furniture
selected_furniture = st.sidebar.selectbox("Select Furniture", filtered_furniture["Name"])
add_furniture = st.sidebar.button("Add to Room")

if add_furniture:
    # Find selected furniture in the catalog
    furniture_data = furniture_df[furniture_df["Name"] == selected_furniture].iloc[0]
    
    # Add to session state list
    st.session_state.furniture_list.append({
        "name": furniture_data["Name"],
        "width": furniture_data["Width"],
        "height": furniture_data["Height"],
        "min_wall_distance": furniture_data["Min Wall Distance"],
        "rotatable": furniture_data["Rotatable"],
        "category": furniture_data["Category"]
    })
    st.sidebar.success(f"Added {selected_furniture} to the room")

# Display current furniture list
st.subheader("Furniture Items")
if not st.session_state.furniture_list:
    st.info("No furniture added yet. Use the sidebar to add furniture items.")
else:
    furniture_table = []
    for i, item in enumerate(st.session_state.furniture_list):
        furniture_table.append([
            i+1, 
            item["name"], 
            f"{item['width']} × {item['height']}", 
            item["category"].title()
        ])
    
    df = pd.DataFrame(furniture_table, columns=["#", "Name", "Dimensions (W×H)", "Category"])
    st.table(df)
    
    if st.button("Clear All Furniture"):
        st.session_state.furniture_list = []
        st.experimental_rerun()

# Button to run optimization
col1, col2 = st.columns([3, 1])
optimize_button = col1.button("Optimize Furniture Arrangement", disabled=len(st.session_state.furniture_list) == 0)

# Button to use template room
template_button = col2.button("Use Template Room")

if template_button:
    # Reset furniture list and add template items
    st.session_state.furniture_list = [
        {"name": "Bed (Full)", "width": 4.5, "height": 6.5, "min_wall_distance": 0, "rotatable": True, "category": "bed"},
        {"name": "Nightstand", "width": 1.5, "height": 1.5, "min_wall_distance": 0, "rotatable": False, "category": "storage"},
        {"name": "Desk", "width": 4, "height": 2, "min_wall_distance": 0, "rotatable": True, "category": "table"},
        {"name": "Desk Chair", "width": 2, "height": 2, "min_wall_distance": 0, "rotatable": True, "category": "seating"},
        {"name": "Bookshelf (Tall)", "width": 3, "height": 1, "min_wall_distance": 0, "rotatable": False, "category": "storage"}
    ]
    st.experimental_rerun()

# Main functionality - optimize and display results
if optimize_button:
    st.header("Optimized Layout")
    
    with st.spinner("Optimizing furniture arrangement..."):
        # Run optimization
        room, score = arrange_furniture(
            room_width, 
            room_height, 
            st.session_state.furniture_list,
            iterations=iterations,
            strategy=strategy
        )
        
        # Display score
        st.subheader(f"Optimization Score: {score:.2f}")
        
        # Create visualization
        fig = room.visualize("Optimized Room Layout")
        st.pyplot(fig)
        
        # Display arrangement details
        st.subheader("Furniture Placement Details")
        
        furniture_details = []
        for item in room.furniture:
            width, height = item.get_dimensions()
            rotation = "Rotated" if item.rotated else "Normal"
            furniture_details.append([
                item.name,
                f"({item.x:.2f}, {item.y:.2f})",
                f"{width:.2f} × {height:.2f}",
                rotation
            ])
        
        df = pd.DataFrame(furniture_details, 
                         columns=["Name", "Position (x, y)", "Dimensions (W×H)", "Orientation"])
        st.table(df)
        
        # Add download button for the arrangement
        arrangement_data = {
            "room": {
                "width": room_width,
                "height": room_height
            },
            "furniture": []
        }
        
        for item in room.furniture:
            arrangement_data["furniture"].append({
                "name": item.name,
                "width": item.width,
                "height": item.height,
                "x": item.x,
                "y": item.y,
                "rotated": item.rotated,
                "category": item.category
            })
        
        json_str = json.dumps(arrangement_data, indent=2)
        st.download_button(
            label="Download Arrangement as JSON",
            data=json_str,
            file_name="furniture_arrangement.json",
            mime="application/json"
        )

# Footer
st.markdown("""
---
This app was created with Streamlit using a simulated annealing algorithm to optimize furniture placement.
""")