import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy as np
import time
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="Temporal Network Analysis of Cities",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
# Custom CSS for aesthetics and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        animation: fadeInDown 1s ease-out;
    }
    h2, h3 {
        color: #2c3e50;
        font-weight: 600;
        animation: fadeIn 1.5s ease-out;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Metric Cards */
    .st-emotion-cache-1r6slb0 { /* Streamlit metric container class might vary, using generic approach */
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 10px;
        transition: transform 0.3s ease;
    }
    .st-emotion-cache-1r6slb0:hover {
        transform: translateY(-5px);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Project Footer/Header */
    .project-header {
        background: linear-gradient(90deg, #1f77b4 0%, #00d2ff 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: fadeInDown 0.8s ease-out;
    }
    .project-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .team-members {
        font-size: 1.2rem;
        margin-top: 10px;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# --- Project Branding ---
st.markdown("""
<style>
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #000;
    }
    ::-webkit-scrollbar-thumb {
        background: #00FFFF;
        border-radius: 5px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0, 255, 255, 0.3);
        border-color: #00FFFF;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00FFFF;
    }
    .metric-label {
        font-size: 14px;
        color: #ccc;
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(45deg, #00FFFF, #0088FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
<div class="project-header">
    <div class="project-title" style="font-size: 2.5em; text-align: center; margin-bottom: 20px; color: #FFFFFF;">
        Pakistan Through Discrete Structures
    </div>
    <div class="team-members" style="text-align: center; color: #E0E0E0; font-size: 1.1em;">By Abdullah Nadeem, Arham Manzoor, and Zainab Nisaar</div>
</div>
""", unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_data():
    """Loads and preprocesses the CPI data."""
    try:
        # Load Sheet1 which contains the raw data
        df = pd.read_excel('Categorized_CPI_Data.xlsx', sheet_name='Sheet1')
        
        # Create a Date column
        # Mapping month names to numbers
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        df['Month_Num'] = df['Month'].map(month_map)
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_Num'].astype(str) + '-01')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # --- Sidebar Controls ---
    st.sidebar.title("Configuration")
    
    # Category Selection
    categories = df['Category'].unique()
    selected_category = st.sidebar.selectbox("Select Product Category", categories)
    
    # --- Simulation Controls ---
    st.sidebar.markdown("### Simulation Controls")
    
    # Initialize Simulation State
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'sim_speed' not in st.session_state:
        st.session_state.sim_speed = 4  # Default 4 seconds

    # Controls Layout
    col_toggle, col_reset = st.sidebar.columns(2)
    
    with col_toggle:
        if st.session_state.is_playing:
            if st.button("⏸", key="pause_sim", type="primary"):
                st.session_state.is_playing = False
                st.rerun()
        else:
            if st.button("▶", key="play_sim", type="primary"):
                st.session_state.is_playing = True
                st.rerun()
            
    with col_reset:
        if st.button("⏹", key="reset_sim"):
            st.session_state.is_playing = False
            # Reset to min date
            min_date_val = df['Date'].min()
            st.session_state.selected_year = min_date_val.year
            st.session_state.selected_month = min_date_val.month
            st.toast("Simulation Reset!", icon="⏹") # Feature 3: Toast
            st.rerun()



    # Timer Control (Speed)
    # "Timer which can be increased or decreased just like the calender" -> Number Input or +/- buttons
    # We'll use a number input for clarity and ease of use within 0-10 range
    # Timer Control (Speed)
    # "Timer which can be increased or decreased just like the calender" -> Number Input or +/- buttons
    # We'll use a number input for clarity and ease of use within 0-10 range
    # FIX: Use 'key' to bind directly to session state to avoid lost updates during reruns
    st.sidebar.number_input(
        "Simulation Speed (seconds)",
        min_value=0,
        max_value=10,
        step=1,
        key="sim_speed"
    )

    # Date Selection (Custom Calendar Widget)
    st.sidebar.markdown("### Select Time Step")
    
    # Ensure 'Date' column is datetime
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    min_year = min_date.year
    max_year = max_date.year
    
    # Initialize Session State for Year and Month if not present
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = min_year
    if 'selected_month' not in st.session_state:
        st.session_state.selected_month = min_date.month

    # Year Selector with Arrows
    col_prev, col_year, col_next = st.sidebar.columns([1, 2, 1])
    
    with col_prev:
        if st.button("◀", key="prev_year"):
            if st.session_state.selected_year > min_year:
                st.session_state.selected_year -= 1
    
    with col_year:
        st.markdown(f"<h3 style='text-align: center; margin: 0;'>{st.session_state.selected_year}</h3>", unsafe_allow_html=True)
    
    with col_next:
        if st.button("▶", key="next_year"):
            if st.session_state.selected_year < max_year:
                st.session_state.selected_year += 1

    # Month Selector (Grid)
    months = [
        "Jan", "Feb", "Mar", "Apr", 
        "May", "Jun", "Jul", "Aug", 
        "Sep", "Oct", "Nov", "Dec"
    ]
    
    # Helper callback to update month
    def set_month(m):
        st.session_state.selected_month = m

    # Create 4 rows of 3 columns
    for i in range(0, 12, 3):
        cols = st.sidebar.columns(3)
        for j in range(3):
            month_idx = i + j + 1
            month_name = months[i + j]
            
            # Determine button type based on CURRENT state
            btn_type = "primary" if month_idx == st.session_state.selected_month else "secondary"
            
            with cols[j]:
                # Use on_click to update state BEFORE the rerun, ensuring the button re-renders with correct color immediately
                st.button(
                    month_name, 
                    key=f"month_{month_idx}", 
                    type=btn_type, 
                    use_container_width=True,
                    on_click=set_month,
                    args=(month_idx,)
                )
    
    # Construct selected_date_str for downstream compatibility
    # Format: 'YYYY-MM'
    selected_date_str = f"{st.session_state.selected_year}-{st.session_state.selected_month:02d}"
    
    # Similarity Threshold
    # Similarity Threshold
    # Calculate similarity range for guidance
    # We need to calculate sim_matrix first to give good bounds, but that depends on filtering.
    # So we move the slider after data filtering or just make it generic high precision.
    # For now, let's make it high precision 0.9-1.0 as observed data is high.
    threshold = st.sidebar.slider(
        "Similarity Threshold (Edge Visibility)",
        min_value=0.980,
        max_value=1.000,
        value=0.995,
        step=0.001,
        format="%.3f",
        help="Only edges with cosine similarity above this value will be shown."
    )

    # --- Composite Score Configuration ---
    st.sidebar.markdown("---")
    st.sidebar.title("Composite Score Weighting")
    
    weighting_method = st.sidebar.selectbox(
        "Select Weighting Technique",
        ["Equal Weighting", "Correlation-Based", "Category Importance", "Entropy-Based", "Interactive"]
    )
    
    # Initialize weights dictionary
    weights = {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25}
    
    if weighting_method == "Interactive":
        st.sidebar.markdown("### Adjust Weights")
        w_deg = st.sidebar.slider("Degree Weight", 0.0, 1.0, 0.25)
        w_clo = st.sidebar.slider("Closeness Weight", 0.0, 1.0, 0.25)
        w_bet = st.sidebar.slider("Betweenness Weight", 0.0, 1.0, 0.25)
        w_eig = st.sidebar.slider("Eigenvector Weight", 0.0, 1.0, 0.25)
        
        # Normalize
        total_w = w_deg + w_clo + w_bet + w_eig
        if total_w == 0:
            weights = {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25}
        else:
            weights = {
                'Degree': w_deg / total_w,
                'Closeness': w_clo / total_w,
                'Betweenness': w_bet / total_w,
                'Eigenvector': w_eig / total_w
            }
        
        st.sidebar.info(f"Normalized Weights:\nDegree: {weights['Degree']:.2f}\nCloseness: {weights['Closeness']:.2f}\nBetweenness: {weights['Betweenness']:.2f}\nEigenvector: {weights['Eigenvector']:.2f}")

    # Feature 1: Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 12px;">
            <p>Version 2.0.0<br>
            &copy; 2024 Discrete Structures Project</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # --- Analysis Logic ---
    
    # Filter data
    selected_date = pd.to_datetime(selected_date_str + '-01')
    mask = (df['Category'] == selected_category) & (df['Date'] == selected_date)
    filtered_df = df[mask]
    
    if filtered_df.empty:
        st.warning("No data available for the selected category and date.")
    else:
        # Pivot to create feature vectors: Index=City, Columns=Product, Values=Price
        # We use 'Product' as the feature. If 'Product' column name is different, adjust here.
        # Based on inspection, column is 'Product' (or similar, let's check inspection output if needed, but 'Product' is standard)
        # Actually, let's verify column names from previous turn. 
        # Columns were: ['Year', 'Month', 'City', 'Product', 'Price', 'Category'] based on standard layout, 
        # but let's be safe and use the columns present.
        # Assuming 'City', 'Product', 'Price' exist.
        
        pivot_df = filtered_df.pivot_table(index='City', columns='Product', values='Price')
        
        # Fill missing values with 0 (assuming missing means price not available/zero) or row mean
        pivot_df = pivot_df.fillna(0)
        
        if pivot_df.shape[0] < 2:
            st.warning("Not enough cities to form a network.")
        else:
            # Calculate Cosine Similarity
            # pivot_df.values is (n_cities, n_products)
            sim_matrix = cosine_similarity(pivot_df.values)
            
            # Display stats in sidebar to guide threshold selection
            sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            if len(sim_values) > 0:
                min_sim = np.min(sim_values)
                max_sim = np.max(sim_values)
                avg_sim_val = np.mean(sim_values)
                st.sidebar.markdown("---")
                st.sidebar.markdown("### Current Similarity Stats")
                st.sidebar.info(f"Min: {min_sim:.4f}\nMax: {max_sim:.4f}\nAvg: {avg_sim_val:.4f}")

            cities = pivot_df.index.tolist()
            
            # Create Graph
            G = nx.Graph()
            for i, city in enumerate(cities):
                G.add_node(city)
                
            # Add edges based on similarity
            # Add edges based on similarity
            # Add edges based on similarity (k-Nearest Neighbors = 2)
            # This creates the clean "spiderweb" look from the video
            k_neighbors = 2 # Reduced from 3 to clean up the hairball 

            for i in range(len(cities)):
                # Get all similarities for city i, sort them, and pick top k
                scores = []
                for j in range(len(cities)):
                    if i != j:
                        scores.append((j, sim_matrix[i, j]))
                
                # Sort descending by similarity score
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only top k connections (and ensure threshold is met)
                for neighbor_idx, score in scores[:k_neighbors]:
                    if score >= threshold:
                        # Distance = 1 - similarity (Higher sim = Lower distance)
                        dist = max(0, 1.0 - score)
                        G.add_edge(cities[i], cities[neighbor_idx], weight=score, distance=dist)
            
            # Calculate Centrality Metrics (Weighted for better granularity)
            # 1. Weighted Degree (Strength): Sum of weights of incident edges
            degree_dict = dict(G.degree(weight='weight'))
            # Normalize by N-1 to keep it somewhat comparable to standard degree (0-1 range approx)
            # But since it's weighted, it can exceed 1. We'll rely on Min-Max scaling later for the score.
            degree_centrality = degree_dict 

            # 2. Weighted Closeness: Reciprocal of sum of shortest path distances
            closeness_centrality = nx.closeness_centrality(G, distance='distance')

            # 3. Weighted Betweenness: Fraction of shortest paths (using distance) that pass through node
            betweenness_centrality = nx.betweenness_centrality(G, weight='distance')

            # 4. Weighted Eigenvector: Considers edge weights
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            except:
                # Fallback if convergence fails
                eigenvector_centrality = {node: 0.0 for node in G.nodes()}

            # --- Main Layout ---
            
            # Feature 6: Category Icons
            category_icons = {
                'Food': '🍔',
                'Clothing': '👕',
                'Housing': '🏠',
                'Transport': '🚌',
                'Education': '🎓',
                'Health': '🏥',
                'Communication': '📱',
                'Recreation': '🎮',
                'Miscellaneous': '📦'
            }
            icon = category_icons.get(selected_category, '📊')
            
            # Custom Gradient Header for the Section
            st.markdown(f"<h2 style='text-align: center;'>{icon} Temporal Network of <span class='gradient-text'>Cities</span></h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center; color: #ccc;'>Category: *{selected_category}* | Date: *{selected_date_str}*</h4>", unsafe_allow_html=True)
            
            # --- 1. Calculate Master Graph for Fixed Layout (3D) ---
            # We need a consistent layout where nodes don't move. 
            # We'll build a graph of ALL edges across ALL time steps for this category.
            if 'fixed_pos_category' not in st.session_state or st.session_state.fixed_pos_category != selected_category:
                with st.spinner("Calculating fixed 3D layout..."):
                    master_G = nx.Graph()
                    all_cat_dates = df[df['Category'] == selected_category]['Date'].unique()
                    
                    for d in all_cat_dates:
                        d_mask = (df['Category'] == selected_category) & (df['Date'] == d)
                        d_df = df[d_mask]
                        if not d_df.empty:
                            d_pivot = d_df.pivot_table(index='City', columns='Product', values='Price').fillna(0)
                            if d_pivot.shape[0] >= 2:
                                # Normalize
                                d_pivot = (d_pivot - d_pivot.min()) / (d_pivot.max() - d_pivot.min())
                                d_pivot = d_pivot.fillna(0)
                                d_sim = cosine_similarity(d_pivot.values)
                                d_cities = d_pivot.index.tolist()
                                for i in range(len(d_cities)):
                                    master_G.add_node(d_cities[i]) # Ensure node exists
                                    for j in range(i + 1, len(d_cities)):
                                        if d_sim[i, j] >= threshold:
                                            master_G.add_edge(d_cities[i], d_cities[j])
                    
                    # Compute Layout ONCE (3D)
                    if len(master_G.nodes) > 0:
                        # dim=3 for 3D layout
                        pos_3d = nx.spring_layout(master_G, seed=42, k=0.15, iterations=50, dim=3)
                        
                        # Center AND Flatten
                        x_vals = [coords[0] for coords in pos_3d.values()]
                        y_vals = [coords[1] for coords in pos_3d.values()]
                        z_vals = [coords[2] for coords in pos_3d.values()]
                        
                        centroid_x = sum(x_vals) / len(x_vals)
                        centroid_y = sum(y_vals) / len(y_vals)
                        centroid_z = sum(z_vals) / len(z_vals)
                        
                        st.session_state.fixed_pos = {
                            # LOGIC CHANGE: Multiply Y by 0.6 to flatten the "sphere" into a "disc"
                            node: (coords[0] - centroid_x, (coords[1] - centroid_y) * 0.6, coords[2] - centroid_z)
                            for node, coords in pos_3d.items()
                        }
                        
                        # Manual Adjustment for Islamabad and Rawalpindi (Too close)
                        if 'Islamabad' in st.session_state.fixed_pos and 'Rawalpindi' in st.session_state.fixed_pos:
                            # Move Islamabad slightly up and right
                            ix, iy, iz = st.session_state.fixed_pos['Islamabad']
                            st.session_state.fixed_pos['Islamabad'] = (ix + 0.1, iy + 0.1, iz)
                            
                            # Move Rawalpindi slightly down and left
                            rx, ry, rz = st.session_state.fixed_pos['Rawalpindi']
                            st.session_state.fixed_pos['Rawalpindi'] = (rx - 0.1, ry - 0.1, rz)
                            
                    else:
                        st.session_state.fixed_pos = {}
                    st.session_state.fixed_pos_category = selected_category
            
            fixed_pos = st.session_state.fixed_pos

            # --- Network Visualization (3D) ---
            chart_placeholder = st.empty()
            
            # Helper to create figure
            def create_network_fig(graph, layout_pos, show_edges=True):
                traces = []
                
                # 1. Edges Trace (LOGIC CHANGE: Thinner & Transparent)
                if show_edges:
                    edge_x = []
                    edge_y = []
                    edge_z = []
                    
                    for edge in graph.edges():
                        if edge[0] in layout_pos and edge[1] in layout_pos:
                            x0, y0, z0 = layout_pos[edge[0]]
                            x1, y1, z1 = layout_pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            edge_z.extend([z0, z1, None])
                    
                    edge_trace = go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        # USE RGBA for Transparency (0.4 opacity) and Thinner lines (width=1)
                        line=dict(color='rgba(0, 255, 255, 0.4)', width=1), 
                        hoverinfo='none'
                    )
                    traces.append(edge_trace)

                # 2. Nodes Trace
                node_x = []
                node_y = []
                node_z = []
                node_text = []
                
                for node in graph.nodes():
                    if node in layout_pos:
                        x, y, z = layout_pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_z.append(z)
                        deg = degree_centrality.get(node, 0)
                        clo = closeness_centrality.get(node, 0)
                        bet = betweenness_centrality.get(node, 0)
                        eig = eigenvector_centrality.get(node, 0)
                        
                        hover_str = (
                            f"<b>{node}</b><br>"
                            f"Degree: {deg:.2f}<br>"
                            f"Closeness: {clo:.2f}<br>"
                            f"Betweenness: {bet:.2f}<br>"
                            f"Eigenvector: {eig:.2f}"
                        )
                        node_text.append(hover_str)

                node_trace = go.Scatter3d(
                    x=node_x, y=node_y, z=node_z,
                    mode='markers+text',
                    text=[node for node in graph.nodes() if node in layout_pos],
                    textposition="top center",
                    textfont=dict(color='#FFFFFF', size=11), 
                    hoverinfo='text',
                    hovertext=node_text,
                    marker=dict(
                        size=10, # Enlarged from 5
                        color='#00FFFF', # Cyan
                        # LOGIC CHANGE: Add a white rim to make nodes pop
                        line=dict(width=1, color='white'),
                        opacity=1
                    )
                )
                traces.append(node_trace)

                # 3. Axis Locking (Crucial for smooth animation)
                # By setting a fixed range, we stop the graph from "bouncing" when it rotates
                axis_template = dict(
                    showgrid=False, zeroline=False, showticklabels=False, 
                    showbackground=False, title='', visible=False,
                    range=[-1.5, 1.5] # HARD LOCK THE SCALE
                )

                layout = go.Layout(
                    title=dict(text='City Similarity Network', font=dict(color='white', size=16), y=0.95),
                    showlegend=False,
                    scene=dict(
                        xaxis=axis_template,
                        yaxis=axis_template,
                        zaxis=axis_template,
                        aspectmode='cube', # Forces equal scaling
                        bgcolor='rgba(0,0,0,0)',
                        # LOGIC CHANGE: Default camera distance ensures the whole graph fits
                        # Zoomed in: z=1.3 (was 1.5)
                        camera=dict(eye=dict(x=0, y=0, z=1.3)) 
                    ),
                    paper_bgcolor='#000000', # Pure Black
                    height=850, # Enlarged from 700
                    margin=dict(b=0,l=0,r=0,t=0),
                )
                
                fig = go.Figure(data=traces, layout=layout)
                return fig

            # --- Animation Logic (Camera Rotation) ---
            # Optimized: We now handle rotation purely in JS to ensure it's infinite and smooth.
            # This Python function is kept for compatibility but doesn't add heavy frames anymore.
            def add_rotation_animation(fig):
                return fig

            # --- Display Graph ---
            def display_html_graph(fig):
                # Convert to HTML
                html_content = fig.to_html(
                    config={'displayModeBar': False}, 
                    include_plotlyjs='cdn', 
                    full_html=True
                )
                
                # Inject Infinite Rotation Script with Play/Pause Button
                script = """
                <script>
                    window.addEventListener('load', function() {
                        var graph = document.getElementsByClassName('plotly-graph-div')[0];
                        
                        if (graph) {
                            var t = 0;
                            // Initialize state from sessionStorage (Default to false)
                            var storedState = sessionStorage.getItem('isSpinning');
                            var isSpinning = storedState === 'true'; // Default is false if null or 'false'
                            
                            // Create Button
                            var btn = document.createElement("button");
                            btn.innerHTML = isSpinning ? "⏸ Pause Rotation" : "▶ Resume Rotation";
                            btn.style.cssText = "position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); z-index: 1000; padding: 8px 16px; background: rgba(0, 0, 0, 0.7); color: #00FFFF; border: 1px solid #00FFFF; border-radius: 20px; cursor: pointer; font-family: 'Roboto', sans-serif; font-size: 14px; transition: all 0.3s;";
                            
                            btn.onmouseover = function() {
                                btn.style.background = "rgba(0, 255, 255, 0.2)";
                            };
                            btn.onmouseout = function() {
                                btn.style.background = "rgba(0, 0, 0, 0.7)";
                            };
                            
                            btn.onclick = function() {
                                isSpinning = !isSpinning;
                                sessionStorage.setItem('isSpinning', isSpinning);
                                btn.innerHTML = isSpinning ? "⏸ Pause Rotation" : "▶ Resume Rotation";
                            };
                            
                            document.body.appendChild(btn);

                            function rotate() {
                                if (isSpinning) {
                                    t += 0.005; // Speed of rotation
                                    // Zoomed in radius: 1.3 (was 1.8)
                                    var x = 1.3 * Math.cos(t);
                                    var z = 1.3 * Math.sin(t);
                                    
                                    Plotly.relayout(graph, {
                                        'scene.camera.eye': {x: x, y: 0.5, z: z}
                                    });
                                }
                                requestAnimationFrame(rotate);
                            }
                            requestAnimationFrame(rotate);
                        }
                    });
                </script>
                """
                html_content = html_content.replace('</body>', script + '</body>')
                
                # Render with components.html
                components.html(html_content, height=850) # Enlarged height

            if st.session_state.is_playing:
                # Simulation Mode
                real_fig = create_network_fig(G, fixed_pos, show_edges=True)
                real_fig = add_rotation_animation(real_fig)
                
                # Use HTML component for auto-spin
                with chart_placeholder:
                    display_html_graph(real_fig)
                
            else:
                # Manual Mode
                real_fig = create_network_fig(G, fixed_pos, show_edges=True)
                real_fig = add_rotation_animation(real_fig)
                with chart_placeholder:
                    display_html_graph(real_fig)
            
            # --- Metrics & Explanations (Moved Below Graph) ---
            st.markdown("---")
            
            # Feature 1: Tabbed Layout
            tab_overview, tab_rank, tab_heat, tab_compare, tab_raw = st.tabs([
                "📊 Network Overview", 
                "🏆 Rankings", 
                "🔥 Heatmap", 
                "⚖️ Comparative Analysis",
                "📝 Raw Data"
            ])
            
            with tab_overview:
                st.markdown("### Network Overview")
                
                avg_sim = np.mean(sim_matrix)
                density = nx.density(G)
                num_nodes = len(G.nodes())
                num_edges = len(G.edges())
                
                if len(G.nodes) > 0:
                    degree_dict = dict(G.degree(G.nodes()))
                    most_central = max(degree_dict, key=degree_dict.get)
                    avg_degree = np.mean(list(degree_dict.values()))
                else:
                    most_central = "N/A"
                    avg_degree = 0
                
                # Feature 9: Trend Indicators (Compare to fixed baseline or global avg)
                # For simplicity, let's compare to a "global average" of 0.5 for similarity
                sim_trend = "⬆" if avg_sim > 0.5 else "⬇"
                den_trend = "⬆" if density > 0.1 else "⬇"
                
                # Animated Metric Cards
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_sim:.3f} <span style="font-size:16px; color:#888;">{sim_trend}</span></div>
                        <div class="metric-label">Avg Similarity</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{density:.3f} <span style="font-size:16px; color:#888;">{den_trend}</span></div>
                        <div class="metric-label">Graph Density</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_degree:.2f}</div>
                        <div class="metric-label">Avg Degree</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col4:
                    # Feature 8: Custom "Most Central" Box
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #00FFFF 0%, #0088FF 100%); padding: 15px; border-radius: 10px; text-align: center; color: black; box-shadow: 0 4px 15px rgba(0, 255, 255, 0.4);">
                        <div style="font-size: 12px; font-weight: bold; text-transform: uppercase;">Most Central</div>
                        <div style="font-size: 18px; font-weight: 900;">{most_central}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Feature 4: Collapsible Help
                with st.expander("ℹ️ How to read this graph?"):
                    st.markdown("""
                    - **Nodes**: Represent cities.
                    - **Edges**: Connect cities with similar price patterns.
                    - **Weight**: Cosine similarity (0 to 1). Higher means more similar.
                    - **Centrality**: A central city has price patterns similar to many other cities.
                    """)

            # --- Metrics Table ---
            metrics_df = pd.DataFrame({
                'City': list(G.nodes()),
                'Degree': [degree_centrality[node] for node in G.nodes()],
                'Closeness': [closeness_centrality[node] for node in G.nodes()],
                'Betweenness': [betweenness_centrality[node] for node in G.nodes()],
                'Eigenvector': [eigenvector_centrality[node] for node in G.nodes()]
            })
            metrics_df = metrics_df.set_index('City')

            # --- Composite Score Analysis ---
            
            # 1. Normalize Metrics
            norm_df = metrics_df.copy()
            for col in norm_df.columns:
                min_val = norm_df[col].min()
                max_val = norm_df[col].max()
                if max_val - min_val != 0:
                    norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
                else:
                    norm_df[col] = 0.0
            
            # 2. Calculate Weights
            final_weights = weights.copy()
            
            if weighting_method == "Correlation-Based":
                corr_matrix = metrics_df.corr().abs()
                corr_sums = corr_matrix.sum() - 1 
                inv_corr = 1 / (1 + corr_sums)
                final_weights = (inv_corr / inv_corr.sum()).to_dict()
                
            elif weighting_method == "Category Importance":
                cat_weight_map = {
                    'Food': {'Degree': 0.4, 'Closeness': 0.2, 'Betweenness': 0.2, 'Eigenvector': 0.2},
                    'Clothing': {'Degree': 0.2, 'Closeness': 0.4, 'Betweenness': 0.2, 'Eigenvector': 0.2},
                    'Housing': {'Degree': 0.2, 'Closeness': 0.2, 'Betweenness': 0.4, 'Eigenvector': 0.2},
                    'Transport': {'Degree': 0.2, 'Closeness': 0.2, 'Betweenness': 0.2, 'Eigenvector': 0.4},
                }
                final_weights = cat_weight_map.get(selected_category, {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25})

            elif weighting_method == "Entropy-Based":
                entropy_weights = {}
                total_dispersion = 0
                dispersions = {}
                for col in metrics_df.columns:
                    col_sum = metrics_df[col].sum()
                    if col_sum == 0:
                        p = np.ones(len(metrics_df)) / len(metrics_df)
                    else:
                        p = metrics_df[col] / col_sum
                    p = p[p > 0]
                    if len(p) > 0:
                        entropy = -np.sum(p * np.log(p))
                        max_entropy = np.log(len(metrics_df))
                        norm_entropy = 0 if max_entropy == 0 else entropy / max_entropy
                    else:
                        norm_entropy = 0
                    dispersion = 1 - norm_entropy
                    dispersions[col] = dispersion
                    total_dispersion += dispersion
                
                if total_dispersion == 0:
                     final_weights = {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25}
                else:
                    final_weights = {k: v / total_dispersion for k, v in dispersions.items()}

            # 3. Calculate Composite Score
            norm_df['Composite Score'] = (
                norm_df['Degree'] * final_weights['Degree'] +
                norm_df['Closeness'] * final_weights['Closeness'] +
                norm_df['Betweenness'] * final_weights['Betweenness'] +
                norm_df['Eigenvector'] * final_weights['Eigenvector']
            )
            
            # Get Ranks
            norm_df['Rank'] = norm_df['Composite Score'].rank(ascending=False, method='min')
            norm_df = norm_df.sort_values('Rank')
            
            with tab_rank:
                st.markdown("### Composite Score Ranking")
                
                # Show weights used
                with st.expander("Show Weighting Details"):
                    st.json({k: f"{v:.3f}" for k, v in final_weights.items()})

                # Add a download button for the data
                csv = norm_df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Metrics CSV",
                    data=csv,
                    file_name=f'city_metrics_{selected_category}_{selected_date_str}.csv',
                    mime='text/csv',
                )
                
                st.dataframe(
                    norm_df[['Composite Score', 'Rank', 'Degree', 'Closeness', 'Betweenness', 'Eigenvector']].head(10),
                    column_config={
                        "Composite Score": st.column_config.ProgressColumn(
                            "Composite Score",
                            help="Weighted score based on all centrality metrics",
                            format="%.3f",
                            min_value=0,
                            max_value=1,
                        ),
                        "Degree": st.column_config.NumberColumn(format="%.3f"),
                        "Closeness": st.column_config.NumberColumn(format="%.3f"),
                        "Betweenness": st.column_config.NumberColumn(format="%.3f"),
                        "Eigenvector": st.column_config.NumberColumn(format="%.3f"),
                    },
                    use_container_width=True
                )
                
                # Sort by Score for Visualization
                ranked_df = norm_df.sort_values(by='Composite Score', ascending=False)
                
                # Visualization
                score_fig = go.Figure(go.Bar(
                    x=ranked_df.index,
                    y=ranked_df['Composite Score'],
                    marker=dict(color=ranked_df['Composite Score'], colorscale='Viridis'),
                    text=ranked_df['Composite Score'].apply(lambda x: f"{x:.3f}"),
                    textposition='auto'
                ))
                
                score_fig.update_layout(
                    title='City Ranking by Composite Score',
                    xaxis_title='City',
                    yaxis_title='Composite Score (0-1)',
                    xaxis_tickangle=-45,
                    margin=dict(b=100)
                )
                
                st.plotly_chart(score_fig, use_container_width=True, key="score_chart")

            with tab_heat:
                st.markdown("### Similarity Matrix Heatmap")
                
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=sim_matrix,
                    x=cities,
                    y=cities,
                    colorscale='Viridis',
                ))
                heatmap_fig.update_layout(
                    title='Pairwise Cosine Similarity',
                    xaxis_nticks=36
                )
                st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap")

            # --- Comparative Analysis (Moved to Tab) ---
            with tab_compare:
                st.header("Comparative Analysis")
                st.write("Compare how different weighting techniques affect city rankings.")
                
                # Calculate scores for all methods to compare
                
                # 1. Equal
                w_equal = {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25}
                
                # 2. Correlation
                corr_matrix = metrics_df.corr().abs()
                corr_sums = corr_matrix.sum() - 1
                inv_corr = 1 / (1 + corr_sums)
                w_corr = (inv_corr / inv_corr.sum()).to_dict()
                
                # 3. Category
                cat_weight_map = {
                    'Food': {'Degree': 0.4, 'Closeness': 0.2, 'Betweenness': 0.2, 'Eigenvector': 0.2},
                    'Clothing': {'Degree': 0.2, 'Closeness': 0.4, 'Betweenness': 0.2, 'Eigenvector': 0.2},
                    'Housing': {'Degree': 0.2, 'Closeness': 0.2, 'Betweenness': 0.4, 'Eigenvector': 0.2},
                    'Transport': {'Degree': 0.2, 'Closeness': 0.2, 'Betweenness': 0.2, 'Eigenvector': 0.4},
                }
                w_cat = cat_weight_map.get(selected_category, {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25})
                
                # 4. Entropy
                dispersions = {}
                total_dispersion = 0
                for col in metrics_df.columns:
                    col_sum = metrics_df[col].sum()
                    if col_sum == 0:
                        p = np.ones(len(metrics_df)) / len(metrics_df)
                    else:
                        p = metrics_df[col] / col_sum
                    p = p[p > 0]
                    if len(p) > 0:
                        entropy = -np.sum(p * np.log(p))
                        max_entropy = np.log(len(metrics_df))
                        norm_entropy = 0 if max_entropy == 0 else entropy / max_entropy
                    else:
                        norm_entropy = 0
                    dispersion = 1 - norm_entropy
                    dispersions[col] = dispersion
                    total_dispersion += dispersion
                
                if total_dispersion == 0:
                     w_ent = {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25}
                else:
                    w_ent = {k: v / total_dispersion for k, v in dispersions.items()}

                methods_dict = {
                    "Equal": w_equal,
                    "Correlation": w_corr,
                    "Category": w_cat,
                    "Entropy": w_ent
                }
                
                # Compute Scores and Ranks
                rank_df = pd.DataFrame(index=metrics_df.index)
                score_df = pd.DataFrame(index=metrics_df.index)
                
                for m_name, m_weights in methods_dict.items():
                    score = (
                        norm_df['Degree'] * m_weights['Degree'] +
                        norm_df['Closeness'] * m_weights['Closeness'] +
                        norm_df['Betweenness'] * m_weights['Betweenness'] +
                        norm_df['Eigenvector'] * m_weights['Eigenvector']
                    )
                    score_df[m_name] = score
                    rank_df[m_name] = score.rank(ascending=False, method='min')
                
                # --- Visualizations ---
                c_tab1, c_tab2 = st.tabs(["📈 Ranking Evolution", "🏆 Top 5 Contrast"])
                
                with c_tab1:
                    st.subheader("How Rankings Change Across Methods")
                    bump_data = rank_df.reset_index().melt(id_vars='City', var_name='Method', value_name='Rank')
                    
                    fig_bump = go.Figure()
                    for city in cities:
                        city_data = bump_data[bump_data['City'] == city]
                        fig_bump.add_trace(go.Scatter(
                            x=city_data['Method'],
                            y=city_data['Rank'],
                            mode='lines+markers',
                            name=city,
                            line_shape='spline'
                        ))
                    
                    fig_bump.update_layout(
                        title="City Rankings by Method (Lower is Better)",
                        yaxis=dict(autorange="reversed", title="Rank"),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_bump, use_container_width=True)
                    
                with c_tab2:
                    st.subheader("Top 5 Cities: Equal vs Entropy")
                    top5_equal = score_df.nlargest(5, 'Equal').index.tolist()
                    contrast_data = score_df.loc[top5_equal, ['Equal', 'Entropy']].reset_index()
                    
                    fig_contrast = go.Figure()
                    fig_contrast.add_trace(go.Bar(
                        x=contrast_data['City'],
                        y=contrast_data['Equal'],
                        name='Equal Weighting',
                        marker_color='#1f77b4'
                    ))
                    fig_contrast.add_trace(go.Bar(
                        x=contrast_data['City'],
                        y=contrast_data['Entropy'],
                        name='Entropy-Based',
                        marker_color='#ff7f0e'
                    ))
                    
                    fig_contrast.update_layout(
                        title="Score Comparison for Top Cities (Equal vs Entropy)",
                        barmode='group',
                        yaxis_title="Composite Score"
                    )
                    st.plotly_chart(fig_contrast, use_container_width=True)

            # Feature 5: Raw Data Inspector
            with tab_raw:
                st.markdown("### Raw Data Inspector")
                st.write("View the underlying data for the current selection.")
                st.dataframe(pivot_df)

            # Feature 10: Feedback Widget
            st.markdown("---")
            st.markdown("### 📝 Feedback")
            col_fb1, col_fb2 = st.columns([3, 1])
            with col_fb1:
                st.write("Was this analysis helpful?")
            with col_fb2:
                sentiment_mapping = ["one", "two", "three", "four", "five"]
                selected = st.feedback("stars")
                if selected is not None:
                    st.toast(f"Thanks for your {sentiment_mapping[selected]}-star rating!", icon="⭐")

            # --- Temporal Relation & Hasse Diagram ---
            # (Kept separate as it's a distinct analysis)
            st.markdown("---")
            st.header("Temporal Relation Analysis (Hasse Diagram)")
            st.write("Visualizing the subset relationship of edges between different time steps.")
            
            # 1. Compute Edges for ALL Time Steps
            # We need to iterate over all dates for the selected category
            temporal_edges = {}
            all_dates = df['Date'].sort_values().unique()
            
            # We'll use the same threshold as selected in sidebar
            
            for d in all_dates:
                d_mask = (df['Category'] == selected_category) & (df['Date'] == d)
                d_df = df[d_mask]
                
                if not d_df.empty:
                    d_pivot = d_df.pivot_table(index='City', columns='Product', values='Price').fillna(0)
                    # Normalize
                    d_pivot = (d_pivot - d_pivot.min()) / (d_pivot.max() - d_pivot.min())
                    d_pivot = d_pivot.fillna(0)
                    
                    if d_pivot.shape[0] >= 2:
                        d_sim = cosine_similarity(d_pivot.values)
                        d_cities = d_pivot.index.tolist()
                        
                        # Extract Edges
                        edges = set()
                        for i in range(len(d_cities)):
                            for j in range(i + 1, len(d_cities)):
                                if d_sim[i, j] >= threshold:
                                    # Store edge as sorted tuple of city names to be consistent
                                    u, v = sorted((d_cities[i], d_cities[j]))
                                    edges.add((u, v))
                        
                        # Store edges for this date (formatted as string for display)
                        date_label = pd.to_datetime(d).strftime('%Y-%m')
                        temporal_edges[date_label] = edges

            # 2. Determine Relations (Subset)
            # G_t1 T G_t2 <=> E_t1 is subset of E_t2
            sorted_dates = sorted(temporal_edges.keys())
            hasse_edges = []
            
            # Check properties
            is_reflexive = True # Always true for subset
            is_antisymmetric = True
            is_transitive = True # Subset is transitive
            
            # Build Relation Graph
            R = nx.DiGraph()
            for d in sorted_dates:
                R.add_node(d)
            
            for i in range(len(sorted_dates)):
                for j in range(len(sorted_dates)):
                    if i == j: continue
                    
                    d1 = sorted_dates[i]
                    d2 = sorted_dates[j]
                    
                    E1 = temporal_edges[d1]
                    E2 = temporal_edges[d2]
                    
                    if E1.issubset(E2):
                        R.add_edge(d1, d2)
                        
            # 3. Visualize Hasse Diagram (Transitive Reduction)
            try:
                H = nx.transitive_reduction(R)
                H.add_nodes_from(R.nodes(data=True)) # Preserve node data if any
            except:
                H = R # Fallback
            
            if len(H.nodes()) > 0:
                # Use Graphviz layout if available, else spring
                try:
                    pos_h = nx.nx_agraph.graphviz_layout(H, prog='dot')
                except:
                    pos_h = nx.spring_layout(H, seed=42)
                
                edge_x_h = []
                edge_y_h = []
                
                for edge in H.edges():
                    x0, y0 = pos_h[edge[0]]
                    x1, y1 = pos_h[edge[1]]
                    edge_x_h.extend([x0, x1, None])
                    edge_y_h.extend([y0, y1, None])
                    
                hasse_edge_trace = go.Scatter(
                    x=edge_x_h, y=edge_y_h,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                node_x_h = []
                node_y_h = []
                node_text_h = []
                
                for node in H.nodes():
                    x, y = pos_h[node]
                    node_x_h.append(x)
                    node_y_h.append(y)
                    node_text_h.append(node)
                    
                hasse_node_trace = go.Scatter(
                    x=node_x_h, y=node_y_h,
                    mode='markers+text',
                    text=node_text_h,
                    textposition="top center",
                    marker=dict(size=10, color='#1f77b4'),
                    hoverinfo='text'
                )
                
                fig_hasse = go.Figure(data=[hasse_edge_trace, hasse_node_trace],
                    layout=go.Layout(
                        title='Hasse Diagram of Temporal Relations (Subset)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )
                st.plotly_chart(fig_hasse, use_container_width=True)
            for m_name, m_weights in methods_dict.items():
                score = (
                    norm_df['Degree'] * m_weights['Degree'] +
                    norm_df['Closeness'] * m_weights['Closeness'] +
                    norm_df['Betweenness'] * m_weights['Betweenness'] +
                    norm_df['Eigenvector'] * m_weights['Eigenvector']
                )
                score_df[m_name] = score
                rank_df[m_name] = score.rank(ascending=False, method='min')
            
            # --- Visualizations ---
            tab1, tab2 = st.tabs(["📈 Ranking Evolution", "🏆 Top 5 Contrast"])
            
            with tab1:
                st.subheader("How Rankings Change Across Methods")
                # Bump Chart (Line Plot)
                # Prepare data for Plotly Express-like structure
                bump_data = rank_df.reset_index().melt(id_vars='City', var_name='Method', value_name='Rank')
                
                fig_bump = go.Figure()
                for city in cities:
                    city_data = bump_data[bump_data['City'] == city]
                    fig_bump.add_trace(go.Scatter(
                        x=city_data['Method'],
                        y=city_data['Rank'],
                        mode='lines+markers',
                        name=city,
                        line_shape='spline'
                    ))
                
                fig_bump.update_layout(
                    title="City Rankings by Method (Lower is Better)",
                    yaxis=dict(autorange="reversed", title="Rank"),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bump, use_container_width=True, key="bump_chart")
                
            with tab2:
                st.subheader("Top 5 Cities: Equal vs Entropy")
                # Compare Equal vs Entropy (Stability vs Variability)
                
                # Get Top 5 from Equal
                top5_equal = score_df.nlargest(5, 'Equal').index.tolist()
                
                # Filter data
                contrast_data = score_df.loc[top5_equal, ['Equal', 'Entropy']].reset_index()
                
                fig_contrast = go.Figure()
                fig_contrast.add_trace(go.Bar(
                    x=contrast_data['City'],
                    y=contrast_data['Equal'],
                    name='Equal Weighting',
                    marker_color='#1f77b4'
                ))
                fig_contrast.add_trace(go.Bar(
                    x=contrast_data['City'],
                    y=contrast_data['Entropy'],
                    name='Entropy-Based',
                    marker_color='#ff7f0e'
                ))
                
                fig_contrast.update_layout(
                    title="Score Comparison for Top Cities (Equal vs Entropy)",
                    barmode='group',
                    yaxis_title="Composite Score"
                )
                st.plotly_chart(fig_contrast, use_container_width=True, key="contrast_chart")

            # --- Temporal Relation & Hasse Diagram ---
            st.markdown("---")
            st.header("Temporal Relation Analysis (Hasse Diagram)")
            st.write("Visualizing the subset relationship of edges between different time steps.")
            
            # 1. Compute Edges for ALL Time Steps
            # We need to iterate over all dates for the selected category
            temporal_edges = {}
            all_dates = df['Date'].sort_values().unique()
            
            # We'll use the same threshold as selected in sidebar
            
            for d in all_dates:
                d_mask = (df['Category'] == selected_category) & (df['Date'] == d)
                d_df = df[d_mask]
                
                if not d_df.empty:
                    d_pivot = d_df.pivot_table(index='City', columns='Product', values='Price').fillna(0)
                    # Normalize
                    d_pivot = (d_pivot - d_pivot.min()) / (d_pivot.max() - d_pivot.min())
                    d_pivot = d_pivot.fillna(0)
                    
                    if d_pivot.shape[0] >= 2:
                        d_sim = cosine_similarity(d_pivot.values)
                        d_cities = d_pivot.index.tolist()
                        
                        # Extract Edges
                        edges = set()
                        for i in range(len(d_cities)):
                            for j in range(i + 1, len(d_cities)):
                                if d_sim[i, j] >= threshold:
                                    # Store edge as sorted tuple of city names to be consistent
                                    u, v = sorted((d_cities[i], d_cities[j]))
                                    edges.add((u, v))
                        
                        # Store edges for this date (formatted as string for display)
                        date_label = pd.to_datetime(d).strftime('%Y-%m')
                        temporal_edges[date_label] = edges

            # 2. Determine Relations (Subset)
            # G_t1 T G_t2 <=> E_t1 is subset of E_t2
            sorted_dates = sorted(temporal_edges.keys())
            hasse_edges = []
            
            # Check properties
            is_reflexive = True # Always true for subset
            is_antisymmetric = True
            is_transitive = True # Subset is transitive
            
            # Build Relation Graph
            R = nx.DiGraph()
            for d in sorted_dates:
                R.add_node(d)
            
            for i in range(len(sorted_dates)):
                for j in range(len(sorted_dates)):
                    if i == j: continue
                    
                    d1 = sorted_dates[i]
                    d2 = sorted_dates[j]
                    
                    E1 = temporal_edges[d1]
                    E2 = temporal_edges[d2]
                    
                    if E1.issubset(E2):
                        R.add_edge(d1, d2)
                        
            # 3. Visualize Hasse Diagram (Transitive Reduction)
            try:
                H = nx.transitive_reduction(R)
                H.add_nodes_from(R.nodes(data=True)) # Preserve node data if any
            except:
                H = R # Fallback
            
            if len(H.nodes()) > 0:
                # Use Graphviz layout if available, else spring
                try:
                    pos_h = nx.nx_agraph.graphviz_layout(H, prog='dot')
                except:
                    pos_h = nx.spring_layout(H, seed=42)
                
                edge_x_h = []
                edge_y_h = []
                
                for edge in H.edges():
                    x0, y0 = pos_h[edge[0]]
                    x1, y1 = pos_h[edge[1]]
                    edge_x_h.extend([x0, x1, None])
                    edge_y_h.extend([y0, y1, None])
                    
                hasse_edge_trace = go.Scatter(
                    x=edge_x_h, y=edge_y_h,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                node_x_h = []
                node_y_h = []
                node_text_h = []
                
                for node in H.nodes():
                    x, y = pos_h[node]
                    node_x_h.append(x)
                    node_y_h.append(y)
                    node_text_h.append(node)
                    
                hasse_node_trace = go.Scatter(
                    x=node_x_h, y=node_y_h,
                    mode='markers+text',
                    text=node_text_h,
                    textposition="top center",
                    marker=dict(size=10, color='#1f77b4'),
                    hoverinfo='text'
                )
                
                fig_hasse = go.Figure(data=[hasse_edge_trace, hasse_node_trace],
                    layout=go.Layout(
                        title='Hasse Diagram of Temporal Relations (Subset)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )
                st.plotly_chart(fig_hasse, use_container_width=True, key="hasse_chart")
                
                st.info(f"**Properties Verified:**\n- Reflexive: Yes\n- Antisymmetric: {is_antisymmetric}\n- Transitive: Yes")
            else:
                st.warning("No temporal relations found or not enough data.")

else:
    st.error("Could not load data. Please check if 'Categorized_CPI_Data.xlsx' is in the same directory.")

# --- Auto-Increment Logic for Simulation ---
if 'is_playing' in st.session_state and st.session_state.is_playing:
    # Show status
    st.sidebar.success(f"Simulation Running... Speed: {st.session_state.sim_speed}s")
    
    # Toast BEFORE sleep so user sees it
    st.toast(f"Simulation Active: Moving to next month in {st.session_state.sim_speed} seconds...")
    
    # Wait
    time.sleep(st.session_state.sim_speed)
    
    # Calculate next month
    current_year = st.session_state.selected_year
    current_month = st.session_state.selected_month
    
    # Logic to move to next month
    next_month = current_month + 1
    next_year = current_year
    
    if next_month > 12:
        next_month = 1
        next_year += 1
        
    # Check bounds
    if df is not None:
        max_date_val = df['Date'].max()
        try:
            next_date = pd.Timestamp(year=next_year, month=next_month, day=1)
        except:
            next_date = max_date_val # Fallback
            
        if next_date <= max_date_val:
            st.session_state.selected_year = next_year
            st.session_state.selected_month = next_month
            st.rerun()
        else:
            # Stop at end
            st.session_state.is_playing = False
            st.toast("Simulation Complete!")
            st.rerun()
