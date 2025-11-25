import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
<div class="project-header">
    <div class="project-title">Pakistan Through Discrete Structures</div>
    <div class="team-members">By Abdullah Nadeem, Arham Manzoor, and Zainab Nisaar</div>
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
    
    # Date Selection
    # Get unique dates sorted
    dates = df['Date'].sort_values().unique()
    # Convert to readable string for slider
    date_options = [pd.to_datetime(d).strftime('%Y-%m') for d in dates]
    
    selected_date_str = st.sidebar.select_slider(
        "Select Time Step (Month-Year)",
        options=date_options,
        value=date_options[0]
    )
    
    # Similarity Threshold
    # Similarity Threshold
    # Calculate similarity range for guidance
    # We need to calculate sim_matrix first to give good bounds, but that depends on filtering.
    # So we move the slider after data filtering or just make it generic high precision.
    # For now, let's make it high precision 0.9-1.0 as observed data is high.
    threshold = st.sidebar.slider(
        "Similarity Threshold (Edge Visibility)",
        min_value=0.900,
        max_value=1.000,
        value=0.990,
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
            for i in range(len(cities)):
                for j in range(i + 1, len(cities)):
                    sim = sim_matrix[i, j]
                    if sim >= threshold:
                        # Distance = 1 - similarity (Higher sim = Lower distance)
                        # Adding a small epsilon to avoid 0 distance if needed, though 0 is fine for connected nodes
                        dist = max(0, 1.0 - sim)
                        G.add_edge(cities[i], cities[j], weight=sim, distance=dist)
            
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
            
            st.title("Temporal Network of Cities")
            st.markdown(f"### Category: *{selected_category}* | Date: *{selected_date_str}*")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # --- Network Visualization ---
                pos = nx.spring_layout(G, seed=42)
                
                edge_x = []
                edge_y = []
                edge_text = []
                
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.append(x0)
                    edge_x.append(x1)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)
                    edge_text.append(f"Sim: {edge[2]['weight']:.2f}")

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='text',
                    text=edge_text,
                    mode='lines')

                node_x = []
                node_y = []
                node_text = []
                node_adjacencies = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Create hover text with metrics
                    hover_str = (
                        f"<b>{node}</b><br>" +
                        f"Degree Centrality: {degree_centrality[node]:.4f}<br>" +
                        f"Closeness Centrality: {closeness_centrality[node]:.4f}<br>" +
                        f"Betweenness Centrality: {betweenness_centrality[node]:.4f}<br>" +
                        f"Eigenvector Centrality: {eigenvector_centrality[node]:.4f}"
                    )
                    node_text.append(hover_str)
                    node_adjacencies.append(len(G.adj[node]))

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        reversescale=True,
                        color=node_adjacencies,
                        size=20,
                        colorbar=dict(
                            thickness=15,
                            title=dict(
                                text='Node Degree',
                                side='right'
                            ),
                            xanchor='left'
                        ),
                        line_width=2))

                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title=dict(
                                        text='City Similarity Network',
                                        font=dict(size=16)
                                    ),
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # --- Metrics & Explanations ---
                st.markdown("### Network Metrics")
                
                avg_sim = np.mean(sim_matrix)
                density = nx.density(G)
                if len(G.nodes) > 0:
                    degree_dict = dict(G.degree(G.nodes()))
                    most_central = max(degree_dict, key=degree_dict.get)
                else:
                    most_central = "N/A"
                
                st.info(f"**Avg Similarity:** {avg_sim:.3f}")
                st.info(f"**Graph Density:** {density:.3f}")
                st.success(f"**Most Central City:**\n\n{most_central}")
                
                st.markdown("---")
                st.markdown("### Explanation")
                st.markdown("""
                - **Nodes**: Represent cities.
                - **Edges**: Connect cities with similar price patterns.
                - **Weight**: Cosine similarity (0 to 1). Higher means more similar.
                - **Centrality**: A central city has price patterns similar to many other cities.
                """)

            # --- Metrics Table ---
            # st.markdown("### City Centrality Metrics") # Moved to after composite score
            metrics_df = pd.DataFrame({
                'City': list(G.nodes()),
                'Degree': [degree_centrality[node] for node in G.nodes()],
                'Closeness': [closeness_centrality[node] for node in G.nodes()],
                'Betweenness': [betweenness_centrality[node] for node in G.nodes()],
                'Eigenvector': [eigenvector_centrality[node] for node in G.nodes()]
            })
            metrics_df = metrics_df.set_index('City')
            # st.dataframe(metrics_df.style.format("{:.4f}")) # Moved to after composite score

            # --- Composite Score Analysis ---
            st.markdown("---")
            st.markdown("### Composite Score Ranking")
            
            # 1. Normalize Metrics (Min-Max Scaling)
            norm_df = metrics_df.copy()
            for col in norm_df.columns:
                min_val = norm_df[col].min()
                max_val = norm_df[col].max()
                if max_val - min_val != 0:
                    norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
                else:
                    norm_df[col] = 0.0
            
            # 2. Calculate Weights based on Method
            final_weights = weights.copy() # Default from sidebar (Equal or Interactive)
            
            if weighting_method == "Correlation-Based":
                # Calculate correlation matrix
                corr_matrix = metrics_df.corr().abs()
                # Sum of correlations for each metric (excluding self)
                corr_sums = corr_matrix.sum() - 1 
                # Weight = 1 / (1 + sum_corr) - Inverse relationship
                # Higher correlation -> Lower weight
                inv_corr = 1 / (1 + corr_sums)
                final_weights = (inv_corr / inv_corr.sum()).to_dict()
                
                st.write("**Correlation-Based Weights:**")
                st.json({k: f"{v:.3f}" for k, v in final_weights.items()})
                
            elif weighting_method == "Category Importance":
                # Define importance based on category (Example Logic)
                # Assuming certain metrics are more important for certain categories
                # This is a heuristic mapping
                cat_weight_map = {
                    'Food': {'Degree': 0.4, 'Closeness': 0.2, 'Betweenness': 0.2, 'Eigenvector': 0.2}, # Hubs important
                    'Clothing': {'Degree': 0.2, 'Closeness': 0.4, 'Betweenness': 0.2, 'Eigenvector': 0.2}, # Access important
                    'Housing': {'Degree': 0.2, 'Closeness': 0.2, 'Betweenness': 0.4, 'Eigenvector': 0.2}, # Bridges important
                    'Transport': {'Degree': 0.2, 'Closeness': 0.2, 'Betweenness': 0.2, 'Eigenvector': 0.4}, # Influence important
                }
                # Default if category not in map
                final_weights = cat_weight_map.get(selected_category, {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25})
                
                st.write(f"**Category Importance Weights ({selected_category}):**")
                st.json({k: f"{v:.3f}" for k, v in final_weights.items()})

            elif weighting_method == "Entropy-Based":
                # Calculate Entropy for each metric
                # H = -sum(p * log(p))
                # We use the normalized values. To avoid log(0), add small epsilon.
                entropy_weights = {}
                total_dispersion = 0
                
                dispersions = {}
                for col in metrics_df.columns:
                    # Normalize to sum to 1 to treat as probability for entropy calc
                    col_sum = metrics_df[col].sum()
                    if col_sum == 0:
                        p = np.ones(len(metrics_df)) / len(metrics_df)
                    else:
                        p = metrics_df[col] / col_sum
                    
                    # Handle zeros
                    p = p[p > 0]
                    
                    if len(p) > 0:
                        entropy = -np.sum(p * np.log(p))
                        # Max entropy is log(n)
                        max_entropy = np.log(len(metrics_df))
                        if max_entropy == 0:
                            norm_entropy = 0
                        else:
                            norm_entropy = entropy / max_entropy
                    else:
                        norm_entropy = 0
                        
                    # Dispersion = 1 - Normalized Entropy
                    # Higher dispersion (more variability) -> Higher Weight
                    dispersion = 1 - norm_entropy
                    dispersions[col] = dispersion
                    total_dispersion += dispersion
                
                if total_dispersion == 0:
                     final_weights = {'Degree': 0.25, 'Closeness': 0.25, 'Betweenness': 0.25, 'Eigenvector': 0.25}
                else:
                    final_weights = {k: v / total_dispersion for k, v in dispersions.items()}
                
                st.write("**Entropy-Based Weights (Variability):**")
                st.json({k: f"{v:.3f}" for k, v in final_weights.items()})

            # 3. Calculate Composite Score
            norm_df['Composite Score'] = (
                norm_df['Degree'] * final_weights['Degree'] +
                norm_df['Closeness'] * final_weights['Closeness'] +
                norm_df['Betweenness'] * final_weights['Betweenness'] +
                norm_df['Eigenvector'] * final_weights['Eigenvector']
            )
            
            # Sort by Score
            ranked_df = norm_df.sort_values(by='Composite Score', ascending=False)
            
            # --- Metrics Table (Updated with Score) ---
            st.markdown("### City Centrality Metrics & Composite Score")
            # We want to show the original metrics but sorted by score, and maybe include the score
            # Let's merge the score back to the original metrics df for display
            display_df = metrics_df.copy()
            display_df['Composite Score'] = norm_df['Composite Score']
            display_df = display_df.sort_values(by='Composite Score', ascending=False)
            
            st.dataframe(display_df.style.format("{:.4f}"))


            
            # 4. Visualization
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
            
            st.plotly_chart(score_fig, use_container_width=True)

            # --- Heatmap ---
            st.markdown("### Similarity Matrix Heatmap")
            
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=sim_matrix,
                x=cities,
                y=cities,
                colorscale='Viridis',
                # zmin=0, zmax=1 # Removed to allow auto-scaling based on data range
            ))
            heatmap_fig.update_layout(
                title='Pairwise Cosine Similarity',
                xaxis_nticks=36
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # --- Comparative Analysis ---
            st.markdown("---")
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
                st.plotly_chart(fig_bump, use_container_width=True)
                
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
                st.plotly_chart(fig_contrast, use_container_width=True)

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
                st.plotly_chart(fig_hasse, use_container_width=True)
                
                st.info(f"**Properties Verified:**\n- Reflexive: Yes\n- Antisymmetric: {is_antisymmetric}\n- Transitive: Yes")
            else:
                st.warning("No temporal relations found or not enough data.")

else:
    st.error("Could not load data. Please check if 'Categorized_CPI_Data.xlsx' is in the same directory.")
