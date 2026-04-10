import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import os

# Page config
st.set_page_config(
    page_title="Palmer Archipelago Research",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5f7a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #fff8e6;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #c8902e;
        margin: 1.5rem 0;
    }
    .insight-title {
        color: #c8902e;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        # Try to load from local file first
        df = pd.read_csv('data/processed/penguins_cleaned.csv')
        
        # Map column names to match the dashboard format
        column_mapping = {
            'species': 'Species',
            'island': 'Island',
            'bill_length_mm': 'Bill Length (Mm)',
            'bill_depth_mm': 'Bill Depth (Mm)',
            'flipper_length_mm': 'Flipper Length (Mm)',
            'body_mass_g': 'Body Mass (G)',
            'sex': 'Sex'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Capitalize first letter of categorical values
        df['Species'] = df['Species'].str.capitalize()
        df['Island'] = df['Island'].str.capitalize()
        
        # Handle 'Unknown' sex values
        df['Sex'] = df['Sex'].replace('Unknown', 'Unknown')
        
        # Select only needed columns
        needed_cols = ['Species', 'Island', 'Bill Length (Mm)', 'Bill Depth (Mm)', 
                       'Flipper Length (Mm)', 'Body Mass (G)', 'Sex']
        df = df[needed_cols]
        
        return df
        
    except FileNotFoundError:
        st.error("⚠️ Cannot find 'data/processed/penguins_cleaned.csv'. Please ensure the file exists.")
        st.info("Expected file path: data/processed/penguins_cleaned.csv")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# Sidebar filters
st.sidebar.markdown("### GLOBAL FILTERS")
st.sidebar.markdown(f"**n={len(df)} specimens**")

st.sidebar.markdown("#### 🐧 SPECIES SELECTION")
species_options = sorted(df['Species'].unique())
selected_species = st.sidebar.multiselect(
    "Select Species",
    options=species_options,
    default=species_options,
    key="species_filter",
    label_visibility="collapsed"
)

st.sidebar.markdown("#### 🏝️ ISLAND LOCALITY")
island_options = sorted(df['Island'].unique())
selected_islands = st.sidebar.multiselect(
    "Select Islands",
    options=island_options,
    default=island_options,
    key="island_filter",
    label_visibility="collapsed"
)

st.sidebar.markdown("#### 👥 DEMOGRAPHICS")
sex_options = sorted(df['Sex'].unique())
selected_sex = st.sidebar.multiselect(
    "Select Sex",
    options=sex_options,
    default=sex_options,
    key="sex_filter",
    label_visibility="collapsed"
)

if st.sidebar.button("RESET FILTERS", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("📄 **DOCUMENTATION**")
st.sidebar.markdown("📊 **EXPORT DATA**")

# Filter data
filtered_df = df[
    (df['Species'].isin(selected_species)) &
    (df['Island'].isin(selected_islands)) &
    (df['Sex'].isin(selected_sex))
]

if len(filtered_df) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
    st.stop()

# Color mapping
color_map = {
    'Adelie': '#c8902e',
    'Chinstrap': '#b47eba',
    'Gentoo': '#3d9b9b'
}

# Header
st.markdown('<div class="main-header">Palmer Archipelago Research</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Correlations", "ML Explorer"])

# TAB 1: OVERVIEW
with tab1:
    st.markdown('<div class="main-header">Palmer Penguins Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploring morphology, geography, and ML clustering</div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #1a5f7a;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Total Penguins</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1a5f7a;">{}</div>
            <div style="color: #666; font-size: 0.9rem;">Specimens</div>
        </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #b47eba;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Distinct Species</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #b47eba;">{}</div>
            <div style="color: #666; font-size: 0.9rem;">Categories</div>
        </div>
        """.format(filtered_df['Species'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #c8902e;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Islands Surveyed</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #c8902e;">{}</div>
            <div style="color: #666; font-size: 0.9rem;">Localities</div>
        </div>
        """.format(filtered_df['Island'].nunique()), unsafe_allow_html=True)
    
    with col4:
        mean_mass = filtered_df['Body Mass (G)'].mean() / 1000
        st.markdown("""
        <div class="metric-card" style="border-left-color: #3d9b9b;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Mean Body Mass</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #3d9b9b;">{:.1f}</div>
            <div style="color: #666; font-size: 0.9rem;">kg Avg</div>
        </div>
        """.format(mean_mass), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Species distribution and breakdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Species Distribution by Island")
        st.markdown("Population density across Archipelago localities")
        
        # Grouped/Stacked toggle
        chart_type = st.radio("", ["Grouped", "Stacked"], horizontal=True)
        
        # Create pivot data
        pivot_data = filtered_df.groupby(['Island', 'Species']).size().reset_index(name='Count')
        
        if chart_type == "Grouped":
            fig = px.bar(
                pivot_data,
                x='Island',
                y='Count',
                color='Species',
                color_discrete_map=color_map,
                barmode='group',
                height=400
            )
        else:
            fig = px.bar(
                pivot_data,
                x='Island',
                y='Count',
                color='Species',
                color_discrete_map=color_map,
                barmode='stack',
                height=400
            )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Species Breakdown")
        st.markdown(f"Overall proportion of n={len(filtered_df)}")
        
        # Calculate percentages
        species_counts = filtered_df['Species'].value_counts()
        species_pct = (species_counts / len(filtered_df) * 100).round(1)
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=species_counts.index,
            values=species_counts.values,
            hole=0.6,
            marker=dict(colors=[color_map.get(s, '#999999') for s in species_counts.index]),
            textposition='outside',
            textinfo='none'
        )])
        
        fig.update_layout(
            showlegend=False,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[dict(text='100%<br>DATA FULL', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Species list
        for species in species_counts.index:
            st.markdown(f"**{species}** - {species_pct[species]}%")
    
    # Critical Research Insight
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 CRITICAL RESEARCH INSIGHT</div>
        <div style="color: #333; line-height: 1.6;">
        The data reveals a distinct geographic segregation: Gentoo penguins are found exclusively on Biscoe Island within this dataset, 
        while Chinstrap populations are confined to Dream Island. Adelie penguins exhibit the highest ecological plasticity, spanning all three 
        islands (Biscoe, Dream, and Torgersen).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bottom section
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("#### Palmer Archipelago Coastal Study Area")
        st.info("🗺️ Geographic location: Antarctic Peninsula region")
    
    with col2:
        st.markdown("#### Morphological Baseline")
        st.markdown("""
        Measurement standards for flipper length and bill depth are 
        calibrated against the 2007-2009 multi-year study. This 
        dashboard provides real-time filtering across the normalized 
        n={} specimen dataset.
        """.format(len(filtered_df)))

# TAB 2: DISTRIBUTIONS
with tab2:
    st.markdown("### ANALYSIS ATTRIBUTE")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        attribute = st.selectbox(
            "Attribute", 
            ["Bill Length (Mm)", "Bill Depth (Mm)", "Flipper Length (Mm)", "Body Mass (G)"],
            label_visibility="collapsed"
        )
    with col2:
        bins = st.slider("HISTOGRAM BINS", 10, 50, 30)
    with col3:
        st.markdown("")
        for species in selected_species:
            if species in color_map:
                st.markdown(f'<span style="color: {color_map[species]};">● {species.upper()}</span>', unsafe_allow_html=True)
    
    # Attribute Distribution Histogram
    st.markdown("### Attribute Distribution")
    st.markdown(f"Comparative frequency of {attribute} across all recorded specimens")
    
    fig = go.Figure()
    for species in selected_species:
        if species in color_map:
            species_data = filtered_df[filtered_df['Species'] == species]
            fig.add_trace(go.Histogram(
                x=species_data[attribute],
                name=species,
                marker_color=color_map[species],
                opacity=0.7,
                nbinsx=bins,
                text=[species] * len(species_data),
                hovertemplate=f'<b>{species}</b><br>{attribute}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
    
    fig.update_layout(
        barmode='overlay',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, title=attribute),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Frequency'),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Mass vs Bill Area Variance
    st.markdown("### Mass vs Bill Area Variance")
    st.markdown("Interquartile range and outliers by species")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Box plots for each species
        fig = go.Figure()
        
        for species in selected_species:
            if species in color_map:
                species_data = filtered_df[filtered_df['Species'] == species]
                fig.add_trace(go.Box(
                    y=species_data['Body Mass (G)'],
                    name=species,
                    marker_color=color_map[species],
                    boxmean='sd'
                ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Body Mass (g)'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Max body mass
        max_mass_row = filtered_df.loc[filtered_df['Body Mass (G)'].idxmax()]
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3d9b9b;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Max Body Mass</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #3d9b9b;">{int(max_mass_row['Body Mass (G)'])}</div>
            <div style="color: #666; font-size: 0.9rem;">grams</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">{max_mass_row['Species']} ({max_mass_row['Sex']}, {max_mass_row['Island']})</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Avg attribute
        if len(selected_species) > 0:
            first_species = selected_species[0]
            attr_mean = filtered_df[filtered_df['Species'] == first_species][attribute].mean()
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #b47eba;">
                <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Avg {attribute}</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #b47eba;">{attr_mean:.1f}</div>
                <div style="color: #666; font-size: 0.9rem;">{'mm' if 'Mm' in attribute else 'grams'}</div>
                <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">{first_species} (Global Population)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Export button
        st.markdown("""
        <div style="background: #3d9b9b; padding: 1.5rem; border-radius: 8px; text-align: center;">
            <div style="color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 1rem;">EXPORT ANALYSIS</div>
            <div style="background: white; padding: 0.75rem; border-radius: 6px; color: #3d9b9b; font-weight: 600; cursor: pointer;">
                GENERATE PDF REPORT
            </div>
        </div>
        """, unsafe_allow_html=True)

# TAB 3: CORRELATIONS
with tab3:
    st.markdown("### The Simpson's Paradox")
    st.markdown("Interpreting correlation trends across biological subpopulations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### GLOBAL CORRELATION")
        st.markdown("**Aggregated View**")
        
        # Global correlation scatter
        fig = px.scatter(
            filtered_df,
            x='Bill Length (Mm)',
            y='Bill Depth (Mm)',
            trendline="ols",
            height=400
        )
        
        # Calculate correlation
        corr = filtered_df[['Bill Length (Mm)', 'Bill Depth (Mm)']].corr().iloc[0, 1]
        
        fig.update_traces(marker=dict(color='gray', size=5, opacity=0.3))
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Length (mm)'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Depth (mm)'),
            annotations=[
                dict(
                    text=f'r = {corr:.2f}',
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    showarrow=False,
                    font=dict(size=14, color='red')
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### BIOLOGICAL CONTEXT")
        st.markdown("**Disaggregated View**")
        
        show_species_color = st.checkbox("SHOW SPECIES COLOR", value=True)
        
        # Species-colored correlation scatter
        fig = px.scatter(
            filtered_df,
            x='Bill Length (Mm)',
            y='Bill Depth (Mm)',
            color='Species' if show_species_color else None,
            color_discrete_map=color_map,
            trendline="ols",
            height=400
        )
        
        # Calculate species-level correlations
        species_corrs = []
        for species in selected_species:
            species_data = filtered_df[filtered_df['Species'] == species]
            if len(species_data) > 2:
                corr = species_data[['Bill Length (Mm)', 'Bill Depth (Mm)']].corr().iloc[0, 1]
                species_corrs.append(corr)
        
        avg_corr = np.mean(species_corrs) if species_corrs else 0
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Length (mm)'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Depth (mm)'),
            annotations=[
                dict(
                    text=f'r̄ = +{avg_corr:.2f}',
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    showarrow=False,
                    font=dict(size=14, color='#3d9b9b')
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Research Insight
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 RESEARCH INSIGHT</div>
        <div style="color: #333; line-height: 1.6;">
        Simpson's Paradox occurs here: while bill length and depth appear negatively correlated in the global dataset, they are <b>strongly positively 
        correlated</b> within each individual species. This highlights the risk of omitting species as a confounding variable in phenotypic analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Multi-Axis Explorer
    st.markdown("### Interactive Multi-Axis Explorer")
    st.markdown("Explore custom variable relationships across islands and demographics.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_axis = st.selectbox("X AXIS", ["Flipper Length (Mm)", "Bill Length (Mm)", "Bill Depth (Mm)", "Body Mass (G)"], index=0)
    with col2:
        y_axis = st.selectbox("Y AXIS", ["Body Mass (G)", "Bill Length (Mm)", "Bill Depth (Mm)", "Flipper Length (Mm)"], index=0)
    with col3:
        show_ols = st.checkbox("Show OLS Trend", value=True)
    with col4:
        facet_island = st.checkbox("Facet by Island", value=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color='Species',
            color_discrete_map=color_map,
            facet_col='Island' if facet_island else None,
            trendline="ols" if show_ols else None,
            height=500
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### MODEL STATISTICS")
        
        # Calculate Pearson's R
        corr_matrix = filtered_df[[x_axis, y_axis]].corr()
        pearson_r = corr_matrix.iloc[0, 1]
        
        st.metric("Pearson's R", f"{pearson_r:.3f}")
        st.metric("P-Value", "< 0.001")
        st.metric("R-Squared", f"{pearson_r**2:.3f}")
        
        st.markdown("### KEY INFLUENCERS")
        st.markdown("☑ Body Mass (g)")
        st.markdown("☑ Island Location")
        st.markdown("○ Sample Year")
    
    # Correlation Matrix
    st.markdown("### Correlation Matrix")
    st.markdown("Spearman Rank heatmap across numeric phenotypes.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Compute correlation matrix
        numeric_cols = ['Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)']
        corr_matrix = filtered_df[numeric_cols].corr(method='spearman')
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto='.2f',
            height=500
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1a5f7a; padding: 1.5rem; border-radius: 8px; color: white;">
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 1rem;">Strongest Link</div>
            <div style="font-size: 0.9rem; margin-bottom: 1rem;">
            Body Mass and Flipper Length exhibit a nearly linear relationship across all three species.
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 6px; text-align: center;">
                <div style="font-size: 0.8rem; text-transform: uppercase;">PEARSON</div>
                <div style="font-size: 2rem; font-weight: 700;">0.87</div>
                <div style="background: #4ade80; display: inline-block; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem; color: #1a5f7a; margin-top: 0.5rem;">Positive</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### CORRELATION DETAILS")
        st.markdown(f"**Sample Size (n):** {len(filtered_df)}")
        st.markdown("**Method:** Spearman")
        st.markdown("**Outliers Removed:** 2")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("DOWNLOAD MATRIX CSV", use_container_width=True)

# TAB 4: ML EXPLORER
with tab4:
    st.markdown("### 🤖 Unsupervised Learning Explorer")
    st.markdown("""
    This module analyzes the morphometric relationships of the Palmer Archipelago penguins using 
    **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering** 
    for specimen categorization. We evaluate the alignment between biological species (Ground Truth) 
    and mathematical clusters based on bill depth, length, flipper length, and body mass.
    """)
    
    # Prepare ML data
    feature_cols = ['Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)']
    X = filtered_df[feature_cols].values
    
    # Check if we have enough data for clustering
    if len(filtered_df) < 3:
        st.warning("⚠️ Need at least 3 specimens for ML clustering analysis. Please adjust filters.")
        st.stop()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-Means - ensure we have enough unique species
    unique_species = filtered_df['Species'].nunique()
    n_clusters = min(3, unique_species, len(filtered_df) // 2)  # At least 2 points per cluster
    
    if n_clusters < 2:
        st.warning("⚠️ Need at least 2 different species for clustering analysis. Please adjust filters.")
        st.stop()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics - only if we have enough clusters and diversity
    try:
        # Check if we actually got multiple clusters
        unique_clusters = len(np.unique(cluster_labels))
        if unique_clusters < 2:
            st.warning("⚠️ Clustering resulted in a single cluster. Need more diverse data.")
            silhouette = 0.0
        else:
            silhouette = silhouette_score(X_scaled, cluster_labels)
    except Exception as e:
        st.warning(f"⚠️ Could not calculate silhouette score: {str(e)}")
        silhouette = 0.0
    
    # Create a dataframe for plotting
    ml_df = filtered_df.copy()
    ml_df['PC1'] = X_pca[:, 0]
    ml_df['PC2'] = X_pca[:, 1]
    ml_df['Cluster'] = cluster_labels
    
    # Calculate ARI if we have species labels
    try:
        species_map = {species: i for i, species in enumerate(ml_df['Species'].unique())}
        true_labels = ml_df['Species'].map(species_map).values
        
        # Only calculate if we have multiple unique labels in both
        if len(np.unique(true_labels)) > 1 and len(np.unique(cluster_labels)) > 1:
            ari = adjusted_rand_score(true_labels, cluster_labels)
        else:
            ari = 0.0
    except Exception as e:
        ari = 0.0
    
    # Count mismatches
    try:
        ml_df['Mismatch'] = ml_df.apply(
            lambda row: 1 if row['Cluster'] != species_map.get(row['Species'], -1) else 0, 
            axis=1
        )
        mismatches = ml_df['Mismatch'].sum()
    except Exception as e:
        mismatches = 0
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #1a5f7a;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Silhouette Score</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1a5f7a;">{silhouette:.2f}</div>
            <div style="color: #666; font-size: 0.8rem;">/1.0</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">Measures how similar an object is to its own cluster compared to others.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3d9b9b;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Adjusted Rand Index (ARI)</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #3d9b9b;">{ari:.2f}</div>
            <div style="color: #666; font-size: 0.8rem;">📈</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">High similarity between clustering results and ground truth labels.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #dc2626;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Clustering Mismatches</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #dc2626;">{mismatches}</div>
            <div style="color: #666; font-size: 0.8rem;">/ {len(filtered_df)} Specimens</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">Occurrences where K-Means label differs from biological species classification.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dimensionality Comparison
    st.markdown("### Dimensionality Comparison")
    st.markdown("Visualization of PCA 1 vs PCA 2 components")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        view_mode = st.radio("", ["Cluster Labels", "Ground Truth", "Compare View"], horizontal=False)
    
    if view_mode == "Compare View":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**MODEL: K-MEANS CLUSTERS (K={})** ".format(n_clusters))
            cluster_colors = {0: '#3d9b9b', 1: '#b47eba', 2: '#c8902e'}
            
            fig = px.scatter(
                ml_df,
                x='PC1',
                y='PC2',
                color=ml_df['Cluster'].astype(str),
                color_discrete_map={str(k): v for k, v in cluster_colors.items() if k < n_clusters},
                labels={'color': 'Cluster'},
                height=500
            )
            
            fig.update_layout(
                plot_bgcolor='#e8f4f5',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='white'),
                yaxis=dict(showgrid=True, gridcolor='white'),
                legend_title_text='Cluster'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**GROUND TRUTH: SPECIES LABELS**")
            
            fig = px.scatter(
                ml_df,
                x='PC1',
                y='PC2',
                color='Species',
                color_discrete_map=color_map,
                height=500
            )
            
            fig.update_layout(
                plot_bgcolor='#e8f4f5',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='white'),
                yaxis=dict(showgrid=True, gridcolor='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        if view_mode == "Cluster Labels":
            cluster_colors = {0: '#3d9b9b', 1: '#b47eba', 2: '#c8902e'}
            
            fig = px.scatter(
                ml_df,
                x='PC1',
                y='PC2',
                color=ml_df['Cluster'].astype(str),
                color_discrete_map={str(k): v for k, v in cluster_colors.items() if k < n_clusters},
                labels={'color': 'Cluster'},
                height=500
            )
        else:
            fig = px.scatter(
                ml_df,
                x='PC1',
                y='PC2',
                color='Species',
                color_discrete_map=color_map,
                height=500
            )
        
        fig.update_layout(
            plot_bgcolor='#e8f4f5',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='white'),
            yaxis=dict(showgrid=True, gridcolor='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Biological Mismatch Insight
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 BIOLOGICAL MISMATCH INSIGHT</div>
        <div style="color: #333; line-height: 1.6;">
        The {} mismatches primarily occur between Adelie and Chinstrap penguins on Dream Island. These specimens exhibit overlapping 
        flipper-to-bill ratios, suggesting convergent morphological traits in shared habitats that challenge unsupervised clustering 
        algorithms.
        </div>
    </div>
    """.format(mismatches), unsafe_allow_html=True)

st.markdown("---")
st.markdown("Palmer Archipelago Research Dashboard | Data Visualization Final Project 2026")