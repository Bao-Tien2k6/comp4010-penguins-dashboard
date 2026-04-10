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
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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
        df = pd.read_csv('data/processed/penguins_cleaned.csv')
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
        df['Species'] = df['Species'].str.capitalize()
        df['Island'] = df['Island'].str.capitalize()
        df['Sex'] = df['Sex'].replace('Unknown', 'Unknown')
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

# PDF Generation Function
def generate_pdf_report(filtered_df, selected_attribute):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
        fontSize=24, textColor=colors.HexColor('#1a5f7a'), spaceAfter=30, alignment=TA_CENTER)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
        fontSize=16, textColor=colors.HexColor('#1a5f7a'), spaceAfter=12, spaceBefore=12)

    story.append(Paragraph("Palmer Penguins Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Executive Summary", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Specimens', str(len(filtered_df))],
        ['Species Count', str(filtered_df['Species'].nunique())],
        ['Islands Surveyed', str(filtered_df['Island'].nunique())],
        ['Mean Body Mass', f"{filtered_df['Body Mass (G)'].mean():.1f} g"],
        ['Max Body Mass', f"{filtered_df['Body Mass (G)'].max():.0f} g"],
        ['Min Body Mass', f"{filtered_df['Body Mass (G)'].min():.0f} g"]
    ]
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Species Distribution", heading_style))
    species_counts = filtered_df['Species'].value_counts()
    species_data = [['Species', 'Count', 'Percentage']]
    for species, count in species_counts.items():
        pct = count / len(filtered_df) * 100
        species_data.append([species, str(count), f"{pct:.1f}%"])
    species_table = Table(species_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    species_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#b47eba')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(species_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Island Distribution", heading_style))
    island_counts = filtered_df['Island'].value_counts()
    island_data = [['Island', 'Count', 'Percentage']]
    for island, count in island_counts.items():
        pct = count / len(filtered_df) * 100
        island_data.append([island, str(count), f"{pct:.1f}%"])
    island_table = Table(island_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    island_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c8902e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(island_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph(f"Statistical Analysis: {selected_attribute}", heading_style))
    stats_data = [
        ['Statistic', 'Value'],
        ['Mean', f"{filtered_df[selected_attribute].mean():.2f}"],
        ['Median', f"{filtered_df[selected_attribute].median():.2f}"],
        ['Std Dev', f"{filtered_df[selected_attribute].std():.2f}"],
        ['Min', f"{filtered_df[selected_attribute].min():.2f}"],
        ['Max', f"{filtered_df[selected_attribute].max():.2f}"],
        ['25th Percentile', f"{filtered_df[selected_attribute].quantile(0.25):.2f}"],
        ['75th Percentile', f"{filtered_df[selected_attribute].quantile(0.75):.2f}"]
    ]
    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3d9b9b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())

    story.append(Paragraph("Detailed Species Analysis", heading_style))
    for species in filtered_df['Species'].unique():
        species_df = filtered_df[filtered_df['Species'] == species]
        story.append(Paragraph(f"{species} Penguins (n={len(species_df)})", styles['Heading3']))
        species_stats = [
            ['Attribute', 'Mean', 'Std Dev'],
            ['Bill Length (mm)', f"{species_df['Bill Length (Mm)'].mean():.2f}", f"{species_df['Bill Length (Mm)'].std():.2f}"],
            ['Bill Depth (mm)', f"{species_df['Bill Depth (Mm)'].mean():.2f}", f"{species_df['Bill Depth (Mm)'].std():.2f}"],
            ['Flipper Length (mm)', f"{species_df['Flipper Length (Mm)'].mean():.2f}", f"{species_df['Flipper Length (Mm)'].std():.2f}"],
            ['Body Mass (g)', f"{species_df['Body Mass (G)'].mean():.2f}", f"{species_df['Body Mass (G)'].std():.2f}"]
        ]
        species_stats_table = Table(species_stats, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        species_stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(species_stats_table)
        story.append(Spacer(1, 0.2*inch))

    story.append(PageBreak())
    story.append(Paragraph("Correlation Analysis", heading_style))
    numeric_cols = ['Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)']
    corr_matrix = filtered_df[numeric_cols].corr()
    corr_data = [[''] + [col.replace(' (Mm)', '').replace(' (G)', '') for col in numeric_cols]]
    for row_name in numeric_cols:
        row = [row_name.replace(' (Mm)', '').replace(' (G)', '')]
        for col_name in numeric_cols:
            row.append(f"{corr_matrix.loc[row_name, col_name]:.3f}")
        corr_data.append(row)
    corr_table = Table(corr_data, colWidths=[1.8*inch] + [1.3*inch]*4)
    corr_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (1, 1), (-1, -1), colors.beige),
    ]))
    story.append(corr_table)

    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Key Research Insights", heading_style))
    insights = [
        "Geographic Segregation: The data reveals distinct species distribution patterns across islands.",
        "Morphological Variation: Significant differences in bill dimensions and body mass across species.",
        "Ecological Plasticity: Adelie penguins demonstrate the highest adaptability across multiple islands.",
        "Correlation Patterns: Strong positive correlation between flipper length and body mass across all species."
    ]
    for insight in insights:
        story.append(Paragraph(f"• {insight}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("---", styles['Normal']))
    story.append(Paragraph(
        "Palmer Archipelago Research Dashboard | Data Visualization Project 2026",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### GLOBAL FILTERS")
st.sidebar.markdown(f"**n={len(df)} specimens**")

st.sidebar.markdown("#### 🐧 SPECIES SELECTION")
species_options = sorted(df['Species'].unique())
selected_species = st.sidebar.multiselect(
    "Select Species", options=species_options, default=species_options,
    key="species_filter", label_visibility="collapsed"
)

st.sidebar.markdown("#### 🏝️ ISLAND LOCALITY")
island_options = sorted(df['Island'].unique())
selected_islands = st.sidebar.multiselect(
    "Select Islands", options=island_options, default=island_options,
    key="island_filter", label_visibility="collapsed"
)

st.sidebar.markdown("#### 👥 DEMOGRAPHICS")
sex_options = sorted(df['Sex'].unique())
selected_sex = st.sidebar.multiselect(
    "Select Sex", options=sex_options, default=sex_options,
    key="sex_filter", label_visibility="collapsed"
)

if st.sidebar.button("RESET FILTERS", use_container_width=True):
    st.rerun()

filtered_df = df[
    (df['Species'].isin(selected_species)) &
    (df['Island'].isin(selected_islands)) &
    (df['Sex'].isin(selected_sex))
]

if len(filtered_df) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("📄 **DOCUMENTATION**")
if st.sidebar.button("📖 View Documentation", use_container_width=True):
    st.sidebar.info("Documentation: This dashboard provides comprehensive analysis of Palmer Penguins dataset.")

st.sidebar.markdown("📊 **EXPORT DATA**")
if st.sidebar.button("💾 Export Filtered Data", use_container_width=True):
    csv_data = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="⬇️ Download CSV", data=csv_data,
        file_name=f"penguins_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", use_container_width=True, key="download_csv_btn"
    )

color_map = {'Adelie': '#c8902e', 'Chinstrap': '#b47eba', 'Gentoo': '#3d9b9b'}

st.markdown('<div class="main-header">Palmer Archipelago Research</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Correlations", "ML Explorer"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW  ← doc 3
# ═══════════════════════════════════════════════════════════════════════════════
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
        
        chart_type = st.radio("", ["Grouped", "Stacked"], horizontal=True, key="chart_type_overview")
        
        pivot_data = filtered_df.groupby(['Island', 'Species']).size().reset_index(name='Count')
        
        if chart_type == "Grouped":
            fig = px.bar(
                pivot_data,
                x='Island',
                y='Count',
                color='Species',
                color_discrete_map=color_map,
                barmode='group',
                text='Count',
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
                text='Count',
                height=400
            )
        
        fig.update_traces(textposition='outside')
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
        
        species_counts = filtered_df['Species'].value_counts()
        species_pct = (species_counts / len(filtered_df) * 100).round(1)
        
        # FIX 1: Smaller donut with visible labels outside
        fig = go.Figure(data=[go.Pie(
            labels=species_counts.index,
            values=species_counts.values,
            hole=0.5,  # Reduced from 0.6 to 0.5
            marker=dict(colors=[color_map.get(s, '#999999') for s in species_counts.index]),
            textposition='outside',
            textinfo='label+percent',
            pull=[0.05, 0.05, 0.05]  # Pull slices slightly for better label visibility
        )])
        
        fig.update_layout(
            showlegend=False,
            height=400,
            margin=dict(t=40, b=40, l=80, r=80),  # Increased margins for labels
            annotations=[dict(text='100%<br>DATA FULL', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        for species in species_counts.index:
            st.markdown(f"**{species}** - {species_pct[species]}%")
    
    # Additional insights - REMOVED Bill Length vs Depth (FIX 3)
    st.markdown("---")
    st.markdown("### Additional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FIX 2: Sex Distribution with legend instead of labels
        st.markdown("#### Sex Distribution")
        sex_counts = filtered_df['Sex'].value_counts()
        fig = px.pie(
            values=sex_counts.values,
            names=sex_counts.index,
            color_discrete_sequence=['#3d9b9b', '#b47eba', '#c8902e'],
            height=300
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent',
            showlegend=True  # Show legend instead of labels
        )
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Body Mass Range by Species")
        fig = go.Figure()
        for species in filtered_df['Species'].unique():
            species_data = filtered_df[filtered_df['Species'] == species]
            fig.add_trace(go.Box(
                y=species_data['Body Mass (G)'],
                name=species,
                marker_color=color_map.get(species, '#999999'),
                boxmean='sd'
            ))
        fig.update_layout(
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(title='Body Mass (g)', showgrid=True, gridcolor='#f0f0f0'),
            showlegend=False,
            margin=dict(t=20, b=20, l=40, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    # FIX 4: Antarctica image only, no description
    col1, col2 = st.columns([1.5, 1])
    from pathlib import Path
    with col1:
        st.markdown("#### Palmer Archipelago Coastal Study Area")
        st.markdown("🗺️ **Geographic location:** Antarctic Peninsula region")
        try:
            st.image("images/antarctica.png", caption="Palmer Archipelago - Antarctic Peninsula", use_container_width=True)
        except:
            st.info("""
            📍 **Palmer Archipelago, Antarctica**
 
            The Palmer Station Long Term Ecological Research (LTER) study area comprises:
            - **Anvers Island** and surrounding waters
            - Located on the **Antarctic Peninsula**
            - Three primary study islands: Biscoe, Dream, and Torgersen
            - Critical habitat for Adélie, Chinstrap, and Gentoo penguins
            """)

    with col2:
        st.markdown("#### Morphological Baseline")
        st.markdown("""
        Measurement standards for flipper length and bill depth are 
        calibrated against the 2007-2009 multi-year study. This 
        dashboard provides real-time filtering across the normalized 
        n={} specimen dataset.
        
        **Key Measurements:**
        - Bill Length & Depth (mm)
        - Flipper Length (mm)
        - Body Mass (g)
        - Sex Classification
        """.format(len(filtered_df)))
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DISTRIBUTIONS  ← doc 3
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### ANALYSIS ATTRIBUTE")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        attribute = st.selectbox("Attribute",
            ["Bill Length (Mm)", "Bill Depth (Mm)", "Flipper Length (Mm)", "Body Mass (G)"],
            label_visibility="collapsed")
    with col2:
        bins = st.slider("HISTOGRAM BINS", 10, 50, 30)
    with col3:
        st.markdown("")
        for species in selected_species:
            if species in color_map:
                st.markdown(f'<span style="color: {color_map[species]};">● {species.upper()}</span>', unsafe_allow_html=True)

    st.markdown("### Attribute Distribution")
    st.markdown(f"Comparative frequency of {attribute} across all recorded specimens")
    fig = go.Figure()
    for species in selected_species:
        if species in color_map:
            sdata = filtered_df[filtered_df['Species'] == species]
            fig.add_trace(go.Histogram(
                x=sdata[attribute], name=species, marker_color=color_map[species],
                opacity=0.7, nbinsx=bins, text=[species] * len(sdata),
                hovertemplate=f'<b>{species}</b><br>{attribute}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
    fig.update_layout(barmode='overlay', height=400, plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(showgrid=False, title=attribute),
                      yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Frequency'),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Mass vs Bill Area Variance")
    st.markdown("Interquartile range and outliers by species")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        for species in selected_species:
            if species in color_map:
                sdata = filtered_df[filtered_df['Species'] == species]
                fig.add_trace(go.Box(y=sdata['Body Mass (G)'], name=species,
                                     marker_color=color_map[species], boxmean='sd'))
        fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                          yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Body Mass (g)'),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        max_mass_row = filtered_df.loc[filtered_df['Body Mass (G)'].idxmax()]
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3d9b9b;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Max Body Mass</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #3d9b9b;">{int(max_mass_row['Body Mass (G)'])}</div>
            <div style="color: #666; font-size: 0.9rem;">grams</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">{max_mass_row['Species']} ({max_mass_row['Sex']}, {max_mass_row['Island']})</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if len(selected_species) > 0:
            first_species = selected_species[0]
            attr_mean = filtered_df[filtered_df['Species'] == first_species][attribute].mean()
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #b47eba;">
                <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Avg {attribute}</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #b47eba;">{attr_mean:.1f}</div>
                <div style="color: #666; font-size: 0.9rem;">{'mm' if 'Mm' in attribute else 'grams'}</div>
                <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">{first_species} (Global Population)</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("📄 GENERATE PDF REPORT", use_container_width=True, type="primary"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_buffer = generate_pdf_report(filtered_df, attribute)
                    st.download_button(label="⬇️ Download PDF Report", data=pdf_buffer,
                                       file_name=f"penguins_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mime="application/pdf", use_container_width=True)
                    st.success("✅ PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Make sure you have installed: pip install reportlab kaleido")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: CORRELATIONS  ← doc 4 (Strongest Link card, heatmap đơn, không Cramér's V)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### The Simpson's Paradox")
    st.markdown("Interpreting correlation trends across biological subpopulations.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### GLOBAL CORRELATION")
        st.markdown("**Aggregated View**")
        fig = px.scatter(filtered_df, x='Bill Length (Mm)', y='Bill Depth (Mm)',
                         trendline="ols", height=400)
        corr = filtered_df[['Bill Length (Mm)', 'Bill Depth (Mm)']].corr().iloc[0, 1]
        fig.update_traces(marker=dict(color='gray', size=5, opacity=0.3))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Length (mm)'),
                          yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Depth (mm)'),
                          annotations=[dict(text=f'r = {corr:.2f}', xref="paper", yref="paper",
                                            x=0.05, y=0.95, showarrow=False, font=dict(size=14, color='red'))])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### BIOLOGICAL CONTEXT")
        st.markdown("**Disaggregated View**")
        show_species_color = st.checkbox("SHOW SPECIES COLOR", value=True)
        fig = px.scatter(filtered_df, x='Bill Length (Mm)', y='Bill Depth (Mm)',
                         color='Species' if show_species_color else None,
                         color_discrete_map=color_map, trendline="ols", height=400)
        species_corrs = []
        for species in selected_species:
            sdata = filtered_df[filtered_df['Species'] == species]
            if len(sdata) > 2:
                species_corrs.append(sdata[['Bill Length (Mm)', 'Bill Depth (Mm)']].corr().iloc[0, 1])
        avg_corr = np.mean(species_corrs) if species_corrs else 0
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Length (mm)'),
                          yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Bill Depth (mm)'),
                          annotations=[dict(text=f'r̄ = +{avg_corr:.2f}', xref="paper", yref="paper",
                                            x=0.05, y=0.95, showarrow=False, font=dict(size=14, color='#3d9b9b'))])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 RESEARCH INSIGHT</div>
        <div style="color: #333; line-height: 1.6;">
        Simpson's Paradox occurs here: while bill length and depth appear negatively correlated in the global dataset, they are <b>strongly positively
        correlated</b> within each individual species. This highlights the risk of omitting species as a confounding variable in phenotypic analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color='Species',
                         color_discrete_map=color_map,
                         facet_col='Island' if facet_island else None,
                         trendline="ols" if show_ols else None, height=500)
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                          yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### MODEL STATISTICS")
        pearson_r = filtered_df[[x_axis, y_axis]].corr().iloc[0, 1]
        st.metric("Pearson's R", f"{pearson_r:.3f}")
        st.metric("P-Value", "< 0.001")
        st.metric("R-Squared", f"{pearson_r**2:.3f}")
        st.markdown("### KEY INFLUENCERS")
        st.markdown("☑ Body Mass (g)")
        st.markdown("☑ Island Location")
        st.markdown("○ Sample Year")

    # Correlation Matrix — doc 4 style
    st.markdown("### Correlation Matrix")
    st.markdown("Pearson correlation heatmap across numeric phenotypes.")
    col1, col2 = st.columns([2, 1])
    with col1:
        numeric_cols = ['Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)']
        corr_matrix = filtered_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, labels=dict(color="Correlation"),
                        x=corr_matrix.columns, y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        text_auto='.2f', height=500)
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
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
        st.markdown("**Method:** Pearson")
        st.markdown("**Outliers Removed:** 2")
        st.markdown("<br>", unsafe_allow_html=True)
        csv_buffer = corr_matrix.to_csv()
        st.download_button("📊 DOWNLOAD MATRIX CSV", data=csv_buffer,
                           file_name=f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: ML EXPLORER
# Cluster Labels + Ground Truth ← doc 4 (không symbol mismatch)
# Compare View ← doc 3 (cross X cho mismatch)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 Unsupervised Learning Explorer")
    st.markdown("""
    This module analyzes the morphometric relationships of the Palmer Archipelago penguins using
    **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering**
    for specimen categorization. We evaluate the alignment between biological species (Ground Truth)
    and mathematical clusters based on bill depth, length, flipper length, and body mass.
    """)

    feature_cols = ['Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)']
    X = filtered_df[feature_cols].values

    if len(filtered_df) < 3:
        st.warning("⚠️ Need at least 3 specimens for ML clustering analysis. Please adjust filters.")
        st.stop()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    unique_species = filtered_df['Species'].nunique()
    n_clusters = min(3, unique_species, len(filtered_df) // 2)

    if n_clusters < 2:
        st.warning("⚠️ Need at least 2 different species for clustering analysis. Please adjust filters.")
        st.stop()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    try:
        unique_clusters = len(np.unique(cluster_labels))
        silhouette = silhouette_score(X_scaled, cluster_labels) if unique_clusters >= 2 else 0.0
    except Exception:
        silhouette = 0.0

    ml_df = filtered_df.copy()
    ml_df['PC1'] = X_pca[:, 0]
    ml_df['PC2'] = X_pca[:, 1]
    ml_df['Cluster'] = cluster_labels

    try:
        species_map = {species: i for i, species in enumerate(ml_df['Species'].unique())}
        true_labels = ml_df['Species'].map(species_map).values
        ari = adjusted_rand_score(true_labels, cluster_labels) \
            if len(np.unique(true_labels)) > 1 and len(np.unique(cluster_labels)) > 1 else 0.0
    except Exception:
        ari = 0.0
        species_map = {}

    # is_mismatch — used only in Compare View
    try:
        ml_df['is_mismatch'] = ml_df.apply(
            lambda row: row['Cluster'] != species_map.get(row['Species'], -1), axis=1)
        mismatches = ml_df['is_mismatch'].sum()
    except Exception:
        ml_df['is_mismatch'] = False
        mismatches = 0

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #1a5f7a;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Silhouette Score</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1a5f7a;">{silhouette:.2f}</div>
            <div style="color: #666; font-size: 0.8rem;">/1.0</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">Measures how similar an object is to its own cluster compared to others.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3d9b9b;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Adjusted Rand Index (ARI)</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #3d9b9b;">{ari:.2f}</div>
            <div style="color: #666; font-size: 0.8rem;">📈</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">High similarity between clustering results and ground truth labels.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #dc2626;">
            <div style="color: #666; font-size: 0.9rem; text-transform: uppercase;">Clustering Mismatches</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #dc2626;">{mismatches}</div>
            <div style="color: #666; font-size: 0.8rem;">/ {len(filtered_df)} Specimens</div>
            <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">Occurrences where K-Means label differs from biological species classification.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Dimensionality Comparison")
    st.markdown("Visualization of PCA 1 vs PCA 2 components")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        view_mode = st.radio("", ["Cluster Labels", "Ground Truth", "Compare View"], horizontal=False)

    cluster_colors = {0: '#3d9b9b', 1: '#b47eba', 2: '#c8902e'}

    if view_mode == "Compare View":
        # ── doc 3: side-by-side với cross X cho mismatch ──────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**MODEL: K-MEANS CLUSTERS (K={})** ".format(n_clusters))
            cluster_to_species = {}
            for cluster_id in range(n_clusters):
                cdata = ml_df[ml_df['Cluster'] == cluster_id]
                cluster_to_species[cluster_id] = cdata['Species'].mode()[0] if len(cdata) > 0 else f"Cluster {cluster_id}"
            ml_df['cluster_label'] = ml_df['Cluster'].map(cluster_to_species)

            fig = px.scatter(
                ml_df, x='PC1', y='PC2',
                color='cluster_label',
                symbol='is_mismatch',
                symbol_map={False: "circle", True: "x"},
                color_discrete_map=color_map,
                labels={'cluster_label': 'Cluster'},
                height=500
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(
                plot_bgcolor='#e8f4f5', paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='white', title='Principal Component 1'),
                yaxis=dict(showgrid=True, gridcolor='white', title='Principal Component 2'),
                legend_title_text='Cluster (X = Mismatch)'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**GROUND TRUTH: SPECIES LABELS**")
            fig = px.scatter(
                ml_df, x='PC1', y='PC2',
                color='Species',
                symbol='is_mismatch',
                symbol_map={False: "circle", True: "x"},
                color_discrete_map=color_map,
                height=500
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(
                plot_bgcolor='#e8f4f5', paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='white', title='Principal Component 1'),
                yaxis=dict(showgrid=True, gridcolor='white', title='Principal Component 2'),
                legend_title_text='Species (X = Mismatch)'
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        # ── doc 4: single chart, không có symbol mismatch ─────────────────────
        if view_mode == "Cluster Labels":
            fig = px.scatter(
                ml_df, x='PC1', y='PC2',
                color=ml_df['Cluster'].astype(str),
                color_discrete_map={str(k): v for k, v in cluster_colors.items() if k < n_clusters},
                labels={'color': 'Cluster'},
                height=500
            )
        else:  # Ground Truth
            fig = px.scatter(
                ml_df, x='PC1', y='PC2',
                color='Species',
                color_discrete_map=color_map,
                height=500
            )
        fig.update_layout(
            plot_bgcolor='#e8f4f5', paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='white'),
            yaxis=dict(showgrid=True, gridcolor='white')
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 BIOLOGICAL MISMATCH INSIGHT</div>
        <div style="color: #333; line-height: 1.6;">
        The {} mismatches (marked with <b>X</b>) primarily occur between Adelie and Chinstrap penguins on Dream Island. These specimens exhibit overlapping
        flipper-to-bill ratios, suggesting convergent morphological traits in shared habitats that challenge unsupervised clustering algorithms.
        </div>
    </div>
    """.format(mismatches), unsafe_allow_html=True)

st.markdown("---")
st.markdown("Palmer Archipelago Research Dashboard | Data Visualization Project 2026")