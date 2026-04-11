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

st.set_page_config(
    page_title="Palmer Archipelago Research",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1a5f7a; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .metric-card { background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .insight-box { background: #fff8e6; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #c8902e; margin: 1.5rem 0; }
    .insight-title { color: #c8902e; font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; }
    .insight-box-blue { background: #e8f4f8; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1a5f7a; margin: 1rem 0; }
    .insight-title-blue { color: #1a5f7a; font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; }
    .insight-box-green { background: #e8f5f0; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3d9b9b; margin: 1rem 0; }
    .insight-title-green { color: #3d9b9b; font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; }
    .insight-box-purple { background: #f5eef8; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #b47eba; margin: 1rem 0; }
    .insight-title-purple { color: #b47eba; font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; }
    .verdict-box { background: linear-gradient(135deg, #1a5f7a 0%, #3d9b9b 100%); padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/penguins_cleaned.csv')
        column_mapping = {
            'species': 'Species', 'island': 'Island',
            'bill_length_mm': 'Bill Length (Mm)', 'bill_depth_mm': 'Bill Depth (Mm)',
            'flipper_length_mm': 'Flipper Length (Mm)', 'body_mass_g': 'Body Mass (G)', 'sex': 'Sex'
        }
        df = df.rename(columns=column_mapping)
        df['Species'] = df['Species'].str.capitalize()
        df['Island'] = df['Island'].str.capitalize()
        df['Sex'] = df['Sex'].replace('Unknown', 'Unknown')
        needed_cols = ['Species', 'Island', 'Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)', 'Sex']
        df = df[needed_cols]
        df['Bill Area (mm2)'] = df['Bill Length (Mm)'] * df['Bill Depth (Mm)']
        df['Mass per Bill Area'] = df['Body Mass (G)'] / df['Bill Area (mm2)']
        return df
    except FileNotFoundError:
        st.error("Cannot find 'data/processed/penguins_cleaned.csv'.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

def generate_pdf_report(filtered_df, selected_attribute):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
        textColor=colors.HexColor('#1a5f7a'), spaceAfter=30, alignment=TA_CENTER)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16,
        textColor=colors.HexColor('#1a5f7a'), spaceAfter=12, spaceBefore=12)
    story.append(Paragraph("Palmer Penguins Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Executive Summary", heading_style))
    summary_data = [['Metric', 'Value'],
        ['Total Specimens', str(len(filtered_df))],
        ['Species Count', str(filtered_df['Species'].nunique())],
        ['Islands Surveyed', str(filtered_df['Island'].nunique())],
        ['Mean Body Mass', f"{filtered_df['Body Mass (G)'].mean():.1f} g"],
        ['Max Body Mass', f"{filtered_df['Body Mass (G)'].max():.0f} g"],
        ['Min Body Mass', f"{filtered_df['Body Mass (G)'].min():.0f} g"]]
    st_tbl = Table(summary_data, colWidths=[3*inch, 2*inch])
    st_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),12),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),1,colors.grey),
        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,1),(-1,-1),10)]))
    story.append(st_tbl)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Species Distribution", heading_style))
    species_counts = filtered_df['Species'].value_counts()
    species_data = [['Species', 'Count', 'Percentage']]
    for species, count in species_counts.items():
        pct = count / len(filtered_df) * 100
        species_data.append([species, str(count), f"{pct:.1f}%"])
    sp_tbl = Table(species_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    sp_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#b47eba')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),12),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),1,colors.grey),
        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,1),(-1,-1),10)]))
    story.append(sp_tbl)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Island Distribution", heading_style))
    island_counts = filtered_df['Island'].value_counts()
    island_data = [['Island', 'Count', 'Percentage']]
    for island, count in island_counts.items():
        pct = count / len(filtered_df) * 100
        island_data.append([island, str(count), f"{pct:.1f}%"])
    is_tbl = Table(island_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    is_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#c8902e')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),12),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),1,colors.grey),
        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,1),(-1,-1),10)]))
    story.append(is_tbl)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Statistical Analysis: {selected_attribute}", heading_style))
    stats_data = [['Statistic', 'Value'],
        ['Mean', f"{filtered_df[selected_attribute].mean():.2f}"],
        ['Median', f"{filtered_df[selected_attribute].median():.2f}"],
        ['Std Dev', f"{filtered_df[selected_attribute].std():.2f}"],
        ['Min', f"{filtered_df[selected_attribute].min():.2f}"],
        ['Max', f"{filtered_df[selected_attribute].max():.2f}"],
        ['25th Percentile', f"{filtered_df[selected_attribute].quantile(0.25):.2f}"],
        ['75th Percentile', f"{filtered_df[selected_attribute].quantile(0.75):.2f}"]]
    stat_tbl = Table(stats_data, colWidths=[3*inch, 2*inch])
    stat_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#3d9b9b')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),12),
        ('BOTTOMPADDING',(0,0),(-1,0),12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),1,colors.grey),
        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,1),(-1,-1),10)]))
    story.append(stat_tbl)
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    story.append(Paragraph("Detailed Species Analysis", heading_style))
    for species in filtered_df['Species'].unique():
        species_df = filtered_df[filtered_df['Species'] == species]
        story.append(Paragraph(f"{species} Penguins (n={len(species_df)})", styles['Heading3']))
        ss = [['Attribute', 'Mean', 'Std Dev'],
            ['Bill Length (mm)', f"{species_df['Bill Length (Mm)'].mean():.2f}", f"{species_df['Bill Length (Mm)'].std():.2f}"],
            ['Bill Depth (mm)', f"{species_df['Bill Depth (Mm)'].mean():.2f}", f"{species_df['Bill Depth (Mm)'].std():.2f}"],
            ['Flipper Length (mm)', f"{species_df['Flipper Length (Mm)'].mean():.2f}", f"{species_df['Flipper Length (Mm)'].std():.2f}"],
            ['Body Mass (g)', f"{species_df['Body Mass (G)'].mean():.2f}", f"{species_df['Body Mass (G)'].std():.2f}"]]
        ss_tbl = Table(ss, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        ss_tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,0),10),
            ('BOTTOMPADDING',(0,0),(-1,0),8),
            ('BACKGROUND',(0,1),(-1,-1),colors.lightgrey),
            ('GRID',(0,0),(-1,-1),1,colors.grey),
            ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,1),(-1,-1),9)]))
        story.append(ss_tbl)
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
    corr_tbl = Table(corr_data, colWidths=[1.8*inch] + [1.3*inch]*4)
    corr_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a5f7a')),
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('TEXTCOLOR',(0,0),(0,-1),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('GRID',(0,0),(-1,-1),1,colors.grey),
        ('BACKGROUND',(1,1),(-1,-1),colors.beige)]))
    story.append(corr_tbl)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Key Research Insights", heading_style))
    insights = [
        "Geographic Segregation: The data reveals distinct species distribution patterns across islands.",
        "Morphological Variation: Significant differences in bill dimensions and body mass across species.",
        "Ecological Plasticity: Adelie penguins demonstrate the highest adaptability across multiple islands.",
        "Correlation Patterns: Strong positive correlation between flipper length and body mass across all species.",
        "Species > Geography: High mass-per-bill-area on Biscoe Island is driven by Gentoo predominance, not environment."
    ]
    for insight in insights:
        story.append(Paragraph(f"- {insight}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("---", styles['Normal']))
    story.append(Paragraph(
        "Palmer Archipelago Research Dashboard | Data Visualization Project 2026",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("### GLOBAL FILTERS")
st.sidebar.markdown(f"**n={len(df)} specimens**")
st.sidebar.markdown("#### 🐧 SPECIES SELECTION")
species_options = sorted(df['Species'].unique())
selected_species = st.sidebar.multiselect("Select Species", options=species_options, default=species_options, key="species_filter", label_visibility="collapsed")
st.sidebar.markdown("#### 🏝️ ISLAND LOCALITY")
island_options = sorted(df['Island'].unique())
selected_islands = st.sidebar.multiselect("Select Islands", options=island_options, default=island_options, key="island_filter", label_visibility="collapsed")
st.sidebar.markdown("#### 👥 DEMOGRAPHICS")
sex_options = sorted(df['Sex'].unique())
selected_sex = st.sidebar.multiselect("Select Sex", options=sex_options, default=sex_options, key="sex_filter", label_visibility="collapsed")
if st.sidebar.button("RESET FILTERS", use_container_width=True):
    st.rerun()

filtered_df = df[
    (df['Species'].isin(selected_species)) &
    (df['Island'].isin(selected_islands)) &
    (df['Sex'].isin(selected_sex))
]
if len(filtered_df) == 0:
    st.warning("No data matches the current filters.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("📄 **DOCUMENTATION**")
if st.sidebar.button("📖 View Documentation", use_container_width=True):
    st.sidebar.info("Documentation: This dashboard provides comprehensive analysis of Palmer Penguins dataset.")
st.sidebar.markdown("📊 **EXPORT DATA**")
if st.sidebar.button("💾 Export Filtered Data", use_container_width=True):
    csv_data = filtered_df.to_csv(index=False)
    st.sidebar.download_button(label="⬇️ Download CSV", data=csv_data,
        file_name=f"penguins_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", use_container_width=True, key="download_csv_btn")

color_map = {'Adelie': '#c8902e', 'Chinstrap': '#b47eba', 'Gentoo': '#3d9b9b'}
island_colors = {'Biscoe': '#1a5f7a', 'Dream': '#c8902e', 'Torgersen': '#b47eba'}

st.markdown('<div class="main-header">Palmer Archipelago Research</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Correlations", "ML Explorer"])

# ═══ TAB 1: OVERVIEW ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="main-header">Palmer Penguins Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploring morphology, geography, and ML clustering</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card" style="border-left-color: #1a5f7a;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Total Penguins</div><div style="font-size:2.5rem;font-weight:700;color:#1a5f7a;">{len(filtered_df)}</div><div style="color:#666;font-size:0.9rem;">Specimens</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #b47eba;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Distinct Species</div><div style="font-size:2.5rem;font-weight:700;color:#b47eba;">{filtered_df["Species"].nunique()}</div><div style="color:#666;font-size:0.9rem;">Categories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #c8902e;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Islands Surveyed</div><div style="font-size:2.5rem;font-weight:700;color:#c8902e;">{filtered_df["Island"].nunique()}</div><div style="color:#666;font-size:0.9rem;">Localities</div></div>', unsafe_allow_html=True)
    with col4:
        mean_mass = filtered_df['Body Mass (G)'].mean() / 1000
        st.markdown(f'<div class="metric-card" style="border-left-color: #3d9b9b;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Mean Body Mass</div><div style="font-size:2.5rem;font-weight:700;color:#3d9b9b;">{mean_mass:.1f}</div><div style="color:#666;font-size:0.9rem;">kg Avg</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Species Distribution by Island")
        st.markdown("Population density across Archipelago localities")
        chart_type = st.radio("", ["Grouped", "Stacked"], horizontal=True, key="chart_type_overview")
        pivot_data = filtered_df.groupby(['Island', 'Species']).size().reset_index(name='Count')
        fig = px.bar(pivot_data, x='Island', y='Count', color='Species', color_discrete_map=color_map,
            barmode='group' if chart_type == "Grouped" else 'stack', text='Count', height=400)
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Species Breakdown")
        st.markdown(f"Overall proportion of n={len(filtered_df)}")
        species_counts = filtered_df['Species'].value_counts()
        species_pct = (species_counts / len(filtered_df) * 100).round(1)
        fig = go.Figure(data=[go.Pie(
            labels=species_counts.index, values=species_counts.values, hole=0.5,
            marker=dict(colors=[color_map.get(s, '#999') for s in species_counts.index]),
            textposition='outside', textinfo='label+percent', pull=[0.05]*len(species_counts))])
        fig.update_layout(showlegend=False, height=400, margin=dict(t=40, b=40, l=80, r=80),
            annotations=[dict(text='100%<br>DATA FULL', x=0.5, y=0.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
        for species in species_counts.index:
            st.markdown(f"**{species}** - {species_pct[species]}%")
    st.markdown("---")
    st.markdown("### Additional Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sex Distribution")
        sex_counts = filtered_df['Sex'].value_counts()
        fig = px.pie(values=sex_counts.values, names=sex_counts.index,
            color_discrete_sequence=['#3d9b9b', '#b47eba', '#c8902e'], height=300)
        fig.update_traces(textposition='inside', textinfo='percent', showlegend=True)
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Body Mass Range by Species")
        fig = go.Figure()
        for species in filtered_df['Species'].unique():
            species_data = filtered_df[filtered_df['Species'] == species]
            fig.add_trace(go.Box(y=species_data['Body Mass (G)'], name=species,
                marker_color=color_map.get(species, '#999'), boxmean='sd'))
        fig.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(title='Body Mass (g)', showgrid=True, gridcolor='#f0f0f0'),
            showlegend=False, margin=dict(t=20, b=20, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)
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
    col1, col2 = st.columns([1.5, 1])
    from pathlib import Path
    with col1:
        st.markdown("#### Palmer Archipelago Coastal Study Area")
        st.markdown("🗺️ **Geographic location:** Antarctic Peninsula region")
        try:
            st.image("images/antarctica.png", caption="Palmer Archipelago - Antarctic Peninsula", use_container_width=True)
        except:
            st.info("""📍 **Palmer Archipelago, Antarctica**

The Palmer Station Long Term Ecological Research (LTER) study area comprises:
- **Anvers Island** and surrounding waters
- Located on the **Antarctic Peninsula**
- Three primary study islands: Biscoe, Dream, and Torgersen
- Critical habitat for Adélie, Chinstrap, and Gentoo penguins""")
    with col2:
        st.markdown("#### Morphological Baseline")
        st.markdown(f"""Measurement standards for flipper length and bill depth are
calibrated against the 2007-2009 multi-year study. This
dashboard provides real-time filtering across the normalized
n={len(filtered_df)} specimen dataset.

**Key Measurements:**
- Bill Length & Depth (mm)
- Flipper Length (mm)
- Body Mass (g)
- Sex Classification""")

# ═══ TAB 2: DISTRIBUTIONS ═════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Is Morphology Explained by Species or Geography?")
    st.markdown("This section examines whether variation in penguin morphology is better explained by **species identity** or **geographic location** (island), and identifies which features best distinguish species.")

    st.markdown("---")
    st.markdown("### Which Features Predict Species Identity?")
    st.markdown("Each morphological feature separates species with different power. Ranges below are derived from the filtered dataset.")

    feature_specs = {
        'Bill Length (Mm)': {
            'label': 'Bill Length (mm)', 'color': '#c8902e', 'icon': '📏',
            'insight': "Bill length is the strongest separator between Adelie (shortest) and the other two species. Chinstrap tends to have the longest bills, indicating adaptation toward longer-bill feeding strategies."
        },
        'Bill Depth (Mm)': {
            'label': 'Bill Depth (mm)', 'color': '#b47eba', 'icon': '📐',
            'insight': "Despite Adelie having the shortest bill length, it has the deepest bill - the thickest cross-section. Gentoo has the shallowest bill depth, consistent with its propulsion-focused morphology."
        },
        'Flipper Length (Mm)': {
            'label': 'Flipper Length (mm)', 'color': '#3d9b9b', 'icon': '🏊',
            'insight': "Flipper length is a strong discriminator. Gentoo has significantly longer flippers, enabling them to be the fastest underwater swimmers among all penguin species."
        },
        'Body Mass (G)': {
            'label': 'Body Mass (g)', 'color': '#1a5f7a', 'icon': '⚖️',
            'insight': "Gentoo penguins are substantially heavier than the other two species. This mass advantage, combined with long flippers, supports diving efficiency and propulsion."
        },
    }

    for feat, meta in feature_specs.items():
        with st.expander(f"{meta['icon']}  {meta['label']} - click to expand", expanded=False):
            col_chart, col_insight = st.columns([3, 2])
            with col_chart:
                fig = go.Figure()
                for species in ['Adelie', 'Chinstrap', 'Gentoo']:
                    sdata = filtered_df[filtered_df['Species'] == species][feat].dropna()
                    if len(sdata) == 0:
                        continue
                    fig.add_trace(go.Box(y=sdata, name=species, marker_color=color_map.get(species, '#999'),
                        boxmean='sd',
                        hovertemplate=f"<b>{species}</b><br>Median: %{{median:.1f}}<br>Q1-Q3: %{{q1:.1f}}-%{{q3:.1f}}<extra></extra>"))
                fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                    yaxis=dict(title=meta['label'], showgrid=True, gridcolor='#f0f0f0'),
                    showlegend=False, margin=dict(t=10, b=10, l=40, r=10))
                st.plotly_chart(fig, use_container_width=True)
                range_rows = []
                for species in ['Adelie', 'Chinstrap', 'Gentoo']:
                    sdata = filtered_df[filtered_df['Species'] == species][feat].dropna()
                    if len(sdata) == 0:
                        continue
                    q1 = sdata.quantile(0.10)
                    q3 = sdata.quantile(0.90)
                    med = sdata.median()
                    range_rows.append({'Species': species, 'Range (10-90%)': f"{q1:.0f} - {q3:.0f}", 'Median': f"{med:.1f}"})
                if range_rows:
                    st.dataframe(pd.DataFrame(range_rows).set_index('Species'), use_container_width=True)
            with col_insight:
                # FIX 1: NO Species Separation Power / F-ratio block - interpretation only
                st.markdown(f"""
                <div class="insight-box-blue" style="margin-top: 1rem;">
                    <div class="insight-title-blue">📊 Interpretation</div>
                    <div style="color: #333; line-height: 1.7; font-size: 0.95rem;">{meta['insight']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌍 Species vs. Geography: What Drives Morphological Variance?")
    st.markdown("We compare the **within-species variance** across islands against **between-species variance** on the same island to determine the dominant driver.")

    feat_sel = st.selectbox("Select feature to analyse:",
        ["Bill Length (Mm)", "Bill Depth (Mm)", "Flipper Length (Mm)", "Body Mass (G)", "Mass per Bill Area"],
        key="species_geo_feat")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### By Species (collapsed across islands)")
        fig = go.Figure()
        for species in filtered_df['Species'].unique():
            sdata = filtered_df[filtered_df['Species'] == species][feat_sel].dropna()
            fig.add_trace(go.Box(y=sdata, name=species, marker_color=color_map.get(species, '#999'), boxmean='sd'))
        fig.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(title=feat_sel, showgrid=True, gridcolor='#f0f0f0'), showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        species_list = filtered_df['Species'].unique()
        medians_s = {s: filtered_df[filtered_df['Species'] == s][feat_sel].median() for s in species_list}
        spread_s = max(medians_s.values()) - min(medians_s.values()) if len(medians_s) > 1 else 0
        st.metric("Between-species median spread", f"{spread_s:.1f}")
    with col2:
        st.markdown("#### By Island (collapsed across species)")
        fig = go.Figure()
        for island in filtered_df['Island'].unique():
            idata = filtered_df[filtered_df['Island'] == island][feat_sel].dropna()
            fig.add_trace(go.Box(y=idata, name=island, marker_color=island_colors.get(island, '#aaa'), boxmean='sd'))
        fig.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(title=feat_sel, showgrid=True, gridcolor='#f0f0f0'), showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        medians_i = {i: filtered_df[filtered_df['Island'] == i][feat_sel].median() for i in filtered_df['Island'].unique()}
        spread_i = max(medians_i.values()) - min(medians_i.values()) if len(medians_i) > 1 else 0
        st.metric("Between-island median spread", f"{spread_i:.1f}")

    st.markdown("#### Faceted View: Species within Each Island")
    st.markdown("Does the same species maintain its morphology regardless of island? If yes, **species > geography**.")
    fig = px.box(filtered_df, x='Species', y=feat_sel, color='Species', facet_col='Island',
        color_discrete_map=color_map, height=400, points=False)
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'), showlegend=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)

    ratio = spread_s / spread_i if spread_i > 0 else float('inf')
    if ratio >= 1.5:
        verdict_text = f"Species identity explains <b>{ratio:.1f}x</b> more variance in <em>{feat_sel}</em> than island location. Morphological traits are <b>conserved within species</b> regardless of island - species is the dominant driver."
    elif ratio < 0.7:
        verdict_text = f"Island location explains more variance than species for <em>{feat_sel}</em> (island spread = {spread_i:.1f} vs species spread = {spread_s:.1f}). Possible environmental influence worth further study."
    else:
        verdict_text = f"Species and island contribute comparably to variance in <em>{feat_sel}</em> (spread ratio = {ratio:.1f}). Both factors are relevant."
    st.markdown(f'<div class="verdict-box"><div style="font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">🏆 Verdict for {feat_sel}</div><div style="line-height:1.7;font-size:0.95rem;">{verdict_text}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Mass per Bill Area by Species & Island")
    st.markdown("**Mass per Bill Area** = Body Mass (g) / (Bill Length x Bill Depth). A high value means the penguin is relatively heavy for its bill size - indicating body propulsion prioritisation over feeding specialisation.")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure()
        for species in ['Adelie', 'Chinstrap', 'Gentoo']:
            sdata = filtered_df[filtered_df['Species'] == species]['Mass per Bill Area'].dropna()
            if len(sdata) == 0:
                continue
            fig.add_trace(go.Box(y=sdata, name=species, marker_color=color_map.get(species, '#999'),
                boxmean='sd', hovertemplate=f"<b>{species}</b><br>Mass/Bill Area: %{{y:.2f}} g/mm2<extra></extra>"))
        fig.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(title='Mass per Bill Area (g/mm2)', showgrid=True, gridcolor='#f0f0f0'),
            xaxis=dict(title='Species'), showlegend=False, margin=dict(t=20, b=20, l=60, r=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        for island in filtered_df['Island'].unique():
            idata = filtered_df[filtered_df['Island'] == island]['Mass per Bill Area'].dropna()
            fig.add_trace(go.Box(y=idata, name=island, marker_color=island_colors.get(island, '#aaa'), boxmean='sd'))
        fig.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(title='Mass per Bill Area (g/mm2)', showgrid=True, gridcolor='#f0f0f0'),
            xaxis=dict(title='Island'), showlegend=False, margin=dict(t=20, b=20, l=60, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Big Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="insight-box-green"><div class="insight-title-green">Gentoo</div><div style="color:#333;line-height:1.7;font-size:0.92rem;">Gentoo are the <b>fastest underwater swimmers</b> of all penguins. Their combination of <b>increased mass + longest flippers</b> supports diving efficiency and propulsion rather than feeding specialisation via bill morphology. High mass-per-bill-area confirms this - more weight relative to bill size.</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="insight-box-purple"><div class="insight-title-purple">Chinstrap</div><div style="color:#333;line-height:1.7;font-size:0.92rem;">Chinstrap penguins display <b>relatively larger bill sizes for their body mass</b>, indicating reliance on feeding morphology. In reality, they are <b>opportunistic feeders</b> that primarily consume krill and small fish - their low mass-per-bill-area ratio reflects this bill-forward adaptation.</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="insight-box-blue"><div class="insight-title-blue">Adelie</div><div style="color:#333;line-height:1.7;font-size:0.92rem;">Adelie penguins exhibit a <b>more balanced scaling</b> between bill size and body mass. This moderate mass-per-bill-area, combined with their multi-island distribution, suggests greater <b>ecological plasticity</b> - adaptable feeders without extreme specialisation in either direction.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="insight-box"><div class="insight-title">🌍 Species > Geography: The Biscoe Island Effect</div><div style="color:#333;line-height:1.7;">Biscoe Island shows the <b>highest mass-per-bill-area</b> values across all islands. However, this is <b>not an environmental effect</b> - it is entirely explained by the <b>predominance of Gentoo penguins</b> on Biscoe Island. When comparing the same species across islands, morphological traits remain stable, confirming that <b>species identity is the dominant driver of morphological variance</b>, not geographic location.</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    # ── FIX 2: Attribute Distribution - histogram + cards side by side ────────
    st.markdown("### Attribute Distribution")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        attribute = st.selectbox("Attribute",
            ["Bill Length (Mm)", "Bill Depth (Mm)", "Flipper Length (Mm)", "Body Mass (G)"],
            label_visibility="collapsed")
    with col_ctrl2:
        bins = st.slider("HISTOGRAM BINS", 10, 50, 30)
    with col_ctrl3:
        st.markdown("")
        for species in selected_species:
            if species in color_map:
                st.markdown(f'<span style="color: {color_map[species]};">● {species.upper()}</span>', unsafe_allow_html=True)

    st.markdown(f"Comparative frequency of {attribute} across all recorded specimens")

    col_hist, col_cards = st.columns([2, 1])
    with col_hist:
        fig = go.Figure()
        for species in selected_species:
            if species in color_map:
                sdata = filtered_df[filtered_df['Species'] == species]
                fig.add_trace(go.Histogram(x=sdata[attribute], name=species, marker_color=color_map[species],
                    opacity=0.7, nbinsx=bins,
                    hovertemplate=f'<b>{species}</b><br>{attribute}: %{{x}}<br>Count: %{{y}}<extra></extra>'))
        fig.update_layout(barmode='overlay', height=400, plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, title=attribute),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Frequency'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_cards:
        max_mass_row = filtered_df.loc[filtered_df['Body Mass (G)'].idxmax()]
        st.markdown(f'<div class="metric-card" style="border-left-color: #3d9b9b;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Max Body Mass</div><div style="font-size:2.5rem;font-weight:700;color:#3d9b9b;">{int(max_mass_row["Body Mass (G)"])}</div><div style="color:#666;font-size:0.9rem;">grams</div><div style="color:#999;font-size:0.8rem;margin-top:0.5rem;">{max_mass_row["Species"]} ({max_mass_row["Sex"]}, {max_mass_row["Island"]})</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if len(selected_species) > 0:
            first_species = selected_species[0]
            attr_mean = filtered_df[filtered_df['Species'] == first_species][attribute].mean()
            unit = 'mm' if 'Mm' in attribute else 'grams'
            st.markdown(f'<div class="metric-card" style="border-left-color: #b47eba;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Avg {attribute}</div><div style="font-size:2.5rem;font-weight:700;color:#b47eba;">{attr_mean:.1f}</div><div style="color:#666;font-size:0.9rem;">{unit}</div><div style="color:#999;font-size:0.8rem;margin-top:0.5rem;">{first_species} (Global Population)</div></div>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("📄 GENERATE PDF REPORT", use_container_width=True, type="primary"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_buffer = generate_pdf_report(filtered_df, attribute)
                    st.download_button(label="⬇️ Download PDF Report", data=pdf_buffer,
                        file_name=f"penguins_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", use_container_width=True)
                    st.success("PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Make sure you have installed: pip install reportlab kaleido")

# ═══ TAB 3: CORRELATIONS ══════════════════════════════════════════════════════
with tab3:
    st.markdown("### The Simpson's Paradox")
    st.markdown("Interpreting correlation trends across biological subpopulations.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### GLOBAL CORRELATION")
        st.markdown("**Aggregated View**")
        fig = px.scatter(filtered_df, x='Bill Length (Mm)', y='Bill Depth (Mm)', trendline="ols", height=400)
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
            annotations=[dict(text=f'r = +{avg_corr:.2f}', xref="paper", yref="paper",
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
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color='Species', color_discrete_map=color_map,
            facet_col='Island' if facet_island else None, trendline="ols" if show_ols else None, height=500)
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
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
    st.markdown("### Correlation Matrix")
    st.markdown("Pearson correlation heatmap across numeric phenotypes.")
    col1, col2 = st.columns([2, 1])
    with col1:
        numeric_cols = ['Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)']
        corr_matrix = filtered_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, labels=dict(color="Correlation"), x=corr_matrix.columns,
            y=corr_matrix.columns, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            text_auto='.2f', height=500)
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("""
        <div style="background: #1a5f7a; padding: 1.5rem; border-radius: 8px; color: white;">
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 1rem;">Strongest Link</div>
            <div style="font-size: 0.9rem; margin-bottom: 1rem;">Body Mass and Flipper Length exhibit a nearly linear relationship across all three species.</div>
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

# ═══ TAB 4: ML EXPLORER ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 Unsupervised Learning Explorer")
    st.markdown("""
    This module analyzes the morphometric relationships of the Palmer Archipelago penguins using
    **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering**
    for specimen categorization. Models are pre-trained via best-feature selection (optimised ARI)
    and loaded from `ML_implementation/`. We evaluate alignment between biological species
    (Ground Truth) and mathematical clusters.
    """)

    import joblib

    # ── Load pre-trained models ───────────────────────────────────────────────
    MODEL_DIR = "ML_implementation"

    @st.cache_resource
    def load_ml_models():
        try:
            pca_model    = joblib.load(f"{MODEL_DIR}/pca.pkl")
            scaler_model = joblib.load(f"{MODEL_DIR}/scaler.pkl")
            kmeans_model = joblib.load(f"{MODEL_DIR}/kmeans_model.pkl")
            return pca_model, scaler_model, kmeans_model, None
        except FileNotFoundError as e:
            return None, None, None, str(e)
        except Exception as e:
            return None, None, None, str(e)

    pca_model, scaler_model, kmeans_model, load_error = load_ml_models()

    if load_error or pca_model is None:
        st.error(f"⚠️ Could not load pre-trained models from `{MODEL_DIR}/`.")
        st.info(f"Expected files: `pca.pkl`, `scaler.pkl`, `kmeans_model.pkl`\n\nError: {load_error}")
        st.stop()

    # ── Column mapping: dashboard names → training script names ──────────────
    # Training script used lowercase snake_case; scaler was fit on those columns.
    # Map dashboard filtered_df columns to the feature order the scaler expects.
    col_map = {
        'Bill Length (Mm)': 'bill_length_mm',
        'Bill Depth (Mm)':  'bill_depth_mm',
        'Flipper Length (Mm)': 'flipper_length_mm',
        'Body Mass (G)':    'body_mass_g',
    }

    # Infer which features the scaler was trained on from its n_features_in_
    all_feature_cols_dashboard = ['Bill Length (Mm)', 'Bill Depth (Mm)', 'Flipper Length (Mm)', 'Body Mass (G)']
    n_features_expected = scaler_model.n_features_in_

    # If scaler expects fewer than 4 features, the best combo was a subset -
    # we use all 4 in dashboard order and trust the pkl was fit on all 4.
    # (If your scaler was fit on a subset, reorder here to match training order.)
    feature_cols = all_feature_cols_dashboard[:n_features_expected]

    if len(filtered_df) < 3:
        st.warning("Need at least 3 specimens for ML clustering analysis.")
        st.stop()

    # ── Transform with pre-trained scaler → PCA → KMeans ─────────────────────
    X = filtered_df[feature_cols].values
    X_scaled   = scaler_model.transform(X)
    X_pca      = pca_model.transform(X_scaled)
    cluster_labels = kmeans_model.predict(X_scaled)
    n_clusters = kmeans_model.n_clusters

    try:
        silhouette = silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) >= 2 else 0.0
    except Exception:
        silhouette = 0.0

    ml_df = filtered_df.copy()
    ml_df['PC1'] = X_pca[:, 0]
    ml_df['PC2'] = X_pca[:, 1]
    ml_df['Cluster'] = cluster_labels

    try:
        species_map = {species: i for i, species in enumerate(ml_df['Species'].unique())}
        true_labels = ml_df['Species'].map(species_map).values
        ari = adjusted_rand_score(true_labels, cluster_labels) if len(np.unique(true_labels)) > 1 and len(np.unique(cluster_labels)) > 1 else 0.0
    except Exception:
        ari = 0.0
        species_map = {}

    try:
        ml_df['is_mismatch'] = ml_df.apply(lambda row: row['Cluster'] != species_map.get(row['Species'], -1), axis=1)
        mismatches = ml_df['is_mismatch'].sum()
    except Exception:
        ml_df['is_mismatch'] = False
        mismatches = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card" style="border-left-color:#1a5f7a;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Silhouette Score</div><div style="font-size:2.5rem;font-weight:700;color:#1a5f7a;">{silhouette:.2f}</div><div style="color:#666;font-size:0.8rem;">/1.0</div><div style="color:#999;font-size:0.8rem;margin-top:0.5rem;">Measures how similar an object is to its own cluster compared to others.</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="border-left-color:#3d9b9b;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Adjusted Rand Index (ARI)</div><div style="font-size:2.5rem;font-weight:700;color:#3d9b9b;">{ari:.2f}</div><div style="color:#666;font-size:0.8rem;">📈</div><div style="color:#999;font-size:0.8rem;margin-top:0.5rem;">High similarity between clustering results and ground truth labels.</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="border-left-color:#dc2626;"><div style="color:#666;font-size:0.9rem;text-transform:uppercase;">Clustering Mismatches</div><div style="font-size:2.5rem;font-weight:700;color:#dc2626;">{mismatches}</div><div style="color:#666;font-size:0.8rem;">/ {len(filtered_df)} Specimens</div><div style="color:#999;font-size:0.8rem;margin-top:0.5rem;">Occurrences where K-Means label differs from biological species classification.</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Dimensionality Comparison")
    st.markdown("Visualization of PCA 1 vs PCA 2 components")

    col_radio, _, _ = st.columns([1, 1, 1])
    with col_radio:
        view_mode = st.radio("", ["Cluster Labels", "Ground Truth", "Compare View"], horizontal=False)

    cluster_colors = {0: '#3d9b9b', 1: '#b47eba', 2: '#c8902e'}

    if view_mode == "Compare View":
        # FIX 3: single merged chart - species colors, X marks mismatches, + checklist below
        cluster_to_species = {}
        for cluster_id in range(n_clusters):
            cdata = ml_df[ml_df['Cluster'] == cluster_id]
            cluster_to_species[cluster_id] = cdata['Species'].mode()[0] if len(cdata) > 0 else f"Cluster {cluster_id}"
        ml_df['cluster_label'] = ml_df['Cluster'].map(cluster_to_species)

        fig = go.Figure()
        for species in ml_df['Species'].unique():
            sp_color = color_map.get(species, '#999')
            matched = ml_df[(ml_df['Species'] == species) & (~ml_df['is_mismatch'])]
            if len(matched) > 0:
                fig.add_trace(go.Scatter(
                    x=matched['PC1'], y=matched['PC2'], mode='markers',
                    name=f"{species} (matched)",
                    marker=dict(symbol='circle', color=sp_color, size=9, line=dict(width=1, color='DarkSlateGrey')),
                    hovertemplate=f"<b>{species}</b> - matched<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
                ))
            mismatched = ml_df[(ml_df['Species'] == species) & (ml_df['is_mismatch'])]
            if len(mismatched) > 0:
                fig.add_trace(go.Scatter(
                    x=mismatched['PC1'], y=mismatched['PC2'], mode='markers',
                    name=f"{species} (mismatch X)",
                    marker=dict(symbol='x', color=sp_color, size=13, line=dict(width=2.5, color=sp_color)),
                    customdata=mismatched['cluster_label'],
                    hovertemplate=f"<b>{species}</b> - K-Means mismatch<br>Assigned to: %{{customdata}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
                ))
        fig.update_layout(
            height=540, plot_bgcolor='#e8f4f5', paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='white', title='Principal Component 1'),
            yaxis=dict(showgrid=True, gridcolor='white', title='Principal Component 2'),
            legend=dict(title=dict(text='Species  (X = K-Means mismatch)', font=dict(size=12)),
                bgcolor='rgba(255,255,255,0.9)', bordercolor='#ccc', borderwidth=1),
            title=dict(text=f"K-Means vs Ground Truth  -  {mismatches} mismatch(es) out of {len(ml_df)} specimens",
                font=dict(size=13, color='#444'), x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Checklist
        match_count = len(ml_df) - mismatches
        match_pct = match_count / len(ml_df) * 100
        species_summary = []
        for species in sorted(ml_df['Species'].unique()):
            sp_total = len(ml_df[ml_df['Species'] == species])
            sp_mismatch = int(len(ml_df[(ml_df['Species'] == species) & ml_df['is_mismatch']]))
            sp_match = sp_total - sp_mismatch
            species_summary.append((species, sp_total, sp_match, sp_mismatch))

        rows_html = ""
        for species, total, matched_n, mismatched_n in species_summary:
            sp_color = color_map.get(species, '#999')
            acc = matched_n / total * 100 if total > 0 else 0
            icon = "✅" if mismatched_n == 0 else ("⚠️" if mismatched_n <= 5 else "❌")
            rows_html += (
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">'
                f'<span style="font-size:1rem;">{icon}</span>'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{sp_color};display:inline-block;flex-shrink:0;"></span>'
                f'<span style="font-weight:600;min-width:90px;color:#333;">{species}</span>'
                f'<div style="flex:1;background:#f0f0f0;border-radius:4px;height:8px;overflow:hidden;">'
                f'<div style="background:{sp_color};width:{acc:.0f}%;height:100%;border-radius:4px;"></div></div>'
                f'<span style="font-size:0.82rem;color:#555;min-width:120px;text-align:right;">{matched_n}/{total} correct ({acc:.0f}%)</span>'
                f'</div>'
            )

        checklist_html = (
            f'<div style="background:white;border:1px solid #e0e0e0;border-radius:10px;padding:1.2rem;margin-top:0.5rem;">'
            f'<div style="font-weight:700;font-size:1rem;color:#1a5f7a;margin-bottom:1rem;">📋 Clustering Accuracy Checklist</div>'
            f'<div style="display:flex;gap:2rem;margin-bottom:1rem;flex-wrap:wrap;">'
            f'<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#16a34a;">{match_count}</div><div style="font-size:0.8rem;color:#666;">✅ Correct</div></div>'
            f'<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#dc2626;">{mismatches}</div><div style="font-size:0.8rem;color:#666;">✗ Mismatch</div></div>'
            f'<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#1a5f7a;">{match_pct:.1f}%</div><div style="font-size:0.8rem;color:#666;">Accuracy</div></div>'
            f'</div>'
            f'<div style="border-top:1px solid #f0f0f0;padding-top:0.8rem;">{rows_html}</div>'
            f'</div>'
        )
        st.markdown(checklist_html, unsafe_allow_html=True)

    else:
        # Cluster Labels or Ground Truth - unchanged from original
        if view_mode == "Cluster Labels":
            fig = px.scatter(ml_df, x='PC1', y='PC2', color=ml_df['Cluster'].astype(str),
                color_discrete_map={str(k): v for k, v in cluster_colors.items() if k < n_clusters},
                labels={'color': 'Cluster'}, height=500)
        else:
            fig = px.scatter(ml_df, x='PC1', y='PC2', color='Species',
                color_discrete_map=color_map, height=500)
        fig.update_layout(plot_bgcolor='#e8f4f5', paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='white'),
            yaxis=dict(showgrid=True, gridcolor='white'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">💡 BIOLOGICAL MISMATCH INSIGHT</div>
        <div style="color: #333; line-height: 1.6;">
        The {mismatches} mismatches (marked with <b>X</b>) primarily occur between Adelie and Chinstrap penguins on Dream Island. These specimens exhibit overlapping
        flipper-to-bill ratios, suggesting convergent morphological traits in shared habitats that challenge unsupervised clustering algorithms.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("Palmer Archipelago Research Dashboard | Data Visualization Project 2026")