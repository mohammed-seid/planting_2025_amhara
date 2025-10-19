import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from requests.auth import HTTPBasicAuth
import requests
import time
import io

# Page configuration
st.set_page_config(
    page_title="ETH 2025 Planting Survey Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for modern look
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stMetric > label {
        color: #1f77b4;
        font-size: 16px;
        font-weight: 600;
    }
    
    .stMetric > div > div > div {
        font-size: 28px;
        font-weight: 700;
        color: #2e7d32;
    }
    
    /* Header styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    
    h2 {
        color: #1e40af;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #1e40af;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 15px;
        border-radius: 10px;
        color: #1e3a8a;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Password protection
def check_password():
    """Returns `True` if the user has the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Custom login screen
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1 style='color: #1e3a8a; font-size: 3rem;'>🌳</h1>
            <h2 style='color: #1e3a8a;'>ETH 2025 Planting Survey Dashboard</h2>
            <p style='color: #64748b; font-size: 1.1rem;'>Please enter your password to access the dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Enter your password"
        )
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("🔒 Password incorrect. Please try again.")
    
    return False

if not check_password():
    st.stop()

def fetch_commcare_data():
    """Fetch data from CommCare API with pagination"""
    
    try:
        USERNAME = st.secrets["commcare"]["username"]
        API_KEY = st.secrets["commcare"]["api_key"]
        PROJECT_SPACE = st.secrets["commcare"]["project_space"]
        FORM_ID = st.secrets["commcare"]["form_id"]
    except KeyError as e:
        st.error(f"❌ Missing secret: {e}. Please check your secrets.toml file.")
        return None
    
    url = f"https://www.commcarehq.org/a/{PROJECT_SPACE}/api/v0.5/odata/forms/{FORM_ID}/feed"
    
    limit = 2000
    offset = 0
    all_records = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Starting data export from CommCare...")
    
    while True:
        params = {
            'limit': limit,
            'offset': offset
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                auth=HTTPBasicAuth(USERNAME, API_KEY),
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
                break
            
            data = response.json()
            records = data.get('value', [])
            
            if len(records) == 0:
                break
            
            all_records.extend(records)
            
            if offset == 0 and len(records) > 0:
                estimated_total = len(records) + limit
                progress_estimate = min(offset / estimated_total, 0.99)
            else:
                progress_estimate = min(offset / (offset + len(records)), 0.99)
                
            progress_bar.progress(progress_estimate)
            status_text.text(f"📥 Fetched {len(all_records):,} records so far...")
            
            if len(records) < limit:
                break
            
            offset += limit
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
            break
    
    progress_bar.progress(1.0)
    status_text.empty()
    
    return all_records

def clean_data(df):
    """Clean and preprocess the data"""
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # 1. Remove 'form.' from all column names
    df_clean.columns = df_clean.columns.str.replace('form.', '', regex=False)
    
    # 2. Replace '---' with NaN
    df_clean = df_clean.replace('---', np.nan)
    
    # 3. Convert date columns to datetime
    date_columns = ['completed_time', 'started_time', 'received_on']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # 4. Convert numeric columns
    numeric_columns = ['intro_consent.consent', 'age', 'hh_size', 'land_own_total']
    # Add species columns
    species_columns = [
        'oaf_trees.oaf_gesho.num_planted_private_hid',
        'oaf_trees.oaf_grev.num_planted_private_hid',
        'oaf_trees.oaf_dec.num_planted_private_hid',
        'oaf_trees.oaf_wanza.num_planted_private_hid',
        'oaf_trees.oaf_papaya.num_planted_private_hid',
        'oaf_trees.oaf_coffee.num_planted_private_hid',
        'oaf_trees.oaf_moringa.num_planted_private_hid'
    ]
    numeric_columns.extend(species_columns)
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def get_target_sample_data():
    """Return the target sample sizes for each species and site"""
    target_data = {
        'site': ['core', 'core', 'core', 'core', 'core', 'core', 'core', 
                'nep', 'nep', 'nep', 'nep', 'nep', 'nep'],
        'species': ['coffee', 'dec', 'gesho', 'grev', 'moringa', 'papaya', 'wanza',
                   'dec', 'gesho', 'grev', 'moringa', 'papaya', 'wanza'],
        'target_sample': [247, 32, 215, 715, 105, 81, 279,
                         114, 440, 270, 141, 587, 97]
    }
    return pd.DataFrame(target_data)

def calculate_species_progress(df):
    """Calculate survey progress for each species by site"""
    
    # Define species mapping to column names
    species_mapping = {
        'gesho': 'oaf_trees.oaf_gesho.num_planted_private_hid',
        'grev': 'oaf_trees.oaf_grev.num_planted_private_hid',
        'dec': 'oaf_trees.oaf_dec.num_planted_private_hid',
        'wanza': 'oaf_trees.oaf_wanza.num_planted_private_hid',
        'papaya': 'oaf_trees.oaf_papaya.num_planted_private_hid',
        'coffee': 'oaf_trees.oaf_coffee.num_planted_private_hid',
        'moringa': 'oaf_trees.oaf_moringa.num_planted_private_hid'
    }
    
    target_df = get_target_sample_data()
    progress_data = []
    
    for site in ['core', 'nep']:
        site_data = df[df['site'] == site] if 'site' in df.columns else pd.DataFrame()
        
        for species, column in species_mapping.items():
            if column in df.columns and not site_data.empty:
                # Count records where the species column > 0 and not NA
                actual_count = site_data[site_data[column] > 0].shape[0]
            else:
                actual_count = 0
            
            # Get target for this species and site
            target_row = target_df[(target_df['site'] == site) & (target_df['species'] == species)]
            target_sample = target_row['target_sample'].iloc[0] if not target_row.empty else 0
            
            progress_pct = (actual_count / target_sample * 100) if target_sample > 0 else 0
            
            progress_data.append({
                'site': site,
                'species': species,
                'actual_count': actual_count,
                'target_sample': target_sample,
                'progress_pct': progress_pct,
                'remaining': max(0, target_sample - actual_count)
            })
    
    return pd.DataFrame(progress_data)

def calculate_enumerator_progress(df):
    """Calculate progress for each enumerator"""
    
    if 'username' not in df.columns:
        return pd.DataFrame()
    
    # Count consented surveys per enumerator
    if 'intro_consent.consent' in df.columns:
        enumerator_data = df[df['intro_consent.consent'] == 1]['username'].value_counts().reset_index()
    else:
        enumerator_data = df['username'].value_counts().reset_index()
    
    enumerator_data.columns = ['username', 'actual_count']
    
    # Add target and progress
    enumerator_target = 190
    enumerator_data['target'] = enumerator_target
    enumerator_data['progress_pct'] = (enumerator_data['actual_count'] / enumerator_target * 100).round(1)
    enumerator_data['remaining'] = enumerator_target - enumerator_data['actual_count']
    
    # Sort by actual count descending
    enumerator_data = enumerator_data.sort_values('actual_count', ascending=False)
    
    return enumerator_data

def create_overview_tab(df):
    """Create the overview tab with overall summary"""
    
    st.markdown("### 📊 Overall Survey Summary")
    
    # Value boxes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_rows = len(df)
        st.metric(
            label="📋 Total Surveys", 
            value=f"{total_rows:,}",
            delta="All Records"
        )
    
    with col2:
        # Count consent = 1 with progress percentage
        if 'intro_consent.consent' in df.columns:
            consented_count = df[df['intro_consent.consent'] == 1].shape[0]
            consent_target = 5000
            consent_progress = (consented_count / consent_target * 100)
            st.metric(
                label="✅ Consented Surveys", 
                value=f"{consented_count:,}",
                delta=f"{consent_progress:.1f}% of target"
            )
        else:
            st.metric(
                label="✅ Consented Surveys", 
                value="N/A",
                delta="Column missing"
            )
    
    with col3:
        # Enumerator progress summary
        if 'username' in df.columns:
            unique_enumerators = df['username'].nunique()
            total_consented = df[df['intro_consent.consent'] == 1].shape[0] if 'intro_consent.consent' in df.columns else len(df)
            enumerator_target_total = unique_enumerators * 190
            enumerator_progress = (total_consented / enumerator_target_total * 100) if enumerator_target_total > 0 else 0
            
            st.metric(
                label="👥 Enumerator Progress", 
                value=f"{unique_enumerators} enum.",
                delta=f"{enumerator_progress:.1f}% overall"
            )
        else:
            st.metric(
                label="👥 Enumerators", 
                value="N/A",
                delta="Column missing"
            )
    
    with col4:
        # Date range
        if 'completed_time' in df.columns:
            date_range = f"{df['completed_time'].min().strftime('%Y-%m-%d')} to {df['completed_time'].max().strftime('%Y-%m-%d')}"
            st.metric(
                label="📅 Date Range", 
                value=date_range
            )
        else:
            st.metric(
                label="📅 Date Range", 
                value="N/A"
            )
    
    st.markdown("---")
    
    # Quick progress overview
    col1, col2 = st.columns(2)
    
    with col1:
        # Enumerator quick overview
        st.markdown("#### 👥 Enumerator Progress Overview")
        if 'username' in df.columns:
            enum_progress = calculate_enumerator_progress(df)
            if not enum_progress.empty:
                avg_progress = enum_progress['progress_pct'].mean()
                max_progress = enum_progress['progress_pct'].max()
                min_progress = enum_progress['progress_pct'].min()
                
                st.metric("Average Progress", f"{avg_progress:.1f}%")
                st.metric("Highest Progress", f"{max_progress:.1f}%")
                st.metric("Lowest Progress", f"{min_progress:.1f}%")
            else:
                st.info("No enumerator data available")
        else:
            st.info("No username column found")
    
    with col2:
        # Species quick overview
        st.markdown("#### 🌱 Species Progress Overview")
        species_progress = calculate_species_progress(df)
        if not species_progress.empty:
            avg_species_progress = species_progress['progress_pct'].mean()
            max_species_progress = species_progress['progress_pct'].max()
            completed_species = species_progress[species_progress['progress_pct'] >= 100].shape[0]
            
            st.metric("Average Species Progress", f"{avg_species_progress:.1f}%")
            st.metric("Highest Species Progress", f"{max_species_progress:.1f}%")
            st.metric("Completed Species Targets", f"{completed_species}/{len(species_progress)}")
        else:
            st.info("No species progress data available")
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily submissions chart
        st.markdown("#### 📈 Daily Survey Submissions")
        if 'completed_time' in df.columns:
            daily_data = df.groupby(df['completed_time'].dt.date).size().reset_index(name='count')
            fig_daily = px.bar(
                daily_data, 
                x='completed_time', 
                y='count',
                title="Surveys Completed by Day",
                labels={'completed_time': 'Date', 'count': 'Number of Surveys'},
                color='count',
                color_continuous_scale='blues'
            )
            fig_daily.update_layout(showlegend=False)
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("No completion time data available")
    
    with col2:
        # Site distribution
        st.markdown("#### 🗺️ Surveys by Site")
        if 'site' in df.columns:
            site_data = df['site'].value_counts().reset_index()
            site_data.columns = ['site', 'count']
            fig_site = px.pie(
                site_data,
                values='count',
                names='site',
                title="Survey Distribution by Site"
            )
            st.plotly_chart(fig_site, use_container_width=True)
        else:
            st.info("No site data available")

def create_enumerator_tab(df):
    """Create the enumerator progress tab"""
    
    st.markdown("### 👥 Enumerator Progress")
    
    # Calculate enumerator progress
    enum_progress = calculate_enumerator_progress(df)
    
    if not enum_progress.empty:
        # Overall progress metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_enum = len(enum_progress)
        total_surveys = enum_progress['actual_count'].sum()
        target_total = total_enum * 190
        overall_progress = (total_surveys / target_total * 100) if target_total > 0 else 0
        
        with col1:
            st.metric("Total Enumerators", total_enum)
        with col2:
            st.metric("Total Surveys", f"{total_surveys:,}")
        with col3:
            st.metric("Target Total", f"{target_total:,}")
        with col4:
            st.metric("Overall Progress", f"{overall_progress:.1f}%")
        
        st.markdown("---")
        
        # Progress chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Progress by Enumerator")
            # Show top 20 enumerators for better visualization
            display_data = enum_progress.head(20).copy()
            fig_enum = px.bar(
                display_data,
                x='username',
                y='progress_pct',
                title="Progress Percentage by Enumerator (Top 20)",
                labels={'progress_pct': 'Progress (%)', 'username': 'Enumerator'},
                color='progress_pct',
                color_continuous_scale='viridis'
            )
            fig_enum.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 100])
            fig_enum.update_traces(hovertemplate='<b>%{x}</b><br>Progress: %{y:.1f}%<extra></extra>')
            st.plotly_chart(fig_enum, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Surveys Completed")
            fig_surveys = px.bar(
                display_data,
                x='username',
                y='actual_count',
                title="Surveys Completed by Enumerator (Top 20)",
                labels={'actual_count': 'Surveys Completed', 'username': 'Enumerator'},
                color='actual_count',
                color_continuous_scale='blues'
            )
            fig_surveys.update_layout(xaxis_tickangle=-45)
            fig_surveys.add_hline(y=190, line_dash="dash", line_color="red", 
                                annotation_text="Target (190)", annotation_position="top left")
            st.plotly_chart(fig_surveys, use_container_width=True)
        
        # Detailed table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Enumerator Progress")
        
        with st.expander("View All Enumerator Data", expanded=True):
            display_df = enum_progress.copy()
            display_df['Progress'] = display_df['progress_pct'].round(1).astype(str) + '%'
            display_df['Completed/Target'] = display_df['actual_count'].astype(str) + '/' + display_df['target'].astype(str)
            display_df = display_df[['username', 'Completed/Target', 'Progress', 'remaining']]
            display_df.columns = ['Enumerator', 'Completed/Target', 'Progress (%)', 'Remaining']
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download enumerator data
            csv = enum_progress.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Enumerator Progress Data",
                data=csv,
                file_name=f"enumerator_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No enumerator progress data available. Check if 'username' column exists.")

def create_species_tab(df):
    """Create the species progress tab"""
    
    st.markdown("### 🌱 Species Survey Progress")
    
    # Calculate species progress
    progress_df = calculate_species_progress(df)
    
    if not progress_df.empty:
        # Overall species progress metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_target = progress_df['target_sample'].sum()
        total_actual = progress_df['actual_count'].sum()
        overall_species_progress = (total_actual / total_target * 100) if total_target > 0 else 0
        completed_targets = progress_df[progress_df['progress_pct'] >= 100].shape[0]
        
        with col1:
            st.metric("Total Species Targets", len(progress_df))
        with col2:
            st.metric("Total Actual Surveys", f"{total_actual:,}")
        with col3:
            st.metric("Total Target", f"{total_target:,}")
        with col4:
            st.metric("Overall Progress", f"{overall_species_progress:.1f}%")
        
        st.markdown("---")
        
        # Progress charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Progress by species and site
            fig_progress = px.bar(
                progress_df,
                x='species',
                y='progress_pct',
                color='site',
                barmode='group',
                title="Survey Progress by Species and Site (%)",
                labels={'progress_pct': 'Progress (%)', 'species': 'Species', 'site': 'Site'},
                text='progress_pct'
            )
            fig_progress.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_progress.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_progress, use_container_width=True)
        
        with col2:
            # Actual vs Target by species and site
            fig_actual = px.bar(
                progress_df,
                x='species',
                y=['actual_count', 'target_sample'],
                color='site',
                barmode='group',
                title="Actual vs Target by Species and Site",
                labels={'value': 'Number of Surveys', 'species': 'Species', 'variable': 'Type'},
            )
            fig_actual.update_layout(legend_title_text='Type')
            st.plotly_chart(fig_actual, use_container_width=True)
        
        # Detailed progress table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Species Progress")
        
        with st.expander("View All Species Data", expanded=True):
            display_df = progress_df.copy()
            display_df['Progress'] = display_df['progress_pct'].round(1).astype(str) + '%'
            display_df['Actual/Target'] = display_df['actual_count'].astype(str) + '/' + display_df['target_sample'].astype(str)
            display_df = display_df[['site', 'species', 'Actual/Target', 'Progress', 'remaining']]
            display_df.columns = ['Site', 'Species', 'Actual/Target', 'Progress (%)', 'Remaining']
            display_df = display_df.sort_values(['Site', 'Species'])
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download species data
            csv = progress_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Species Progress Data",
                data=csv,
                file_name=f"species_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No species progress data available. Check if site and species columns are present.")

def create_data_tab(df):
    """Create the data exploration tab"""
    
    st.markdown("### 🔍 Data Exploration")
    
    # Data preview
    st.markdown("#### 📋 Data Preview")
    st.dataframe(
        df.head(10), 
        use_container_width=True,
        height=400
    )
    
    # Two column layout for additional info
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔍 Column Information", expanded=False):
            col_info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info_df, use_container_width=True, height=300)
    
    with col2:
        with st.expander("📊 Data Statistics", expanded=False):
            st.write("**Numerical Summary:**")
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                st.dataframe(df.describe(), use_container_width=True)
            else:
                st.info("No numerical columns found")
    
    # Download section
    st.markdown("### 💾 Export Data")
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv,
            file_name=f"eth_planting_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        excel_buffer.seek(0)
        
        st.download_button(
            label="⬇️ Download as Excel",
            data=excel_buffer,
            file_name=f"eth_planting_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Full data view
    with st.expander("📊 View Full Dataset", expanded=False):
        st.dataframe(df, use_container_width=True, height=600)

def main():
    # Sidebar - Simplified to only show filters
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Logo", width=150)
        st.markdown("---")
        st.markdown("### 📊 Dashboard Controls")
        
        # Fetch Latest Data button
        fetch_button = st.button("🔄 Fetch Latest Data", type="primary", use_container_width=True)
        
        # Show success message in sidebar after fetching data
        if st.session_state.get('data_loaded', False):
            st.success("✅ Data Successfully Loaded and Cleaned!")
            # Reset the flag after showing the message
            st.session_state.data_loaded = False
        
        # Add refresh timestamp
        if 'last_refresh' in st.session_state:
            st.info(f"🕒 Last updated: {st.session_state.last_refresh}")
        
        st.markdown("---")
        
        # Data filters section
        st.markdown("### 🔍 Data Filters")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # Treatment filter
            if 'treatment' in df.columns:
                treatments = ['All'] + sorted(df['treatment'].dropna().unique().tolist())
                selected_treatment = st.selectbox(
                    "Select Treatment:",
                    treatments
                )
            else:
                selected_treatment = 'All'
                st.info("No treatment column found")
            
            # Site filter
            if 'site' in df.columns:
                sites = ['All'] + sorted(df['site'].dropna().unique().tolist())
                selected_site = st.selectbox(
                    "Select Site:",
                    sites
                )
            else:
                selected_site = 'All'
                st.info("No site column found")
            
            # Apply filters
            if selected_treatment != 'All' or selected_site != 'All':
                filtered_df = df.copy()
                if selected_treatment != 'All':
                    filtered_df = filtered_df[filtered_df['treatment'] == selected_treatment]
                if selected_site != 'All':
                    filtered_df = filtered_df[filtered_df['site'] == selected_site]
                st.session_state['filtered_df'] = filtered_df
                
                # Show filter summary
                st.info(f"Showing {len(filtered_df)} of {len(df)} records")
            else:
                st.session_state['filtered_df'] = df
        else:
            selected_treatment = 'All'
            selected_site = 'All'
            st.info("No data loaded. Click 'Fetch Latest Data' to begin.")
    
    # Fetch data from sidebar button
    if fetch_button:
        with st.spinner("🔄 Fetching data from CommCare..."):
            records = fetch_commcare_data()
            
            if records:
                df = pd.DataFrame(records)
                # Clean the data
                df_clean = clean_data(df)
                st.session_state['df'] = df_clean
                st.session_state['filtered_df'] = df_clean
                st.session_state['last_refresh'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state['data_loaded'] = True
                st.rerun()
    
    # Main header
    st.markdown("""
        <div class='info-box'>
            <h2 style='color: white; margin: 0;'>🌳 ETH 2025 Planting Survey Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Display data if available
    if 'filtered_df' in st.session_state:
        df = st.session_state['filtered_df']
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Enumerator Progress", "🌱 Species Progress", "🔍 Data Explorer"])
        
        with tab1:
            create_overview_tab(df)
        
        with tab2:
            create_enumerator_tab(df)
        
        with tab3:
            create_species_tab(df)
        
        with tab4:
            create_data_tab(df)
    
    else:
        # Empty state
        st.markdown("""
            <div style='text-align: center; padding: 60px; background-color: #f8fafc; border-radius: 10px; margin-top: 30px;'>
                <h2 style='color: #64748b;'>📊 No Data Loaded Yet</h2>
                <p style='color: #94a3b8; font-size: 1.1rem;'>
                    Click the "Fetch Latest Data" button in the sidebar to load data from CommCare
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()