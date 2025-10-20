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
    
    /* Custom progress colors */
    .progress-high {
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
    }
    
    .progress-medium {
        background-color: #f59e0b;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
    }
    
    .progress-low {
        background-color: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
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
    
    # 3. Remove data collected by test users
    test_users = ['test', 'addisu']
    if 'username' in df_clean.columns:
        df_clean = df_clean[~df_clean['username'].str.lower().isin(test_users)]
    
        # 4. Filter only consented data (intro_consent.consent == 1)
    if 'intro_consent.consent' in df_clean.columns:
        # Convert to numeric to handle any string values
        df_clean['intro_consent.consent'] = pd.to_numeric(df_clean['intro_consent.consent'], errors='coerce')
        # Keep only records with consent = 1
        df_clean = df_clean[df_clean['intro_consent.consent'] == 1]
        st.info(f"✅ Filtered to {len(df_clean)} consented surveys (intro_consent.consent == 1)")
    else:
        st.warning("⚠️ 'intro_consent.consent' column not found. Using all data without consent filtering.")
        
    # 4. Convert date columns to datetime
    date_columns = ['completed_time', 'started_time', 'received_on']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # 5. Convert numeric columns and handle NA values for species data
    numeric_columns = ['intro_consent.consent', 'age', 'hh_size', 'land_own_total']
    
    # Define species columns for planting analysis
    species_columns_got = [
        'oaf_trees.oaf_gesho.num_got',
        'oaf_trees.oaf_grev.num_got',
        'oaf_trees.oaf_dec.num_got',
        'oaf_trees.oaf_wanza.num_got',
        'oaf_trees.oaf_papaya.num_got',
        'oaf_trees.oaf_coffee.num_got',
        'oaf_trees.oaf_moringa.num_got'
    ]
    
    species_columns_planted = [
        'oaf_trees.oaf_gesho.num_planted_private',
        'oaf_trees.oaf_grev.num_planted_private',
        'oaf_trees.oaf_dec.num_planted_private',
        'oaf_trees.oaf_wanza.num_planted_private',
        'oaf_trees.oaf_papaya.num_planted_private',
        'oaf_trees.oaf_coffee.num_planted_private',
        'oaf_trees.oaf_moringa.num_planted_private'
    ]
    
    # Troster columns
    troster_columns = [
        'num_oaf_gesho_troster',
        'num_oaf_grev_troster',
        'num_oaf_dec_troster',
        'num_oaf_wanza_troster',
        'num_oaf_papaya_troster',
        'num_oaf_coffee_troster',
        'num_oaf_moringa_troster'
    ]
    
    all_species_columns = species_columns_got + species_columns_planted + troster_columns
    numeric_columns.extend(all_species_columns)
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 6. Replace NA with 0 for specific species columns
    for col in species_columns_got + species_columns_planted:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
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
        'gesho': 'oaf_trees.oaf_gesho.num_planted_private',
        'grev': 'oaf_trees.oaf_grev.num_planted_private',
        'dec': 'oaf_trees.oaf_dec.num_planted_private',
        'wanza': 'oaf_trees.oaf_wanza.num_planted_private',
        'papaya': 'oaf_trees.oaf_papaya.num_planted_private',
        'coffee': 'oaf_trees.oaf_coffee.num_planted_private',
        'moringa': 'oaf_trees.oaf_moringa.num_planted_private'
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

def calculate_planting_rates(df):
    """Calculate planting rates for each species"""
    
    species_mapping = {
        'gesho': {
            'got': 'oaf_trees.oaf_gesho.num_got',
            'planted': 'oaf_trees.oaf_gesho.num_planted_private'
        },
        'grev': {
            'got': 'oaf_trees.oaf_grev.num_got',
            'planted': 'oaf_trees.oaf_grev.num_planted_private'
        },
        'dec': {
            'got': 'oaf_trees.oaf_dec.num_got',
            'planted': 'oaf_trees.oaf_dec.num_planted_private'
        },
        'wanza': {
            'got': 'oaf_trees.oaf_wanza.num_got',
            'planted': 'oaf_trees.oaf_wanza.num_planted_private'
        },
        'papaya': {
            'got': 'oaf_trees.oaf_papaya.num_got',
            'planted': 'oaf_trees.oaf_papaya.num_planted_private'
        },
        'coffee': {
            'got': 'oaf_trees.oaf_coffee.num_got',
            'planted': 'oaf_trees.oaf_coffee.num_planted_private'
        },
        'moringa': {
            'got': 'oaf_trees.oaf_moringa.num_got',
            'planted': 'oaf_trees.oaf_moringa.num_planted_private'
        }
    }
    
    planting_data = []
    
    for species, columns in species_mapping.items():
        got_col = columns['got']
        planted_col = columns['planted']
        
        if got_col in df.columns and planted_col in df.columns:
            # Filter out rows where both are 0 (no data)
            valid_data = df[(df[got_col] > 0) | (df[planted_col] > 0)]
            
            if len(valid_data) > 0:
                # Calculate planting rate for each observation
                planting_rates = []
                total_got = 0
                total_planted = 0
                
                for idx, row in valid_data.iterrows():
                    got = row[got_col]
                    planted = row[planted_col]
                    
                    if got > 0:
                        planting_rate = planted / got
                        planting_rates.append(planting_rate)
                        total_got += got
                        total_planted += planted
                
                if len(planting_rates) > 0:
                    avg_planting_rate = np.mean(planting_rates)
                    overall_planting_rate = total_planted / total_got if total_got > 0 else 0
                    
                    planting_data.append({
                        'species': species,
                        'average_planting_rate': avg_planting_rate,
                        'overall_planting_rate': overall_planting_rate,
                        'total_got': total_got,
                        'total_planted': total_planted,
                        'observations': len(planting_rates),
                        'min_rate': np.min(planting_rates),
                        'max_rate': np.max(planting_rates),
                        'std_rate': np.std(planting_rates)
                    })
    
    return pd.DataFrame(planting_data)

def find_planting_outliers(df):
    """Find planting rate outliers (more than 3 standard deviations from mean)"""
    
    species_mapping = {
        'gesho': {
            'got': 'oaf_trees.oaf_gesho.num_got',
            'planted': 'oaf_trees.oaf_gesho.num_planted_private'
        },
        'grev': {
            'got': 'oaf_trees.oaf_grev.num_got',
            'planted': 'oaf_trees.oaf_grev.num_planted_private'
        },
        'dec': {
            'got': 'oaf_trees.oaf_dec.num_got',
            'planted': 'oaf_trees.oaf_dec.num_planted_private'
        },
        'wanza': {
            'got': 'oaf_trees.oaf_wanza.num_got',
            'planted': 'oaf_trees.oaf_wanza.num_planted_private'
        },
        'papaya': {
            'got': 'oaf_trees.oaf_papaya.num_got',
            'planted': 'oaf_trees.oaf_papaya.num_planted_private'
        },
        'coffee': {
            'got': 'oaf_trees.oaf_coffee.num_got',
            'planted': 'oaf_trees.oaf_coffee.num_planted_private'
        },
        'moringa': {
            'got': 'oaf_trees.oaf_moringa.num_got',
            'planted': 'oaf_trees.oaf_moringa.num_planted_private'
        }
    }
    
    outliers_data = []
    
    for species, columns in species_mapping.items():
        got_col = columns['got']
        planted_col = columns['planted']
        
        if got_col in df.columns and planted_col in df.columns:
            # Filter valid data (got > 0)
            valid_data = df[(df[got_col] > 0) & (df[planted_col].notna())].copy()
            
            if len(valid_data) > 0:
                # Calculate planting rate
                valid_data['planting_rate'] = valid_data[planted_col] / valid_data[got_col]
                
                # Calculate mean and standard deviation
                mean_rate = valid_data['planting_rate'].mean()
                std_rate = valid_data['planting_rate'].std()
                
                # Find outliers (more than 3 standard deviations from mean)
                outlier_threshold = 3 * std_rate
                outliers = valid_data[
                    (valid_data['planting_rate'] < mean_rate - outlier_threshold) | 
                    (valid_data['planting_rate'] > mean_rate + outlier_threshold)
                ]
                
                for idx, row in outliers.iterrows():
                    outliers_data.append({
                        'username': row.get('username', 'N/A'),
                        'farmer_name': row.get('farmer_name', 'N/A'),
                        'phone_no': row.get('phone_no', 'N/A'),
                        'unique_id': row.get('id', 'N/A'),
                        'species': species,
                        'trees_got': row[got_col],
                        'trees_planted': row[planted_col],
                        'planting_rate': row['planting_rate'],
                        'mean_rate': mean_rate,
                        'std_dev': std_rate,
                        'site': row.get('site', 'N/A')
                    })
    
    return pd.DataFrame(outliers_data)

def compare_with_troster(df):
    """Compare survey data with troster data"""
    
    species_mapping = {
        'gesho': {
            'survey': 'oaf_trees.oaf_gesho.num_got',
            'troster': 'num_oaf_gesho_troster'
        },
        'grev': {
            'survey': 'oaf_trees.oaf_grev.num_got',
            'troster': 'num_oaf_grev_troster'
        },
        'dec': {
            'survey': 'oaf_trees.oaf_dec.num_got',
            'troster': 'num_oaf_dec_troster'
        },
        'wanza': {
            'survey': 'oaf_trees.oaf_wanza.num_got',
            'troster': 'num_oaf_wanza_troster'
        },
        'papaya': {
            'survey': 'oaf_trees.oaf_papaya.num_got',
            'troster': 'num_oaf_papaya_troster'
        },
        'coffee': {
            'survey': 'oaf_trees.oaf_coffee.num_got',
            'troster': 'num_oaf_coffee_troster'
        },
        'moringa': {
            'survey': 'oaf_trees.oaf_moringa.num_got',
            'troster': 'num_oaf_moringa_troster'
        }
    }
    
    discrepancies = []
    summary_data = []
    
    for species, columns in species_mapping.items():
        survey_col = columns['survey']
        troster_col = columns['troster']
        
        if survey_col in df.columns and troster_col in df.columns:
            # Filter valid comparisons
            valid_data = df[(df[survey_col].notna()) & (df[troster_col].notna()) & 
                           ((df[survey_col] > 0) | (df[troster_col] > 0))]
            
            if len(valid_data) > 0:
                total_survey = valid_data[survey_col].sum()
                total_troster = valid_data[troster_col].sum()
                
                # Calculate overall difference percentage (fixed to avoid infinity)
                if total_troster > 0:
                    overall_difference_pct = ((total_survey - total_troster) / total_troster) * 100
                else:
                    overall_difference_pct = 0
                
                summary_data.append({
                    'species': species,
                    'total_survey': total_survey,
                    'total_troster': total_troster,
                    'difference_pct': overall_difference_pct,
                    'comparisons': len(valid_data)
                })
                
                # Find individual discrepancies > 25%
                for idx, row in valid_data.iterrows():
                    survey_val = row[survey_col]
                    troster_val = row[troster_col]
                    
                    if troster_val > 0:  # Avoid division by zero
                        difference_pct = abs(survey_val - troster_val) / troster_val * 100
                        
                        if difference_pct > 25:
                            discrepancies.append({
                                'username': row.get('username', 'N/A'),
                                'farmer_name': row.get('farmer_name', 'N/A'),
                                'phone_no': row.get('phone_no', 'N/A'),
                                'species': species,
                                'survey_value': survey_val,
                                'troster_value': troster_val,
                                'difference_pct': difference_pct,
                                'site': row.get('site', 'N/A'),
                                'unique_id': row.get('id', 'N/A')
                            })
    
    return pd.DataFrame(discrepancies), pd.DataFrame(summary_data)

def analyze_demographics(df):
    """Analyze demographic variables (columns starting with 'demo.')"""
    
    # Find all demographic columns
    demo_columns = [col for col in df.columns if col.startswith('demo.')]
    
    if not demo_columns:
        return None
    
    demographic_data = {}
    
    for col in demo_columns:
        # Clean column name for display
        clean_name = col.replace('demo.', '').replace('_', ' ').title()
        
        # Basic statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            demo_stats = {
                'type': 'numeric',
                'count': df[col].count(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isna().sum()
            }
        else:
            # For categorical columns
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            demo_stats = {
                'type': 'categorical',
                'count': df[col].count(),
                'unique_values': df[col].nunique(),
                'top_values': value_counts.to_dict(),
                'missing': df[col].isna().sum()
            }
        
        demographic_data[clean_name] = demo_stats
    
    return demographic_data

def create_overview_tab(df):
    """Create the overview tab with overall summary"""
    
    st.markdown("### 📊 Overall Survey Summary")
    
    # Value boxes
    col1, col2, col3 = st.columns(3)
    
       
    with col1:
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
    
    with col2:
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
    
    with col3:
        # Calculate days remaining until October 20, 2025
        target_date = datetime(2025, 10, 20)
        today = datetime.now()
        days_remaining = (today - target_date).days
        
        st.metric(
            label="📅 Survey Days", 
            value=f"{days_remaining} days"
        )
    
    st.markdown("---")
    
    # Quick progress overview - Single column layout
    st.markdown("#### 👥 Enumerator Progress Overview")
    if 'username' in df.columns:
        enum_progress = calculate_enumerator_progress(df)
        if not enum_progress.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_progress = enum_progress['progress_pct'].mean()
                st.metric("Average Progress", f"{avg_progress:.1f}%")
            
            with col2:
                max_progress = enum_progress['progress_pct'].max()
                st.metric("Highest Progress", f"{max_progress:.1f}%")
            
            with col3:
                min_progress = enum_progress['progress_pct'].min()
                st.metric("Lowest Progress", f"{min_progress:.1f}%")
        else:
            st.info("No enumerator data available")
    else:
        st.info("No username column found")
    
    st.markdown("---")
    
    # Species quick overview
    st.markdown("#### 🌱 Species Progress Overview")
    species_progress = calculate_species_progress(df)
    if not species_progress.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_species_progress = species_progress['progress_pct'].mean()
            st.metric("Average Species Progress", f"{avg_species_progress:.1f}%")
        
        with col2:
            max_species_progress = species_progress['progress_pct'].max()
            st.metric("Highest Species Progress", f"{max_species_progress:.1f}%")
        
        with col3:
            completed_species = species_progress[species_progress['progress_pct'] >= 100].shape[0]
            st.metric("Completed Species Targets", f"{completed_species}/{len(species_progress)}")
    else:
        st.info("No species progress data available")
    
    st.markdown("---")
    
    # Daily submissions chart - Fixed to show per day
    st.markdown("#### 📈 Daily Survey Submissions")
    if 'completed_time' in df.columns:
        # Extract date only (remove time component)
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
        
        # Detailed enumerator progress table at the TOP
        st.markdown("#### 📋 Detailed Enumerator Progress")
        
        # Add color coding for progress
        def color_progress(val):
            if val >= 80:
                return 'background-color: #10b981; color: white; font-weight: bold;'
            elif val >= 50:
                return 'background-color: #f59e0b; color: white; font-weight: bold;'
            else:
                return 'background-color: #ef4444; color: white; font-weight: bold;'
        
        display_df = enum_progress.copy()
        display_df['Progress (%)'] = display_df['progress_pct'].round(1)
        display_df['Completed/Target'] = display_df['actual_count'].astype(str) + '/' + display_df['target'].astype(str)
        display_df = display_df[['username', 'Completed/Target', 'Progress (%)', 'remaining']]
        display_df.columns = ['Enumerator', 'Completed/Target', 'Progress (%)', 'Remaining']
        
        # Apply styling
        styled_df = display_df.style.applymap(color_progress, subset=['Progress (%)'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download enumerator data
        csv = enum_progress.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Enumerator Progress Data",
            data=csv,
            file_name=f"enumerator_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Progress chart - Below the table
        st.markdown("#### 📊 Progress by Enumerator")
        # Show top 20 enumerators for better visualization
        display_data = enum_progress.head(20).copy()
        
        # Add color based on progress for the chart
        display_data['color'] = display_data['progress_pct'].apply(
            lambda x: '#10b981' if x >= 80 else '#f59e0b' if x >= 50 else '#ef4444'
        )
        
        fig_enum = px.bar(
            display_data,
            x='username',
            y='progress_pct',
            title="Progress Percentage by Enumerator (Top 20)",
            labels={'progress_pct': 'Progress (%)', 'username': 'Enumerator'},
            color='color',
            color_discrete_map='identity'
        )
        fig_enum.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 100], showlegend=False)
        fig_enum.update_traces(hovertemplate='<b>%{x}</b><br>Progress: %{y:.1f}%<br>Surveys: %{customdata}', 
                              customdata=display_data['actual_count'])
        st.plotly_chart(fig_enum, use_container_width=True)
        
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
        
        # Progress chart - Single column layout
        st.markdown("#### 📊 Survey Progress by Species and Site (%)")
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

def create_planting_rate_tab(df):
    """Create the planting rate analysis tab"""
    
    st.markdown("### 🌱 Planting Rate Analysis")
    
    # Calculate planting rates
    planting_rates_df = calculate_planting_rates(df)
    
    if not planting_rates_df.empty:
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_overall_rate = planting_rates_df['average_planting_rate'].mean()
            st.metric("Average Planting Rate", f"{avg_overall_rate:.1%}")
        
        with col2:
            total_planted = planting_rates_df['total_planted'].sum()
            st.metric("Total Planted", f"{total_planted:,}")
        
        with col3:
            total_got = planting_rates_df['total_got'].sum()
            st.metric("Total Received", f"{total_got:,}")
        
        with col4:
            overall_rate = total_planted / total_got if total_got > 0 else 0
            st.metric("Overall Planting Rate", f"{overall_rate:.1%}")
        
        st.markdown("---")
        
        # Planting rate by species - Single column layout
        st.markdown("#### 📊 Planting Rate by Species")
        fig_rates = px.bar(
            planting_rates_df,
            x='species',
            y='average_planting_rate',
            title="Average Planting Rate by Species",
            labels={'average_planting_rate': 'Planting Rate', 'species': 'Species'},
            color='average_planting_rate',
            color_continuous_scale='greens'
        )
        fig_rates.update_traces(texttemplate='%{y:.1%}', textposition='outside')
        fig_rates.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig_rates, use_container_width=True)
        
        st.markdown("---")
        
        # NEW: Tree Planting Validation Section
        st.markdown("#### 🔍 Tree Planting Validation")
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Username filter
            if 'username' in df.columns:
                usernames = ['All'] + sorted(df['username'].dropna().unique().tolist())
                selected_username = st.selectbox("Filter by Enumerator:", usernames, key="validation_username")
            else:
                selected_username = 'All'
                st.info("No username column found")
        
        with col2:
            # Species filter
            species_options = ['All', 'gesho', 'grev', 'dec', 'wanza', 'papaya', 'coffee', 'moringa']
            selected_species = st.selectbox("Filter by Species:", species_options, key="validation_species")
        
        with col3:
            # Tree threshold filter
            tree_threshold = st.number_input("Minimum Trees Planted:", min_value=0, value=2000, step=100, key="validation_threshold")
        
        # Apply filters
        filtered_validation_df = df.copy()
        
        if selected_username != 'All':
            filtered_validation_df = filtered_validation_df[filtered_validation_df['username'] == selected_username]
        
        # Species mapping for column names
        species_columns = {
            'gesho': {
                'got': 'oaf_trees.oaf_gesho.num_got',
                'planted': 'oaf_trees.oaf_gesho.num_planted_private',
                'troster': 'num_oaf_gesho_troster'
            },
            'grev': {
                'got': 'oaf_trees.oaf_grev.num_got',
                'planted': 'oaf_trees.oaf_grev.num_planted_private',
                'troster': 'num_oaf_grev_troster'
            },
            'dec': {
                'got': 'oaf_trees.oaf_dec.num_got',
                'planted': 'oaf_trees.oaf_dec.num_planted_private',
                'troster': 'num_oaf_dec_troster'
            },
            'wanza': {
                'got': 'oaf_trees.oaf_wanza.num_got',
                'planted': 'oaf_trees.oaf_wanza.num_planted_private',
                'troster': 'num_oaf_wanza_troster'
            },
            'papaya': {
                'got': 'oaf_trees.oaf_papaya.num_got',
                'planted': 'oaf_trees.oaf_papaya.num_planted_private',
                'troster': 'num_oaf_papaya_troster'
            },
            'coffee': {
                'got': 'oaf_trees.oaf_coffee.num_got',
                'planted': 'oaf_trees.oaf_coffee.num_planted_private',
                'troster': 'num_oaf_coffee_troster'
            },
            'moringa': {
                'got': 'oaf_trees.oaf_moringa.num_got',
                'planted': 'oaf_trees.oaf_moringa.num_planted_private',
                'troster': 'num_oaf_moringa_troster'
            }
        }
        
        # Create validation data
        validation_data = []
        
        for idx, row in filtered_validation_df.iterrows():
            if selected_species == 'All':
                # Check all species
                for species, columns in species_columns.items():
                    got_col = columns['got']
                    planted_col = columns['planted']
                    troster_col = columns['troster']
                    
                    if (got_col in row and planted_col in row and 
                        pd.notna(row[got_col]) and pd.notna(row[planted_col]) and
                        row[planted_col] >= tree_threshold):
                        
                        validation_data.append({
                            'username': row.get('username', 'N/A'),
                            'farmer_name': row.get('farmer_name', 'N/A'),
                            'phone_no': row.get('phone_no', 'N/A'),
                            'unique_id': row.get('id', 'N/A'),
                            'species': species,
                            'trees_got': row[got_col],
                            'trees_planted': row[planted_col],
                            'trees_troster': row.get(troster_col, 0) if troster_col in row else 0,
                            'site': row.get('site', 'N/A')
                        })
            else:
                # Check specific species
                columns = species_columns[selected_species]
                got_col = columns['got']
                planted_col = columns['planted']
                troster_col = columns['troster']
                
                if (got_col in row and planted_col in row and 
                    pd.notna(row[got_col]) and pd.notna(row[planted_col]) and
                    row[planted_col] >= tree_threshold):
                    
                    validation_data.append({
                        'username': row.get('username', 'N/A'),
                        'farmer_name': row.get('farmer_name', 'N/A'),
                        'phone_no': row.get('phone_no', 'N/A'),
                        'unique_id': row.get('id', 'N/A'),
                        'species': selected_species,
                        'trees_got': row[got_col],
                        'trees_planted': row[planted_col],
                        'trees_troster': row.get(troster_col, 0) if troster_col in row else 0,
                        'site': row.get('site', 'N/A')
                    })
        
        validation_df = pd.DataFrame(validation_data)
        
        if not validation_df.empty:
            st.markdown(f"**Found {len(validation_df)} records with {tree_threshold}+ trees planted**")
            
            # Display the validation table
            display_validation = validation_df[[
                'username', 'farmer_name', 'unique_id', 'phone_no', 'species', 
                'trees_got', 'trees_planted', 'trees_troster', 'site'
            ]].copy()
            display_validation.columns = [
                'Enumerator', 'Farmer Name', 'Unique ID', 'Phone', 'Species',
                'Trees Got', 'Trees Planted', 'Trees Troster', 'Site'
            ]
            
            st.dataframe(display_validation, use_container_width=True, height=400)
            
            # Download validation data
            csv_validation = validation_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Validation Data",
                data=csv_validation,
                file_name=f"tree_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No records found with {tree_threshold}+ trees planted for the selected filters.")
        
        st.markdown("---")
        
        # NEW: Planting Rate Outliers Section
        st.markdown("#### ⚠️ Planting Rate Outliers (3+ Standard Deviations)")
        
        # Find outliers
        outliers_df = find_planting_outliers(df)
        
        if not outliers_df.empty:
            st.warning(f"Found {len(outliers_df)} planting rate outliers")
            
            # Filter outliers by enumerator
            if 'username' in outliers_df.columns:
                outlier_enumerators = ['All'] + sorted(outliers_df['username'].unique().tolist())
                selected_outlier_enum = st.selectbox("Filter outliers by Enumerator:", outlier_enumerators, key="outlier_enum")
                
                if selected_outlier_enum != 'All':
                    filtered_outliers = outliers_df[outliers_df['username'] == selected_outlier_enum]
                else:
                    filtered_outliers = outliers_df
            else:
                filtered_outliers = outliers_df
            
            # Display outliers table
            display_outliers = filtered_outliers[[
                'username', 'farmer_name', 'unique_id', 'phone_no', 'species', 'site',
                'trees_got', 'trees_planted', 'planting_rate', 'mean_rate', 'std_dev'
            ]].copy()
            display_outliers['planting_rate'] = (display_outliers['planting_rate'] * 100).round(1)
            display_outliers['mean_rate'] = (display_outliers['mean_rate'] * 100).round(1)
            display_outliers['std_dev'] = (display_outliers['std_dev'] * 100).round(1)
            
            display_outliers.columns = [
                'Enumerator', 'Farmer Name', 'Unique ID', 'Phone', 'Species', 'Site',
                'Trees Got', 'Trees Planted', 'Planting Rate (%)', 'Mean Rate (%)', 'Std Dev (%)'
            ]
            
            st.dataframe(display_outliers, use_container_width=True, height=400)
            
            # Download outliers data
            csv_outliers = outliers_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Outliers Data",
                data=csv_outliers,
                file_name=f"planting_outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No planting rate outliers found")
        
        # Detailed planting rate table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Planting Rate Analysis")
        
        with st.expander("View Detailed Planting Data", expanded=False):
            display_df = planting_rates_df.copy()
            display_df['Average Rate'] = (display_df['average_planting_rate'] * 100).round(1).astype(str) + '%'
            display_df['Overall Rate'] = (display_df['overall_planting_rate'] * 100).round(1).astype(str) + '%'
            display_df['Min Rate'] = (display_df['min_rate'] * 100).round(1).astype(str) + '%'
            display_df['Max Rate'] = (display_df['max_rate'] * 100).round(1).astype(str) + '%'
            display_df['Std Dev'] = (display_df['std_rate'] * 100).round(1).astype(str) + '%'
            
            display_df = display_df[[
                'species', 'Average Rate', 'Overall Rate', 'Min Rate', 'Max Rate', 'Std Dev',
                'total_got', 'total_planted', 'observations'
            ]]
            display_df.columns = [
                'Species', 'Avg Rate', 'Overall Rate', 'Min Rate', 'Max Rate', 'Std Dev',
                'Total Received', 'Total Planted', 'Observations'
            ]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download planting rate data
            csv = planting_rates_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Planting Rate Data",
                data=csv,
                file_name=f"planting_rates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No planting rate data available. Check if species columns are present.")

def create_troster_comparison_tab(df):
    """Create the troster comparison tab"""
    
    st.markdown("### 📊 Survey vs Troster Data Comparison")
    
    # Calculate troster comparisons
    discrepancies_df, summary_df = compare_with_troster(df)
    
    if not summary_df.empty:
        # Overall comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_discrepancies = len(discrepancies_df)
            st.metric("Total Discrepancies", f"{total_discrepancies:,}")
        
        with col2:
            avg_difference = summary_df['difference_pct'].abs().mean()
            st.metric("Avg Difference", f"{avg_difference:.1f}%")
        
        with col3:
            total_survey = summary_df['total_survey'].sum()
            st.metric("Total Survey", f"{total_survey:,}")
        
        with col4:
            total_troster = summary_df['total_troster'].sum()
            st.metric("Total Troster", f"{total_troster:,}")
        
        st.markdown("---")
        
        # Comparison summary table
        st.markdown("#### 📋 Comparison Summary by Species")
        
        display_summary = summary_df.copy()
        display_summary['Difference (%)'] = display_summary['difference_pct'].round(1).astype(str) + '%'
        display_summary = display_summary[[
            'species', 'total_survey', 'total_troster', 'Difference (%)', 'comparisons'
        ]]
        display_summary.columns = [
            'Species', 'Survey Total', 'Troster Total', 'Difference (%)', 'Valid Comparisons'
        ]
        
        st.dataframe(display_summary, use_container_width=True)
        
        # Discrepancies table
        st.markdown("---")
        st.markdown("#### ⚠️ Significant Discrepancies (>25% Difference)")
        
        if not discrepancies_df.empty:
            st.warning(f"Found {len(discrepancies_df)} records with >25% difference between survey and troster data")
            
            # NEW: Enumerator filter for discrepancies
            if 'username' in discrepancies_df.columns:
                unique_enumerators = ['All'] + sorted(discrepancies_df['username'].unique().tolist())
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    selected_enum = st.selectbox("Filter by Enumerator:", unique_enumerators, key="discrepancy_enum")
                
                # Filter discrepancies by selected enumerator
                if selected_enum != 'All':
                    filtered_discrepancies = discrepancies_df[discrepancies_df['username'] == selected_enum]
                else:
                    filtered_discrepancies = discrepancies_df
                
                st.markdown(f"**Showing {len(filtered_discrepancies)} discrepancies for {selected_enum}**")
            else:
                filtered_discrepancies = discrepancies_df
            
            with st.expander("View All Discrepancies", expanded=True):
                display_df = filtered_discrepancies.copy()
                display_df['Difference'] = display_df['difference_pct'].round(1).astype(str) + '%'
                display_df = display_df[[
                    'username', 'farmer_name', 'phone_no', 'unique_id', 'species', 'site',
                    'survey_value', 'troster_value', 'Difference'
                ]]
                display_df.columns = [
                    'Enumerator', 'Farmer Name', 'Phone', 'Unique ID', 'Species', 'Site',
                    'Survey Value', 'Troster Value', 'Difference (%)'
                ]
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download discrepancies data
                csv = discrepancies_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Discrepancies Data",
                    data=csv,
                    file_name=f"troster_discrepancies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Enumerator analysis
            st.markdown("#### 👥 Enumerator Discrepancy Analysis")
            enum_analysis = discrepancies_df['username'].value_counts().reset_index()
            enum_analysis.columns = ['username', 'discrepancy_count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Enumerators with Discrepancies:**")
                st.dataframe(enum_analysis.head(10), use_container_width=True)
            
            with col2:
                st.markdown("**Species with Most Discrepancies:**")
                species_analysis = discrepancies_df['species'].value_counts().reset_index()
                species_analysis.columns = ['species', 'discrepancy_count']
                st.dataframe(species_analysis, use_container_width=True)
        else:
            st.success("✅ No significant discrepancies found (>25% difference)")
    
    else:
        st.info("No troster comparison data available. Check if troster columns are present.")

def create_demographics_tab(df):
    """Create the demographics analysis tab"""
    
    st.markdown("### 👥 Respondent Demographics")
    
    # Analyze demographics
    demo_data = analyze_demographics(df)
    
    if demo_data:
        st.markdown("#### 📊 Demographic Overview")
        
        # Display demographic variables in an organized way
        for demo_var, stats in demo_data.items():
            with st.expander(f"📈 {demo_var}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Responses", f"{stats['count']:,}")
                    st.metric("Missing Values", f"{stats['missing']:,}")
                    
                    if stats['type'] == 'numeric':
                        st.metric("Mean", f"{stats['mean']:.2f}")
                        st.metric("Median", f"{stats['median']:.2f}")
                    else:
                        st.metric("Unique Values", stats['unique_values'])
                
                with col2:
                    if stats['type'] == 'numeric':
                        st.metric("Standard Deviation", f"{stats['std']:.2f}")
                        st.metric("Minimum", f"{stats['min']:.2f}")
                        st.metric("Maximum", f"{stats['max']:.2f}")
                    else:
                        st.markdown("**Top Values:**")
                        for value, count in list(stats['top_values'].items())[:5]:
                            st.write(f"- {value}: {count:,}")
        
        # Create visualizations for key demographic variables
        st.markdown("---")
        st.markdown("#### 📈 Demographic Visualizations")
        
        # Select a demographic variable to visualize
        demo_vars = list(demo_data.keys())
        selected_demo = st.selectbox("Select demographic variable to visualize:", demo_vars)
        
        if selected_demo:
            # Find the original column name
            original_col = None
            for col in df.columns:
                if col.replace('demo.', '').replace('_', ' ').title() == selected_demo:
                    original_col = col
                    break
            
            if original_col and original_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[original_col]):
                    # Create histogram for numeric variables
                    fig = px.histogram(
                        df, 
                        x=original_col,
                        title=f"Distribution of {selected_demo}",
                        labels={original_col: selected_demo},
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Create bar chart for categorical variables (top 15)
                    value_counts = df[original_col].value_counts().head(15)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top Values for {selected_demo}",
                        labels={'x': selected_demo, 'y': 'Count'},
                        color=value_counts.values,
                        color_continuous_scale='blues'
                    )
                    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No demographic variables found (columns starting with 'demo.').")

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
            
            # Enumerator filter
            if 'username' in df.columns:
                enumerators = ['All'] + sorted(df['username'].dropna().unique().tolist())
                selected_enumerator = st.selectbox(
                    "Select Enumerator:",
                    enumerators
                )
            else:
                selected_enumerator = 'All'
                st.info("No username column found")
            
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
            filtered_df = df.copy()
            if selected_enumerator != 'All':
                filtered_df = filtered_df[filtered_df['username'] == selected_enumerator]
            if selected_treatment != 'All':
                filtered_df = filtered_df[filtered_df['treatment'] == selected_treatment]
            if selected_site != 'All':
                filtered_df = filtered_df[filtered_df['site'] == selected_site]
            
            st.session_state['filtered_df'] = filtered_df
                
            # Show filter summary
            st.info(f"Showing {len(filtered_df)} of {len(df)} records")
        else:
            selected_enumerator = 'All'
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
        
        # Create tabs - Added Demographics tab
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview", "👥 Enumerator Progress", "🌱 Species Progress", 
            "🌿 Planting Rates", "📈 Troster Comparison", "👤 Demographics", "🔍 Data Explorer"
        ])
        
        with tab1:
            create_overview_tab(df)
        
        with tab2:
            create_enumerator_tab(df)
        
        with tab3:
            create_species_tab(df)
        
        with tab4:
            create_planting_rate_tab(df)
        
        with tab5:
            create_troster_comparison_tab(df)
        
        with tab6:
            create_demographics_tab(df)
        
        with tab7:
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