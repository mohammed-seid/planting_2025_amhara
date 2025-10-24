import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth
import requests
import time
import io
import base64
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ET 2025 Amhara Planting Survey Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# One Acre Fund Branded CSS
st.markdown("""
    <style>
    /* Main container styling - OAF Colors */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* OAF Color Scheme */
    :root {
        --oaf-green: #2E8B57;
        --oaf-dark-green: #1e5631;
        --oaf-light-green: #4CAF50;
        --oaf-orange: #FF6B35;
        --oaf-blue: #1E88E5;
        --oaf-dark-blue: #0D47A1;
    }
    
    /* Metrics styling - OAF Theme */
    .stMetric {
        background-color: #f8fff9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stMetric > label {
        color: #1e5631;
        font-size: 14px;
        font-weight: 600;
    }
    
    .stMetric > div > div > div {
        font-size: 28px;
        font-weight: 700;
        color: #2E8B57;
    }
    
    /* Header styling */
    h1 {
        color: #1e5631;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2E8B57;
    }
    
    h2 {
        color: #1e5631;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #1e5631;
        font-weight: 500;
    }
    
    /* Button styling - OAF Green */
    .stButton > button {
        background: linear-gradient(90deg, #2E8B57 0%, #1e5631 100%);
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
        background: linear-gradient(90deg, #1e5631 0%, #2E8B57 100%);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #2E8B57 0%, #1e5631 100%);
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
        color: #1e5631;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fff9;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: 600;
        border-left: 3px solid #2E8B57;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Custom progress colors */
    .progress-high {
        background-color: #2E8B57;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
    }
    
    .progress-medium {
        background-color: #FF6B35;
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
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fff9;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white;
    }
    
    /* Logo header styling */
    .logo-header {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 10px 0;
        margin-bottom: 20px;
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
            <h1 style='color: #1e5631; font-size: 3rem;'>🌳</h1>
            <h2 style='color: #1e5631;'>ET 2025 Amhara Planting Survey Dashboard</h2>
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

def load_logo():
    """Load OAF logo"""
    try:
        # Try to load the logo from file
        with open("oaf.png", "rb") as f:
            logo_bytes = f.read()
        logo_base64 = base64.b64encode(logo_bytes).decode()
        return f"data:image/png;base64,{logo_base64}"
    except:
        # Return placeholder if logo not found
        return None

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
    """Clean and preprocess the data with gender validation"""
    
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
    
    # 5. Filter out records with missing gender data (demo.gender)
    if 'demo.gender' in df_clean.columns:
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['demo.gender'])
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            st.info(f"✅ Removed {removed_count} records with missing gender data")
            st.info(f"📊 Final cleaned dataset: {len(df_clean)} records")
    else:
        st.warning("⚠️ 'demo.gender' column not found in dataset")
        
    # 6. Convert date columns to datetime
    date_columns = ['completed_time', 'started_time', 'received_on']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # 7. Convert ALL numeric columns and handle NA values
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
    
    species_columns_sold = [
        'oaf_trees.oaf_gesho.num_sold',
        'oaf_trees.oaf_grev.num_sold',
        'oaf_trees.oaf_dec.num_sold',
        'oaf_trees.oaf_wanza.num_sold',
        'oaf_trees.oaf_papaya.num_sold',
        'oaf_trees.oaf_coffee.num_sold',
        'oaf_trees.oaf_moringa.num_sold'
    ]
    
    species_columns_comm = [
        'oaf_trees.oaf_gesho.num_planted_comm',
        'oaf_trees.oaf_grev.num_planted_comm',
        'oaf_trees.oaf_dec.num_planted_comm',
        'oaf_trees.oaf_wanza.num_planted_comm',
        'oaf_trees.oaf_papaya.num_planted_comm',
        'oaf_trees.oaf_coffee.num_planted_comm',
        'oaf_trees.oaf_moringa.num_planted_comm'
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
    
    all_numeric_columns = (species_columns_got + species_columns_planted + 
                          species_columns_sold + species_columns_comm + troster_columns +
                          ['intro_consent.consent', 'demo.age', 'demo.hh_size', 'asset.land_own_total'])
    
    for col in all_numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 8. Replace NA with 0 for specific species columns
    for col in species_columns_got + species_columns_planted + species_columns_sold + species_columns_comm:
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
                
                # Find individual discrepancies > 35% (changed from 25%)
                for idx, row in valid_data.iterrows():
                    survey_val = row[survey_col]
                    troster_val = row[troster_col]
                    
                    if troster_val > 0:  # Avoid division by zero
                        difference_pct = abs(survey_val - troster_val) / troster_val * 100
                        
                        if difference_pct > 35:  # Changed from 25 to 35
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

def create_overview_tab(df):
    """Create the overview tab with overall summary"""
    
    st.markdown("### 📊 Overall Survey Summary")
    
    # Value boxes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Count consent = 1 with progress gauge
        if 'intro_consent.consent' in df.columns:
            consented_count = df[df['intro_consent.consent'] == 1].shape[0]
            consent_target = 5000
            consent_progress = (consented_count / consent_target * 100)

            # Render as a compact gauge to better reflect progress
            fig_consented = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=consented_count,
                number={'valueformat':',d'},
                delta={'reference': consent_target, 'relative': False, 'valueformat':'.1f'},
                title={'text': "✅ Consented Surveys"},
                gauge={
                    'axis': {'range': [0, consent_target]},
                    'bar': {'color': '#2E8B57'},
                    'steps': [
                        {'range': [0, consent_target * 0.5], 'color': '#ffcccc'},
                        {'range': [consent_target * 0.5, consent_target * 0.9], 'color': '#ffdd99'},
                        {'range': [consent_target * 0.9, consent_target], 'color': '#c6f6d5'}
                    ],
                }
            ))

            st.plotly_chart(fig_consented, use_container_width=True, key="consented_surveys_gauge")
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
            st.metric(
                label="👥 Active Enumerators", 
                value=f"{unique_enumerators}",
            )
        else:
            st.metric(
                label="👥 Enumerators", 
                value="N/A",
                delta="Column missing"
            )
    
    with col3:
        # Species coverage
        species_progress = calculate_species_progress(df)
        if not species_progress.empty:
            completed_species = species_progress[species_progress['progress_pct'] >= 100].shape[0]
            st.metric(
                label="🌱 Completed Species", 
                value=f"{completed_species}/{len(species_progress)}"
            )
        else:
            st.metric(
                label="🌱 Species Progress", 
                value="N/A"
            )
    
    with col4:
        # Data quality indicator
        if 'intro_consent.consent' in df.columns:
            quality_rate = (df[df['intro_consent.consent'] == 1].shape[0] / len(df) * 100) if len(df) > 0 else 0
            st.metric(
                label="📈 Data Quality Score", 
                value=f"{quality_rate:.1f}%"
            )
        else:
            st.metric(
                label="📈 Data Quality", 
                value="N/A"
            )
    
    st.markdown("---")
    
    # Site Distribution Chart
    st.markdown("#### 📍 Site Distribution")
    if 'site' in df.columns:
        site_counts = df['site'].value_counts()
        fig_site = px.pie(
            values=site_counts.values,
            names=site_counts.index,
            title="Survey Distribution by Site",
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        st.plotly_chart(fig_site, use_container_width=True, key="site_distribution_chart")
    else:
        st.info("No site data available")
    
    # Daily survey submissions - full width
    st.markdown("#### 📈 Daily Survey Submissions")
    if 'completed_time' in df.columns:
        daily_data = df.groupby(df['completed_time'].dt.date).size().reset_index(name='count')
        # Ensure column name for x is 'date' for plotting
        daily_data.columns = ['date', 'count']
        fig_daily = px.line(
            daily_data,
            x='date',
            y='count',
            title="Surveys Completed by Day",
            labels={'date': 'Date', 'count': 'Number of Surveys'},
            color_discrete_sequence=['#2E8B57']
        )
        fig_daily.update_traces(mode='lines+markers')
        st.plotly_chart(fig_daily, use_container_width=True, key="daily_submissions_chart_full")
    else:
        st.info("No completion time data available")
    
    # Note: removed the Enumerator Progress Overview in the Overview tab per request.

def create_progress_tab(df):
    """Create combined progress tab with enumerator and species progress"""
    
    tab1, tab2 = st.tabs(["👥 Enumerator Progress", "🌱 Species Progress"])
    
    with tab1:
        st.markdown("### 👥 Enumerator Progress Tracking")
        
        enum_progress = calculate_enumerator_progress(df)
        
        if not enum_progress.empty:
            # Overall metrics
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
            
            # Surveys completed by enumerator - full width
            fig_surveys = px.bar(
                enum_progress,  # Use full dataset, not just top 20
                x='username',
                y='actual_count',
                title="Surveys Completed by Enumerator",
                labels={'actual_count': 'Surveys Completed', 'username': 'Enumerator'},
                color='actual_count',
                color_continuous_scale='blues'
            )
            fig_surveys.update_layout(xaxis_tickangle=-45)
            fig_surveys.add_hline(y=190, line_dash="dash", line_color="red",
                                  annotation_text="Target", annotation_position="top left")
            st.plotly_chart(fig_surveys, use_container_width=True, key="surveys_by_enumerator_chart_full")

            st.markdown("---")
            st.markdown("#### 🎯 Enumerator Progress Gauges (on demand)")
            st.info("Select one or more enumerators below to render their progress gauges. Leave empty to skip rendering.")

            enum_list = enum_progress['username'].dropna().unique().tolist()
            # Provide a searchable selector, allow auto-selection of top N (user preference: default 6), and enforce a max render limit to avoid layout issues
            # Use the enumerator ordering from enum_progress (already sorted by actual_count desc)
            enum_sorted = enum_progress['username'].dropna().tolist()

            # Let user choose Top N for auto-select (min 1, max 12). Default preference: 6
            top_n = st.number_input(
                "Top N to auto-select:",
                min_value=1,
                max_value=12,
                value=6,
                step=1,
                help="Number of top enumerators to auto-select when the checkbox is enabled (max 12)."
            )

            auto_select_top = st.checkbox(
                f"Auto-select top {int(top_n)} enumerators",
                value=False,
                help="Automatically select the top N enumerators by surveys completed"
            )

            default_selection = enum_sorted[:int(top_n)] if auto_select_top else []

            selected_gauges = st.multiselect(
                f"Select enumerators to show gauges (search) — max 12:",
                enum_sorted,
                default=default_selection,
                help="Start typing to search enumerators. Only up to 12 selections will be rendered to avoid layout issues.",
                key="enumerator_gauge_selector"
            )

            # Enforce a hard limit of 12 rendered gauges to protect layout and performance
            if len(selected_gauges) > 12:
                st.warning("You selected more than 12 enumerators; only the first 12 will be displayed to avoid layout issues.")
                selected_gauges = selected_gauges[:12]

            if selected_gauges:
                # Replace gauges with compact Completed / Remaining bars per enumerator for better readability
                gauges_per_row = 4
                cols = None
                for i, username in enumerate(selected_gauges):
                    row = enum_progress[enum_progress['username'] == username].iloc[0]
                    actual = float(row['actual_count'])
                    target = float(row['target']) if 'target' in row else 190.0

                    remaining = max(target - actual, 0.0)

                    if i % gauges_per_row == 0:
                        cols = st.columns(gauges_per_row)

                    # Build a small horizontal bar chart showing Completed vs Remaining
                    bar_df = pd.DataFrame({
                        'Category': ['Completed', 'Remaining'],
                        'Count': [actual, remaining]
                    })

                    fig_g = px.bar(
                        bar_df,
                        x='Count',
                        y='Category',
                        orientation='h',
                        color='Category',
                        color_discrete_map={'Completed': '#2E8B57', 'Remaining': '#d3d3d3'},
                        title=f"{username}",
                        labels={'Count': 'Trees / Surveys', 'Category': ''}
                    )
                    fig_g.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), height=180)

                    # safe key
                    safe_username = str(username).replace(' ', '_')
                    cols[i % gauges_per_row].plotly_chart(fig_g, use_container_width=True, key=f"enum_progress_bar_{i}_{safe_username}")
            
            # Detailed table
            st.markdown("#### 📋 Detailed Progress Table")
            display_df = enum_progress.copy()
            display_df['Progress (%)'] = display_df['progress_pct'].round(1)
            display_df['Completed/Target'] = display_df['actual_count'].astype(str) + '/' + display_df['target'].astype(str)
            display_df = display_df[['username', 'Completed/Target', 'Progress (%)', 'remaining']]
            display_df.columns = ['Enumerator', 'Completed/Target', 'Progress (%)', 'Remaining']
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
        else:
            st.info("No enumerator progress data available.")
    
    with tab2:
        st.markdown("### 🌱 Species Progress Tracking")
        
        progress_df = calculate_species_progress(df)
        
        if not progress_df.empty:
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_target = progress_df['target_sample'].sum()
            total_actual = progress_df['actual_count'].sum()
            overall_species_progress = (total_actual / total_target * 100) if total_target > 0 else 0
            completed_targets = progress_df[progress_df['progress_pct'] >= 100].shape[0]
            
            with col1:
                st.metric("Total Species", len(progress_df))
            with col2:
                st.metric("Total Surveys", f"{total_actual:,}")
            with col3:
                st.metric("Total Target", f"{total_target:,}")
            with col4:
                st.metric("Overall Progress", f"{overall_species_progress:.1f}%")
            
            st.markdown("---")
            
            # Progress visualization as gauges (Actual vs Target) per species and site
            st.markdown("#### 🎯 Species Progress Gauges")
            gauges_per_row = 3
            # Iterate rows in progress_df and create a gauge per row
            for i, r in progress_df.reset_index(drop=True).iterrows():
                if i % gauges_per_row == 0:
                    gauge_cols = st.columns(gauges_per_row)
                species_site_label = f"{r['species'].upper()} ({r['site']})"
                actual = float(r['actual_count'])
                target = float(r['target_sample']) if r.get('target_sample', 0) > 0 else 1

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=actual,
                    number={'valueformat':',d'},
                    title={'text': species_site_label},
                    gauge={
                        'axis': {'range': [0, target]},
                        'bar': {'color': '#2E8B57'},
                        'steps': [
                            {'range': [0, target * 0.5], 'color': '#ffcccc'},
                            {'range': [target * 0.5, target * 0.9], 'color': '#ffdd99'},
                            {'range': [target * 0.9, target], 'color': '#c6f6d5'}
                        ],
                    }
                ))
                gauge_cols[i % gauges_per_row].plotly_chart(fig_g, use_container_width=True, key=f"species_gauge_{i}_{r['species']}_{r['site']}")
            
            # Detailed table
            st.markdown("#### 📋 Species Progress Details")
            display_df = progress_df.copy()
            display_df['Progress'] = display_df['progress_pct'].round(1).astype(str) + '%'
            display_df['Actual/Target'] = display_df['actual_count'].astype(str) + '/' + display_df['target_sample'].astype(str)
            display_df = display_df[['site', 'species', 'Actual/Target', 'Progress', 'remaining']]
            display_df.columns = ['Site', 'Species', 'Actual/Target', 'Progress (%)', 'Remaining']
            display_df = display_df.sort_values(['Site', 'Species'])
            
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.info("No species progress data available.")

def create_data_quality_tab(df):
    """Create data quality check tab"""
    
    tab1, tab2 = st.tabs(["📊 Troster Comparison", "✅ Planting Validation"])
    
    with tab1:
        st.markdown("### 📊 Survey vs Troster Data Comparison")
        
        discrepancies_df, summary_df = compare_with_troster(df)
        
        # Top enumerators with significant discrepancies (top 20)
        if not discrepancies_df.empty:
            top_enum_counts = discrepancies_df['username'].value_counts().head(20).reset_index()
            top_enum_counts.columns = ['username', 'count']
            fig_top_disc = px.bar(
                top_enum_counts,
                x='username',
                y='count',
                title='Top 20 Enumerators by Significant Discrepancies',
                labels={'username': 'Enumerator', 'count': 'Discrepancies'},
                color='count',
                color_continuous_scale='Reds'
            )
            fig_top_disc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_top_disc, use_container_width=True, key='top20_discrepancies_chart')

        if not summary_df.empty:
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_discrepancies = len(discrepancies_df)
                st.metric("Total Discrepancies", f"{total_discrepancies:,}")
            
            with col2:
                avg_difference = summary_df['difference_pct'].abs().mean()
                st.metric("Average Difference", f"{avg_difference:.1f}%")
            
            with col3:
                agreement_rate = (1 - len(discrepancies_df) / summary_df['comparisons'].sum()) * 100
                st.metric("Data Agreement Rate", f"{agreement_rate:.1f}%")
            
            st.markdown("---")
            
            # Discrepancies table with filters
            if not discrepancies_df.empty:
                st.markdown("#### ⚠️ Significant Discrepancies")
                min_diff = st.number_input("Minimum difference percentage:", min_value=0, max_value=100, value=35, step=1)
                enum_filter_opts = ['All'] + sorted(discrepancies_df['username'].dropna().unique().tolist())
                enum_filter = st.selectbox("Filter by Enumerator:", enum_filter_opts, key="dq_enum_filter")

                filtered = discrepancies_df.copy()
                filtered = filtered[filtered['difference_pct'] >= float(min_diff)]
                if enum_filter != 'All':
                    filtered = filtered[filtered['username'] == enum_filter]

                if not filtered.empty:
                    display_df = filtered.copy()
                    display_df['Difference'] = display_df['difference_pct'].round(1).astype(str) + '%'
                    display_df = display_df[[
                        'username', 'farmer_name', 'species', 'site',
                        'survey_value', 'troster_value', 'Difference'
                    ]]
                    display_df.columns = [
                        'Enumerator', 'Farmer Name', 'Species', 'Site',
                        'Survey Value', 'Troster Value', 'Difference (%)'
                    ]
                    st.dataframe(display_df, use_container_width=True, height=400)
                else:
                    st.info("No discrepancies match the selected filters")
            else:
                st.success("✅ No significant discrepancies found")
        
        else:
            st.info("No troster comparison data available.")
    
    with tab2:
        st.markdown("### ✅ Planting Number Validation")
        
        # Tree planting validation
        st.markdown("#### 🔍 High Planting Records")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tree_threshold = st.number_input("Minimum Trees Planted:", min_value=0, value=2000, step=100)
        
        with col2:
            if 'username' in df.columns:
                usernames = ['All'] + sorted(df['username'].dropna().unique().tolist())
                selected_username = st.selectbox("Filter by Enumerator:", usernames, key="planting_selected_username")
            else:
                selected_username = 'All'
        
        # Validation logic
        species_columns = {
            'gesho': 'oaf_trees.oaf_gesho.num_planted_private',
            'grev': 'oaf_trees.oaf_grev.num_planted_private',
            'dec': 'oaf_trees.oaf_dec.num_planted_private',
            'wanza': 'oaf_trees.oaf_wanza.num_planted_private',
            'papaya': 'oaf_trees.oaf_papaya.num_planted_private',
            'coffee': 'oaf_trees.oaf_coffee.num_planted_private',
            'moringa': 'oaf_trees.oaf_moringa.num_planted_private'
        }
        
        validation_data = []
        for species, col in species_columns.items():
            if col in df.columns:
                filtered_data = df[df[col] >= tree_threshold]
                if selected_username != 'All':
                    filtered_data = filtered_data[filtered_data['username'] == selected_username]
                
                for idx, row in filtered_data.iterrows():
                    validation_data.append({
                        'username': row.get('username', 'N/A'),
                        'farmer_name': row.get('farmer_name', 'N/A'),
                        'unique_id': row.get('id', 'N/A'),
                        'species': species,
                        'trees_planted': row[col],
                        'site': row.get('site', 'N/A')
                    })
        
        validation_df = pd.DataFrame(validation_data)
        
        if not validation_df.empty:
            st.markdown(f"**Found {len(validation_df)} records with {tree_threshold}+ trees planted**")
            st.dataframe(validation_df, use_container_width=True)
        else:
            st.info(f"No records found with {tree_threshold}+ trees planted")
        
        # Planting outliers
        st.markdown("---")
        st.markdown("#### 📊 Planting Rate Outliers")

        outliers_df = find_planting_outliers(df)
        if not outliers_df.empty:
            # Top 20 enumerators by planting-rate outliers
            top_outlier_counts = outliers_df['username'].value_counts().head(20).reset_index()
            top_outlier_counts.columns = ['username', 'count']
            fig_top_out = px.bar(
                top_outlier_counts,
                x='username',
                y='count',
                title='Top 20 Enumerators by Planting Rate Outliers',
                labels={'username': 'Enumerator', 'count': 'Outlier Count'},
                color='count',
                color_continuous_scale='Oranges'
            )
            fig_top_out.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_top_out, use_container_width=True, key='top20_planting_outliers_chart')

            st.warning(f"Found {len(outliers_df)} planting rate outliers")
            # Outliers filter by enumerator
            outlier_enum_opts = ['All'] + sorted(outliers_df['username'].dropna().unique().tolist())
            selected_outlier_enum = st.selectbox("Filter outliers by Enumerator:", outlier_enum_opts, key="outliers_selected_enum")

            display_outliers = outliers_df[['username', 'farmer_name', 'species', 'trees_got', 'trees_planted', 'planting_rate']].copy()
            display_outliers['planting_rate'] = (display_outliers['planting_rate'] * 100).round(1)
            if selected_outlier_enum != 'All':
                display_outliers = display_outliers[display_outliers['username'] == selected_outlier_enum]

            if not display_outliers.empty:
                st.dataframe(display_outliers, use_container_width=True)
            else:
                st.info('No planting rate outliers match the selected enumerator')
        else:
            st.success("✅ No planting rate outliers found")

def create_preliminary_results_tab(df):
    """Create preliminary results tab with improved structure"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Respondent Profile", "🌳 Trees Distribution", "🌱 Planting Analysis", "📈 Early Survival"])
    
    with tab1:
        st.markdown("### 👤 Respondent Profile")
        
        # Create respondent profile table
        profile_data = []
        
        # Age statistics
        if 'demo.age' in df.columns:
            # Convert to numeric and handle errors
            df['demo.age'] = pd.to_numeric(df['demo.age'], errors='coerce')
            age_data = df['demo.age'].dropna()
            if len(age_data) > 0:
                age_mean = age_data.mean()
                age_min = age_data.min()
                age_max = age_data.max()
                profile_data.append({
                    'Characteristic': 'Age',
                    'Value': f"{age_mean:.1f} years",
                    'Range': f"{age_min:.0f} - {age_max:.0f} years"
                })
        
        # Gender statistics
        if 'demo.gender' in df.columns:
            female_count = (df['demo.gender'] == 'female').sum()
            female_pct = (female_count / len(df)) * 100
            profile_data.append({
                'Characteristic': 'Gender',
                'Value': f"{female_pct:.1f}% female",
                'Range': f"{female_count:,} respondents"
            })
        
        # Household size
        if 'demo.hh_size' in df.columns:
            df['demo.hh_size'] = pd.to_numeric(df['demo.hh_size'], errors='coerce')
            hh_data = df['demo.hh_size'].dropna()
            if len(hh_data) > 0:
                hh_mean = hh_data.mean()
                hh_min = hh_data.min()
                hh_max = hh_data.max()
                profile_data.append({
                    'Characteristic': 'Household Size',
                    'Value': f"{hh_mean:.1f} members",
                    'Range': f"{hh_min:.0f} - {hh_max:.0f} members"
                })
        
        # Marital status
        if 'demo.marital_status' in df.columns:
            marital_counts = df['demo.marital_status'].value_counts()
            total_marital = marital_counts.sum()
            for status, count in marital_counts.items():
                pct = (count / total_marital) * 100
                profile_data.append({
                    'Characteristic': f'Marital Status: {status}',
                    'Value': f"{pct:.1f}%",
                    'Range': f"{count:,} respondents"
                })
        
        # Display profile table
        if profile_data:
            profile_df = pd.DataFrame(profile_data)
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
        else:
            st.info("No demographic data available for respondent profile")
        
        # Education bar chart - FIXED: using demo.education_level
        st.markdown("#### 📚 Education Level Distribution")
        if 'demo.education_level' in df.columns:
            # Map education codes to labels
            education_map = {
                0: 'No formal education',
                1: 'Some primary',
                2: 'Completed primary',
                3: 'Some secondary',
                4: 'Completed secondary',
                5: 'College/University'
            }
            
            df_edu = df.copy()
            df_edu['education_label'] = df_edu['demo.education_level'].map(education_map)
            edu_counts = df_edu['education_label'].value_counts().reset_index()
            edu_counts.columns = ['Education Level', 'Count']
            
            fig_edu = px.bar(
                edu_counts,
                x='Education Level',
                y='Count',
                title="Education Level Distribution",
                color='Count',
                color_continuous_scale='greens'
            )
            fig_edu.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_edu, use_container_width=True, key="education_level_chart")
        else:
            st.info("Education data not available (looking for 'demo.education_level')")
        
        # Household status pie chart
        st.markdown("#### 🏠 Household Status")
        if 'demo.respondent_hh_status' in df.columns:
            hh_status_counts = df['demo.respondent_hh_status'].value_counts()
            fig_hh = px.pie(
                values=hh_status_counts.values,
                names=hh_status_counts.index,
                title="Household Status Distribution"
            )
            st.plotly_chart(fig_hh, use_container_width=True, key="household_status_chart")
        else:
            st.info("Household status data not available")
    
    with tab2:
        st.markdown("### 🌳 Trees Distribution Analysis")
        
        # Trees Received Summary (calculated only for those who received trees)
        st.markdown("#### 📊 Trees Received Summary")
        
        species_mapping = {
            'gesho': 'oaf_trees.oaf_gesho.num_got',
            'grev': 'oaf_trees.oaf_grev.num_got',
            'dec': 'oaf_trees.oaf_dec.num_got',
            'wanza': 'oaf_trees.oaf_wanza.num_got',
            'papaya': 'oaf_trees.oaf_papaya.num_got',
            'coffee': 'oaf_trees.oaf_coffee.num_got',
            'moringa': 'oaf_trees.oaf_moringa.num_got'
        }
        
        trees_received_data = []
        
        for species, col_got in species_mapping.items():
            if col_got in df.columns:
                # Convert to numeric to fix type errors
                df[col_got] = pd.to_numeric(df[col_got], errors='coerce')
                # Filter only those who received trees (num_got > 0)
                received_data = df[df[col_got] > 0]
                if len(received_data) > 0:
                    total_trees = received_data[col_got].sum()
                    avg_trees = received_data[col_got].mean()
                    farmers_received = len(received_data)
                    
                    trees_received_data.append({
                        'Species': species,
                        'Total Trees Received': int(total_trees),
                        'Farmers Received': farmers_received,
                        'Average per Farmer': f"{avg_trees:.1f}"
                    })
        
        if trees_received_data:
            trees_df = pd.DataFrame(trees_received_data)
            st.dataframe(trees_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trees received data available")
        
        # What happened to the trees analysis - FIXED DATA TYPE ISSUES
        st.markdown("#### 📈 Tree Distribution Analysis")
        
        species_analysis = ['coffee', 'gesho', 'grev', 'dec', 'wanza', 'papaya', 'moringa']
        tree_distribution_data = []
        
        for species in species_analysis:
            got_col = f'oaf_trees.oaf_{species}.num_got'
            planted_private_col = f'oaf_trees.oaf_{species}.num_planted_private'
            sold_col = f'oaf_trees.oaf_{species}.num_sold'
            planted_comm_col = f'oaf_trees.oaf_{species}.num_planted_comm'
            
            if all(col in df.columns for col in [got_col, planted_private_col, sold_col, planted_comm_col]):
                # Convert all columns to numeric to fix type errors
                df[got_col] = pd.to_numeric(df[got_col], errors='coerce')
                df[planted_private_col] = pd.to_numeric(df[planted_private_col], errors='coerce')
                df[sold_col] = pd.to_numeric(df[sold_col], errors='coerce')
                df[planted_comm_col] = pd.to_numeric(df[planted_comm_col], errors='coerce')
                
                # Filter farmers who received this species
                received_data = df[df[got_col] > 0]
                if len(received_data) > 0:
                    total_got = received_data[got_col].sum()
                    total_planted_private = received_data[planted_private_col].sum()
                    total_sold = received_data[sold_col].sum()
                    total_planted_comm = received_data[planted_comm_col].sum()
                    
                    # Calculate percentages
                    pct_planted_private = (total_planted_private / total_got * 100) if total_got > 0 else 0
                    pct_sold = (total_sold / total_got * 100) if total_got > 0 else 0
                    pct_planted_comm = (total_planted_comm / total_got * 100) if total_got > 0 else 0
                    
                    # Other category (difference)
                    accounted_for = total_planted_private + total_sold + total_planted_comm
                    other_trees = max(0, total_got - accounted_for)
                    pct_other = (other_trees / total_got * 100) if total_got > 0 else 0
                    
                    tree_distribution_data.append({
                        'Species': species,
                        'Planted Private (%)': f"{pct_planted_private:.1f}%",
                        'Sold (%)': f"{pct_sold:.1f}%", 
                        'Planted Community (%)': f"{pct_planted_comm:.1f}%",
                        'Other (%)': f"{pct_other:.1f}%",
                        'Total Received': int(total_got)
                    })
        
        if tree_distribution_data:
            distribution_df = pd.DataFrame(tree_distribution_data)
            st.dataframe(distribution_df, use_container_width=True, hide_index=True)
            
            # Visualization of tree distribution
            st.markdown("#### 📊 Tree Distribution Visualization")
            
            # Prepare data for stacked bar chart
            viz_data = []
            for item in tree_distribution_data:
                viz_data.append({
                    'Species': item['Species'],
                    'Category': 'Planted Private',
                    'Percentage': float(item['Planted Private (%)'].replace('%', ''))
                })
                viz_data.append({
                    'Species': item['Species'],
                    'Category': 'Sold',
                    'Percentage': float(item['Sold (%)'].replace('%', ''))
                })
                viz_data.append({
                    'Species': item['Species'],
                    'Category': 'Planted Community', 
                    'Percentage': float(item['Planted Community (%)'].replace('%', ''))
                })
                viz_data.append({
                    'Species': item['Species'],
                    'Category': 'Other',
                    'Percentage': float(item['Other (%)'].replace('%', ''))
                })
            
            viz_df = pd.DataFrame(viz_data)
            
            fig_dist = px.bar(
                viz_df,
                x='Species',
                y='Percentage',
                color='Category',
                title="Tree Distribution by Species and Category",
                barmode='stack',
                labels={'Percentage': 'Percentage (%)', 'Species': 'Species'}
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="tree_distribution_chart")
        else:
            st.info("No tree distribution data available for analysis")
    
    with tab3:
        st.markdown("### 🌱 Planting Analysis")

        st.markdown("#### 📋 Species-level Planting Summary (Overall / Treatment / Control)")

        species_list = ['coffee', 'gesho', 'grev', 'dec', 'wanza', 'papaya', 'moringa']

        def compute_metrics(sub_df, species):
            """Compute species metrics using sample-size as denominator (include zeros).
            Sample = farmers who received OAF (got > 0).
            Averages (e.g., Avg OAF Trees Planted) are calculated as (sum over sample) / sample_size.
            """
            got_col = f'oaf_trees.oaf_{species}.num_got'
            planted_col = f'oaf_trees.oaf_{species}.num_planted_private'
            non_oaf_col = f'private_land_plantation.num_nonoaf_{species}_planted'
            survived_new = f'trees_survived.num_alive_{species}'
            survived_old = f'oaf_trees.oaf_{species}.num_survived'

            # Local copy and coerce
            sub_df = sub_df.copy()
            for col in [got_col, planted_col, non_oaf_col, survived_new, survived_old]:
                if col and col in sub_df.columns:
                    sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce').fillna(0)

            # Define sample: those who received OAF trees
            sample_mask = (sub_df[got_col] > 0) if got_col in sub_df.columns else pd.Series([False] * len(sub_df), index=sub_df.index)
            sample_size = int(sample_mask.sum())

            # Totals over sample
            total_got = float(sub_df.loc[sample_mask, got_col].sum()) if (sample_size > 0 and got_col in sub_df.columns) else 0.0
            total_oaf_planted = float(sub_df.loc[sample_mask, planted_col].sum()) if (sample_size > 0 and planted_col in sub_df.columns) else 0.0
            total_non_oaf = float(sub_df.loc[sample_mask, non_oaf_col].sum()) if (sample_size > 0 and non_oaf_col in sub_df.columns) else 0.0

            # Averages: divide totals by sample_size (include zeros)
            avg_got = (total_got / sample_size) if sample_size > 0 else 0.0
            avg_oaf_planted = (total_oaf_planted / sample_size) if sample_size > 0 else 0.0
            avg_non_oaf = (total_non_oaf / sample_size) if sample_size > 0 else 0.0

            avg_total_planted = ((total_oaf_planted + total_non_oaf) / sample_size) if sample_size > 0 else 0.0

            # Survived: sum over sample, then divide by sample_size
            survived_col = survived_new if survived_new in sub_df.columns else (survived_old if survived_old in sub_df.columns else None)
            total_survived = float(sub_df.loc[sample_mask, survived_col].sum()) if (survived_col is not None and sample_size > 0 and survived_col in sub_df.columns) else 0.0
            avg_survived = (total_survived / sample_size) if sample_size > 0 else 0.0

            return {
                'Sample Size (received OAF)': sample_size,
                'Avg OAF Trees Received': round(avg_got, 1),
                'Avg OAF Trees Planted (per farmer)': round(avg_oaf_planted, 1),
                'Avg Non-OAF Trees Planted (per farmer)': round(avg_non_oaf, 1),
                'Avg Total Planted (per planter)': round(avg_total_planted, 1),
                'Avg Survived (per planter)': round(avg_survived, 1)
            }

        # Build a Treatment vs Control table per species (omit Overall as requested)
        for species in species_list:
            st.markdown(f"##### 🌱 {species.capitalize()}")
            if 'treatment' in df.columns:
                treatment_metrics = compute_metrics(df[df['treatment'] == 'treatment'], species)
                control_metrics = compute_metrics(df[df['treatment'] == 'control'], species)
            else:
                # If no treatment column, compute both on full dataset but keep keys
                treatment_metrics = compute_metrics(df, species)
                control_metrics = compute_metrics(df, species)

            tbl = pd.DataFrame({
                'Metric': list(treatment_metrics.keys()),
                'Treatment': list(treatment_metrics.values()),
                'Control': list(control_metrics.values())
            })
            tbl = tbl.set_index('Metric')
            st.dataframe(tbl, use_container_width=True, height=300)
    
    with tab4:
        st.markdown("### 📈 Early Survival Rates")
        
        # Early survival analysis (per request)
        # We'll look for either `trees_survived.num_alive_<species>` or the older `oaf_trees.oaf_<species>.num_survived`.
        survival_data = []
        species_list = ['coffee', 'gesho', 'grev', 'dec', 'wanza', 'papaya', 'moringa']

        for species in species_list:
            planted_private_col = f'oaf_trees.oaf_{species}.num_planted_private'
            non_oaf_col = f'private_land_plantation.num_nonoaf_{species}_planted'
            survived_col_new = f'trees_survived.num_alive_{species}'
            survived_col_old = f'oaf_trees.oaf_{species}.num_survived'

            # Work on a local copy
            df_local = df.copy()

            # Ensure numeric
            if planted_private_col in df_local.columns:
                df_local[planted_private_col] = pd.to_numeric(df_local[planted_private_col], errors='coerce').fillna(0)
            if non_oaf_col in df_local.columns:
                df_local[non_oaf_col] = pd.to_numeric(df_local[non_oaf_col], errors='coerce').fillna(0)

            # Compute per-observation total planted (OAF private + non-OAF private)
            df_local['planted_total'] = 0
            if planted_private_col in df_local.columns:
                df_local['planted_total'] = df_local['planted_total'] + df_local[planted_private_col]
            if non_oaf_col in df_local.columns:
                df_local['planted_total'] = df_local['planted_total'] + df_local[non_oaf_col]

            # Determine survived column
            if survived_col_new in df_local.columns:
                df_local['survived_count'] = pd.to_numeric(df_local[survived_col_new], errors='coerce').fillna(0)
            elif survived_col_old in df_local.columns:
                df_local['survived_count'] = pd.to_numeric(df_local[survived_col_old], errors='coerce').fillna(0)
            else:
                # No survival data for this species
                continue

            # Keep only rows where we have planted or survived info
            valid = df_local[(df_local['planted_total'] > 0) | (df_local['survived_count'] > 0)]

            if len(valid) == 0:
                continue

            total_planted = int(valid['planted_total'].sum())
            total_survived = int(valid['survived_count'].sum())

            # Average per farmer (only among farmers with >0 values)
            avg_planted_per_farmer = valid.loc[valid['planted_total'] > 0, 'planted_total'].mean() if (valid['planted_total'] > 0).any() else 0
            avg_survived_per_farmer = valid.loc[valid['survived_count'] > 0, 'survived_count'].mean() if (valid['survived_count'] > 0).any() else 0

            survival_rate = (total_survived / total_planted * 100) if total_planted > 0 else 0

            survival_data.append({
                'Species': species,
                'Total Planted': total_planted,
                'Total Survived': total_survived,
                'Average Planted (per farmer)': f"{avg_planted_per_farmer:.1f}",
                'Average Survived (per farmer)': f"{avg_survived_per_farmer:.1f}",
                'Survival Rate (%)': f"{survival_rate:.1f}%"
            })

        if survival_data:
            survival_df = pd.DataFrame(survival_data)

            col1, col2 = st.columns(2)
            with col1:
                # Average of species survival rates
                avg_survival = np.mean([float(x['Survival Rate (%)'].replace('%', '')) for x in survival_data])
                st.metric("Average Survival Rate", f"{avg_survival:.1f}%")
            with col2:
                total_planted_all = survival_df['Total Planted'].sum()
                st.metric("Total Trees Planted (all species)", f"{total_planted_all:,}")

            st.markdown("---")
            st.markdown("#### 📋 Survival Rate Details")
            st.dataframe(survival_df[['Species', 'Total Planted', 'Total Survived', 'Average Planted (per farmer)', 'Average Survived (per farmer)', 'Survival Rate (%)']], use_container_width=True, hide_index=True)

            # Visualization of survival rate by species
            survival_numeric = [float(x.replace('%', '')) for x in survival_df['Survival Rate (%)']]
            fig_survival = px.bar(
                survival_df,
                x='Species',
                y=survival_numeric,
                title='Early Survival Rates by Species',
                labels={'y': 'Survival Rate (%)', 'Species': 'Species'},
                color=survival_numeric,
                color_continuous_scale='viridis'
            )
            fig_survival.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_survival, use_container_width=True, key="survival_rates_chart")
        else:
            st.info("No early survival data found. Expected columns like `trees_survived.num_alive_coffee` or `oaf_trees.oaf_coffee.num_survived`.")

def main():
    # Load logo and create header
    logo_url = load_logo()
    
    if logo_url:
        st.markdown(f"""
            <div class='logo-header'>
                <img src='{logo_url}' width='80' height='80'>
                <div>
                    <h1 style='margin: 0; color: #1e5631;'>ET 2025 Amhara Planting Survey</h1>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='info-box'>
                <h2 style='color: white; margin: 0;'>🌳 ET 2025 Amhara Planting Survey</h2>
            </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
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
                    enumerators,
                    key="sidebar_selected_enumerator"
                )
            else:
                selected_enumerator = 'All'
                st.info("No username column found")
            
            # Treatment filter
            if 'treatment' in df.columns:
                treatments = ['All'] + sorted(df['treatment'].dropna().unique().tolist())
                selected_treatment = st.selectbox(
                    "Select Treatment:",
                    treatments,
                    key="sidebar_selected_treatment"
                )
            else:
                selected_treatment = 'All'
                st.info("No treatment column found")
            
            # Site filter
            if 'site' in df.columns:
                sites = ['All'] + sorted(df['site'].dropna().unique().tolist())
                selected_site = st.selectbox(
                    "Select Site:",
                    sites,
                    key="sidebar_selected_site"
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
        
        # Data export removed for security
        st.markdown("---")
        st.markdown("### 🔒 Data Security")
        st.info("Data export functionality has been disabled for security reasons.")
    
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
    
    # Display data if available
    if 'filtered_df' in st.session_state:
        df = st.session_state['filtered_df']
        
        # Create tabs according to new structure
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "📈 Progress", "✅ Data Quality", "🔬 Preliminary Results"
        ])
        
        with tab1:
            create_overview_tab(df)
        
        with tab2:
            create_progress_tab(df)
        
        with tab3:
            create_data_quality_tab(df)
        
        with tab4:
            create_preliminary_results_tab(df)
    
    else:
        # Empty state
        st.markdown("""
            <div style='text-align: center; padding: 60px; background-color: #f8fff9; border-radius: 10px; margin-top: 30px; border: 2px dashed #2E8B57;'>
                <h2 style='color: #1e5631;'>📊 No Data Loaded Yet</h2>
                <p style='color: #2E8B57; font-size: 1.1rem;'>
                    Click the "Fetch Latest Data" button in the sidebar to load data from CommCare
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()