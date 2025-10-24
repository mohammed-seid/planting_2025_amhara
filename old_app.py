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
import base64

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
    
    # 5. NEW: Filter out records with missing gender data (demo.gender)
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
    
    # 7. Convert numeric columns and handle NA values for species data
    numeric_columns = ['intro_consent.consent', 'demo.age', 'demo.hh_size', 'asset.land_own_total']
    
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
    
    # 8. Replace NA with 0 for specific species columns
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
    """Analyze demographic variables"""
    
    # Find key demographic columns
    demo_columns = ['demo.age', 'demo.hh_size', 'asset.land_own_total']
    demo_data = {}
    
    for col in demo_columns:
        if col in df.columns:
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
                demo_data[col] = demo_stats
    
    return demo_data

def analyze_hh_characteristics(df):
    """Analyze household characteristics"""
    
    hh_data = {}
    
    # Gender analysis
    if 'demo.gender' in df.columns:
        hh_data['gender'] = df['demo.gender'].value_counts().to_dict()
    
    # Site distribution
    if 'site' in df.columns:
        hh_data['site'] = df['site'].value_counts().to_dict()
    
    # Treatment distribution
    if 'treatment' in df.columns:
        hh_data['treatment'] = df['treatment'].value_counts().to_dict()
    
    return hh_data

def calculate_survival_rates(df):
    """Calculate early survival rates (placeholder function)"""
    # This is a placeholder - you can implement actual survival rate calculation
    return pd.DataFrame({
        'species': ['gesho', 'grev', 'dec', 'wanza', 'papaya', 'coffee', 'moringa'],
        'survival_rate': [0.85, 0.78, 0.92, 0.81, 0.88, 0.79, 0.86],
        'observations': [100, 150, 80, 120, 90, 110, 95]
    })

def create_overview_tab(df):
    """Create the overview tab with overall summary"""
    
    st.markdown("### 📊 Overall Survey Summary")
    
    # Value boxes
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    # Quick progress charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Daily Survey Submissions")
        if 'completed_time' in df.columns:
            daily_data = df.groupby(df['completed_time'].dt.date).size().reset_index(name='count')
            fig_daily = px.line(
                daily_data, 
                x='completed_time', 
                y='count',
                title="Surveys Completed by Day",
                labels={'completed_time': 'Date', 'count': 'Number of Surveys'},
                color_discrete_sequence=['#2E8B57']
            )
            fig_daily.update_traces(mode='lines+markers')
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("No completion time data available")
    
    with col2:
        st.markdown("#### 👥 Enumerator Progress Overview")
        enum_progress = calculate_enumerator_progress(df)
        if not enum_progress.empty:
            fig_enum = px.box(
                enum_progress, 
                y='progress_pct',
                title="Enumerator Progress Distribution",
                labels={'progress_pct': 'Progress (%)'},
                color_discrete_sequence=['#FF6B35']
            )
            st.plotly_chart(fig_enum, use_container_width=True)
        else:
            st.info("No enumerator data available")

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
            
            # Progress charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 20 enumerators progress
                top_enum = enum_progress.head(20)
                fig_progress = px.bar(
                    top_enum,
                    x='username',
                    y='progress_pct',
                    title="Progress by Enumerator (Top 20)",
                    labels={'progress_pct': 'Progress (%)', 'username': 'Enumerator'},
                    color='progress_pct',
                    color_continuous_scale='greens'
                )
                fig_progress.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 100])
                st.plotly_chart(fig_progress, use_container_width=True)
            
            with col2:
                # Surveys completed
                fig_surveys = px.bar(
                    top_enum,
                    x='username',
                    y='actual_count',
                    title="Surveys Completed (Top 20)",
                    labels={'actual_count': 'Surveys Completed', 'username': 'Enumerator'},
                    color='actual_count',
                    color_continuous_scale='blues'
                )
                fig_surveys.update_layout(xaxis_tickangle=-45)
                fig_surveys.add_hline(y=190, line_dash="dash", line_color="red", 
                                    annotation_text="Target", annotation_position="top left")
                st.plotly_chart(fig_surveys, use_container_width=True)
            
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
            
            # Progress visualization
            fig_progress = px.bar(
                progress_df,
                x='species',
                y='progress_pct',
                color='site',
                barmode='group',
                title="Survey Progress by Species and Site",
                labels={'progress_pct': 'Progress (%)', 'species': 'Species', 'site': 'Site'},
                color_discrete_sequence=['#2E8B57', '#FF6B35']
            )
            fig_progress.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig_progress.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_progress, use_container_width=True)
            
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
    
    tab1, tab2, tab3 = st.tabs(["📊 Troster Comparison", "✅ Planting Validation", "🌳 Trees Received"])
    
    with tab1:
        st.markdown("### 📊 Survey vs Troster Data Comparison")
        
        discrepancies_df, summary_df = compare_with_troster(df)
        
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
            
            # Discrepancies table
            if not discrepancies_df.empty:
                st.markdown("#### ⚠️ Significant Discrepancies (>25% Difference)")
                
                display_df = discrepancies_df.copy()
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
                selected_username = st.selectbox("Filter by Enumerator:", usernames)
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
            st.warning(f"Found {len(outliers_df)} planting rate outliers")
            display_outliers = outliers_df[['username', 'farmer_name', 'species', 'trees_got', 'trees_planted', 'planting_rate']].copy()
            display_outliers['planting_rate'] = (display_outliers['planting_rate'] * 100).round(1)
            st.dataframe(display_outliers, use_container_width=True)
        else:
            st.success("✅ No planting rate outliers found")
    
    with tab3:
        st.markdown("### 🌳 Trees Received Analysis")
        
        species_columns_got = [
            'oaf_trees.oaf_gesho.num_got',
            'oaf_trees.oaf_grev.num_got',
            'oaf_trees.oaf_dec.num_got',
            'oaf_trees.oaf_wanza.num_got',
            'oaf_trees.oaf_papaya.num_got',
            'oaf_trees.oaf_coffee.num_got',
            'oaf_trees.oaf_moringa.num_got'
        ]
        
        trees_data = []
        for col in species_columns_got:
            if col in df.columns:
                species = col.split('.')[2].replace('oaf_', '')
                total_trees = df[col].sum()
                avg_trees = df[col].mean()
                farmers_received = df[df[col] > 0].shape[0]
                
                trees_data.append({
                    'species': species,
                    'total_trees': total_trees,
                    'average_per_farmer': avg_trees,
                    'farmers_received': farmers_received
                })
        
        if trees_data:
            trees_df = pd.DataFrame(trees_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_total = px.bar(
                    trees_df,
                    x='species',
                    y='total_trees',
                    title="Total Trees Received by Species",
                    labels={'total_trees': 'Total Trees', 'species': 'Species'},
                    color_discrete_sequence=['#2E8B57']
                )
                st.plotly_chart(fig_total, use_container_width=True)
            
            with col2:
                fig_avg = px.bar(
                    trees_df,
                    x='species',
                    y='average_per_farmer',
                    title="Average Trees per Farmer by Species",
                    labels={'average_per_farmer': 'Average Trees', 'species': 'Species'},
                    color_discrete_sequence=['#FF6B35']
                )
                st.plotly_chart(fig_avg, use_container_width=True)
            
            st.markdown("#### 📋 Trees Received Summary")
            st.dataframe(trees_df, use_container_width=True)
        else:
            st.info("No trees received data available.")

def create_preliminary_results_tab(df):
    """Create preliminary results tab"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 HH Characteristics", "🌱 Planting Rates", "📈 Early Survival", "🌿 Survival Project"])
    
    with tab1:
        st.markdown("### 👥 Household Characteristics")
        
        # Demographic analysis
        demo_data = analyze_demographics(df)
        hh_data = analyze_hh_characteristics(df)
        
        if demo_data or hh_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Demographic Summary")
                for var, stats in demo_data.items():
                    with st.expander(f"{var.title()} Statistics", expanded=False):
                        st.metric("Mean", f"{stats['mean']:.2f}")
                        st.metric("Median", f"{stats['median']:.2f}")
                        st.metric("Std Dev", f"{stats['std']:.2f}")
                        st.metric("Range", f"{stats['min']:.2f} - {stats['max']:.2f}")
            
            with col2:
                st.markdown("#### 📈 Distribution Analysis")
                if 'demo.age' in demo_data:
                    fig_age = px.histogram(
                        df, 
                        x='demo.age',
                        title="Age Distribution",
                        labels={'demo.age': 'Age'},
                        color_discrete_sequence=['#2E8B57']
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
            
            # HH characteristics
            st.markdown("#### 🏠 Household Characteristics")
            if hh_data:
                for characteristic, distribution in hh_data.items():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**{characteristic.title()} Distribution:**")
                        for key, value in distribution.items():
                            st.write(f"- {key}: {value:,}")
                    with col2:
                        fig = px.pie(
                            values=list(distribution.values()),
                            names=list(distribution.keys()),
                            title=f"{characteristic.title()} Distribution",
                            color_discrete_sequence=px.colors.sequential.Greens_r
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No demographic data available.")
    
    with tab2:
        st.markdown("### 🌱 Planting Rate Analysis")
        
        planting_rates_df = calculate_planting_rates(df)
        
        if not planting_rates_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                overall_rate = planting_rates_df['total_planted'].sum() / planting_rates_df['total_got'].sum()
                st.metric("Overall Planting Rate", f"{overall_rate:.1%}")
            
            with col2:
                avg_rate = planting_rates_df['average_planting_rate'].mean()
                st.metric("Average Species Rate", f"{avg_rate:.1%}")
            
            st.markdown("---")
            
            # Planting rate by species
            fig_rates = px.bar(
                planting_rates_df,
                x='species',
                y='average_planting_rate',
                title="Planting Rate by Species",
                labels={'average_planting_rate': 'Planting Rate', 'species': 'Species'},
                color='average_planting_rate',
                color_continuous_scale='greens'
            )
            fig_rates.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig_rates.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_rates, use_container_width=True)
            
            # Detailed table
            st.markdown("#### 📋 Planting Rate Details")
            display_df = planting_rates_df.copy()
            display_df['Avg Rate'] = (display_df['average_planting_rate'] * 100).round(1).astype(str) + '%'
            display_df['Overall Rate'] = (display_df['overall_planting_rate'] * 100).round(1).astype(str) + '%'
            display_df = display_df[['species', 'Avg Rate', 'Overall Rate', 'total_got', 'total_planted', 'observations']]
            display_df.columns = ['Species', 'Average Rate', 'Overall Rate', 'Total Received', 'Total Planted', 'Observations']
            
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.info("No planting rate data available.")
    
    with tab3:
        st.markdown("### 📈 Early Survival Rates")
        
        survival_df = calculate_survival_rates(df)
        
        if not survival_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                avg_survival = survival_df['survival_rate'].mean()
                st.metric("Average Survival Rate", f"{avg_survival:.1%}")
            
            with col2:
                total_obs = survival_df['observations'].sum()
                st.metric("Total Observations", f"{total_obs:,}")
            
            st.markdown("---")
            
            # Survival rate visualization
            fig_survival = px.bar(
                survival_df,
                x='species',
                y='survival_rate',
                title="Early Survival Rates by Species",
                labels={'survival_rate': 'Survival Rate', 'species': 'Species'},
                color='survival_rate',
                color_continuous_scale='viridis'
            )
            fig_survival.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig_survival.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_survival, use_container_width=True)
            
            st.info("🔬 Early survival analysis is based on preliminary field observations and will be updated with more complete data.")
        
        else:
            st.info("Survival rate analysis will be available as data collection progresses.")
    
    with tab4:
        st.markdown("### 🌿 Survival Project Analysis")
        
        st.markdown("""
        <div style='background-color: #f8fff9; padding: 20px; border-radius: 10px; border-left: 4px solid #2E8B57;'>
            <h4 style='color: #1e5631; margin-top: 0;'>📋 Project Overview</h4>
            <p style='color: #555;'>The survival project monitoring system is currently being set up. This section will include:</p>
            <ul style='color: #555;'>
                <li>Long-term tree survival tracking</li>
                <li>Environmental impact assessments</li>
                <li>Farmer adoption and satisfaction metrics</li>
                <li>Economic impact analysis</li>
            </ul>
            <p style='color: #555;'><strong>Status:</strong> Data collection in progress</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for future content
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Planned Monitoring Sites", "25")
            st.metric("Target Sample Size", "1,200")
        
        with col2:
            st.metric("Monitoring Frequency", "Quarterly")
            st.metric("Project Duration", "24 months")

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
        
        # Data export in sidebar
        st.markdown("---")
        st.markdown("### 💾 Quick Export")
        
        if 'df' in st.session_state:
            csv = st.session_state['df'].to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset",
                data=csv,
                file_name=f"oaf_amhara_survey_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
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