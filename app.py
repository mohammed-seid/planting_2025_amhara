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

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Logo", width=150)
        st.markdown("---")
        st.markdown("### 📊 Dashboard Controls")
        
        # Add refresh timestamp
        if 'last_refresh' in st.session_state:
            st.info(f"🕒 Last updated: {st.session_state.last_refresh}")
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ Help & Info"):
            st.markdown("""
            **Quick Guide:**
            - Click 'Fetch Latest Data' to retrieve data
            - View summary metrics at the top
            - Download data as CSV
            - Explore columns and data types
            
            **Support:** [email protected]
            """)
        
        st.markdown("---")
        st.caption("ETH 2025 Planting Survey v1.0")
    
    # Main header with welcome message
    st.markdown("""
        <div class='info-box'>
            <h2 style='color: white; margin: 0;'>🌳 Welcome to ETH 2025 Planting Survey Dashboard</h2>
            <p style='color: white; margin-top: 10px; opacity: 0.9;'>
                Monitor and analyze planting survey data in real-time
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Action buttons row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        fetch_button = st.button("🔄 Fetch Latest Data", type="primary", use_container_width=True)
    with col2:
        if st.button("🔃 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
    with col3:
        if st.button("📖 Guide", use_container_width=True):
            st.info("Click 'Fetch Latest Data' to begin")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fetch data
    if fetch_button:
        with st.spinner("🔄 Fetching data from CommCare..."):
            records = fetch_commcare_data()
            
            if records:
                df = pd.DataFrame(records)
                st.session_state['df'] = df
                st.session_state['last_refresh'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.markdown("""
                    <div class='success-box'>
                        <h3 style='margin: 0;'>✅ Data Successfully Loaded!</h3>
                    </div>
                """, unsafe_allow_html=True)
    
    # Display data if available
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Metrics row with enhanced styling
        st.markdown("### 📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📋 Total Records", 
                value=f"{len(df):,}",
                delta="Active"
            )
        with col2:
            st.metric(
                label="🔢 Data Columns", 
                value=len(df.columns)
            )
        with col3:
            st.metric(
                label="💾 Memory Usage", 
                value=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            )
        with col4:
            st.metric(
                label="🕐 Last Updated", 
                value=st.session_state.get('last_refresh', 'N/A').split()[1] if 'last_refresh' in st.session_state else 'N/A'
            )
        
        st.markdown("---")
        
        # Data preview section
        st.markdown("### 📋 Data Preview")
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
                file_name=f"commcare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
                file_name=f"commcare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Full data view
        with st.expander("📊 View Full Dataset", expanded=False):
            st.dataframe(df, use_container_width=True, height=600)
    
    else:
        # Empty state - FIXED: Properly terminated string
        st.markdown("""
            <div style='text-align: center; padding: 60px; background-color: #f8fafc; border-radius: 10px; margin-top: 30px;'>
                <h2 style='color: #64748b;'>📊 No Data Loaded Yet</h2>
                <p style='color: #94a3b8; font-size: 1.1rem;'>
                    Click the "Fetch Latest Data" button above to load data from CommCare
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()