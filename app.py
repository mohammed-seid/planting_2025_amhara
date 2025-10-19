import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from snowflake.snowpark import Session

# Page configuration
st.set_page_config(
    page_title="ETH 2025 Tree Distribution Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegance
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stMetric > label {
        color: #1f77b4;
        font-size: 14px;
    }
    .stMetric > div > div > div {
        font-size: 24px;
        color: #2e7d32;
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
            del st.session_state["password"]  # Don't store password.
        else:
            st.session_state["password_correct"] = False

    # Return `True` if the user has already entered the correct password.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😞 Password incorrect")
    return False

if not check_password():
    st.stop()  # Do not continue if the password is wrong.

# Snowflake connection with OAuth
@st.cache_resource
def get_snowflake_session():
    """Create Snowflake session using OAuth authentication."""
    try:
        connection_parameters = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "authenticator": "oauth",
            "token": st.secrets["snowflake"]["token"],
            "role": st.secrets["snowflake"]["role"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
            "database": st.secrets["snowflake"]["database"],
            "schema": st.secrets["snowflake"]["schema"]
        }
        session = Session.builder.configs(connection_parameters).create()
        return session
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        st.stop()

# Load data from Snowflake
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Loads the T-Roster data from Snowflake, cleans it, and returns a Pandas DataFrame."""
    try:
        session = get_snowflake_session()
        
        # Construct the fully qualified table name
        database_name = st.secrets["snowflake"]["database"]
        schema_name = st.secrets["snowflake"]["schema"]
        table_name = st.secrets["snowflake"]["table_name"]
        
        fully_qualified_table_name = f'"{database_name}"."{schema_name}"."{table_name}"'
        
        # Load data from Snowflake
        snowpark_df = session.table(fully_qualified_table_name)
        df = snowpark_df.to_pandas()

        # Data cleaning for robustness
        tree_columns = [
            'GESHO', 'GREVILLEA', 'DECURRENS', 'WANZA', 'PAPAYA', 'MORINGA', 
            'COFFEE', 'ARZELIBANOS', 'GUAVA', 'PEPPER', 'AVOCADO', 'LEMON'
        ]
        
        # Clean tree columns
        for col in tree_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Clean other numeric columns
        numeric_columns = ['TOTAL_SEEDLING', 'HAS_PHONE', 'AGE', 'MALE_YOUTH', 'FEMALE_YOUTH']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure HAS_PHONE is boolean/numeric
        if 'HAS_PHONE' in df.columns:
            df['HAS_PHONE'] = df['HAS_PHONE'].astype(int)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data from Snowflake: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

if df.empty:
    st.warning("No data loaded. Please check the Snowflake connection and table name.")
    st.stop()

# Sidebar filters - one per row
st.sidebar.title("🌳 Filters")
st.sidebar.markdown("---")

region = st.sidebar.multiselect("Region", options=sorted(df['REGION'].dropna().unique()), key="region")
purpose = st.sidebar.multiselect("Purpose", options=sorted(df['PURPOSE'].dropna().unique()), key="purpose")
cluster = st.sidebar.multiselect("Cluster", options=sorted(df['CLUSTER'].dropna().unique()), key="cluster")
woreda = st.sidebar.multiselect("Woreda", options=sorted(df['WOREDA'].dropna().unique()), key="woreda")
program = st.sidebar.multiselect("Program", options=sorted(df['PROGRAM'].dropna().unique()), key="program")

# Apply filters
filtered_df = df.copy()
for col, vals in zip(['REGION', 'PURPOSE', 'CLUSTER', 'WOREDA', 'PROGRAM'], [region, purpose, cluster, woreda, program]):
    if vals:
        filtered_df = filtered_df[filtered_df[col].isin(vals)]

st.sidebar.markdown("---")
st.sidebar.info(f"Showing {len(filtered_df):,} records")

if st.sidebar.button("🔄 Reset"):
    st.rerun()

# Main title
st.title("🌳 ETH 2025 Tree Distribution Dashboard")
st.markdown("---")

# Tree columns for reference
tree_columns = [
    'GESHO', 'GREVILLEA', 'DECURRENS', 'WANZA', 'PAPAYA', 'MORINGA', 
    'COFFEE', 'ARZELIBANOS', 'GUAVA', 'PEPPER', 'AVOCADO', 'LEMON'
]
existing_tree_cols = [col for col in tree_columns if col in filtered_df.columns]

# Tabs for structure
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🌿 Tree Distribution", "📱 Phone Analysis", "👥 Demographics", "📋 Data", "🔍 Species Stats"])

with tab1:
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_farmers = len(filtered_df)
        st.metric("Farmers Served", f"{total_farmers:,}")
    
    with col2:
        total_trees = int(filtered_df['TOTAL_SEEDLING'].sum())
        st.metric("Total Trees", f"{total_trees:,}")
    
    with col3:
        avg_trees = round(total_trees / total_farmers, 1) if total_farmers > 0 else 0
        st.metric("Avg Trees/Farmer", avg_trees)
    
    with col4:
        phone_owners = int(filtered_df['HAS_PHONE'].sum())
        phone_pct = round((phone_owners / total_farmers) * 100, 1) if total_farmers > 0 else 0
        st.metric("Phone Ownership", f"{phone_pct}%")
    
    with col5:
        unique_woredas = filtered_df['WOREDA'].nunique()
        st.metric("Unique Woredas", unique_woredas)
    
    with col6:
        unique_kebeles = filtered_df['KEBELE'].nunique()
        st.metric("Unique Kebeles", unique_kebeles)
    
    # Purpose analysis
    if 'PURPOSE' in filtered_df.columns and not filtered_df.empty:
        purpose_farmers = filtered_df['PURPOSE'].value_counts()
        purpose_pct_farmers = (purpose_farmers / total_farmers * 100).round(1)
        
        purpose_trees = filtered_df.groupby('PURPOSE')['TOTAL_SEEDLING'].sum()
        purpose_pct_trees = (purpose_trees / total_trees * 100).round(1) if total_trees > 0 else pd.Series(0, index=purpose_trees.index)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Farmers by Purpose (%)")
            fig_pie_farmers = px.pie(
                values=purpose_pct_farmers.values, 
                names=purpose_pct_farmers.index,
                title="Farmers % by Purpose", 
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie_farmers, use_container_width=True)
        
        with col2:
            st.subheader("Trees by Purpose (%)")
            fig_pie_trees = px.pie(
                values=purpose_pct_trees.values, 
                names=purpose_pct_trees.index,
                title="Trees % by Purpose", 
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie_trees, use_container_width=True)

with tab2:
    st.subheader("Tree Species Insights")
    
    if existing_tree_cols and not filtered_df.empty:
        tree_data = filtered_df[existing_tree_cols].fillna(0).astype(float)
        tree_totals = tree_data.sum().sort_values(ascending=False)
        
        # Compact bar chart
        fig_bar = px.bar(
            x=tree_totals.index, 
            y=tree_totals.values,
            title="Seedlings by Species", 
            color=tree_totals.values,
            color_continuous_scale='Greens', 
            labels={'x': 'Species', 'y': 'Count'}
        )
        fig_bar.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Pie chart for percentages
        fig_pie = px.pie(
            values=tree_totals.values, 
            names=tree_totals.index,
            title="Species Distribution (%)", 
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed stats with min, max, avg, std, median, 75%
        with st.expander("Detailed Stats per Species"):
            stats_data = []
            for species in existing_tree_cols:
                took = filtered_df[filtered_df[species] > 0][species]
                n_took = len(took)
                if n_took > 0:
                    desc = took.describe()
                    stats_data.append({
                        'Species': species,
                        'Farmers': n_took,
                        'Min': round(desc['min'], 1),
                        'Max': round(desc['max'], 1),
                        'Avg': round(desc['mean'], 1),
                        'Std': round(desc['std'], 1),
                        'Median': round(desc['50%'], 1),
                        '75%': round(desc['75%'], 1),
                        'Total': int(took.sum())
                    })
                else:
                    stats_data.append({
                        'Species': species, 
                        'Farmers': 0, 
                        'Min': 0, 
                        'Max': 0, 
                        'Avg': 0, 
                        'Std': 0, 
                        'Median': 0, 
                        '75%': 0, 
                        'Total': 0
                    })
            
            stats_df = pd.DataFrame(stats_data).sort_values('Farmers', ascending=False)
            st.dataframe(stats_df, use_container_width=True)

with tab3:
    st.subheader("Phone Ownership Comparison")
    
    if 'HAS_PHONE' in filtered_df.columns and existing_tree_cols and not filtered_df.empty:
        owners = filtered_df[filtered_df['HAS_PHONE'] == 1]
        non_owners = filtered_df[filtered_df['HAS_PHONE'] == 0]
        
        n_owners, n_non = len(owners), len(non_owners)
        avg_trees_o = round(owners['TOTAL_SEEDLING'].sum() / n_owners, 1) if n_owners > 0 else 0
        avg_trees_n = round(non_owners['TOTAL_SEEDLING'].sum() / n_non, 1) if n_non > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Owners", n_owners)
            st.metric("Avg Trees", avg_trees_o)
        with col2:
            st.metric("Non-Owners", n_non)
            st.metric("Avg Trees", avg_trees_n)
        
        # Overall stats for owners and non-owners
        with st.expander("Overall Detailed Stats"):
            o_desc = owners['TOTAL_SEEDLING'].describe() if n_owners > 0 else pd.Series([0]*8, index=['count','mean','std','min','25%','50%','75%','max'])
            n_desc = non_owners['TOTAL_SEEDLING'].describe() if n_non > 0 else pd.Series([0]*8, index=['count','mean','std','min','25%','50%','75%','max'])
            
            overall_stats = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std', 'Min', 'Median', '75%', 'Max'],
                'Owners': [
                    int(o_desc['count']), 
                    round(o_desc['mean'], 1), 
                    round(o_desc['std'], 1), 
                    round(o_desc['min'], 1), 
                    round(o_desc['50%'], 1), 
                    round(o_desc['75%'], 1), 
                    round(o_desc['max'], 1)
                ],
                'Non-Owners': [
                    int(n_desc['count']), 
                    round(n_desc['mean'], 1), 
                    round(n_desc['std'], 1), 
                    round(n_desc['min'], 1), 
                    round(n_desc['50%'], 1), 
                    round(n_desc['75%'], 1), 
                    round(n_desc['max'], 1)
                ]
            })
            
            st.dataframe(overall_stats, use_container_width=True)
with tab4:
    st.subheader("Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'SEX' in filtered_df.columns:
            gender_counts = filtered_df['SEX'].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values, 
                names=gender_counts.index,
                title="Gender Distribution", 
                color_discrete_sequence=['#ff7f0e', '#1f77b4']
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        if 'AGE' in filtered_df.columns:
            fig_age = px.histogram(
                filtered_df, 
                x='AGE', 
                nbins=20, 
                title="Age Distribution",
                color_discrete_sequence=['#2e7d32']
            )
            fig_age.update_layout(
                xaxis_title="Age",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_age, use_container_width=True)
    
    # Youth participation bar chart
    if 'MALE_YOUTH' in filtered_df.columns and 'FEMALE_YOUTH' in filtered_df.columns:
        male_y = int(filtered_df['MALE_YOUTH'].sum())
        female_y = int(filtered_df['FEMALE_YOUTH'].sum())
        youth_data = pd.DataFrame({
            'Category': ['Male Youth', 'Female Youth'], 
            'Count': [male_y, female_y]
        })
        fig_youth = px.bar(
            youth_data, 
            x='Category', 
            y='Count', 
            title="Youth Participation",
            color='Category', 
            color_discrete_map={'Male Youth': '#ff7f0e', 'Female Youth': '#d62728'}
        )
        fig_youth.update_layout(showlegend=False)
        st.plotly_chart(fig_youth, use_container_width=True)
    
    # Region distribution
    if 'REGION' in filtered_df.columns:
        region_counts = filtered_df['REGION'].value_counts()
        fig_region = px.bar(
            x=region_counts.index, 
            y=region_counts.values, 
            title="Distribution by Region",
            color=region_counts.values, 
            color_continuous_scale='Blues',
            labels={'x': 'Region', 'y': 'Count'}
        )
        fig_region.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_region, use_container_width=True)
    
    # Additional demographic insights
    col1, col2 = st.columns(2)
    
    with col1:
        if 'CLUSTER' in filtered_df.columns:
            cluster_counts = filtered_df['CLUSTER'].value_counts().head(10)
            fig_cluster = px.bar(
                x=cluster_counts.index, 
                y=cluster_counts.values,
                title="Top 10 Clusters by Farmers",
                color=cluster_counts.values,
                color_continuous_scale='Greens',
                labels={'x': 'Cluster', 'y': 'Farmers'}
            )
            fig_cluster.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        if 'WOREDA' in filtered_df.columns:
            woreda_counts = filtered_df['WOREDA'].value_counts().head(10)
            fig_woreda = px.bar(
                x=woreda_counts.index, 
                y=woreda_counts.values,
                title="Top 10 Woredas by Farmers",
                color=woreda_counts.values,
                color_continuous_scale='Oranges',
                labels={'x': 'Woreda', 'y': 'Farmers'}
            )
            fig_woreda.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_woreda, use_container_width=True)

with tab5:
    st.subheader("Raw Data")
    
    # Add search functionality
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search in data", "", placeholder="Search by name, location, etc.")
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        show_all = st.checkbox("Show all columns", value=True)
    
    # Filter data based on search
    display_df = filtered_df.copy()
    if search_term:
        # Search across all string columns
        mask = display_df.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
        st.info(f"Found {len(display_df)} matching records")
    
    # Column selection
    if not show_all:
        key_columns = ['REGION', 'WOREDA', 'KEBELE', 'PURPOSE', 'TOTAL_SEEDLING', 'HAS_PHONE']
        available_key_cols = [col for col in key_columns if col in display_df.columns]
        display_df = display_df[available_key_cols]
    
    # Display dataframe with formatting
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records Shown", f"{len(display_df):,}")
    with col2:
        st.metric("Total Columns", len(display_df.columns))
    with col3:
        st.metric("Memory Usage", f"{display_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Download options
    st.markdown("---")
    st.subheader("Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Data as CSV",
            data=csv,
            file_name=f"eth_tree_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel download (optional - requires openpyxl)
        try:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Tree Data')
            buffer.seek(0)
            
            st.download_button(
                label="📊 Download Full Data as Excel",
                data=buffer,
                file_name=f"eth_tree_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.info("Excel download requires openpyxl package")

with tab6:
    st.subheader("Species Stats Selector")
    
    if existing_tree_cols:
        # Species selection with select all option
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_species = st.multiselect(
                "Select Species to Analyze", 
                options=existing_tree_cols, 
                default=existing_tree_cols[:3]
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("Select All"):
                selected_species = existing_tree_cols
                st.rerun()
        
        if selected_species:
            # Calculate statistics
            stats_data = []
            for species in selected_species:
                took = filtered_df[filtered_df[species] > 0]
                n_took = len(took)
                
                if n_took > 0:
                    taken_values = took[species]
                    desc = taken_values.describe()
                    phone_pct = round((took['HAS_PHONE'].sum() / n_took) * 100, 1)
                    
                    stats_data.append({
                        'Species': species,
                        'Farmers Took': n_took,
                        'Avg Taken': round(desc['mean'], 1),
                        'Min': int(desc['min']),
                        'Max': int(desc['max']),
                        '% Phone Owners': phone_pct,
                        'Median': round(desc['50%'], 1),
                        'Std': round(desc['std'], 1),
                        'Total Taken': int(taken_values.sum())
                    })
                else:
                    stats_data.append({
                        'Species': species,
                        'Farmers Took': 0,
                        'Avg Taken': 0,
                        'Min': 0,
                        'Max': 0,
                        '% Phone Owners': 0,
                        'Median': 0,
                        'Std': 0,
                        'Total Taken': 0
                    })
            
            species_stats_df = pd.DataFrame(stats_data)
            
            # Display stats table
            st.dataframe(
                species_stats_df.style.background_gradient(
                    subset=['Farmers Took', 'Total Taken'], 
                    cmap='Greens'
                ),
                use_container_width=True
            )
            
            # Visual comparisons
            st.markdown("---")
            st.subheader("Visual Comparisons")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Farmers who took each species
                fig_farmers = px.bar(
                    species_stats_df,
                    x='Species',
                    y='Farmers Took',
                    title="Number of Farmers per Species",
                    color='Farmers Took',
                    color_continuous_scale='Blues'
                )
                fig_farmers.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig_farmers, use_container_width=True)
            
            with col2:
                # Average taken per species
                fig_avg = px.bar(
                    species_stats_df,
                    x='Species',
                    y='Avg Taken',
                    title="Average Seedlings Taken per Species",
                    color='Avg Taken',
                    color_continuous_scale='Greens'
                )
                fig_avg.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig_avg, use_container_width=True)
            
            # Distribution analysis
            st.markdown("---")
            st.subheader("Distribution Analysis")
            
            # Create box plots for selected species
            if len(selected_species) > 0:
                box_data = []
                for species in selected_species:
                    species_values = filtered_df[filtered_df[species] > 0][species]
                    for val in species_values:
                        box_data.append({'Species': species, 'Count': val})
                
                if box_data:
                    box_df = pd.DataFrame(box_data)
                    fig_box = px.box(
                        box_df,
                        x='Species',
                        y='Count',
                        title="Distribution of Seedlings per Species (Box Plot)",
                        color='Species'
                    )
                    fig_box.update_layout(xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Phone ownership correlation
            st.markdown("---")
            st.subheader("Phone Ownership Analysis")
            
            phone_corr_data = []
            for species in selected_species:
                took_species = filtered_df[filtered_df[species] > 0]
                if len(took_species) > 0:
                    phone_owners = took_species['HAS_PHONE'].sum()
                    phone_pct = (phone_owners / len(took_species)) * 100
                    phone_corr_data.append({
                        'Species': species,
                        'Phone Ownership %': round(phone_pct, 1)
                    })
            
            if phone_corr_data:
                phone_corr_df = pd.DataFrame(phone_corr_data)
                fig_phone = px.bar(
                    phone_corr_df,
                    x='Species',
                    y='Phone Ownership %',
                    title="Phone Ownership Rate by Species",
                    color='Phone Ownership %',
                    color_continuous_scale='RdYlGn'
                )
                fig_phone.update_layout(xaxis_tickangle=-45, showlegend=False)
                fig_phone.add_hline(
                    y=filtered_df['HAS_PHONE'].mean() * 100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Overall Avg"
                )
                st.plotly_chart(fig_phone, use_container_width=True)
        else:
            st.info("👆 Please select at least one species to view statistics")

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: gray;'>"
    f"ETH 2025 Tree Dashboard • Powered by Streamlit & Snowflake • "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</p>", 
    unsafe_allow_html=True
)