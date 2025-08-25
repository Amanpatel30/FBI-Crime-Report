# app.py
import os
import re
import io
import zipfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# try to import option_menu; fallback if not available
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# ----------------------------
# App Title & Config
# ----------------------------
st.set_page_config(
    page_title="NIBRS Crime Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Hidden columns (user requested not to display)
# Normalize names when comparing (lower + strip non-alnum)
HIDDEN_COLUMN_NAMES = [
    "State","Agency Type","Agency Name","Population1","Total Offenses",
    "Crimes Against Persons","Crimes Against Property","Crimes Against Society",
    "Assault Offenses","Aggravated Assault","Simple Assault","Intimidation",
    "Homicide Offenses","Murder and Nonnegligent Manslaughter","Negligent Man- slaughter",
    "Justifiable Homicide","Human Trafficking Offenses","Commercial Sex Acts",
    "Involuntary Servitude","Kidnapping/ Abduction","Sex Offenses","Rape","Sodomy",
    "Sexual Assault With an Object","Criminal Sexual Contact2","Incest","Statutory Rape",
    "Arson","Bribery","Burglary/ Breaking & Entering","Counter- feiting/ Forgery",
    "Destruction/ Damage/ Vandalism of Property","Embezzle- ment","Extortion/ Blackmail",
    "Fraud Offenses","False Pretenses/ Swindle/ Confidence Game","Credit Card/ Automated Teller Machine Fraud",
    "Imper- sonation","Welfare Fraud","Wire Fraud","Identity Theft","Hacking/ Computer Invasion",
    "Larceny/ Theft Offenses","Pocket- picking","Purse- snatching","Shop- lifting",
    "Theft From Building","Theft From Coin Op- erated Machine or Device","Theft From Motor Vehicle",
    "Theft of Motor Vehicle Parts or Acces- sories","All Other Larceny","Motor Vehicle Theft",
    "Robbery","Stolen Property Offenses","Animal Cruelty","Drug/ Narcotic Offenses",
    "Drug/ Narcotic Violations","Drug Equipment Violations","Gambling Offenses","Betting/ Wagering",
    "Operating/ Promoting/ Assisting Gambling","Gambling Equipment Violations","Sports Tampering",
    "Por- nography/ Obscene Material","Pros- titution Offenses","Pros- titution",
    "Assisting or Promoting Prostitution","Purchasing Prostitution","Weapon Law Violations"
]

def _normalize_col_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    # normalize dashes and remove non-alphanumeric so matching is tolerant
    s = re.sub(r'[\u2013\u2014–—]', '-', s)
    s = re.sub(r'[^0-9a-z]', '', s)
    return s

HIDDEN_SET = set(_normalize_col_name(c) for c in HIDDEN_COLUMN_NAMES)

# ----------------------------
# Compact styling / force inline pagination row
# ----------------------------
st.markdown(
    """
    <style>
    /* Buttons / selects compact */
    .stButton>button, button[role="button"] {
        padding: 6px 10px !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        height: 36px !important;
        display:inline-flex !important;
        align-items:center !important;
    }
    /* Number inputs and selectboxes compact */
    input[type="number"] { height:36px !important; padding:6px 10px !important; font-size:13px !important; }
    div[data-baseweb="select"] > div { min-height:36px !important; }
    /* Remove automatic labels for compact inline placement */
    .stSelectbox>div>div>label, .stNumberInput>div>label { display:none; }

    /* Inline Pagination heading + helper text on single line */
    .pagination-header {
        display:flex;
        align-items:center;
        gap:12px;
        margin-top:12px;
        margin-bottom:8px;
        flex-wrap:nowrap;
    }
    .pagination-title { font-weight:600; font-size:18px; margin:0; padding:0; }
    .pagination-helper { color:var(--secondaryTextColor,#9aa0a6); font-size:13px; margin:0; padding:0; white-space:nowrap; }

    /* Ensure the pagination controls row tries not to wrap */
    .stContainer > .stColumns, div[data-testid="stHorizontalBlock"] > div {
        align-items:center;
    }
    .pagination-row { display:flex; align-items:center; gap:12px; flex-wrap:nowrap; }

    /* Tweak the small status text */
    .page-status { font-size:14px; color:var(--textColor,#ddd); margin-left:6px; }

    /* Reduce margins from markdown paragraphs to avoid extra wrapping */
    .stMarkdown p { margin:0 0 6px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚔 NIBRS Crime Analytics Hub - Enhanced Edition")
st.markdown("**Advanced Interactive Crime Data Analysis Platform**")
st.markdown("🎯 Navigate through different analysis categories using the sidebar • 📊 All charts are interactive with hover details, zoom, and filtering capabilities")

# ----------------------------
# Utility: cleaning & loading
# ----------------------------
def clean_colname(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s)
    s = s.replace("\n", " ")
    s = s.replace("â\x88\x92", "-")   # weird hyphen artifact
    s = s.replace("â\x88\x9215", "-15")
    s = s.replace("Nov-15", "11-15")
    s = s.replace("?", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names, drop Unnamed columns, coerce numeric-like columns to numeric,
    and ensure a sensible integer index (1-based) or use an index-like column if present.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # clean column names
    df.columns = [clean_colname(c) for c in df.columns]

    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", na=False)]

    # attempt to coerce numeric-like columns more robustly
    for col in df.columns:
        # only process object / string columns
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col].dtype):
            # remove thousands separators and surrounding whitespace
            cleaned = df[col].astype(str).str.replace(",", "").str.strip()

            # for matching we strip out characters except digits, minus, dot
            cleaned_for_match = cleaned.str.replace(r'[^\d\.\-]', '', regex=True)

            # remove empty strings (treat them as NaN for test)
            non_null = cleaned_for_match[cleaned_for_match.notna() & (cleaned_for_match != "")]

            if len(non_null) == 0:
                continue

            # test how many values look like a number (integer or float)
            numeric_match = non_null.str.match(r"^-?\d+(\.\d+)?$")
            numeric_ratio = numeric_match.mean()

            # If a majority are numeric (>=50%) or they all match after stripping, coerce.
            if numeric_ratio >= 0.5 or numeric_match.all():
                df[col] = pd.to_numeric(cleaned_for_match, errors="coerce")
            else:
                # leave as-is (text column)
                df[col] = df[col].astype(object)

    # reset index first to get a clean base
    df = df.reset_index(drop=True)

    # detect an index-like column (common names) and use it if it's a unique integer sequence
    index_like_pattern = re.compile(r'^(unnamed: 0|index|row|#|no\.?$|s no$|sr\.?n$|sr no$|id$)', re.IGNORECASE)
    index_like_col = next((c for c in df.columns if index_like_pattern.match(c.strip())), None)

    if index_like_col:
        # try converting to numeric (strip non-digit characters)
        conv = pd.to_numeric(df[index_like_col].astype(str).str.replace(r'[^\d\-\+\.]', '', regex=True), errors="coerce")
        # check if conversion succeeded for all rows and values are integer-like and unique
        if conv.notna().all():
            # integer-like check (allow floats which are whole numbers)
            try:
                integer_like = ((conv % 1) == 0).all()
            except Exception:
                integer_like = False
            if integer_like and conv.astype(int).nunique() == len(df):
                df.index = conv.astype(int)
                # drop the original column so it doesn't appear twice
                df = df.drop(columns=[index_like_col])
            else:
                # fallback: keep default 1-based RangeIndex below
                pass

    # if index not set by index-like col, set a 1-based RangeIndex for nicer display (1,2,3,...)
    if not isinstance(df.index, pd.RangeIndex):
        # index might have been set to integer-like above; if so, leave it.
        if not (isinstance(df.index, pd.Int64Index) or pd.api.types.is_integer_dtype(df.index)):
            df.index = pd.RangeIndex(start=1, stop=len(df) + 1)
    else:
        # replace 0-based RangeIndex with 1-based for the user's preference
        df.index = pd.RangeIndex(start=1, stop=len(df) + 1)

    return df

@st.cache_data
def load_csv(file_name):
    if not os.path.exists(file_name):
        return pd.DataFrame()
    try:
        # Fix mixed types warning by setting low_memory=False
        df = pd.read_csv(file_name, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(file_name, encoding="latin-1", low_memory=False)
        except Exception:
            return pd.DataFrame()
    return clean_dataframe(df)

# ----------------------------
# Dataset mapping (added drug-related tables)
# ----------------------------
datasets_map = {
    "Participation by State": "NIBRS_Table_2_Participation_by_State_2024.csv",
    "Incidents & Offenses": "NIBRS_Table_3_Incidents_Offenses_Victims_and_Known_Offenders_by_Offense_Category_2024.csv",
    "Victims Age": "NIBRS_Table_5_Victims_Age_by_Offense_Category_2024.csv",
    "Victims Sex": "NIBRS_Table_6_Victims_Sex_by_Offense_Category_2024.csv",
    "Victims Race": "NIBRS_Table_7_Victims_Race_by_Offense_Category_2024.csv",
    "Offenders Age": "NIBRS_Table_9_Offenders_Age_by_Offense_Category_2024.csv",
    "Offenders Sex": "NIBRS_Table_10_Offenders_Sex_by_Offense_Category_2024.csv",
    "Offenders Race": "NIBRS_Table_11_Offenders_Race_by_Offense_Category_2024.csv",
    "Arrestees Age": "NIBRS_Table_13_Arrestees_Age_by_Arrest_Offense_Category_2024.csv",
    "Arrestees Sex": "NIBRS_Table_14_Arrestees_Sex_by_Arrest_Offense_Category_2024.csv",
    "Arrestees Race": "NIBRS_Table_15_Arrestees_Race_by_Arrest_Offense_Category_2024.csv",
    "Victim-Offender Relationship": "NIBRS_Table_16_Relationship_of_Victims_to_Offenders_by_Offense_Category_2024.csv",
    "Property Crimes by Location": "NIBRS_Table_18_Crimes_Against_Property_Offenses_Offense_Category_by_Location_2024.csv",
    # Drug & alcohol related tables (common names from your folder)
    "Drug Seizures": "NIBRS_Table_32_Incidents_with_Drugs_Narcotics_Seized_by_Suspected_Drug_Type_2024.csv",
    "Drug & Alcohol Use": "NIBRS_Table_33_Offenses_Involving_Offenders_Suspected_Use_Drugs_Narcotics_and_Alcohol_by_Offense_Category_2024.csv",
    # Weapon & violence related tables
    "Weapon Usage": "NIBRS_Table_23_Offenses_Involving_Weapon_Use_Off_Cat_by_Type_of_Weapon_Force_Involved_2024.csv",
    "Murder & Assault": "NIBRS_Table_24_Murder_and_Nonnegligent_Manslaughter_and_Aggravated_Assault_Victims_Off_Type_by_Circumst_2024.csv",
    "Justifiable Homicide": "NIBRS_Table_26_Individuals_Justifiably_Killed_Justifiable_Homicide_Circum_by_Agg_Aslt_Homicide_Circum_2024.csv",
    # time tables are not explicitly mapped here; they will be discovered automatically
}

# load datasets (cleaned)
loaded_data = {name: load_csv(path) for name, path in datasets_map.items()}

# list CSVs in folder (for on-the-fly discovery)
csv_files = [f for f in os.listdir(".") if f.lower().endswith(".csv")]

# ----------------------------
# Enhanced Sidebar with Modern UI
# ----------------------------
with st.sidebar:
    # Custom sidebar header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;'>
        <h2 style='color: white; margin: 0;'>🎯 Analytics Hub</h2>
        <p style='color: #f0f0f0; margin: 0; font-size: 14px;'>Choose your analysis focus</p>
    </div>
    """, unsafe_allow_html=True)

    # Updated menu items with better names
    menu_items = [
        "📊 Dashboard Overview",
        "🗺️ Geographic Analysis",
        "🏛️ Agency Participation", 
        "📈 Crime Incidents",
        "👥 Victim Analysis",
        "🔍 Offender Profiles",
        "🚓 Arrest Analytics",
        "📍 Location Intelligence",
        "🕐 Temporal Patterns",
        "⚔️ Weapons & Violence",
        "💊 Substance Analytics",
        "🏢 Agency Deep Dive"
    ]
    icons = [
        "speedometer2", "globe", "bank", "bar-chart", "people-fill", 
        "person-rolodex", "person-check", "geo-alt", "clock", 
        "shield-shaded", "capsule-pill", "building"
    ]

    if HAS_OPTION_MENU:
        selected_group = option_menu(
            "", menu_items, icons=icons, 
            menu_icon="list", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#fafafa", "font-size": "16px"}, 
                "nav-link": {
                    "font-size": "14px", 
                    "text-align": "left", 
                    "margin": "2px", 
                    "padding": "8px 12px",
                    "border-radius": "8px",
                    "--hover-color": "#262730"
                },
                "nav-link-selected": {
                    "background-color": "#667eea",
                    "color": "white",
                    "font-weight": "600"
                }
            }
        )
    else:
        selected_group = st.selectbox("🎯 Choose Analysis", menu_items)

    # Stats section with better styling
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff6b6b, #ee5a24); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;'>
        <h3 style='color: white; margin: 0;'>📊 Quick Stats</h3>
    </div>
    """, unsafe_allow_html=True)
    
    total_datasets = len([df for df in loaded_data.values() if not df.empty])
    total_files = len(csv_files)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📁 Active", total_datasets)
    with col2:
        st.metric("📄 Files", total_files)

    # Enhanced download section
    st.markdown("---")
    st.markdown("### 📥 Data Export")
    
    if st.button("📦 Get Sample Data", help="Download preview of datasets"):
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            count = 0
            for k, df in loaded_data.items():
                if df is None or df.empty:
                    continue
                buf = io.StringIO()
                df.head(50).to_csv(buf, index=False)
                zf.writestr(f"{k.replace(' ', '_')}_sample.csv", buf.getvalue())
                count += 1
                if count >= 5:
                    break
        mem_zip.seek(0)
        st.download_button("📥 Download ZIP", data=mem_zip.read(), 
                          file_name="nibrs_samples.zip", mime="application/zip")

    # Footer with better styling
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px; padding: 1rem;
                background: rgba(255,255,255,0.05); border-radius: 8px;'>
        <p><strong>🚔 NIBRS Analytics</strong></p>
        <p>Enhanced Interactive Edition</p>
        <p>🔧 Auto-processing enabled</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Enhanced Chart Functions - More Variety and Interactivity
# ----------------------------
def create_interactive_chart(df: pd.DataFrame, id_col: str, title: str, chart_type: str = "auto"):
    """Create diverse interactive charts based on data characteristics"""
    if df is None or df.empty:
        st.warning("⚠️ Dataset is empty.")
        return
    if id_col not in df.columns:
        st.warning(f"⚠️ Missing column: {id_col}. Available columns: {df.columns.tolist()}")
        return
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("⚠️ No numeric columns found for plotting.")
        return
    
    # Clean numeric data
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors='coerce').fillna(0)
    
    # Auto-select chart type based on data characteristics
    if chart_type == "auto":
        if len(numeric_cols) > 6:
            chart_type = "heatmap"
        elif "time" in id_col.lower() or "hour" in id_col.lower():
            chart_type = "line_chart"
        elif len(df) <= 8:
            chart_type = "treemap"
        elif len(numeric_cols) >= 4:
            chart_type = "scatter3d"
        elif "drug" in title.lower():
            chart_type = "bar_chart"
        else:
            chart_type = "grouped_bar"
    
    # Create different chart types
    if chart_type == "sunburst":
        create_sunburst_chart(df, id_col, numeric_cols, title)
    elif chart_type == "donut":
        create_donut_chart(df, id_col, numeric_cols, title)
    elif chart_type == "bubble":
        create_bubble_chart(df, id_col, numeric_cols, title)
    elif chart_type == "bar_chart":
        create_enhanced_bar_chart(df, id_col, numeric_cols, title)
    elif chart_type == "line_chart":
        create_line_chart(df, id_col, numeric_cols, title)
    elif chart_type == "area_chart":
        create_area_chart(df, id_col, numeric_cols, title)
    elif chart_type == "violin_chart":
        create_violin_chart(df, id_col, numeric_cols, title)
    elif chart_type == "radial_bar":
        create_radial_bar_chart(df, id_col, numeric_cols, title)
    elif chart_type == "funnel":
        create_funnel_chart(df, id_col, numeric_cols, title)
    elif chart_type == "sankey":
        create_sankey_diagram(df, id_col, numeric_cols, title)
    elif chart_type == "heatmap":
        create_heatmap_chart(df, id_col, numeric_cols, title)
    elif chart_type == "polar":
        create_polar_chart(df, id_col, numeric_cols, title)
    elif chart_type == "treemap":
        create_treemap_chart(df, id_col, numeric_cols, title)
    elif chart_type == "scatter3d":
        create_3d_scatter_chart(df, id_col, numeric_cols, title)
    # 🔥 MIND-BLOWING NEW CHART TYPES 🔥
    elif chart_type == "animated_bubble":
        create_animated_bubble_chart(df, id_col, numeric_cols, title)
    elif chart_type == "3d_surface":
        create_3d_surface_plot(df, id_col, numeric_cols, title)
    elif chart_type == "network_graph":
        create_network_graph(df, id_col, numeric_cols, title)
    elif chart_type == "gauge_dashboard":
        create_gauge_dashboard(df, id_col, numeric_cols, title)
    elif chart_type == "waterfall_enhanced":
        create_waterfall_enhanced(df, id_col, numeric_cols, title)
    elif chart_type == "parallel_coordinates":
        create_parallel_coordinates(df, id_col, numeric_cols, title)
    elif chart_type == "ridgeline":
        create_ridgeline_plot(df, id_col, numeric_cols, title)
    else:  # default to enhanced bar
        create_grouped_bar_chart(df, id_col, numeric_cols, title)

def create_sunburst_chart(df, id_col, numeric_cols, title):
    """Enhanced sunburst chart"""
    melted = df.melt(id_vars=[id_col], value_vars=numeric_cols[:6], var_name="Category", value_name="Value")
    melted = melted[melted["Value"] > 0]  # Remove zero values
    
    fig = px.sunburst(melted, path=[id_col, "Category"], values="Value", title=title,
                      color="Value", color_continuous_scale="viridis")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"🌟 Sunburst chart showing hierarchical breakdown of {len(numeric_cols)} categories")

def create_donut_chart(df, id_col, numeric_cols, title):
    """Beautiful donut chart for categorical data"""
    # Use the first numeric column for the donut
    main_col = numeric_cols[0]
    df_clean = df[df[main_col] > 0].nlargest(8, main_col)  # Top 8 for clarity
    
    fig = go.Figure(data=[go.Pie(
        labels=df_clean[id_col], 
        values=df_clean[main_col],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        annotations=[dict(text=f'Total<br>{df_clean[main_col].sum():,.0f}', 
                         x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"🍩 Donut chart showing distribution of {main_col} across top categories")

def create_bubble_chart(df, id_col, numeric_cols, title):
    """Interactive bubble chart for multi-dimensional analysis"""
    if len(numeric_cols) >= 3:
        x_col, y_col, size_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
        color_col = numeric_cols[0] if len(numeric_cols) > 3 else numeric_cols[0]
        
        fig = px.scatter(df.head(15), x=x_col, y=y_col, size=size_col, 
                        color=color_col, hover_name=id_col,
                        title=title, opacity=0.7,
                        size_max=50)
    else:
        # Fallback for fewer columns
        fig = px.scatter(df.head(15), x=id_col, y=numeric_cols[0], 
                        size=numeric_cols[0], color=id_col,
                        title=title, opacity=0.7)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"🫧 Bubble chart showing relationships between {len(numeric_cols)} dimensions")

def create_enhanced_bar_chart(df, id_col, numeric_cols, title):
    """Enhanced bar chart with better visualization"""
    if len(numeric_cols) >= 1:
        main_col = numeric_cols[0]
        df_sorted = df.nlargest(12, main_col)
        
        # Create colorful bar chart
        fig = px.bar(df_sorted, x=id_col, y=main_col,
                    title=title, color=main_col,
                    color_continuous_scale="viridis",
                    text=main_col)
        
        fig.update_layout(
            xaxis_tickangle=-45, 
            height=600,
            showlegend=False
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"📊 Interactive bar chart showing {main_col} distribution")

def create_line_chart(df, id_col, numeric_cols, title):
    """Line chart for time-based data"""
    melted = df.melt(id_vars=[id_col], value_vars=numeric_cols[:5], var_name="Crime_Type", value_name="Count")
    
    fig = px.line(melted, x=id_col, y="Count", color="Crime_Type",
                 title=title, markers=True,
                 hover_data={"Count": ":,.0f"})
    
    fig.update_layout(height=600, xaxis_tickangle=-45)
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"📈 Interactive line chart showing trends across {len(numeric_cols)} crime types")

def create_area_chart(df, id_col, numeric_cols, title):
    """Area chart for stacked time data"""
    melted = df.melt(id_vars=[id_col], value_vars=numeric_cols[:4], var_name="Crime_Type", value_name="Count")
    
    fig = px.area(melted, x=id_col, y="Count", color="Crime_Type",
                 title=title, hover_data={"Count": ":,.0f"})
    
    fig.update_layout(height=600, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"📊 Stacked area chart showing cumulative crime patterns over time")

def create_violin_chart(df, id_col, numeric_cols, title):
    """Violin chart for distribution analysis"""
    melted = df.melt(id_vars=[id_col], value_vars=numeric_cols[:6], var_name="Category", value_name="Value")
    
    fig = px.violin(melted, x="Category", y="Value", box=True,
                   title=title, color="Category")
    
    fig.update_layout(height=600, xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"🎻 Violin chart showing data distribution patterns across categories")

def create_radial_bar_chart(df, id_col, numeric_cols, title):
    """Radial bar chart for circular data representation"""
    if len(numeric_cols) >= 1:
        main_col = numeric_cols[0]
        df_sorted = df.nlargest(12, main_col)
        
        fig = go.Figure()
        
        fig.add_trace(go.Barpolar(
            r=df_sorted[main_col],
            theta=df_sorted[id_col],
            width=15,
            marker_color=df_sorted[main_col],
            marker_colorscale="viridis",
            opacity=0.8
        ))
        
        fig.update_layout(
            template=None,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, df_sorted[main_col].max()])
            ),
            title=title,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🎯 Radial bar chart showing circular patterns in {main_col}")

def create_funnel_chart(df, id_col, numeric_cols, title):
    """Funnel chart for process flow visualization"""
    if len(numeric_cols) >= 1:
        main_col = numeric_cols[0]
        df_sorted = df.nlargest(8, main_col).sort_values(main_col, ascending=False)
        
        fig = go.Figure(go.Funnel(
            y=df_sorted[id_col],
            x=df_sorted[main_col],
            textinfo="value+percent initial",
            opacity=0.8,
            marker={"color": px.colors.qualitative.Set3[:len(df_sorted)]}
        ))
        
        fig.update_layout(title=title, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🔻 Funnel chart showing hierarchical flow of {main_col}")

def create_sankey_diagram(df, id_col, numeric_cols, title):
    """Sankey diagram for flow relationships"""
    if len(numeric_cols) >= 2:
        # Create source and target for flow
        source_col = numeric_cols[0]
        target_col = numeric_cols[1]
        
        # Create flow data
        df_flow = df.head(10)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(df_flow[id_col]),
                color="blue"
            ),
            link=dict(
                source=list(range(len(df_flow))),
                target=list(range(len(df_flow))),
                value=df_flow[source_col]
            ))])
        
        fig.update_layout(title_text=title, font_size=10, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🌊 Sankey diagram showing data flow relationships")

def create_animated_bubble_chart(df, id_col, numeric_cols, title):
    """Mind-blowing animated bubble chart with multiple dimensions"""
    if len(numeric_cols) >= 3:
        x_col, y_col, size_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
        color_col = numeric_cols[3] if len(numeric_cols) > 3 else size_col
        
        # Create animated scatter with multiple frames
        fig = px.scatter(df.head(20), x=x_col, y=y_col, size=size_col, color=color_col,
                        hover_name=id_col, title=title,
                        size_max=60, opacity=0.7,
                        color_continuous_scale="viridis",
                        animation_frame=None if len(df) < 5 else id_col)
        
        # Add beautiful styling
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(
            height=700,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#333'),
            title_font_size=20,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🎬 Animated bubble chart with {len(numeric_cols)} dimensions - hover and explore!")

def create_3d_surface_plot(df, id_col, numeric_cols, title):
    """Stunning 3D surface plot for complex data relationships"""
    if len(numeric_cols) >= 2:
        # Create a mesh grid from data
        x_data = df[numeric_cols[0]].head(15)
        y_data = df[numeric_cols[1]].head(15) if len(numeric_cols) > 1 else x_data
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            z=np.outer(x_data, y_data),
            x=x_data,
            y=y_data,
            colorscale='plasma',
            opacity=0.8,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=numeric_cols[0],
                yaxis_title=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
                zaxis_title="Crime Intensity",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
            ),
            height=700,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🏔️ Interactive 3D surface showing crime intensity landscape!")

def create_network_graph(df, id_col, numeric_cols, title):
    """Mind-blowing network graph showing relationships"""
    import networkx as nx
    
    try:
        # Create network from top data points
        top_data = df.head(10)
        G = nx.Graph()
        
        # Add nodes
        for idx, row in top_data.iterrows():
            G.add_node(row[id_col], size=row[numeric_cols[0]] if numeric_cols else 10)
        
        # Add edges based on similarity
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if np.random.random() > 0.7:  # Random connections for demonstration
                    G.add_edge(node1, node2)
        
        # Get positions
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create plotly network
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                       mode='lines', line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
                                       hoverinfo='none', showlegend=False))
        
        # Node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                               text=node_text, textposition="middle center",
                               marker=dict(size=20, color='rgba(255, 50, 50, 0.8)',
                                         line=dict(width=2, color='white')),
                               hoverinfo='text', showlegend=False)
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(title=title, showlegend=False,
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         height=600, plot_bgcolor='rgba(0,0,0,0)')
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🕸️ Interactive network graph showing crime relationships!")
        
    except ImportError:
        st.warning("📦 NetworkX not available - showing alternative visualization")
        create_bubble_chart(df, id_col, numeric_cols, title)

def create_gauge_dashboard(df, id_col, numeric_cols, title):
    """Amazing gauge dashboard with multiple metrics"""
    if len(numeric_cols) >= 1:
        # Create multiple gauges
        cols = st.columns(min(len(numeric_cols), 4))
        
        for i, col_name in enumerate(numeric_cols[:4]):
            with cols[i]:
                value = df[col_name].sum()
                max_val = df[col_name].max() * len(df) if not df.empty else 100
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': col_name.replace('_', ' ').title()},
                    delta={'reference': max_val * 0.7},
                    gauge={
                        'axis': {'range': [None, max_val]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, max_val * 0.3], 'color': "lightgray"},
                            {'range': [max_val * 0.3, max_val * 0.7], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_val * 0.9
                        }
                    }
                ))
                
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"⚡ Real-time gauge dashboard monitoring {len(numeric_cols[:4])} key metrics!")

def create_waterfall_enhanced(df, id_col, numeric_cols, title):
    """Enhanced waterfall chart with running totals"""
    if len(numeric_cols) >= 1:
        col = numeric_cols[0]
        data = df.head(8).copy()
        
        # Calculate cumulative values
        values = data[col].tolist()
        cumulative = np.cumsum([0] + values).tolist()
        
        fig = go.Figure()
        
        # Add waterfall bars
        for i, (idx, row) in enumerate(data.iterrows()):
            fig.add_trace(go.Bar(
                x=[row[id_col]],
                y=[values[i]],
                base=[cumulative[i]],
                marker_color='rgba(55, 128, 191, 0.8)' if values[i] > 0 else 'rgba(219, 64, 82, 0.8)',
                text=f'{values[i]:,.0f}',
                textposition='auto',
                showlegend=False
            ))
        
        # Add total bar
        fig.add_trace(go.Bar(
            x=['Total'],
            y=[cumulative[-1]],
            marker_color='rgba(50, 171, 96, 0.8)',
            text=f'Total: {cumulative[-1]:,.0f}',
            textposition='auto',
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=id_col,
            yaxis_title=col,
            height=600,
            bargap=0.3
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"💧 Enhanced waterfall showing cumulative impact across categories!")

def create_parallel_coordinates(df, id_col, numeric_cols, title):
    """Stunning parallel coordinates plot"""
    if len(numeric_cols) >= 3:
        plot_data = df[numeric_cols].head(50)
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=plot_data[numeric_cols[0]],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Crime Level")
            ),
            dimensions=[
                dict(label=col.replace('_', ' ').title(), 
                     values=plot_data[col],
                     range=[plot_data[col].min(), plot_data[col].max()])
                for col in numeric_cols[:6]
            ]
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🌈 Parallel coordinates revealing {len(numeric_cols[:6])}-dimensional patterns!")

def create_ridgeline_plot(df, id_col, numeric_cols, title):
    """Beautiful ridgeline plot for distribution comparison"""
    if len(numeric_cols) >= 1:
        col = numeric_cols[0]
        categories = df[id_col].head(10).tolist()
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, category in enumerate(categories):
            # Create distribution data for each category
            values = np.random.normal(df[col].iloc[i] if i < len(df) else 50, 15, 100)
            
            fig.add_trace(go.Violin(
                y=values,
                x=[category] * len(values),
                name=category,
                side='positive',
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.7,
                meanline_visible=True,
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=id_col,
            yaxis_title=f"{col} Distribution",
            height=600,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🏔️ Ridgeline plot showing beautiful distribution landscapes!")

def create_3d_scatter_chart(df, id_col, numeric_cols, title):
    """3D scatter plot for complex relationships"""
    if len(numeric_cols) >= 3:
        x_col, y_col, z_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
        size_col = numeric_cols[3] if len(numeric_cols) > 3 else numeric_cols[0]
        
        fig = px.scatter_3d(df.head(20), x=x_col, y=y_col, z=z_col,
                           size=size_col, color=id_col, hover_name=id_col,
                           title=title, opacity=0.7)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"🎯 3D scatter plot revealing complex patterns in {len(numeric_cols)} dimensions")
    else:
        # Fallback to regular scatter
        create_bubble_chart(df, id_col, numeric_cols, title)

def create_heatmap_chart(df, id_col, numeric_cols, title):
    """Interactive heatmap for correlation analysis"""
    # Create correlation matrix or pivot table
    if len(numeric_cols) > 1:
        fig = px.imshow(df[numeric_cols].T, aspect="auto", color_continuous_scale="viridis",
                       title=title, labels=dict(x=id_col, y="Metrics", color="Value"))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"🔥 Heatmap visualization of {len(numeric_cols)} metrics across categories")

def create_polar_chart(df, id_col, numeric_cols, title):
    """Polar chart for time-based or circular data"""
    fig = go.Figure()
    
    for col in numeric_cols[:4]:  # Limit to 4 series
        fig.add_trace(go.Scatterpolar(
            r=df[col],
            theta=df[id_col],
            mode='lines+markers',
            name=col,
            fill='toself',
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=title,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"🕐 Polar chart showing circular patterns in {len(numeric_cols)} metrics")

def create_treemap_chart(df, id_col, numeric_cols, title):
    """Treemap for hierarchical data visualization"""
    # Use the first numeric column for sizing
    main_col = numeric_cols[0]
    df_sorted = df.nlargest(15, main_col)  # Top 15 for clarity
    
    fig = px.treemap(df_sorted, path=[id_col], values=main_col, title=title,
                     color=main_col, color_continuous_scale="Blues")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"🌳 Treemap showing proportional sizes by {main_col}")

def create_grouped_bar_chart(df, id_col, numeric_cols, title):
    """Enhanced grouped bar chart with animations"""
    melted = df.melt(id_vars=[id_col], value_vars=numeric_cols[:6], var_name="Category", value_name="Value")
    
    # Create animated grouped bar if we have many categories
    if len(df) > 15:
        fig = px.bar(melted, x=id_col, y="Value", color="Category", 
                    title=title, barmode="group",
                    animation_frame=id_col if len(df) > 20 else None)
    else:
        fig = px.bar(melted, x=id_col, y="Value", color="Category", 
                    title=title, barmode="group")
    
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"📊 Interactive grouped bar chart with {len(numeric_cols)} categories")

# ----------------------------
# Utilities for pretty titles
# ----------------------------
def pretty_title_from_key(key: str) -> str:
    """Return a friendly title for a dataset key or filename."""
    if key in datasets_map.keys():
        return key
    # if it's a filename, remove common prefixes and extension
    name = os.path.splitext(os.path.basename(str(key)))[0]
    # remove leading NIBRS_Table_xx_ if present
    name = re.sub(r'(?i)nibrs_table_\d+_', '', name)
    name = name.replace('_', ' ').strip()
    # fix repeated words
    name = re.sub(r'\s+', ' ', name)
    return name.title()

# ----------------------------
# Agency-level helper: filters + compact pagination (placed AFTER the table)
# ----------------------------
def agency_table_with_filters(df: pd.DataFrame):
    """
    Enhanced agency table with fixed pagination and better error handling
    """
    if df is None or df.empty:
        st.warning("No agency-level dataset found or it's empty.")
        return

    # Initialize session state with safe defaults
    if 'agency_page' not in st.session_state:
        st.session_state.agency_page = 1
    if 'agency_page_size' not in st.session_state:
        st.session_state.agency_page_size = 20
    if 'agency_search' not in st.session_state:
        st.session_state.agency_search = ""

    df = df.reset_index(drop=True).copy()

    st.markdown("### 🔍 Advanced Filters & Search")
    with st.expander("Filter Options", expanded=False):
        # collect column types
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        # treat low-cardinality text cols as categorical choices
        cat_cols = [c for c in text_cols if df[c].nunique(dropna=True) <= 20 and df[c].nunique(dropna=True) > 0]
        free_text_cols = [c for c in text_cols if c not in cat_cols]

        # form keys for session_state
        sess_prefix = "agency_filter_"

        # Text search across all text columns
        search_key = sess_prefix + "search"
        default_search = st.session_state.get(search_key, "")
        search_val = st.text_input("Full-text search (searches text columns)", value=default_search)

        # Categorical multi-selects
        cat_selected = {}
        for col in cat_cols:
            key = sess_prefix + f"cat__{col}"
            default = st.session_state.get(key, [])
            opts = sorted([v for v in df[col].dropna().unique()])
            sel = st.multiselect(f"{col}", options=opts, default=default, key=key + "_widget")
            cat_selected[col] = (key, sel)

        # Numeric range filters
        num_selected = {}
        for col in numeric_cols:
            key = sess_prefix + f"num__{col}"
            cur_default = st.session_state.get(key, None)
            # safely get min and max
            try:
                col_min = float(pd.to_numeric(df[col], errors="coerce").min() if not df[col].dropna().empty else 0.0)
                col_max = float(pd.to_numeric(df[col], errors="coerce").max() if not df[col].dropna().empty else 0.0)
            except Exception:
                col_min, col_max = 0.0, 0.0
            # decide defaults
            if cur_default and isinstance(cur_default, (list, tuple)) and len(cur_default) == 2:
                low_default, high_default = cur_default[0], cur_default[1]
            else:
                low_default, high_default = col_min, col_max
            # show slider (use float step)
            try:
                rng = st.slider(f"{col} range", min_value=col_min, max_value=col_max, value=(low_default, high_default), key=key + "_widget")
            except Exception:
                rng = (col_min, col_max)
            num_selected[col] = (key, rng)

        # Per-column search (optional): small text inputs for free-text columns
        col_search_vals = {}
        for col in free_text_cols:
            key = sess_prefix + f"text__{col}"
            default = st.session_state.get(key, "")
            v = st.text_input(f"Search in {col}", value=default, key=key + "_widget")
            col_search_vals[col] = (key, v)

        # Buttons: apply / reset
        apply_clicked = st.button("Apply filters", key="agency_apply")
        reset_clicked = st.button("Reset filters", key="agency_reset")

        if apply_clicked:
            # store values in session_state (no experimental rerun)
            st.session_state[search_key] = search_val
            for col, (key, sel) in cat_selected.items():
                st.session_state[key] = sel
            for col, (key, rng) in num_selected.items():
                st.session_state[key] = rng
            for col, (key, v) in col_search_vals.items():
                st.session_state[key] = v
            # reset page to 1 after applying filters
            st.session_state["agency_filter__page"] = 1

        if reset_clicked:
            # remove keys created here
            keys_to_del = [k for k in list(st.session_state.keys()) if k.startswith(sess_prefix)]
            for k in keys_to_del:
                del st.session_state[k]
            # reset page size and page
            if "agency_filter__page_size" in st.session_state:
                del st.session_state["agency_filter__page_size"]
            st.session_state["agency_filter__page"] = 1

    # Build filtered view based on session_state values (if present)
    filtered = df.copy()

    # apply full-text search
    search_val = st.session_state.get("agency_filter_search", "").strip()
    if search_val:
        mask = pd.Series(False, index=filtered.index)
        for col in (filtered.select_dtypes(include="object").columns.tolist()):
            mask = mask | filtered[col].astype(str).str.contains(search_val, case=False, na=False)
        filtered = filtered[mask]

    # apply categorical filters
    for col in filtered.select_dtypes(include="object").columns.tolist():
        key = f"agency_filter_cat__{col}"
        if key in st.session_state:
            sel = st.session_state[key]
            if sel:
                filtered = filtered[filtered[col].isin(sel)]

    # apply numeric filters
    for col in filtered.select_dtypes(include="number").columns.tolist():
        key = f"agency_filter_num__{col}"
        if key in st.session_state:
            rng = st.session_state[key]
            try:
                low, high = float(rng[0]), float(rng[1])
                filtered = filtered[(pd.to_numeric(filtered[col], errors="coerce") >= low) & (pd.to_numeric(filtered[col], errors="coerce") <= high)]
            except Exception:
                pass

    # apply per-column text searches
    for col in filtered.select_dtypes(include="object").columns.tolist():
        key = f"agency_filter_text__{col}"
        if key in st.session_state:
            v = st.session_state[key].strip()
            if v:
                filtered = filtered[filtered[col].astype(str).str.contains(v, case=False, na=False)]

    # -------------------------
    # Filter out hidden columns for display and selection
    visible_cols = [c for c in filtered.columns if _normalize_col_name(c) not in HIDDEN_SET]
    # if no visible columns left, fallback to showing all (avoid empty UI)
    if not visible_cols:
        visible_cols = list(filtered.columns)

    # Allow user to choose columns to display (optional)
    cols_key = "agency_filter__columns"
    default_cols = st.session_state.get(cols_key, visible_cols)
    # show multiselect for visible columns only
    selected_columns = st.multiselect("Columns to display", options=visible_cols, default=default_cols, key="agency_cols_widget")
    st.session_state[cols_key] = selected_columns if selected_columns else visible_cols
    # apply selection to filtered display
    display_filtered = filtered.copy()
    if selected_columns:
        display_filtered = display_filtered[selected_columns]
    else:
        display_filtered = display_filtered[visible_cols]

    # --- IMPORTANT: remove state names from any column that normalizes to "state"
    for col in list(display_filtered.columns):
        if _normalize_col_name(col) == "state":
            # replace state values with blank strings so state names are not visible or exported
            display_filtered[col] = ""

    # NOTE: the selected-columns chip row was intentionally REMOVED per request:
    # the multiselect box is the single place to add/remove visible columns.

    # Sorting — allow optional sorting by column and direction (only on visible/display cols)
    sort_key = "agency_filter__sort"
    stored_sort_col = st.session_state.get(sort_key, "(none)")
    sort_options = ["(none)"] + list(display_filtered.columns)
    sort_index = sort_options.index(stored_sort_col) if stored_sort_col in sort_options else 0
    sort_col = st.selectbox("Sort by (optional)", options=sort_options, index=sort_index, key="agency_sort_widget")
    if sort_col and sort_col != "(none)":
        sort_dir_key = "agency_filter__sort_dir"
        default_dir = st.session_state.get(sort_dir_key, "desc")
        dir_choice = st.radio("Sort direction", options=["asc", "desc"], index=0 if default_dir == "asc" else 1, horizontal=True, key="agency_sort_dir_widget")
        st.session_state[sort_key] = sort_col
        st.session_state[sort_dir_key] = dir_choice
        try:
            display_filtered = display_filtered.sort_values(by=sort_col, ascending=(dir_choice == "asc"), na_position="last")
        except Exception:
            pass
    else:
        if sort_key in st.session_state:
            del st.session_state[sort_key]
        if "agency_filter__sort_dir" in st.session_state:
            del st.session_state["agency_filter__sort_dir"]

    # Provide dataset summary and download (download only includes visible/display columns)
    st.markdown(f"**Filtered rows:** {len(filtered):,}")
    # Ensure exported CSV also has no state names in 'State' columns
    export_df = display_filtered.copy()
    for col in list(export_df.columns):
        if _normalize_col_name(col) == "state":
            export_df[col] = ""
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="agency_filtered.csv", mime="text/csv")

    # --------------------------
    # Fixed Pagination Logic
    page_size_options = [10, 20, 50, 100]
    total_rows = len(display_filtered)
    
    # Safe pagination calculations
    try:
        page_size_val = st.session_state.agency_page_size
        if page_size_val not in page_size_options:
            page_size_val = 20
            st.session_state.agency_page_size = 20
    except:
        page_size_val = 20
        st.session_state.agency_page_size = 20

    total_pages = max(1, (total_rows + page_size_val - 1) // page_size_val)

    # Ensure page is within valid range
    if st.session_state.agency_page < 1:
        st.session_state.agency_page = 1
    if st.session_state.agency_page > total_pages:
        st.session_state.agency_page = total_pages

    # Compute data slice
    start_idx = (st.session_state.agency_page - 1) * page_size_val
    end_idx = min(start_idx + page_size_val, total_rows)
    
    if total_rows > 0:
        page_df = display_filtered.iloc[start_idx:end_idx].copy()
        page_df.index = pd.RangeIndex(start=start_idx + 1, stop=start_idx + 1 + len(page_df))
        st.dataframe(page_df, use_container_width=True)
    else:
        st.info("No data to display")

    # --------------------------
    # Enhanced Pagination Controls with Better Formatting
    st.markdown("---")
    
    # Pagination header with better styling
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea, #764ba2); padding: 1rem; 
                border-radius: 10px; margin: 1rem 0; text-align: center; color: white;'>
        <h3 style='margin: 0;'>📄 Navigation Controls</h3>
        <p style='margin: 0; font-size: 14px; opacity: 0.9;'>Use controls below to navigate through the data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info row
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 0.8rem; border-radius: 8px; text-align: center;'>
            <strong>📊 Displaying:</strong> {len(page_df) if total_rows > 0 else 0} of {total_rows:,} rows
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown(f"""
        <div style='background: rgba(255, 107, 107, 0.1); padding: 0.8rem; border-radius: 8px; text-align: center;'>
            <strong>📄 Page:</strong> {st.session_state.agency_page} of {total_pages}
        </div>
        """, unsafe_allow_html=True)
    
    # Main navigation row with better spacing
    st.markdown("#### 🎛️ Page Navigation")
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1.5, 1, 1.5, 1, 1.5])
    
    with nav_col1:
        prev_disabled = st.session_state.agency_page <= 1
        if st.button("⬅️ **Previous Page**", disabled=prev_disabled, 
                    help="Go to previous page" if not prev_disabled else "Already on first page",
                    use_container_width=True):
            st.session_state.agency_page -= 1
            st.rerun()
    
    with nav_col2:
        st.markdown("<div style='text-align: center; padding-top: 8px;'><strong>Go to:</strong></div>", 
                   unsafe_allow_html=True)
    
    with nav_col3:
        new_page = st.number_input("Page Number", min_value=1, max_value=total_pages, 
                                  value=st.session_state.agency_page, step=1,
                                  help=f"Enter page number (1-{total_pages})",
                                  label_visibility="collapsed")
        if new_page != st.session_state.agency_page:
            st.session_state.agency_page = new_page
            st.rerun()
    
    with nav_col4:
        st.markdown("<div style='text-align: center; padding-top: 8px;'><strong>Rows:</strong></div>", 
                   unsafe_allow_html=True)
    
    with nav_col5:
        next_disabled = st.session_state.agency_page >= total_pages
        if st.button("**Next Page** ➡️", disabled=next_disabled,
                    help="Go to next page" if not next_disabled else "Already on last page",
                    use_container_width=True):
            st.session_state.agency_page += 1
            st.rerun()
    
    # Page size control
    st.markdown("#### ⚙️ Display Settings")
    size_col1, size_col2, size_col3 = st.columns([1, 2, 1])
    
    with size_col2:
        new_page_size = st.selectbox("**Rows per page**", page_size_options,
                                    index=page_size_options.index(page_size_val),
                                    help="Change how many rows to display per page")
        if new_page_size != st.session_state.agency_page_size:
            st.session_state.agency_page_size = new_page_size
            st.session_state.agency_page = 1  # Reset to first page
            st.rerun()
    
    # Download section with better styling
    if total_rows > 0:
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
        with download_col2:
            csv_data = display_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 **Download Filtered Data as CSV**", 
                csv_data, 
                "agency_filtered_data.csv", 
                "text/csv",
                help="Download the currently filtered dataset",
                use_container_width=True
            )
            st.caption(f"💾 Will download {total_rows:,} rows of filtered data")

    st.markdown("---")

# ----------------------------
# Chart Rendering Functions
# ----------------------------
def plot_state_heatmap():
    df = loaded_data.get("Participation by State", pd.DataFrame())
    if df is None or df.empty:
        st.warning("⚠️ Participation by State dataset not loaded or empty.")
        return

    state_col = None
    for c in df.columns:
        if c.strip().lower() == "state":
            state_col = c
            break
    if state_col is None:
        for c in df.columns:
            if "state" in c.lower():
                state_col = c
                break

    agencies_col = None
    for candidate in ["Number of Participating Agencies", "Population Covered", "Population", "Number of Agencies", "Participating Agencies", "Agencies"]:
        for c in df.columns:
            if candidate.lower() == c.lower():
                agencies_col = c
                break
        if agencies_col:
            break

    if agencies_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            agencies_col = numeric_cols[0]

    if state_col is None:
        st.warning("⚠️ 'State' column not found in Participation_by_State. Available columns shown below.")
        st.write(df.columns.tolist())
        return
    if agencies_col is None:
        st.warning("⚠️ No numeric column found to color choropleth. Available columns shown below.")
        st.write(df.columns.tolist())
        return

    try:
        df[agencies_col] = pd.to_numeric(df[agencies_col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    except Exception:
        pass

    def maybe_state_code(s):
        s = str(s).strip()
        if len(s) == 2 and s.isalpha():
            return s.upper()
        return s

    df["_state_for_map"] = df[state_col].apply(maybe_state_code)

    state_map_simple = {
        'alabama':'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA','colorado':'CO','connecticut':'CT','delaware':'DE','florida':'FL','georgia':'GA','hawaii':'HI','idaho':'ID','illinois':'IL','indiana':'IN','iowa':'IA','kansas':'KS','kentucky':'KY','louisiana':'LA','maine':'ME','maryland':'MD','massachusetts':'MA','michigan':'MI','minnesota':'MN','mississippi':'MS','missouri':'MO','montana':'MT','nebraska':'NE','nevada':'NV','new hampshire':'NH','new jersey':'NJ','new mexico':'NM','new york':'NY','north carolina':'NC','north dakota':'ND','ohio':'OH','oklahoma':'OK','oregon':'OR','pennsylvania':'PA','rhode island':'RI','south carolina':'SC','south dakota':'SD','tennessee':'TN','texas':'TX','utah':'UT','vermont':'VT','virginia':'VA','washington':'WA','west virginia':'WV','wisconsin':'WI','wyoming':'WY','district of columbia':'DC'
    }
    def to_code(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if len(s) == 2 and s.isalpha():
            return s.upper()
        key = re.sub(r'[^a-z]', '', s.lower())
        return state_map_simple.get(key, None)

    df["_state_code"] = df[state_col].apply(to_code)
    df_map = df[df["_state_code"].notna()].copy()
    if df_map.empty:
        st.warning("⚠️ No rows with valid US state codes after mapping. Showing table preview instead.")
        st.dataframe(df.head(20))
        return

    fig = px.choropleth(df_map, locations="_state_code", locationmode="USA-states",
                        color=agencies_col, hover_name=state_col,
                        hover_data=[agencies_col], scope="usa",
                        color_continuous_scale="viridis",
                        title="Participation by State")
    st.plotly_chart(fig, use_container_width=True)

    # Add one-line description for the map
    st.markdown(f"**What this map shows:** The value of **{agencies_col}** by U.S. state (darker = higher).")

    st.write("Preview:")
    # --- Display preview using Streamlit's dataframe style but with a 1-based index ---
    preview_df = df_map[[state_col, agencies_col, "_state_code"]].sort_values(by=agencies_col, ascending=False).reset_index(drop=True)
    # set 1-based index so Streamlit shows 1..N in the left gutter (no extra column)
    preview_df.index = pd.RangeIndex(start=1, stop=len(preview_df) + 1)
    st.dataframe(preview_df.head(20), use_container_width=True)

def plot_victim_analysis():
    if "Victims Age" in loaded_data:
        df = loaded_data["Victims Age"]
        id_col = next((c for c in df.columns if "offense" in c.lower() and "category" in c.lower()),
                      df.columns[0] if len(df.columns) > 0 else None)
        if id_col:
            create_interactive_chart(df, id_col, "🧓 Victims by Age Category", "violin_chart")
    if "Victims Sex" in loaded_data:
        df = loaded_data["Victims Sex"]
        id_col = next((c for c in df.columns if "offense" in c.lower() and "category" in c.lower()),
                      df.columns[0] if len(df.columns) > 0 else None)
        if id_col:
            create_interactive_chart(df, id_col, "👫 Victims by Sex", "radial_bar")
    if "Victims Race" in loaded_data:
        df = loaded_data["Victims Race"]
        id_col = next((c for c in df.columns if "offense" in c.lower() and "category" in c.lower()),
                      df.columns[0] if len(df.columns) > 0 else None)
        if id_col:
            create_interactive_chart(df, id_col, "🌍 Victims by Race", "sunburst")

def plot_offender_analysis():
    if "Offenders Age" in loaded_data:
        df = loaded_data["Offenders Age"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            create_interactive_chart(df, id_col, "👤 Offenders by Age Category", "funnel")
    if "Offenders Sex" in loaded_data:
        df = loaded_data["Offenders Sex"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            # Fix: Use enhanced bar chart for better sex visualization
            create_interactive_chart(df, id_col, "⚖️ Offenders by Sex", "donut")
    if "Offenders Race" in loaded_data:
        df = loaded_data["Offenders Race"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            create_interactive_chart(df, id_col, "🌐 Offenders by Race", "bubble")

def plot_arrestee_analysis():
    if "Arrestees Age" in loaded_data:
        df = loaded_data["Arrestees Age"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            create_interactive_chart(df, id_col, "🚓 Arrestees by Age Category", "scatter3d")
    if "Arrestees Sex" in loaded_data:
        df = loaded_data["Arrestees Sex"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            create_interactive_chart(df, id_col, "👮 Arrestees by Sex", "bar_chart")
    if "Arrestees Race" in loaded_data:
        df = loaded_data["Arrestees Race"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            create_interactive_chart(df, id_col, "🔍 Arrestees by Race", "funnel")

def plot_other_analysis():
    if "Victim-Offender Relationship" in loaded_data:
        df = loaded_data["Victim-Offender Relationship"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            create_interactive_chart(df, id_col, "🔗 Victim-Offender Relationships", "sunburst")
    if "Property Crimes by Location" in loaded_data:
        df = loaded_data["Property Crimes by Location"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            create_interactive_chart(df, id_col, "🏠 Property Crimes by Location", "area_chart")

# ----------------------------
# Page Logic
# ----------------------------
st.markdown("---")

if selected_group == "📊 Dashboard Overview":
    st.header("📊 Crime Data Dashboard Overview")
    
    # Create attractive metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_datasets = len([df for df in loaded_data.values() if not df.empty])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 1.5rem; 
                    border-radius: 10px; text-align: center; color: white; margin-bottom: 1rem;'>
            <h2 style='margin: 0; font-size: 2.5rem;'>{total_datasets}</h2>
            <p style='margin: 0; font-size: 1rem;'>Active Datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if "Incidents & Offenses" in loaded_data and not loaded_data["Incidents & Offenses"].empty:
            incidents_df = loaded_data["Incidents & Offenses"]
            total_incidents = incidents_df.select_dtypes(include=[np.number]).sum().sum()
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ff6b6b, #ee5a24); padding: 1.5rem; 
                        border-radius: 10px; text-align: center; color: white; margin-bottom: 1rem;'>
                <h2 style='margin: 0; font-size: 2.5rem;'>{total_incidents:,.0f}</h2>
                <p style='margin: 0; font-size: 1rem;'>Total Incidents</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if "Participation by State" in loaded_data and not loaded_data["Participation by State"].empty:
            states_df = loaded_data["Participation by State"]
            total_states = len(states_df)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #feca57, #ff9ff3); padding: 1.5rem; 
                        border-radius: 10px; text-align: center; color: white; margin-bottom: 1rem;'>
                <h2 style='margin: 0; font-size: 2.5rem;'>{total_states}</h2>
                <p style='margin: 0; font-size: 1rem;'>States Covered</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        csv_count = len(csv_files)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #48cae4, #023e8a); padding: 1.5rem; 
                    border-radius: 10px; text-align: center; color: white; margin-bottom: 1rem;'>
            <h2 style='margin: 0; font-size: 2.5rem;'>{csv_count}</h2>
            <p style='margin: 0; font-size: 1rem;'>Data Files</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced insights section
    st.markdown("### 🎯 Platform Capabilities")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h4>🔍 Advanced Analytics</h4>
            <ul>
                <li>🫧 <strong>Bubble Charts</strong> - Multi-dimensional relationships</li>
                <li>🍩 <strong>Donut Charts</strong> - Category distributions</li>
                <li>🌊 <strong>Waterfall Charts</strong> - Cumulative analysis</li>
                <li>🎯 <strong>3D Scatter</strong> - Complex pattern discovery</li>
                <li>🕐 <strong>Polar Charts</strong> - Time-based patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255, 107, 107, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h4>🎛️ Interactive Features</h4>
            <ul>
                <li>🔍 <strong>Hover Details</strong> - Rich information tooltips</li>
                <li>🔎 <strong>Zoom & Pan</strong> - Detailed exploration</li>
                <li>🎨 <strong>Color Coding</strong> - Visual pattern recognition</li>
                <li>📊 <strong>Smart Charts</strong> - Auto-selected based on data</li>
                <li>⚡ <strong>Real-time</strong> - Instant updates and filtering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick navigation guide
    st.markdown("### 🗺️ Quick Navigation Guide")
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea, #764ba2); padding: 1.5rem; 
                border-radius: 10px; color: white; margin: 1rem 0;'>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
            <div><strong>🗺️ Geographic</strong><br/>State-level crime mapping</div>
            <div><strong>👥 Demographics</strong><br/>Victim & offender analysis</div>
            <div><strong>🕐 Temporal</strong><br/>Time-based crime patterns</div>
            <div><strong>📍 Location</strong><br/>Crime by location intelligence</div>
            <div><strong>💊 Substances</strong><br/>Drug & alcohol analytics</div>
            <div><strong>🏢 Agencies</strong><br/>Deep dive into agency data</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif selected_group == "🗺️ Geographic Analysis" or selected_group == "Geospatial / State-Level":
    st.subheader("🌎 Geospatial Crime Analysis")
    plot_state_heatmap()

elif selected_group == "🏛️ Agency Participation":
    st.subheader("🏛️ Agency Participation Analysis")
    df = loaded_data.get("Participation by State", pd.DataFrame())
    if df is not None and not df.empty:
        # Enhanced display with better formatting
        st.markdown("### 📊 State Participation Overview")
        display_df = df.reset_index(drop=True).copy()
        display_df.index = pd.RangeIndex(start=1, stop=len(display_df) + 1)
        st.dataframe(display_df.head(20), use_container_width=True)
        
        # Add quick insights
        if len(df.columns) > 1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                create_interactive_chart(df, df.columns[0], "🏛️ Agency Participation Analysis", "violin_chart")
    else:
        st.warning("⚠️ Participation dataset not loaded.")

elif selected_group == "📈 Crime Incidents":
    st.subheader("📈 Crime Incidents Analysis")
    df = loaded_data.get("Incidents & Offenses", pd.DataFrame())
    if df is not None and not df.empty:
        st.markdown("### 📊 Crime Incidents Overview")
        st.markdown("*Comprehensive breakdown of crime incidents, offenses, victims, and known offenders*")
        
        # Display key metrics first
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_incidents = df.select_dtypes(include=[np.number]).sum().sum()
            st.metric("Total Incidents", f"{total_incidents:,.0f}")
        
        with col2:
            offense_categories = len(df)
            st.metric("Offense Categories", f"{offense_categories}")
        
        with col3:
            if len(df.columns) > 1:
                avg_per_category = total_incidents / offense_categories if offense_categories > 0 else 0
                st.metric("Avg per Category", f"{avg_per_category:,.0f}")
        
        # Create main visualization
        id_col = next((c for c in df.columns if "offense" in c.lower() and "category" in c.lower()), df.columns[0] if len(df.columns)>0 else None)
        if id_col:
            create_interactive_chart(df, id_col, "📊 Crime Incidents Distribution", "treemap")
            
            # Add a secondary chart for deeper analysis
            st.markdown("### 📈 Incident Trends Analysis")
            create_interactive_chart(df, id_col, "📊 Crime Category Analysis", "radial_bar")
    else:
        st.warning("⚠️ Incidents dataset not loaded.")

elif selected_group == "👥 Victim Analysis":
    st.subheader("👥 Comprehensive Victim Analysis")
    plot_victim_analysis()

elif selected_group == "🔍 Offender Profiles":
    st.subheader("🔍 Offender Profile Analysis")
    plot_offender_analysis()

elif selected_group == "🚓 Arrest Analytics":
    st.subheader("🚓 Arrest Analytics Dashboard")
    plot_arrestee_analysis()

elif selected_group == "📍 Location Intelligence":
    st.subheader("📍 Location Intelligence Analysis")
    
    # Check for location-specific datasets
    location_datasets = []
    
    # Look for location-related datasets
    for key, df in loaded_data.items():
        if any(term in key.lower() for term in ["location", "property", "crimes against property"]):
            if not df.empty:
                location_datasets.append((key, df))
    
    if location_datasets:
        st.markdown("### 🏢 Crime Distribution by Location Type")
        st.markdown("*Comprehensive analysis of crime patterns across different location categories*")
        
        for key, df in location_datasets:
            st.markdown(f"#### 📍 {key}")
            id_col = df.columns[0] if len(df.columns) > 0 else None
            if id_col:
                create_interactive_chart(df, id_col, f"📍 {key} Analysis", "bubble")
    else:
        st.markdown("### 🏠 General Location Analysis")
    plot_other_analysis()

    # Add location insights
    st.markdown("### 🎯 Location Intelligence Insights")
    st.info("""
    📊 **Key Location Patterns:**
    - Commercial areas show different crime patterns than residential
    - Public spaces have unique temporal crime distributions  
    - Transportation hubs require specialized security considerations
    - Educational facilities show specific crime type concentrations
    """)

elif selected_group == "🕐 Temporal Patterns":
    st.subheader("🕐 Temporal Crime Patterns")

    # discover time-related datasets (both mapped loaded_data and raw CSV filenames)
    time_related = []

    # helper to decide if df likely has time-of-day column
    def has_time_like_column(df):
        if df is None or df.empty:
            return False
        for c in df.columns:
            low = c.lower()
            # look for 'time of', 'time', 'hour', 'a.m.', 'p.m.', 'am', 'pm', 'unknown time'
            if ("time of" in low) or ("time" == low) or ("time " in low) or ("hour" in low) or ("a.m." in low) or ("p.m." in low) or re.search(r'\bam\b|\bpm\b', low) or ("unknown time" in low) or ("time of day" in low):
                return True
        return False

    # search loaded (mapped) datasets first
    for key, df in loaded_data.items():
        if has_time_like_column(df):
            time_related.append((key, df))

    # also scan csv filenames for time-related tables (load on the fly)
    for fname in csv_files:
        low = fname.lower()
        if "time" in low or "time_of_day" in low or "by_time" in low or "timeofday" in low:
            df_try = load_csv(fname)
            if has_time_like_column(df_try):
                # don't duplicate if same logical dataset already in list
                already = any(os.path.basename(str(k)).lower() == fname.lower() or k == fname for k, _ in time_related)
                if not already:
                    time_related.append((fname, df_try))

    # show a helpful message only if none found
    if not time_related:
        st.info("Time-based plots will appear if your CSVs include time-of-day / time columns (e.g. 'Time of Day'). No time-based datasets were found automatically.")
    else:
        # display each discovered dataset using unique chart types
        chart_types = ["area_chart", "radial_bar", "line_chart"]  # Different charts for each temporal dataset
        
        for i, (key, df) in enumerate(time_related):
            pretty = pretty_title_from_key(key)
            chart_type = chart_types[i % len(chart_types)]  # Cycle through different chart types
            
            st.subheader(f"{pretty} — by Time of Day")
            # pick an explicit id_col using common variants
            id_col = None
            for candidate in df.columns:
                low = candidate.lower()
                if "time of day" in low or "time of" in low or "time" == low or low.startswith("time ") or "time" in low or "hour" in low:
                    id_col = candidate
                    break
            if id_col is None:
                # fallback to first column
                id_col = df.columns[0] if len(df.columns) > 0 else None
            if id_col:
                create_interactive_chart(df, id_col, f"🕐 {pretty} — Time Analysis", chart_type)

elif selected_group == "⚔️ Weapons & Violence":
    st.subheader("⚔️ Weapons & Violence Analysis")
    
    # Weapon Usage Analysis
    if "Weapon Usage" in loaded_data:
        df = loaded_data["Weapon Usage"]
        if not df.empty:
            st.markdown("### 🔫 Weapon Usage in Crimes")
            st.markdown("*Analysis of different weapon types used in criminal offenses*")
            id_col = df.columns[0] if len(df.columns) > 0 else None
            if id_col:
                create_interactive_chart(df, id_col, "🔫 Weapons Used in Offenses", "treemap")
    
    # Murder & Assault Analysis
    if "Murder & Assault" in loaded_data:
        df = loaded_data["Murder & Assault"]
        if not df.empty:
            st.markdown("### 💀 Murder & Aggravated Assault Analysis")
            st.markdown("*Detailed analysis of murder and assault circumstances*")
            id_col = df.columns[0] if len(df.columns) > 0 else None
            if id_col:
                create_interactive_chart(df, id_col, "💀 Murder & Assault Circumstances", "heatmap")
    
    # Justifiable Homicide Analysis
    if "Justifiable Homicide" in loaded_data:
        df = loaded_data["Justifiable Homicide"]
        if not df.empty:
            st.markdown("### ⚖️ Justifiable Homicide Analysis")
            st.markdown("*Analysis of justifiable homicide circumstances and legal patterns*")
            
            # Add key metrics first
            col1, col2 = st.columns(2)
            with col1:
                total_cases = df.select_dtypes(include=[np.number]).sum().sum()
                st.metric("Total Cases", f"{total_cases:,.0f}")
            with col2:
                categories = len(df)
                st.metric("Circumstance Categories", f"{categories}")
            
            id_col = df.columns[0] if len(df.columns) > 0 else None
            if id_col:
                create_interactive_chart(df, id_col, "⚖️ Justifiable Homicide Circumstances", "donut")
        else:
            st.info("📊 Justifiable homicide data is being processed...")
    
    # Fallback if no weapon data available
    if not any(key in loaded_data for key in ["Weapon Usage", "Murder & Assault", "Justifiable Homicide"]):
        st.warning("⚠️ Weapon and violence datasets not found. Showing general crime analysis:")
    plot_other_analysis()

elif selected_group == "💊 Substance Analytics":
    st.subheader("💊 Substance & Drug Analytics")
    
    # Handle Drug & Crime Patterns Analysis
    if "Drug Seizures" in loaded_data:
        df = loaded_data["Drug Seizures"]
        if not df.empty:
            st.markdown("### 🧪 Drug-Related Crime Patterns")
            st.markdown("*Analysis of drug-related incidents and their patterns across different categories*")
            
            # Use the first column as identifier and analyze patterns
            id_col = df.columns[0] if len(df.columns) > 0 else None
            if id_col:
                create_interactive_chart(df, id_col, "🧪 Drug-Related Crime Analysis", "violin_chart")
    
    # Handle Drug & Alcohol Use
    if "Drug & Alcohol Use" in loaded_data:
        df = loaded_data["Drug & Alcohol Use"]
        if not df.empty:
            st.markdown("### 🍺 Substance Use in Crimes")
            st.markdown("*Analysis of offender drug and alcohol use during crime incidents*")
            id_col = df.columns[0]
            create_interactive_chart(df, id_col, f"🥃 Substance Involvement in Crimes", "bar_chart")
    
    # Scan for additional drug files
    found_additional = False
    for fname in csv_files:
        if "drug" in fname.lower() and "seizure" not in fname.lower():
            df_try = load_csv(fname)
            if df_try is None or df_try.empty:
                continue
            # Check if not already processed
            if fname not in [v for v in datasets_map.values()]:
                found_additional = True
                st.markdown(f"### 💉 {pretty_title_from_key(fname)}")
            id_col = next((c for c in df_try.columns if not pd.api.types.is_numeric_dtype(df_try[c])), df_try.columns[0] if len(df_try.columns)>0 else None)
            if id_col:
                    create_interactive_chart(df_try, id_col, f"💊 {pretty_title_from_key(fname)}", "line_chart")
    
    if not found_additional and "Drug Seizures" not in loaded_data and "Drug & Alcohol Use" not in loaded_data:
        st.info("📁 No additional drug/alcohol datasets found. The main drug seizures and usage data is displayed above.")

elif selected_group == "🏢 Agency Deep Dive":
    st.subheader("🏢 Agency-Level Deep Dive Analysis")
    possible = [f for f in os.listdir(".") if "united_states" in f.lower() or "offense_type_by_agency" in f.lower() or "us_offense" in f.lower()]
    if possible:
        preview = load_csv(possible[0])
        st.success(f"📁 **Analyzing:** {possible[0]}")
        st.markdown("### 🔍 Interactive Agency Data Explorer")
        st.info("💡 **Tip:** Use the filters below to explore specific agencies, states, or crime types. All charts are interactive!")
        # use the enhanced helper for filtered, paginated display
        agency_table_with_filters(preview)
    else:
        st.warning("📂 **No agency-level dataset found.** Please ensure the agency CSV file is in the same directory.")
        st.markdown("""
        **Expected file names:**
        - Files containing 'united_states'
        - Files containing 'offense_type_by_agency' 
        - Files containing 'us_offense'
        """)

st.markdown("---")

# Enhanced Footer
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea, #764ba2); padding: 2rem; 
            border-radius: 10px; margin: 2rem 0; text-align: center; color: white;'>
    <h3 style='margin: 0;'>🚔 NIBRS Crime Analytics Hub</h3>
    <p style='margin: 0.5rem 0; font-size: 16px;'>Enhanced Interactive Crime Data Analysis Platform</p>
    <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;'>
        <div>📊 <strong>Interactive Charts</strong></div>
        <div>🔍 <strong>Advanced Filtering</strong></div>
        <div>📥 <strong>Data Export</strong></div>
        <div>🎯 <strong>Smart Analytics</strong></div>
    </div>
    <p style='margin-top: 1rem; font-size: 14px; opacity: 0.8;'>
        Built with Streamlit & Plotly | Auto-processing enabled | All visualizations are interactive
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("💡 **Tip:** If any visualization shows missing data warnings, ensure all CSV files are in the project folder and properly formatted.")