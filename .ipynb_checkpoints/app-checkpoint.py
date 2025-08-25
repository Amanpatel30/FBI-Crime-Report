# app.py
import os
import re
import io
import zipfile
import streamlit as st
import pandas as pd
import plotly.express as px

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

st.title("📊 NIBRS Crime Data Interactive Dashboard")
st.markdown("Use the sidebar to navigate between chart groups. (Preprocessing applied automatically)")

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
        df = pd.read_csv(file_name)
    except Exception:
        try:
            df = pd.read_csv(file_name, encoding="latin-1")
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
    # time tables are not explicitly mapped here; they will be discovered automatically
}

# load datasets (cleaned)
loaded_data = {name: load_csv(path) for name, path in datasets_map.items()}

# list CSVs in folder (for on-the-fly discovery)
csv_files = [f for f in os.listdir(".") if f.lower().endswith(".csv")]

# ----------------------------
# Sidebar: compact menu only (no large file-list)
# ----------------------------
with st.sidebar:
    st.markdown("## 🔎 Select Analysis Group")
    st.markdown("")  # spacing

    # main menu items + icons
    menu_items = [
        "Geospatial / State-Level",
        "Participation & Agencies",
        "Incidents & Offenses",
        "Victims",
        "Offenders",
        "Arrestees",
        "Crimes by Location",
        "Crimes by Time",
        "Weapons & Circumstances",
        "Drugs & Alcohol",
        "Agency-level"
    ]
    icons = [
        "globe", "bank", "bar-chart", "people-fill", "person-rolodex", "person-check", "geo-alt", "clock", "shield-shaded", "capsule-pill", "building"
    ]

    if HAS_OPTION_MENU:
        selected_group = option_menu("Main Menu", menu_items, icons=icons, menu_icon="list", default_index=0)
    else:
        selected_group = st.selectbox("Main Menu (no option_menu)", menu_items)

    st.markdown("---")
    # quick download action
    if st.button("Download preview CSVs (zip)"):
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            count = 0
            for k, df in loaded_data.items():
                if df is None or df.empty:
                    continue
                buf = io.StringIO()
                df.head(100).to_csv(buf, index=False)
                zf.writestr(f"{k.replace(' ', '_')}_preview.csv", buf.getvalue())
                count += 1
                if count >= 6:
                    break
        mem_zip.seek(0)
        st.download_button("Download ZIP of previews", data=mem_zip.read(), file_name="nibrs_previews.zip")

    st.markdown("---")
    st.markdown("Project: NIBRS Dashboard")
    st.caption("Preprocessing applied automatically (header fixes, Unnamed drop, numeric coercion).")

# ----------------------------
# Helper: stacked bar from df safely (prevents wide-form error)
# - Added one-line caption generation under each chart
# ----------------------------
def stacked_bar_from_df(df: pd.DataFrame, id_col: str, title: str):
    if df is None or df.empty:
        st.warning("⚠️ Dataset is empty.")
        return
    if id_col not in df.columns:
        st.warning(f"⚠️ Missing column: {id_col}. Available columns: {df.columns.tolist()}")
        return
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("⚠️ No numeric columns found for plotting. Available columns: " + ", ".join(df.columns.tolist()))
        return
    melted = df.melt(id_vars=[id_col], value_vars=numeric_cols, var_name="Category", value_name="Value")
    melted = melted.dropna(subset=["Value"])
    if melted.empty:
        st.warning("⚠️ No numeric values to plot after melting.")
        return
    fig = px.bar(melted, x=id_col, y="Value", color="Category", barmode="stack", title=title)
    st.plotly_chart(fig, use_container_width=True)

    # — Add a concise caption describing the chart
    # Build a short example of numeric categories (sample up to 4)
    sample_cats = numeric_cols[:4]
    if len(numeric_cols) > 4:
        sample_text = ", ".join(sample_cats) + ", ..."
    else:
        sample_text = ", ".join(sample_cats)
    # If id_col looks like time-of-day, mention it as time
    id_lower = id_col.lower() if isinstance(id_col, str) else ""
    if "time" in id_lower or "hour" in id_lower or "day" in id_lower:
        context = f"distribution across time of day ({id_col})"
    else:
        context = f"distribution across {id_col}"
    st.markdown(f"**What this chart shows:** Stacked values for {sample_text} — a {context}.")

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
    Displays the agency-level dataframe with advanced filters and a compact pagination
    placed AFTER the table. Uses st.session_state to persist applied filters until Reset.
    """
    if df is None or df.empty:
        st.warning("No agency-level dataset found or it's empty.")
        return

    # Reset index to simple RangeIndex (we will set left-gutter numbering later)
    df = df.reset_index(drop=True).copy()

    st.markdown("### Advanced Filters")
    with st.expander("Open filters", expanded=False):
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
    # Determine paging values from session_state (use stored page_size if present)
    page_size_options = [10, 20, 50, 100, "All"]
    page_size_key = "agency_filter__page_size"
    stored_page_size = st.session_state.get(page_size_key, 20)
    # coerce stored value to a sane value for computing page_count
    if stored_page_size == "All":
        page_size_val = len(display_filtered) if len(display_filtered) > 0 else 1
    else:
        try:
            page_size_val = int(stored_page_size)
        except Exception:
            page_size_val = 20

    total_pages = max(1, (len(display_filtered) + page_size_val - 1) // page_size_val)

    page_key = "agency_filter__page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    # clamp page
    if st.session_state[page_key] < 1:
        st.session_state[page_key] = 1
    if st.session_state[page_key] > total_pages:
        st.session_state[page_key] = total_pages

    # compute slice using the stored page
    curr_page = st.session_state[page_key]
    start_idx = (curr_page - 1) * page_size_val
    end_idx = start_idx + page_size_val
    page_df = display_filtered.iloc[start_idx:end_idx].copy()

    # set left-gutter index to reflect overall position (1-based)
    page_df.index = pd.RangeIndex(start=start_idx + 1, stop=start_idx + 1 + len(page_df))

    # final display using default Streamlit style (keeps table design)
    st.dataframe(page_df, use_container_width=True)

    # --------------------------
    # PAGINATION HEADER (title + helper on same line)
    st.markdown(
        """<div class="pagination-header">
               <div class="pagination-title">Pagination</div>
               <div class="pagination-helper">Use the controls to navigate pages. Changing rows-per-page resets to page 1.</div>
           </div>""",
        unsafe_allow_html=True,
    )

    # Single-row controls: Prev | Page(label) | Page input | Next | Rows(label) | Rows select | Status
    cols = st.columns([1, 0.6, 1.2, 0.9, 0.8, 1.1, 1.2], gap="small")

    # Prev
    with cols[0]:
        if st.button("Prev", key="pag_prev_final"):
            if st.session_state[page_key] > 1:
                st.session_state[page_key] -= 1

    # Page label
    with cols[1]:
        st.markdown("<div class='inline-label'>Page</div>", unsafe_allow_html=True)

    # Page number input (compact)
    with cols[2]:
        new_page = st.number_input("", min_value=1, max_value=total_pages, value=st.session_state[page_key],
                                   step=1, format="%d", key="agency_page_number_widget_final")
        try:
            new_page = int(new_page)
            if new_page < 1:
                new_page = 1
            if new_page > total_pages:
                new_page = total_pages
            if new_page != st.session_state[page_key]:
                st.session_state[page_key] = new_page
        except Exception:
            pass

    # Next
    with cols[3]:
        if st.button("Next", key="pag_next_final"):
            if st.session_state[page_key] < total_pages:
                st.session_state[page_key] += 1

    # Rows label
    with cols[4]:
        st.markdown("<div class='inline-label'>Rows</div>", unsafe_allow_html=True)

    # Rows select
    with cols[5]:
        page_size_widget = st.selectbox("", options=page_size_options,
                                       index=page_size_options.index(stored_page_size) if stored_page_size in page_size_options else 1,
                                       key="agency_page_size_widget_final")
        # if changed, store and reset to page 1
        if page_size_widget != st.session_state.get(page_size_key, stored_page_size):
            st.session_state[page_size_key] = page_size_widget
            st.session_state[page_key] = 1

    # Status
    with cols[6]:
        st.markdown(f"<span class='page-status'><strong>Page {st.session_state[page_key]} of {total_pages}</strong> &nbsp; • &nbsp; Rows: {page_size_widget}</span>", unsafe_allow_html=True)

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
            stacked_bar_from_df(df, id_col, "Victims by Age Category")
    if "Victims Sex" in loaded_data:
        df = loaded_data["Victims Sex"]
        id_col = next((c for c in df.columns if "offense" in c.lower() and "category" in c.lower()),
                      df.columns[0] if len(df.columns) > 0 else None)
        if id_col:
            stacked_bar_from_df(df, id_col, "Victims by Sex")
    if "Victims Race" in loaded_data:
        df = loaded_data["Victims Race"]
        id_col = next((c for c in df.columns if "offense" in c.lower() and "category" in c.lower()),
                      df.columns[0] if len(df.columns) > 0 else None)
        if id_col:
            stacked_bar_from_df(df, id_col, "Victims by Race")

def plot_offender_analysis():
    if "Offenders Age" in loaded_data:
        df = loaded_data["Offenders Age"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Offenders by Age Category")
    if "Offenders Sex" in loaded_data:
        df = loaded_data["Offenders Sex"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Offenders by Sex")
    if "Offenders Race" in loaded_data:
        df = loaded_data["Offenders Race"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Offenders by Race")

def plot_arrestee_analysis():
    if "Arrestees Age" in loaded_data:
        df = loaded_data["Arrestees Age"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Arrestees by Age Category")
    if "Arrestees Sex" in loaded_data:
        df = loaded_data["Arrestees Sex"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Arrestees by Sex")
    if "Arrestees Race" in loaded_data:
        df = loaded_data["Arrestees Race"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Arrestees by Race")

def plot_other_analysis():
    if "Victim-Offender Relationship" in loaded_data:
        df = loaded_data["Victim-Offender Relationship"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Victim-Offender Relationship by Offense")
    if "Property Crimes by Location" in loaded_data:
        df = loaded_data["Property Crimes by Location"]
        id_col = df.columns[0] if len(df.columns)>0 else None
        if id_col:
            stacked_bar_from_df(df, id_col, "Property Crimes by Location")

# ----------------------------
# Page Logic
# ----------------------------
st.markdown("---")
if selected_group == "Geospatial / State-Level":
    st.subheader("🌎 Geospatial Crime Analysis")
    plot_state_heatmap()

elif selected_group == "Participation & Agencies":
    st.subheader("🏛 Participation & Agency-level")
    df = loaded_data.get("Participation by State", pd.DataFrame())
    if df is not None and not df.empty:
        # show first 20 rows with 1-based left index (no explicit Index column)
        display_df = df.reset_index(drop=True).copy()
        display_df.index = pd.RangeIndex(start=1, stop=len(display_df) + 1)
        st.dataframe(display_df.head(20), use_container_width=True)
    else:
        st.warning("Participation dataset not loaded.")

elif selected_group == "Incidents & Offenses":
    st.subheader("📈 Incidents & Offenses")
    df = loaded_data.get("Incidents & Offenses", pd.DataFrame())
    if df is not None and not df.empty:
        id_col = next((c for c in df.columns if "offense" in c.lower() and "category" in c.lower()), df.columns[0] if len(df.columns)>0 else None)
        if id_col:
            stacked_bar_from_df(df, id_col, "Incidents & Offenses (stacked)")
    else:
        st.warning("Incidents dataset not loaded.")

elif selected_group == "Victims":
    st.subheader("🧍 Victims Analysis")
    plot_victim_analysis()

elif selected_group == "Offenders":
    st.subheader("👥 Offender Analysis")
    plot_offender_analysis()

elif selected_group == "Arrestees":
    st.subheader("🚓 Arrestee Analysis")
    plot_arrestee_analysis()

elif selected_group == "Crimes by Location":
    st.subheader("📍 Crimes by Location")
    plot_other_analysis()

elif selected_group == "Crimes by Time":
    st.subheader("🕒 Crimes by Time")

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
        # display each discovered dataset using a pretty title (no raw filename printed)
        for key, df in time_related:
            pretty = pretty_title_from_key(key)
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
                stacked_bar_from_df(df, id_col, f"{pretty} — by {id_col}")

elif selected_group == "Weapons & Circumstances":
    st.subheader("🛡 Weapons & Circumstances")
    plot_other_analysis()

elif selected_group == "Drugs & Alcohol":
    st.subheader("💊 Drug & Alcohol")
    # find any loaded datasets whose friendly key mentions drug or alcohol
    found = False
    for key, df in loaded_data.items():
        if df is None or df.empty:
            continue
        if "drug" in key.lower() or "alcohol" in key.lower():
            found = True
            st.subheader(pretty_title_from_key(key))
            # try to pick sensible id column (first textual column)
            id_col = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), df.columns[0] if len(df.columns)>0 else None)
            if id_col:
                stacked_bar_from_df(df, id_col, pretty_title_from_key(key))
    # additionally scan CSV filenames (in case some drug files were not in the mapping)
    for fname in csv_files:
        if "drug" in fname.lower() or "alcohol" in fname.lower():
            df_try = load_csv(fname)
            if df_try is None or df_try.empty:
                continue
            # avoid duplicates if we already displayed the mapped one
            already = any(pretty_title_from_key(k).lower() == pretty_title_from_key(fname).lower() for k,_ in loaded_data.items())
            if already:
                continue
            found = True
            st.subheader(pretty_title_from_key(fname))
            id_col = next((c for c in df_try.columns if not pd.api.types.is_numeric_dtype(df_try[c])), df_try.columns[0] if len(df_try.columns)>0 else None)
            if id_col:
                stacked_bar_from_df(df_try, id_col, pretty_title_from_key(fname))
    if not found:
        st.warning("No drug/alcohol datasets detected in the mapped files. Make sure the drug/alcohol CSVs are next to app.py; their filenames often contain 'drug' or 'alcohol'.")

elif selected_group == "Agency-level":
    st.subheader("🏛 Agency-level (US Offense by Agency)")
    possible = [f for f in os.listdir(".") if "united_states" in f.lower() or "offense_type_by_agency" in f.lower() or "us_offense" in f.lower()]
    if possible:
        preview = load_csv(possible[0])
        st.write("File:", possible[0])
        # use the helper for filtered, paginated display (keeps Streamlit table style)
        agency_table_with_filters(preview)
    else:
        st.info("Place the US_Offense_Agency CSV next to app.py with name containing 'united_states' or 'us_offense'.")

st.markdown("---")
st.caption("If a plot warns that numeric columns are missing, check your CSVs in the project folder. If you'd like additional filters (state/time/age/sex), tell me and I'll add them.")
