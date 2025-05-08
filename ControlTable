import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64
import os
import tempfile
from io import StringIO

# Try to import PDF processing libraries with fallbacks
try:
    import tabula
    PDF_SUPPORT = True
except ImportError:
    try:
        import PyPDF2
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False

# Set page configuration
st.set_page_config(
    page_title="Interactive EDA App",
    page_icon="📊",
    layout="wide"
)

# Application title and description
st.title("📊 Interactive Exploratory Data Analysis")
st.markdown("""
This application helps you perform exploratory data analysis (EDA) on your datasets. 
Upload your data file in various formats and explore its properties interactively.
""")

# Sidebar for upload and main controls
with st.sidebar:
    st.header("Upload Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file (Excel, CSV, PDF, TXT)",
        type=["xlsx", "xls", "csv", "pdf", "txt"]
    )
    
    if uploaded_file and uploaded_file.name.endswith('.txt'):
        delimiter_options = {
            "Comma (,)": ",", 
            "Semicolon (;)": ";", 
            "Tab (\\t)": "\t", 
            "Space ( )": " ",
            "Pipe (|)": "|"
        }
        delimiter = st.selectbox(
            "Select delimiter for text file:",
            options=list(delimiter_options.keys())
        )
        delimiter = delimiter_options[delimiter]

# Function to load data from various file formats
@st.cache_data
def load_data(file, file_type, delimiter=","):
    """Load data from different file formats into a pandas DataFrame."""
    try:
        if file_type in ['xlsx', 'xls']:
            return pd.read_excel(file)
        
        elif file_type == 'csv':
            return pd.read_csv(file)
        
        elif file_type == 'txt':
            return pd.read_csv(file, delimiter=delimiter)
        
        elif file_type == 'pdf':
            if not PDF_SUPPORT:
                st.warning("PDF support is not available. Please install tabula-py or PyPDF2.")
                return None
            
            # Save uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.read())
                pdf_path = tmp_file.name
            
            try:
                # Try using tabula if available
                if 'tabula' in globals():
                    dfs = tabula.read_pdf(pdf_path, pages='all')
                    if dfs:
                        # Combine all tables from the PDF
                        return pd.concat(dfs, ignore_index=True)
                    else:
                        st.warning("No tables detected in the PDF.")
                        return None
                # Fallback to PyPDF2 (limited capability for tabular data)
                elif 'PyPDF2' in globals():
                    st.warning("Using PyPDF2 for PDF extraction. Note that complex tables may not be properly extracted.")
                    reader = PyPDF2.PdfReader(pdf_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Try to convert text to DataFrame
                    # This is a very basic approach and may not work well for complex PDFs
                    lines = text.strip().split('\n')
                    data = [line.split() for line in lines if line.strip()]
                    if data:
                        return pd.DataFrame(data[1:], columns=data[0])
                    else:
                        st.warning("Could not extract tabular data from the PDF.")
                        return None
            finally:
                # Clean up temporary file
                os.unlink(pdf_path)
        
        return None
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to display column information
def display_column_info(df):
    """Display information about DataFrame columns."""
    # Create a DataFrame with column information
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Null %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    st.dataframe(col_info, use_container_width=True)
    
    # Identify columns with missing values
    cols_with_nulls = df.columns[df.isnull().any()].tolist()
    
    if cols_with_nulls:
        st.warning(f"Columns with missing values: {', '.join(cols_with_nulls)}")
    else:
        st.success("No missing values detected in any column.")

# Function to analyze categorical columns
def analyze_categorical_columns(df):
    """Analyze and display information about categorical columns."""
    # Identify categorical columns (object, category, or bool)
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if categorical_cols:
        st.subheader("Categorical Columns Analysis")
        
        # Select a categorical column to analyze
        selected_cat_col = st.selectbox(
            "Select a categorical column to analyze:",
            options=categorical_cols
        )
        
        # Get value counts
        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = [selected_cat_col, 'Count']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(f"Unique values in '{selected_cat_col}':")
            st.write(f"- Unique count: {df[selected_cat_col].nunique()}")
            st.dataframe(value_counts, use_container_width=True)
        
        with col2:
            # Plot distribution of categorical values
            fig = px.bar(
                value_counts, 
                x=selected_cat_col, 
                y='Count',
                title=f"Distribution of {selected_cat_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns detected in the dataset.")

# Function to display descriptive statistics
def display_descriptive_stats(df):
    """Calculate and display descriptive statistics for numerical columns."""
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    
    if numeric_cols:
        st.subheader("Descriptive Statistics for Numerical Columns")
        
        # Display basic statistics
        desc_stats = df[numeric_cols].describe().T
        
        # Add additional statistics
        desc_stats['median'] = df[numeric_cols].median()
        desc_stats['variance'] = df[numeric_cols].var()
        desc_stats['skewness'] = df[numeric_cols].skew()
        desc_stats['kurtosis'] = df[numeric_cols].kurtosis()
        
        # Reorder columns
        desc_stats = desc_stats[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'variance', 'skewness', 'kurtosis']]
        
        st.dataframe(desc_stats.round(2), use_container_width=True)
    else:
        st.info("No numerical columns detected in the dataset.")

# Function to handle missing values
def handle_missing_values(df):
    """Provide options for handling missing values in the DataFrame."""
    st.subheader("Missing Value Handling")
    
    # Identify columns with missing values
    cols_with_nulls = df.columns[df.isnull().any()].tolist()
    
    if not cols_with_nulls:
        st.success("No missing values detected in the dataset.")
        return df
    
    # Display information about missing values
    missing_info = pd.DataFrame({
        'Column': cols_with_nulls,
        'Missing Count': [df[col].isnull().sum() for col in cols_with_nulls],
        'Missing %': [(df[col].isnull().sum() / len(df) * 100).round(2) for col in cols_with_nulls]
    }).sort_values('Missing Count', ascending=False)
    
    st.dataframe(missing_info, use_container_width=True)
    
    # Options for handling missing values
    handling_option = st.radio(
        "Choose a method to handle missing values:",
        options=["No action", "Imputation", "Remove rows"]
    )
    
    df_processed = df.copy()
    
    if handling_option == "Imputation":
        # Column selection for imputation
        col_to_impute = st.selectbox(
            "Select column for imputation:",
            options=cols_with_nulls
        )
        
        # Determine data type of selected column
        is_numeric = pd.api.types.is_numeric_dtype(df[col_to_impute])
        
        if is_numeric:
            impute_method = st.selectbox(
                "Select imputation method for numeric column:",
                options=["Mean", "Median", "Mode", "Specific Value"]
            )
            
            if impute_method == "Mean":
                fill_value = df_processed[col_to_impute].mean()
            elif impute_method == "Median":
                fill_value = df_processed[col_to_impute].median()
            elif impute_method == "Mode":
                fill_value = df_processed[col_to_impute].mode()[0]
            else:  # Specific Value
                fill_value = st.number_input(
                    "Enter value to impute:",
                    value=float(df_processed[col_to_impute].mean())
                )
        else:
            impute_method = st.selectbox(
                "Select imputation method for categorical column:",
                options=["Mode", "Specific Value"]
            )
            
            if impute_method == "Mode":
                fill_value = df_processed[col_to_impute].mode()[0]
            else:  # Specific Value
                fill_value = st.text_input(
                    "Enter value to impute:",
                    value="Missing"
                )
        
        # Apply imputation
        if st.button("Apply Imputation"):
            df_processed[col_to_impute] = df_processed[col_to_impute].fillna(fill_value)
            st.success(f"Successfully imputed missing values in '{col_to_impute}' with {impute_method.lower()}: {fill_value}")
    
    elif handling_option == "Remove rows":
        # Column selection for row removal
        cols_to_check = st.multiselect(
            "Select columns to check for missing values (rows with NaN in these columns will be removed):",
            options=cols_with_nulls
        )
        
        if cols_to_check and st.button("Remove Rows with Missing Values"):
            original_rows = len(df_processed)
            df_processed = df_processed.dropna(subset=cols_to_check)
            removed_rows = original_rows - len(df_processed)
            
            st.success(f"Removed {removed_rows} rows with missing values in the selected columns.")
    
    return df_processed

# Function to handle outliers
def handle_outliers(df):
    """Detect and visualize outliers in numerical columns."""
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    
    if not numeric_cols:
        st.info("No numerical columns detected for outlier analysis.")
        return df
    
    st.subheader("Outlier Detection and Handling")
    
    # Column selection for outlier analysis
    col_for_outliers = st.selectbox(
        "Select a numerical column for outlier analysis:",
        options=numeric_cols
    )
    
    # Calculate IQR for selected column
    Q1 = df[col_for_outliers].quantile(0.25)
    Q3 = df[col_for_outliers].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df[col_for_outliers] < lower_bound) | (df[col_for_outliers] > upper_bound)][col_for_outliers]
    outlier_count = len(outliers)
    
    st.write(f"Outlier analysis for '{col_for_outliers}':")
    st.write(f"- IQR: {IQR:.2f}")
    st.write(f"- Lower bound: {lower_bound:.2f}")
    st.write(f"- Upper bound: {upper_bound:.2f}")
    st.write(f"- Number of outliers detected: {outlier_count} ({outlier_count/len(df)*100:.2f}% of data)")
    
    # Visualize outliers with a box plot
    fig, ax = plt.subplots(figsize=(10, 4))
    df.boxplot(column=col_for_outliers, ax=ax)
    ax.set_title(f'Box Plot of {col_for_outliers}')
    st.pyplot(fig)
    
    # Option to handle outliers
    if outlier_count > 0:
        handling_option = st.radio(
            "Choose a method to handle outliers:",
            options=["No action", "Cap outliers", "Remove outliers"]
        )
        
        df_processed = df.copy()
        
        if handling_option == "Cap outliers":
            if st.button("Apply Capping"):
                # Cap the outliers at the boundaries
                df_processed[col_for_outliers] = df_processed[col_for_outliers].clip(lower=lower_bound, upper=upper_bound)
                st.success(f"Successfully capped outliers in '{col_for_outliers}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        elif handling_option == "Remove outliers":
            if st.button("Remove Outlier Rows"):
                # Remove rows with outliers
                original_rows = len(df_processed)
                df_processed = df_processed[(df_processed[col_for_outliers] >= lower_bound) & 
                                           (df_processed[col_for_outliers] <= upper_bound)]
                removed_rows = original_rows - len(df_processed)
                
                st.success(f"Removed {removed_rows} rows with outliers in '{col_for_outliers}'.")
        
        return df_processed
    
    return df

# Function to visualize data
def visualize_data(df):
    """Create visualizations for data exploration."""
    st.subheader("Data Visualization")

    # Get column types
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if not numeric_cols:
        st.warning("No numerical columns available for visualization.")
        return

    # Chart type selection
    chart_type = st.selectbox(
        "Select visualization type:",
        options=["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Correlation Heatmap"]
    )

    if chart_type == "Histogram":
        col = st.selectbox("Select column for histogram:", options=numeric_cols)
        bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30)

        fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        col = st.selectbox("Select column for box plot:", options=numeric_cols)

        group_by = None
        if categorical_cols:
            use_grouping = st.checkbox("Group by categorical variable")
            if use_grouping:
                group_by = st.selectbox("Select grouping variable:", options=categorical_cols)

        if group_by:
            fig = px.box(df, y=col, x=group_by, title=f"Box Plot of {col} by {group_by}")
        else:
            fig = px.box(df, y=col, title=f"Box Plot of {col}")

        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numerical columns for a scatter plot.")
            return

        x_col = st.selectbox("Select X-axis column:", options=numeric_cols)
        y_col = st.selectbox("Select Y-axis column:", options=numeric_cols, index=min(1, len(numeric_cols)-1))

        color_col = None
        if categorical_cols:
            use_color = st.checkbox("Color by categorical variable")
            if use_color:
                color_col = st.selectbox("Select color variable:", options=categorical_cols)

        if color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f"Scatter Plot of {y_col} vs {x_col} (colored by {color_col})")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}")

        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        if not categorical_cols:
            st.warning("Need at least one categorical column for a bar chart.")
            return

        cat_col = st.selectbox("Select categorical column:", options=categorical_cols)

        # Count values or use a numeric column for values
        agg_type = st.radio("Bar chart values:", options=["Count", "Numeric column average"])

        if agg_type == "Count":
            # Simple count of categories
            value_counts = df[cat_col].value_counts().reset_index()
            value_counts.columns = [cat_col, 'Count']  # Rename columns for clarity
            fig = px.bar(
                value_counts,
                x=cat_col,
                y='Count',  # Use the 'Count' column for the y-axis
                title=f"Count of values in {cat_col}"
            )
            fig.update_xaxes(title=cat_col)
            fig.update_yaxes(title="Count")
        else:
            # Average of a numeric column by category
            if not numeric_cols:
                st.warning("Need at least one numeric column for aggregation.")
                return

            num_col = st.selectbox("Select numeric column for average:", options=numeric_cols)

            agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(
                agg_df,
                x=cat_col,
                y=num_col,
                title=f"Average {num_col} by {cat_col}"
            )

        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numerical columns for a correlation heatmap.")
            return

        # Select columns for correlation
        selected_cols = st.multiselect(
            "Select columns for correlation analysis:",
            options=numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )

        if not selected_cols or len(selected_cols) < 2:
            st.warning("Please select at least two columns.")
            return

        # Calculate correlation matrix
        corr_matrix = df[selected_cols].corr()

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

# Main application logic
if uploaded_file is not None:
    # Determine file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Check if PDF support is available
    if file_extension == 'pdf' and not PDF_SUPPORT:
        st.warning("""
        PDF processing libraries (tabula-py or PyPDF2) are not available.
        Please install one of these libraries to process PDF files.
        """)
    
    # Load data based on file type
    with st.spinner('Loading data...'):
        if file_extension == 'txt':
            df = load_data(uploaded_file, file_extension, delimiter)
        else:
            df = load_data(uploaded_file, file_extension)
    
    if df is not None:
        st.success(f"Data loaded successfully from {uploaded_file.name}")
        
        # Display the number of rows and columns
        st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Data Preview", 
            "Column Information", 
            "Descriptive Statistics", 
            "Categorical Analysis", 
            "Missing Values", 
            "Outlier Detection"
        ])
        
        with tab1:
            st.subheader("Data Preview")
            # Display the first few rows of the DataFrame
            rows_to_show = st.slider("Number of rows to display:", 5, 50, 10)
            st.dataframe(df.head(rows_to_show), use_container_width=True)
        
        with tab2:
            st.subheader("Column Information")
            display_column_info(df)
        
        with tab3:
            display_descriptive_stats(df)
        
        with tab4:
            analyze_categorical_columns(df)
        
        with tab5:
            df = handle_missing_values(df)
        
        with tab6:
            df = handle_outliers(df)
        
        # Add a tab for visualizations
        st.markdown("---")
        visualize_data(df)
        
        # Option to download the processed data
        st.markdown("---")
        st.subheader("Download Processed Data")
        
        # Column selection and renaming
        st.write("Select columns and optionally rename them:")
        
        # Create column selection interface
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to include in download:",
            options=all_columns,
            default=all_columns
        )
        
        if not selected_columns:
            st.warning("Please select at least one column for download.")
        else:
            # Column renaming interface
            st.write("Rename selected columns:")
            
            # Initialize dictionary to store column renames
            column_renames = {}
            
            # Initialize session state for rename fields if not exists
            if 'rename_count' not in st.session_state:
                st.session_state.rename_count = 3  # Start with 3 rename fields
            
            # Function to add more rename fields
            def add_more_rename_fields():
                st.session_state.rename_count += 1
            
            # Display initial 3 rename fields with dropdowns
            for i in range(st.session_state.rename_count):
                cols = st.columns([3, 3, 1])
                
                # Column selection dropdown
                original_col = cols[0].selectbox(
                    "Select column:",
                    options=[""] + selected_columns,
                    key=f"original_col_{i}"
                )
                
                if original_col:
                    # New name input field
                    new_name = cols[1].text_input(
                        "New name:",
                        value="",
                        key=f"new_name_{i}"
                    )
                    
                    # Store the rename if both fields are filled
                    if new_name and new_name != original_col:
                        column_renames[original_col] = new_name
            
            # Add button to add more rename fields
            st.button("+ Add more columns to rename", on_click=add_more_rename_fields)
            
            # Get filename for download
            custom_filename = st.text_input(
                "Enter filename (without extension):",
                value="processed_data"
            )
            
            # Get available file formats for download
            download_formats = {
                "CSV (.csv)": "csv",
                "Excel (.xlsx)": "xlsx"
            }
            
            download_format = st.selectbox(
                "Select file format for download:",
                options=list(download_formats.keys())
            )
            
            format_key = download_formats[download_format]
            
            if st.button("Download Processed Data"):
                # Create a copy with selected columns
                df_download = df[selected_columns].copy()
                
                # Apply column renames if any
                if column_renames:
                    df_download = df_download.rename(columns=column_renames)
                
                # Create a buffer
                buffer = io.BytesIO()
                
                # Write data to the buffer
                if format_key == "csv":
                    csv_data = df_download.to_csv(index=False)
                    buffer = StringIO(csv_data)
                    mime_type = "text/csv"
                    file_ext = "csv"
                else:  # Excel
                    df_download.to_excel(buffer, index=False)
                    buffer.seek(0)
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_ext = "xlsx"
                
                # Generate download link
                if format_key == "csv":
                    # For CSV, buffer.getvalue() returns a string that needs to be encoded
                    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
                else:  # Excel
                    # For Excel, buffer.getvalue() already returns bytes, so no need to encode
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                
                filename = f"{custom_filename}.{file_ext}"
                href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Click here to download</a>'
                st.markdown(href, unsafe_allow_html=True)
else:
    # Display instructions when no file is uploaded
    st.info("""
    #### Upload a file to get started
    
    This application supports the following file formats:
    - Excel (.xlsx, .xls)
    - CSV (.csv)
    - PDF (.pdf) - For tabular data extraction
    - Text files (.txt) - For delimited data
    
    Once you upload a file, you can:
    - View and explore your data
    - Get column information and statistics
    - Analyze categorical variables
    - Handle missing values
    - Detect and handle outliers
    - Create visualizations
    
    Use the sidebar to upload your file.
    """)
