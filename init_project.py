#!/usr/bin/env python3
"""
Initialization script for Market Pattern Explorer repo.
This script ensures all necessary directories exist and
sets up the environment for the application.
"""

import os
import sys

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def main():
    """Main initialization function."""
    print("Initializing Market Pattern Explorer project structure...")
    
    # Create necessary directories
    ensure_directory("vector_db")
    ensure_directory("assets")
    ensure_directory(".streamlit")
    
    # Check for configuration files
    streamlit_config = ".streamlit/config.toml"
    streamlit_css = ".streamlit/style.css"
    
    missing_files = []
    
    if not os.path.exists(streamlit_config):
        missing_files.append(streamlit_config)
    
    if not os.path.exists(streamlit_css):
        missing_files.append(streamlit_css)
    
    if missing_files:
        print("\nThe following configuration files are missing. Creating them now:")
        for file in missing_files:
            print(f"  - {file}")
            
            if file == streamlit_config:
                # Create default Streamlit config
                with open(streamlit_config, "w") as f:
                    f.write("""[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F8F8"
textColor = "#222222"
font = "sans serif"
""")
                print(f"    Created default {streamlit_config}")
                
            elif file == streamlit_css:
                # Create default CSS styling
                with open(streamlit_css, "w") as f:
                    f.write("""/* Main styling for Market Pattern Explorer */

/* Custom font and general styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Improved header styling */
h1, h2, h3 {
    font-weight: 600 !important;
}

/* Card styling for sections */
div.stBlock {
    border-radius: 8px;
    padding: 1px 20px 20px 20px;
    background-color: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 1rem;
}

/* Button styling */
button.stButton>button {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s;
    border: none;
}

button.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
}

/* Slider styling */
div.stSlider>div>div>div {
    height: 6px;
}

/* Improved select box and inputs */
div.stSelectbox>div>div,
div.stNumberInput>div>div:first-child {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 6px;
    border: 1px solid #eee;
    padding: 2px 10px;
}

/* Dataframe styling */
.dataframe * {
    font-size: 0.9rem !important;
}

/* Metric label improvement */
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #666;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    height: 40px;
    padding-top: 7px;
    border-radius: 4px 4px 0 0;
    font-weight: 500;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}

.stTabs [aria-selected="true"] {
    background-color: #f0f9ff !important;
}

/* Info and warning boxes */
div.stInfo, div.stWarning, div.stError, div.stSuccess {
    border-radius: 6px;
    padding: 20px !important;
    line-height: 1.5;
}

/* Tooltip hover effects */
div.tooltip-content {
    background-color: rgba(29, 30, 31, 0.85) !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 10px 15px !important;
    font-size: 0.8rem !important;
    backdrop-filter: blur(3px);
}

/* Progress bar */
div.stProgress > div > div > div > div {
    background-color: #1E88E5;
}

/* Charts borders */
div.js-plotly-plot {
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
""")
                print(f"    Created default {streamlit_css}")
                
        print("\nConfiguration files have been created successfully.")
    
    print("\nInitialization complete! You can now run the application with:")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()