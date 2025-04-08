import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA  # Using PCA instead of UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import time
import json
from datetime import datetime, timedelta
import faiss  # For vector database functionality
import pickle
import os
import math
from typing import List, Dict, Tuple, Any, Union, Optional
from scipy import stats

# Load custom CSS for Apple-like styling
def load_custom_css():
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
# Set page configuration
st.set_page_config(
    page_title="Stock Pattern Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
try:
    load_custom_css()
except Exception as e:
    st.write("Note: Custom styling could not be loaded.")
    
# Add a subtle gradient background
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f7 100%);
    }
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Card-like containers for visuals */
    .stPlotlyChart, .stDataFrame {
        background-color: white !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        padding: 1rem !important;
        margin-bottom: 1.5rem !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stPlotlyChart:hover, .stDataFrame:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1) !important;
    }
    /* Improved sidebar styling */
    .css-1d391kg, .css-163ttbj, section[data-testid="stSidebar"] {
        background-color: #fafafa !important;
    }
    /* Custom tab styling */
    button[role="tab"] {
        border-radius: 4px 4px 0 0 !important;
        padding: 10px 24px !important;
        margin-right: 4px !important;
        font-weight: 500 !important;
    }
    button[role="tab"][aria-selected="true"] {
        background-color: #0071e3 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Simplified title with Apple-style
st.title("Market Pattern Explorer")

# Add clean description
st.markdown("Discover hidden patterns in market behavior through AI-driven analysis.")

# Feature boxes with clean styling
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### ✓ Similarity Search")
    st.markdown("Find similar trading days based on technical patterns")
with col2:
    st.markdown("#### ✓ Pattern Recognition")
    st.markdown("AI-powered clustering of market behaviors")
with col3:
    st.markdown("#### ✓ Advanced Visualizations")
    st.markdown("Interactive visualization of market dynamics")

# Add separator
st.markdown("---")

# Sidebar configuration with clean styling
with st.sidebar:
    # Add header
    st.header("Analysis Controls")
    st.markdown("Customize your market pattern analysis")
    
    # Add separator
    st.markdown("---")
    
    # Stock selection
    ticker_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "V", "JNJ", "PG", "SPY", "QQQ", "DIA"]
    ticker = st.selectbox("Select Ticker Symbol", ticker_options, index=0)
    custom_ticker = st.text_input("Or Enter Custom Ticker Symbol")
    if custom_ticker:
        ticker = custom_ticker.upper()
    
    # Date range selection
    today = datetime.now()
    default_start = today - timedelta(days=3650)  # ~10 years
    
    period_options = ["1 Year", "3 Years", "5 Years", "10 Years", "Max", "Custom"]
    period_selection = st.selectbox("Select Time Period", period_options, index=3)
    
    # Set the start date based on period selection
    if period_selection == "1 Year":
        start_date = today - timedelta(days=365)
        end_date = today
    elif period_selection == "3 Years":
        start_date = today - timedelta(days=3*365)
        end_date = today
    elif period_selection == "5 Years":
        start_date = today - timedelta(days=5*365)
        end_date = today
    elif period_selection == "10 Years":
        start_date = today - timedelta(days=10*365)
        end_date = today
    elif period_selection == "Max":
        start_date = datetime(1970, 1, 1)
        end_date = today
    else:  # Custom
        st.write("Select custom date range:")
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)
    
    # Clustering options
    st.subheader("Clustering Options")
    
    # Option to select clustering algorithm
    clustering_algorithm = st.selectbox(
        "Clustering Algorithm", 
        ["K-Means", "DBSCAN"], 
        index=0,
        help="K-Means creates spherical clusters with roughly equal sizes. DBSCAN identifies clusters of arbitrary shape based on density."
    )
    
    if clustering_algorithm == "K-Means":
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4)
    else:  # DBSCAN options
        eps = st.slider("DBSCAN Epsilon", min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                        help="Maximum distance between two samples for them to be considered as in the same neighborhood")
        min_samples = st.slider("DBSCAN Min Samples", min_value=2, max_value=20, value=5,
                                help="Number of samples in a neighborhood for a point to be considered as a core point")
    
    # Feature selection for clustering - simplified to focus on most important indicators
    st.subheader("Feature Selection")
    
    # Core indicators with explanations
    core_features = {
        "Return": "Daily price change percentage",
        "Volume": "Trading volume",
        "Volatility": "Price fluctuation (20-day)",
        "Momentum": "Price trend strength (20-day)",
        "RSI": "Relative Strength Index (overbought/oversold)",
        "MACD": "Moving Average Convergence/Divergence (trend)"
    }
    
    # Simple label for feature selection
    st.markdown("Select key indicators for pattern analysis:")
    
    # Use columns for a more compact layout
    col1, col2 = st.columns(2)
    
    # Pre-selected core features for simplicity
    selected_features = []
    
    with col1:
        if st.checkbox("Return", value=True, help="Daily price change percentage"):
            selected_features.append("Return")
        if st.checkbox("Volume", value=True, help="Trading volume"):
            selected_features.append("Volume")
        if st.checkbox("Volatility", value=True, help="Price fluctuation (20-day standard deviation)"):
            selected_features.append("Volatility_20d")
            
    with col2:
        if st.checkbox("Momentum", value=True, help="Price trend strength (20-day)"):
            selected_features.append("Momentum_20d")
        if st.checkbox("RSI", value=False, help="Relative Strength Index (overbought/oversold indicator)"):
            selected_features.append("RSI")
        if st.checkbox("MACD", value=False, help="Moving Average Convergence/Divergence (trend indicator)"):
            selected_features.append("MACD")
    
    # If nothing selected, use default
    if not selected_features:
        selected_features = ["Return", "Volume", "Volatility_20d", "Momentum_20d"]
        st.info("Using default indicators: Return, Volume, Volatility, and Momentum")
    
    # Visualization options
    st.subheader("Visualization Options")
    dimension_reduction = st.selectbox(
        "Dimension Reduction Method",
        ["PCA", "t-SNE"],
        index=0,
        help="PCA preserves global structure. t-SNE preserves local similarities but is slower."
    )
    
    random_state = 42  # For reproducibility of results
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        normalize_features = st.checkbox("Normalize Features", value=True,
                                        help="Standardize features to have zero mean and unit variance")
        show_centroids = st.checkbox("Show Cluster Centroids", value=True,
                                    help="Display cluster centers in visualizations")
        logarithmic_scale = st.checkbox("Use Log Scale for Volume", value=False,
                                       help="Apply logarithmic transformation to volume data")
    
    # Add a divider
    st.markdown("---")
    
    # Add help information
    with st.expander("Help & Information"):
        st.markdown("""
        ### How to use this app:
        1. Select a stock ticker or enter a custom one
        2. Choose a time period for analysis
        3. Configure clustering parameters
        4. Explore the different visualization tabs
        
        ### Features explained:
        - **Return**: Daily price percentage change
        - **Volume**: Trading volume (normalized)
        - **Volatility**: Standard deviation of returns
        - **Momentum**: Price movement over time periods
        - **RSI**: Relative Strength Index (momentum indicator)
        - **MACD**: Moving Average Convergence Divergence
        - **BB Width**: Bollinger Band Width (volatility)
        - **OBV**: On-Balance Volume (volume flow)
        - **ATR**: Average True Range (volatility)
        - **Relative Volume**: Current volume vs. average
        """)

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    """Calculate various technical indicators for the price data."""
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # If we're missing columns, try to find them in a MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            new_df = pd.DataFrame(index=df.index)
            for col in required_cols:
                matching_cols = [c for c in df.columns if col in c]
                if matching_cols:
                    new_df[col] = df[matching_cols[0]]
            df = new_df
        else:
            # If still missing critical columns, return the original dataframe
            return df
    
    # Price-based indicators
    # Daily returns
    df["Return"] = df["Close"].pct_change()
    
    # Price momentum (different timeframes)
    df["Momentum_5d"] = df["Close"].pct_change(periods=5)
    df["Momentum_10d"] = df["Close"].pct_change(periods=10)
    df["Momentum_20d"] = df["Close"].pct_change(periods=20)
    
    # Moving Averages
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    
    # Moving Average Convergence Divergence (MACD)
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_StdDev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (df["BB_StdDev"] * 2)
    df["BB_Lower"] = df["BB_Middle"] - (df["BB_StdDev"] * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    
    # Volatility indicators
    df["Volatility_5d"] = df["Return"].rolling(window=5).std()
    df["Volatility_10d"] = df["Return"].rolling(window=10).std()
    df["Volatility_20d"] = df["Return"].rolling(window=20).std()
    
    # Volume indicators
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["Relative_Volume"] = df["Volume"] / df["Volume_MA_20"]
    
    # On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    
    # Advanced Price Patterns
    # Calculate True Range
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR_14"] = df["TR"].rolling(window=14).mean()
    
    # Commodity Channel Index (CCI)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["TP_MA_20"] = tp.rolling(window=20).mean()
    mean_deviation = abs(tp - df["TP_MA_20"]).rolling(window=20).mean()
    df["CCI"] = (tp - df["TP_MA_20"]) / (0.015 * mean_deviation)
    
    # Percentage Price Oscillator (PPO)
    df["PPO"] = ((df["EMA_12"] - df["EMA_26"]) / df["EMA_26"]) * 100
    
    # Clean up intermediate columns
    df = df.drop(columns=["H-L", "H-PC", "L-PC", "TR"], errors='ignore')
    
    return df

# Load data function with error handling and technical indicators
@st.cache_data
def load_price_data(ticker, start, end):
    try:
        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start, end=end)
        
        if df.empty:
            st.error(f"No data found for ticker {ticker}. Please check the symbol and try again.")
            return None
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            # Find the key columns
            close_col = [col for col in df.columns if 'Close' in col][0]
            volume_col = [col for col in df.columns if 'Volume' in col][0]
            
            # Create a simplified dataframe with single-level columns for key columns
            simple_df = pd.DataFrame(index=df.index)
            for original_col, new_col in [
                (close_col, "Close"),
                (volume_col, "Volume"),
                ([col for col in df.columns if 'Open' in col][0], "Open"),
                ([col for col in df.columns if 'High' in col][0], "High"),
                ([col for col in df.columns if 'Low' in col][0], "Low"),
            ]:
                simple_df[new_col] = df[original_col]
            
            # Replace the original dataframe with the simplified one
            df = simple_df
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Debug output
        st.write("Data shape:", df.shape)
        st.write("Features calculated:", len(df.columns))
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)  # Print the full traceback for debugging
        return None

# Backtest class for pattern analysis
class PatternBacktester:
    """Class for backtesting trading strategies based on pattern recognition"""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the backtester with a dataframe of price data"""
        self.data = data.copy()
        self.results = None
        
    def backtest_pattern_strategy(self, pattern_function, entry_days=1, exit_days=5, stop_loss_pct=0.02):
        """Backtest a trading strategy based on pattern recognition
        
        Args:
            pattern_function: Function that returns True/False for a pattern at each row
            entry_days: Number of days to enter position after pattern
            exit_days: Number of days to exit position after entry
            stop_loss_pct: Stop loss percentage
            
        Returns:
            DataFrame with backtest results
        """
        # Create a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Apply the pattern detection function
        df['pattern'] = df.apply(pattern_function, axis=1)
        
        # Initialize trading signals
        df['signal'] = 0
        
        # Create entry signals
        df.loc[df['pattern'], 'signal'] = 1
        
        # Shift the signals to implement the entry delay
        df['entry_signal'] = df['signal'].shift(entry_days).fillna(0)
        
        # Track positions and returns
        positions = []
        entry_prices = []
        entry_dates = []
        exit_prices = []
        exit_dates = []
        exit_reasons = []
        returns = []
        
        # Simulate trades
        in_position = False
        entry_price = 0
        entry_date = None
        exit_date = None
        days_in_position = 0
        
        # Loop through the dataframe
        for i, (date, row) in enumerate(df.iterrows()):
            if in_position:
                days_in_position += 1
                current_return = (row['Close'] / entry_price) - 1
                
                # Check exit conditions
                if days_in_position >= exit_days:
                    # Exit due to time limit
                    in_position = False
                    exit_price = row['Close']
                    exit_date = date
                    trade_return = (exit_price / entry_price) - 1
                    
                    positions.append(True)
                    entry_prices.append(entry_price)
                    entry_dates.append(entry_date)
                    exit_prices.append(exit_price)
                    exit_dates.append(exit_date)
                    exit_reasons.append('Time exit')
                    returns.append(trade_return)
                    
                    # Reset position tracking
                    days_in_position = 0
                
                elif current_return <= -stop_loss_pct:
                    # Exit due to stop loss
                    in_position = False
                    exit_price = row['Close']
                    exit_date = date
                    trade_return = (exit_price / entry_price) - 1
                    
                    positions.append(True)
                    entry_prices.append(entry_price)
                    entry_dates.append(entry_date)
                    exit_prices.append(exit_price)
                    exit_dates.append(exit_date)
                    exit_reasons.append('Stop loss')
                    returns.append(trade_return)
                    
                    # Reset position tracking
                    days_in_position = 0
            
            elif row['entry_signal'] == 1:
                # Enter a new position
                in_position = True
                entry_price = row['Close']
                entry_date = date
                days_in_position = 0
        
        # Create results dataframe
        if positions:
            self.results = pd.DataFrame({
                'Entry Date': entry_dates,
                'Entry Price': entry_prices,
                'Exit Date': exit_dates,
                'Exit Price': exit_prices,
                'Exit Reason': exit_reasons,
                'Return': returns
            })
            self.results['Cum Return'] = (1 + self.results['Return']).cumprod() - 1
            
            return self.results
        else:
            return pd.DataFrame()
    
    def get_performance_stats(self):
        """Calculate performance statistics from backtest results"""
        if self.results is None or len(self.results) == 0:
            return {}
        
        stats = {
            'Total Trades': len(self.results),
            'Win Rate': len(self.results[self.results['Return'] > 0]) / len(self.results),
            'Average Return': self.results['Return'].mean(),
            'Cumulative Return': self.results['Cum Return'].iloc[-1],
            'Max Return': self.results['Return'].max(),
            'Min Return': self.results['Return'].min(),
            'Avg Winning Trade': self.results.loc[self.results['Return'] > 0, 'Return'].mean() if len(self.results[self.results['Return'] > 0]) > 0 else 0,
            'Avg Losing Trade': self.results.loc[self.results['Return'] < 0, 'Return'].mean() if len(self.results[self.results['Return'] < 0]) > 0 else 0,
        }
        
        return stats

# Vector Database Class for similarity search
class VectorDatabase:
    def __init__(self, dimension: int = 2):
        """Initialize a vector database using FAISS.
        
        Args:
            dimension: Dimensionality of the vectors to be stored
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.metadata = []  # Store metadata alongside vectors
        self.pattern_groups = {}  # Store pattern groups by similarity
        
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add vectors and their metadata to the database.
        
        Args:
            vectors: NumPy array of shape (n, dimension) with vectors to add
            metadata_list: List of metadata dictionaries for each vector
        """
        # Ensure vectors are in float32 format required by FAISS
        vectors = vectors.astype(np.float32)
        
        # Add vectors to the index
        self.index.add(vectors)
        
        # Store metadata
        self.metadata.extend(metadata_list)
        
        # Group similar patterns using K-means clustering
        if len(vectors) > 5:  # Only cluster if we have enough vectors
            # Determine a reasonable number of clusters - ~sqrt(n) is a rule of thumb
            n_groups = min(int(np.sqrt(len(vectors))), 20)  # Cap at 20 groups max
            n_groups = max(n_groups, 3)  # At least 3 groups
            
            # Create clusters of similar patterns
            kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Store pattern groups
            self.pattern_groups = {
                'labels': cluster_labels.tolist(),
                'centroids': kmeans.cluster_centers_.tolist(),
                'n_groups': n_groups
            }
            
            # Add cluster labels to metadata
            for i, label in enumerate(cluster_labels):
                if i < len(self.metadata):
                    self.metadata[i]['pattern_group'] = int(label)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for the k nearest vectors to the query vector.
        
        Args:
            query_vector: Query vector of shape (1, dimension)
            k: Number of nearest neighbors to return
            
        Returns:
            List of metadata dictionaries for the nearest neighbors
        """
        # Ensure query vector is in float32 format
        query_vector = query_vector.astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_vector, k)
        
        # Return metadata for the nearest neighbors
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):  # Check if index is valid
                result = self.metadata[idx].copy()
                distance = distances[0][i]
                result["distance"] = float(distance)
                
                # Add similarity score (inverse of distance, normalized to 0-100%)
                max_distance = 10.0  # Set a reasonable maximum distance 
                similarity = max(0, min(100, (1 - distance / max_distance) * 100))
                result["similarity"] = float(similarity)
                
                results.append(result)
        
        return results
    
    def get_pattern_group(self, group_id: int) -> List[Dict[str, Any]]:
        """Get all patterns belonging to a specific pattern group.
        
        Args:
            group_id: ID of the pattern group to retrieve
            
        Returns:
            List of metadata dictionaries for the pattern group
        """
        if not self.pattern_groups or 'labels' not in self.pattern_groups:
            return []
            
        if group_id < 0 or group_id >= self.pattern_groups.get('n_groups', 0):
            return []
            
        results = []
        for i, metadata in enumerate(self.metadata):
            if metadata.get('pattern_group') == group_id:
                results.append(metadata)
                
        return results
    
    def get_representative_patterns(self) -> List[Dict[str, Any]]:
        """Get representative patterns from each group (closest to centroids).
        
        Returns:
            List of metadata dictionaries for representative patterns
        """
        if not self.pattern_groups or 'centroids' not in self.pattern_groups:
            return []
            
        centroids = np.array(self.pattern_groups['centroids'], dtype=np.float32)
        n_groups = self.pattern_groups.get('n_groups', 0)
        
        representatives = []
        for i in range(n_groups):
            centroid = centroids[i].reshape(1, -1)
            
            # Find the closest pattern to this centroid
            distances, indices = self.index.search(centroid, 1)
            if len(indices[0]) > 0:
                idx = indices[0][0]
                representative = self.metadata[idx].copy()
                representative['group_id'] = i
                representatives.append(representative)
                
        return representatives
    
    def classify_pattern(self, query_vector: np.ndarray) -> Dict[str, Any]:
        """Classify a pattern into one of the existing pattern groups.
        
        Args:
            query_vector: Query vector of shape (1, dimension)
            
        Returns:
            Dictionary with classification results
        """
        if not self.pattern_groups or 'centroids' not in self.pattern_groups:
            return {"group_id": -1, "similarity": 0.0}
            
        # Ensure query vector is in float32 format
        query_vector = query_vector.astype(np.float32)
            
        # Reshape if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        centroids = np.array(self.pattern_groups['centroids'], dtype=np.float32)
        
        # Calculate distances to all centroids
        distances = np.zeros(len(centroids))
        for i, centroid in enumerate(centroids):
            distances[i] = np.linalg.norm(centroid - query_vector)
        
        # Find the closest centroid
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Calculate similarity score
        max_distance = 10.0  # Set a reasonable maximum distance
        similarity = max(0, min(100, (1 - closest_distance / max_distance) * 100))
        
        return {
            "group_id": int(closest_idx),
            "similarity": float(similarity),
            "distance": float(closest_distance)
        }
    
    def save(self, path: str):
        """Save the vector database to disk.
        
        Args:
            path: Directory path to save the database
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, f"{path}_index")
        
        # Save the metadata and pattern groups
        data_to_save = {
            'metadata': self.metadata,
            'pattern_groups': self.pattern_groups,
            'dimension': self.dimension
        }
        
        with open(f"{path}_data.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
    
    @classmethod
    def load(cls, path: str) -> 'VectorDatabase':
        """Load a vector database from disk.
        
        Args:
            path: Directory path to load the database from
            
        Returns:
            Loaded VectorDatabase instance
        """
        # Load the FAISS index
        index_path = f"{path}_index"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
            
        index = faiss.read_index(index_path)
        
        # Load data
        data_path = f"{path}_data.pkl"
        metadata_path = f"{path}_metadata.pkl"  # For backward compatibility
        info_path = os.path.join(os.path.dirname(path), "info.pkl")  # For backward compatibility
        
        # Try loading using new format first
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                metadata = data.get('metadata', [])
                pattern_groups = data.get('pattern_groups', {})
                dimension = data.get('dimension', index.d)
                
        # Fall back to old format for backward compatibility
        elif os.path.exists(metadata_path) and os.path.exists(info_path):
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                dimension = info.get("dimension", index.d)
                
            pattern_groups = {}
        
        # Fall back to default values if everything fails
        else:
            metadata = []
            pattern_groups = {}
            dimension = index.d
            
        # Create a new instance
        db = cls(dimension=dimension)
        db.index = index
        db.metadata = metadata
        db.pattern_groups = pattern_groups
        
        return db


# Main content
with st.spinner("Loading stock data..."):
    df = load_price_data(ticker, start_date, end_date)

if df is not None:
    st.write(f"Loaded {len(df)} days of data for **{ticker}**")
    
    # Display recent price chart
    st.subheader("Recent Price History")
    
    # Debug print the dataframe columns
    st.write("Data columns:", df.columns)
    
    # Look for the Close column - it might be a multi-level column now
    if isinstance(df.columns, pd.MultiIndex):
        # If multi-index columns, find the Close column
        close_col = [col for col in df.columns if 'Close' in col]
        if close_col:
            close_col = close_col[0]  # Take the first Close column
            price_data = df[close_col][-100:]
        else:
            st.error("Could not find Close price column in the data")
            price_data = pd.Series(index=df[-100:].index)
    else:
        # Regular columns
        price_data = df["Close"][-100:]
    
    # Create a dataframe specifically for plotting
    plot_df = pd.DataFrame({'Date': price_data.index, 'Price': price_data.values})
    
    price_fig = px.line(
        plot_df,
        x='Date',
        y='Price',
        title=f"{ticker} Stock Price"
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # Feature Engineering
    with st.spinner("Analyzing price-volume behavior..."):
        # Map the selected features to the actual dataframe columns if needed
        feature_mapping = {
            "Return": "Return", 
            "Volume": "Volume", 
            "Volatility_20d": "Volatility_20d", 
            "Momentum_20d": "Momentum_20d",
            "RSI": "RSI",
            "MACD": "MACD",
            "BB_Width": "BB_Width",
            "OBV": "OBV",
            "ATR_14": "ATR_14",
            "Relative_Volume": "Relative_Volume"
        }
        
        # Get actual feature column names based on selection
        actual_features = [feature_mapping[f] for f in selected_features if f in feature_mapping]
        
        # Ensure we have at least 2 features
        if len(actual_features) < 2:
            st.warning("Not enough valid features selected. Using Return and Volume as defaults.")
            actual_features = ["Return", "Volume"]
        
        # Extract selected features
        X = df[actual_features].copy()
        
        # Apply log transformation to volume if selected
        if logarithmic_scale and "Volume" in X.columns:
            X["Volume"] = np.log1p(X["Volume"])
        elif "Volume" in X.columns:
            # Otherwise, scale volume to millions for better visibility
            X["Volume"] = X["Volume"] / 1e6
        
        # Scale features
        if normalize_features:
            scaler = StandardScaler()
        else:
            # Just center the data but don't scale to unit variance
            scaler = StandardScaler(with_std=False)
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering based on selected algorithm
        if clustering_algorithm == "K-Means":
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            df["Cluster"] = clusters
            centroids = kmeans.cluster_centers_
        else:
            # DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            
            # DBSCAN can assign -1 for noise points that don't belong to any cluster
            # Remap cluster IDs to ensure they're all non-negative
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters[unique_clusters >= 0])  # Count non-noise clusters
            
            # Create a mapping from original cluster IDs to sequential IDs
            id_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
            if -1 in id_map:  # Map noise points to the last cluster ID
                id_map[-1] = n_clusters
                n_clusters += 1  # Add one more for noise
                
            # Apply the mapping
            df["Cluster"] = np.array([id_map[c] for c in clusters])
            
            # Calculate "centroids" as the mean of each cluster
            centroids = np.zeros((n_clusters, X_scaled.shape[1]))
            for i in range(n_clusters):
                if i < len(unique_clusters):
                    cluster_points = X_scaled[clusters == unique_clusters[i]]
                    if len(cluster_points) > 0:
                        centroids[i] = np.mean(cluster_points, axis=0)
                        
        # Save the features used for later reference
        features = actual_features
        
        # Centroids already calculated earlier, no need to reassign
        # Use the centroids for label assignment
        
        # Create interpretable labels with 8 distinct categories: 
        # (High Return, Low Return) x (High Volume, Low Volume) x (Positive, Negative)
        labels = []
        for i, centroid in enumerate(centroids):
            return_val, volume_val = centroid[0], centroid[1]
            
            # Determine return direction (positive or negative)
            return_direction = "Positive" if return_val > 0 else "Negative"
            
            # Determine return magnitude (high or low) - based on absolute value
            return_magnitude = abs(return_val)
            
            # Calculate median of absolute return values for thresholding
            return_median = np.median([abs(c[0]) for c in centroids])
            return_level = "High" if return_magnitude > return_median else "Low"
            
            # Determine volume level (high or low)
            volume_median = np.median([c[1] for c in centroids])
            volume_level = "High" if volume_val > volume_median else "Low"
            
            # Combine the three dimensions into a single label
            cluster_label = f"{return_level} Return ({return_direction}) / {volume_level} Volume"
                
            labels.append(f"Cluster {i+1}: {cluster_label}")
        
        # Map cluster numbers to labels
        label_map = {i: labels[i] for i in range(n_clusters)}
        df["Behavior"] = df["Cluster"].map(label_map)
        
        # 3D Embedding with PCA (ensure n_components doesn't exceed min dimensions)
        n_components_3d = min(3, X_scaled.shape[1])
        pca = PCA(n_components=n_components_3d, random_state=42)
        embedding = pca.fit_transform(X_scaled)
        
        # If we have fewer than 3 components, pad with zeros
        if n_components_3d < 3:
            padded_embedding = np.zeros((embedding.shape[0], 3))
            padded_embedding[:, :n_components_3d] = embedding
            embedding = padded_embedding
            
        df[["x", "y", "z"]] = embedding
        
        # Get the explained variance ratio for the components
        explained_variance = pca.explained_variance_ratio_
        
        # If we have fewer than 3 components, pad with zeros for the plot
        if len(explained_variance) < 3:
            explained_variance = np.pad(explained_variance, (0, 3 - len(explained_variance)), 'constant')

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "3D Visualization", 
        "Cluster Analysis", 
        "Data Table", 
        "Vector Database",
        "Technical Dashboard",
        "Predictive Analytics",
        "Pattern Backtesting",
        "Pattern Groups"
    ])
    
    with tab1:
        # 3D Plot
        st.subheader("🧭 3D Price-Volume Embedding")
        fig = go.Figure()
        
        # Create a color scale based on the number of clusters
        colors = px.colors.qualitative.Plotly[:n_clusters] 
        
        # Add scatter trace
        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode='markers',
            marker=dict(
                size=4,
                color=df["Cluster"],
                colorscale=colors,
                opacity=0.8,
                colorbar=dict(
                    title="Cluster",
                    tickvals=list(range(n_clusters)),
                    ticktext=[f"Cluster {i+1}" for i in range(n_clusters)]
                )
            ),
            text=df.apply(
                lambda row: (
                    f"Date: {row.name.date()}<br>"
                    f"Return: {row['Return']:.2%}<br>"
                    f"Volume: {row['Volume']:.2f}M<br>"
                    + (f"RSI: {row['RSI']:.1f}<br>" if 'RSI' in row and not pd.isna(row['RSI']) else "")
                    + f"Type: {row['Behavior']}"
                ), 
                axis=1
            ),
            hoverinfo="text"
        ))
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(
                xaxis_title=f"PCA-1 ({explained_variance[0]:.1%})",
                yaxis_title=f"PCA-2 ({explained_variance[1]:.1%})",
                zaxis_title=f"PCA-3 ({explained_variance[2]:.1%})"
            ),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        This 3D visualization shows relationships between trading days based on their price-volume characteristics.
        Points closer together represent days with similar behavior patterns.
        Hover over points to see details about that specific trading day.
        """)
    
    with tab2:
        # Cluster analysis
        st.subheader("📊 Cluster Summary")
        
        # Calculate cluster statistics
        summary = df.groupby("Behavior")["Return"].agg(
            ["count", "mean", "std", "min", "max"]
        ).rename(
            columns={
                "count": "Days", 
                "mean": "Avg Return", 
                "std": "Volatility",
                "min": "Min Return",
                "max": "Max Return"
            }
        )
        
        # Display stats with better formatting
        formatted_summary = summary.copy()
        for col in ["Avg Return", "Min Return", "Max Return"]:
            formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{x:.2%}")
        formatted_summary["Volatility"] = formatted_summary["Volatility"].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(formatted_summary, use_container_width=True)
        
        # Plot cluster distribution
        st.subheader("Cluster Distribution")
        cluster_counts = df["Behavior"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        
        fig = px.pie(
            cluster_counts, 
            values="Count", 
            names="Cluster", 
            title="Distribution of Trading Days by Cluster",
            color_discrete_sequence=px.colors.qualitative.Plotly[:n_clusters]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster centers visualization
        st.subheader("Cluster Centers")
        
        # Create a dataframe of cluster centers
        centers_df = pd.DataFrame(
            scaler.inverse_transform(centroids),
            columns=features
        )
        centers_df["Cluster"] = [f"Cluster {i+1}" for i in range(n_clusters)]
        centers_df = centers_df.set_index("Cluster")
        
        # Format the values for display (with checks for column existence)
        if "Return" in centers_df.columns:
            centers_df["Return"] = centers_df["Return"].apply(lambda x: f"{x:.2%}")
        
        # Handle different column names for Momentum
        for col in ["Momentum_20d", "Momentum_10d", "Momentum_5d", "Momentum"]:
            if col in centers_df.columns:
                centers_df[col] = centers_df[col].apply(lambda x: f"{x:.2%}")
                
        # Handle different column names for Volatility
        for col in ["Volatility_20d", "Volatility_10d", "Volatility_5d", "Volatility"]:
            if col in centers_df.columns:
                centers_df[col] = centers_df[col].apply(lambda x: f"{x:.4f}")
                
        if "Volume" in centers_df.columns:
            centers_df["Volume"] = centers_df["Volume"].apply(lambda x: f"{x:.2f}M")
        
        st.dataframe(centers_df, use_container_width=True)
    
    with tab3:
        # Raw data table
        st.subheader("Recent Data Sample")
        
        # Create a more readable dataframe for display
        display_df = df.copy()
        display_df["Return"] = display_df["Return"].apply(lambda x: f"{x:.2%}")
        
        # Select relevant columns that exist in the dataframe
        cols_to_display = ["Close", "Return", "Volume", "Behavior"]
        
        # Add optional technical indicators if available
        for col in ["Momentum_20d", "Volatility_20d", "RSI"]:
            if col in display_df.columns:
                if col.startswith("Momentum"):
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                elif col.startswith("Volatility"):
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                cols_to_display.append(col)
                
        display_df = display_df[cols_to_display]
        
        st.dataframe(display_df.tail(100), use_container_width=True)
        
        # Download option
        csv = df.to_csv()
        st.download_button(
            label="Download Complete Dataset as CSV",
            data=csv,
            file_name=f"{ticker}_analysis.csv",
            mime="text/csv",
        )
    
    with tab4:
        # Enhanced Vector Database implementation with simplified styling
        st.header("AI-Powered Similarity Search")
        
        st.markdown("""
        Discover market patterns that repeat through time with vector embedding technology. 
        Select any trading day to find historical moments with similar technical characteristics.
        
        > This feature uses FAISS (Facebook AI Similarity Search) to create embeddings of market behavior
        > and find matches based on technical patterns rather than time periods.
        """)
        
        # Vector Database Directory setup
        db_dir = "vector_db"
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, f"{ticker}_vector_db")
        
        # Create an enhanced vector database for the current ticker
        with st.spinner("Building advanced vector database..."):
            # Create 30-day rolling patterns of returns and volume
            st.write("Computing 30-day rolling return-volume patterns for vector embeddings...")
            
            # Create rolling window features for return patterns
            rolling_returns = []
            rolling_volumes = []
            
            # Ensure we have at least 30 days of data
            if len(df) >= 30:
                # Compute rolling 30-day return and volume patterns for each day
                for i in range(30, len(df)):
                    # Get 30-day return pattern (normalized to start at 1.0)
                    returns_window = df['Close'].iloc[i-30:i] / df['Close'].iloc[i-30]
                    returns_window = returns_window.values / returns_window.values[0]  # Normalize to start at 1.0
                    
                    # Get 30-day volume pattern (normalized to average 1.0)
                    volume_window = df['Volume'].iloc[i-30:i].values
                    if np.mean(volume_window) > 0:
                        volume_window = volume_window / np.mean(volume_window)  # Normalize to mean 1.0
                    
                    rolling_returns.append(returns_window)
                    rolling_volumes.append(volume_window)
                
                # Create a dataframe with the windows
                embedding_data = pd.DataFrame(index=df.index[30:])
                embedding_data['Returns_Pattern'] = rolling_returns
                embedding_data['Volume_Pattern'] = rolling_volumes
                
                # Add key indicators at the end of each window
                embedding_data['Final_Return'] = df['Return'].iloc[30:].values  # 1-day return at end of window
                
                if 'Volatility_20d' in df.columns:
                    embedding_data['Volatility'] = df['Volatility_20d'].iloc[30:].values
                
                if 'RSI' in df.columns:
                    embedding_data['RSI'] = df['RSI'].iloc[30:].values
                
                # Extract features from the patterns
                # For returns: extract features like trend strength, volatility, max drawdown
                trend_strength = []  # Linear regression slope
                volatility = []  # Standard deviation of daily returns
                max_drawdown = []  # Maximum drawdown in window
                
                for returns in rolling_returns:
                    # Calculate daily returns from the cumulative returns
                    daily_rets = np.diff(returns) / returns[:-1]
                    
                    # Trend strength (regression slope)
                    days = np.arange(len(returns))
                    slope, _, _, _, _ = stats.linregress(days, returns)
                    trend_strength.append(slope)
                    
                    # Volatility
                    vol = np.std(daily_rets)
                    volatility.append(vol)
                    
                    # Max drawdown
                    cum_max = np.maximum.accumulate(returns)
                    drawdown = (returns / cum_max) - 1
                    max_dd = drawdown.min()
                    max_drawdown.append(max_dd)
                
                # For volume: extract features like trend, spikes, consistency
                volume_trend = []  # Direction of volume
                volume_spikes = []  # Count of volume spikes (>2x avg)
                volume_consistency = []  # Coefficient of variation
                
                for volumes in rolling_volumes:
                    # Volume trend
                    days = np.arange(len(volumes))
                    slope, _, _, _, _ = stats.linregress(days, volumes)
                    volume_trend.append(slope)
                    
                    # Volume spikes
                    spikes = np.sum(volumes > 2.0)  # Count days with >2x average volume
                    volume_spikes.append(spikes)
                    
                    # Volume consistency
                    cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
                    volume_consistency.append(cv)
                
                # Add extracted features to the dataframe
                embedding_data['Trend_Strength'] = trend_strength
                embedding_data['Return_Volatility'] = volatility
                embedding_data['Max_Drawdown'] = max_drawdown
                embedding_data['Volume_Trend'] = volume_trend
                embedding_data['Volume_Spikes'] = volume_spikes
                embedding_data['Volume_Consistency'] = volume_consistency
                
                # Select features for embedding
                embedding_features = [
                    'Trend_Strength', 'Return_Volatility', 'Max_Drawdown',
                    'Volume_Trend', 'Volume_Spikes', 'Volume_Consistency',
                    'Final_Return'
                ]
                
                # Add optional features if available
                if 'Volatility' in embedding_data.columns:
                    embedding_features.append('Volatility')
                
                if 'RSI' in embedding_data.columns:
                    embedding_features.append('RSI')
                
                # Create numerical feature matrix for embedding
                X_features = embedding_data[embedding_features].copy()
                
                # Handle any NaN values
                X_features = X_features.fillna(0)
                
                # Show feature stats
                st.write(f"Created {len(X_features)} rolling window patterns with {len(embedding_features)} features")
                
                # Display some statistics
                st.write("Feature statistics:")
                st.write(X_features.describe().T)
            else:
                st.warning(f"Not enough data for 30-day rolling windows. Need at least 30 days, but only have {len(df)}.")
                # Create a dummy embedding data with original features as fallback
                embedding_features = ["Return", "Volume"]
                if "Volatility_20d" in df.columns:
                    embedding_features.append("Volatility_20d")
                if "Momentum_20d" in df.columns:
                    embedding_features.append("Momentum_20d")
                
                X_features = df[embedding_features].copy()
                embedding_data = X_features.copy()
                embedding_data['Date'] = df.index
            
            # Use PCA to create more meaningful embeddings - reduce dimensions
            embedding_scaler = StandardScaler()
            X_features_scaled = embedding_scaler.fit_transform(X_features)
            
            # Apply PCA for dimensionality reduction (min of 5D or number of features)
            n_components = min(5, X_features_scaled.shape[1])
            embedding_pca = PCA(n_components=n_components, random_state=42)
            vectors = embedding_pca.fit_transform(X_features_scaled)
            
            # Display embedding information
            st.write("Embedding dimensions:", vectors.shape[1])
            st.write("Features used:", embedding_features)
            st.write("Explained variance by embedding components:", 
                     [f"{var:.1%}" for var in embedding_pca.explained_variance_ratio_])
            
            # For text representation, create descriptive strings for each day
            text_descriptions = []
            for idx, row in df.iterrows():
                # Create a descriptive text of the trading day
                description = (
                    f"Trading day for {ticker} on {idx.strftime('%Y-%m-%d')}. "
                    f"Return: {row['Return']:.4f}. "
                    f"Volume: {row['Volume']:.2f}. "
                )
                
                # Add additional metrics if they exist
                if "Volatility_20d" in row:
                    description += f"Volatility: {row['Volatility_20d']:.6f}. "
                elif "Volatility_10d" in row:
                    description += f"Volatility: {row['Volatility_10d']:.6f}. "
                    
                if "Momentum_20d" in row:
                    description += f"Momentum: {row['Momentum_20d']:.4f}. "
                elif "Momentum_10d" in row:
                    description += f"Momentum: {row['Momentum_10d']:.4f}. "
                    
                description += f"Behavior: {row['Behavior']}."
                text_descriptions.append(description)
            
            # Create metadata for each trading day
            metadata_list = []
            for i, (idx, row) in enumerate(df.iterrows()):
                # Start with the basic metadata that's always available
                metadata = {
                    "date": idx.strftime('%Y-%m-%d'),
                    "close": float(row["Close"]),
                    "return": float(row["Return"]),
                    "volume": float(row["Volume"]),
                    "behavior": row["Behavior"],
                    "description": text_descriptions[i]
                }
                
                # Add additional metrics if they exist
                if "Volatility_20d" in row:
                    metadata["volatility"] = float(row["Volatility_20d"])
                elif "Volatility_10d" in row:
                    metadata["volatility"] = float(row["Volatility_10d"])
                else:
                    metadata["volatility"] = 0.0
                    
                if "Momentum_20d" in row:
                    metadata["momentum"] = float(row["Momentum_20d"])
                elif "Momentum_10d" in row:
                    metadata["momentum"] = float(row["Momentum_10d"])
                else:
                    metadata["momentum"] = 0.0
                    
                metadata_list.append(metadata)
            
            # Create the vector database with 8 dimensions
            vector_db = VectorDatabase(dimension=vectors.shape[1])
            vector_db.add_vectors(vectors, metadata_list)
            
            # Save the database
            vector_db.save(db_file)
            
            # Save additional metadata for future search functionality
            extra_data = {
                "text_descriptions": text_descriptions,
                "feature_names": embedding_features,
                "pca_components": embedding_pca.components_.tolist(),
                "pca_mean": embedding_pca.mean_.tolist(),
                "scaler_mean": embedding_scaler.mean_.tolist(),
                "scaler_scale": embedding_scaler.scale_.tolist()
            }
            
            with open(os.path.join(db_dir, f"{ticker}_embedding_data.json"), "w") as f:
                json.dump(extra_data, f, indent=2)
            
            st.success(f"Successfully created enhanced vector database for {ticker} with {len(metadata_list)} trading days using {vectors.shape[1]}-dimensional embeddings")
        
        # Simple header for the time machine feature
        st.subheader("Time Machine Pattern Finder")
        
        # Simple instruction text
        st.markdown("Select any specific trading day to find historical periods with matching technical patterns")
            
        # Available dates in the dataset
        available_dates = df.index.strftime('%Y-%m-%d').tolist()
        
        if available_dates:
            # Date selector
            selected_date = st.selectbox(
                "Select a reference date",
                options=available_dates,
                index=min(len(available_dates)-1, 252)  # Default to ~1 year ago if available
            )
            
            # Number of similar days to find
            num_results = st.slider("Number of similar days to find", 
                                  min_value=1, max_value=20, value=5)
            
            # Explanation
            st.info("""
            This search finds trading days that have similar technical characteristics to the selected date.
            The similarity is calculated based on returns, volume, volatility, momentum, and other available indicators.
            """)
            
            # Search button
            if st.button("Find Similar Trading Days"):
                try:
                    # Get the selected date's data from the DataFrame
                    selected_date_data = df[df.index.strftime('%Y-%m-%d') == selected_date]
                    
                    if selected_date_data.empty:
                        st.error(f"No data found for selected date: {selected_date}")
                    else:
                        # Load embedding transformation data
                        embedding_data_file = os.path.join(db_dir, f"{ticker}_embedding_data.json")
                        
                        if not os.path.exists(embedding_data_file):
                            st.error("Embedding data not found. Please reload the page and try again.")
                        else:
                            with open(embedding_data_file, "r") as f:
                                embedding_info = json.load(f)
                                
                            # Get expected features
                            expected_features = embedding_info.get("feature_names", ["Return", "Volume"])
                            
                            # Create a feature vector with the selected date's data
                            feature_vector = np.zeros((1, len(expected_features)), dtype=np.float32)
                            
                            # Fill in with actual values from the selected date
                            for i, feature in enumerate(expected_features):
                                if feature in selected_date_data.columns:
                                    feature_vector[0, i] = selected_date_data[feature].values[0]
                                elif feature == "Return" and "Return" in selected_date_data.columns:
                                    feature_vector[0, i] = selected_date_data["Return"].values[0]
                                elif feature == "Volume" and "Volume" in selected_date_data.columns:
                                    # Normalize volume if needed
                                    max_volume = df["Volume"].max()
                                    feature_vector[0, i] = selected_date_data["Volume"].values[0] / max_volume if max_volume > 0 else 0
                            
                            # Transform feature vector to match database embeddings
                            # Apply scaler
                            scaler_mean = np.array(embedding_info["scaler_mean"])
                            scaler_scale = np.array(embedding_info["scaler_scale"])
                            feature_vector_scaled = (feature_vector - scaler_mean) / scaler_scale
                            
                            # Apply PCA transformation
                            pca_components = np.array(embedding_info["pca_components"])
                            pca_mean = np.array(embedding_info["pca_mean"])
                            query_vector = np.dot(feature_vector_scaled - pca_mean, pca_components.T)
                        
                            # Load the database
                            vector_db = VectorDatabase.load(db_file)
                            
                            # Search for similar days
                            results = vector_db.search(query_vector.astype(np.float32), k=num_results)
                            
                            if not results:
                                st.error("No similar trading days found.")
                            else:
                                # Convert to DataFrame for display
                                results_df = pd.DataFrame(results)
                                
                                # Format results
                                results_df["return"] = results_df["return"].apply(lambda x: f"{x:.2%}")
                                results_df["momentum"] = results_df["momentum"].apply(lambda x: f"{x:.2%}")
                                results_df["distance"] = results_df["distance"].apply(lambda x: f"{x:.4f}")
                                
                                # Rename columns
                                results_df = results_df.rename(columns={
                                    "date": "Date",
                                    "close": "Close Price",
                                    "return": "Return",
                                    "volume": "Volume (M)",
                                    "volatility": "Volatility",
                                    "momentum": "Momentum",
                                    "behavior": "Behavior",
                                    "distance": "Distance"
                                })
                                
                                # Remove description column from display (it's too long)
                                if "description" in results_df.columns:
                                    results_df = results_df.drop(columns=["description"])
                                
                                # Display results
                                st.dataframe(results_df)
                                
                                # Plot a scatter plot for the first two PCA components with results highlighted
                                st.subheader("PCA Projection of Stock Behavior")
                                
                                # Extract first 2 principal components from all data
                                projection_data = pd.DataFrame(
                                    embedding_pca.transform(X_features_scaled)[:, :2],
                                    columns=["PC1", "PC2"]
                                )
                                projection_data["Date"] = df.index
                                projection_data["Type"] = "All Days"
                                
                                # Mark the similar days
                                for result in results:
                                    result_date = result["date"]
                                    match_idx = projection_data["Date"].astype(str).str.contains(result_date)
                                    projection_data.loc[match_idx, "Type"] = "Similar Day"
                                
                                # Mark the query point (project it onto the PCA space)
                                query_projection = np.dot(feature_vector_scaled - pca_mean, pca_components.T)[:, :2]
                                query_df = pd.DataFrame({
                                    "PC1": [query_projection[0, 0]],
                                    "PC2": [query_projection[0, 1]],
                                    "Date": ["Query"],
                                    "Type": ["Query Point"]
                                })
                                
                                # Combine the data
                                combined_data = pd.concat([projection_data, query_df], ignore_index=True)
                                
                                # Create an enhanced scatter plot with more interactive features
                                scatter_fig = px.scatter(
                                    combined_data,
                                    x="PC1",
                                    y="PC2",
                                    color="Type",
                                    hover_name="Date",
                                    size_max=15,
                                    color_discrete_sequence=["#d3d3d3", "#ff4b4b", "#1f77b4"],
                                    title="PCA Projection of Trading Day Patterns",
                                    labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
                                    hover_data={
                                        "PC1": False,  # Hide PC1 in the hover tooltip
                                        "PC2": False,  # Hide PC2 in the hover tooltip
                                        "Type": True    # Show type in the hover tooltip
                                    },
                                )
                                
                                # Customize the layout
                                scatter_fig.update_layout(
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="center",
                                        x=0.5
                                    ),
                                    margin=dict(l=20, r=20, t=50, b=50),
                                    height=500,
                                    plot_bgcolor='rgba(240,240,240,0.8)',
                                )
                                
                                # Adjust point sizes and opacity for better visualization
                                scatter_fig.update_traces(
                                    marker=dict(size=6, opacity=0.5, line=dict(width=0)),
                                    selector=dict(mode='markers', name="All Days")
                                )
                                
                                scatter_fig.update_traces(
                                    marker=dict(size=12, opacity=0.9, line=dict(width=1, color='black')),
                                    selector=dict(mode='markers', name="Similar Day")
                                )
                                
                                scatter_fig.update_traces(
                                    marker=dict(size=18, opacity=1.0, symbol="star", 
                                              line=dict(width=2, color='black')),
                                    selector=dict(mode='markers', name="Query Point")
                                )
                                
                                st.plotly_chart(scatter_fig, use_container_width=True)
                                
                                # Add EDA section on the findings
                                st.subheader("Exploratory Data Analysis of Similar Days")
                                
                                # Calculate statistics on similar days
                                st.write("### Statistics of Similar Trading Days")
                                
                                # Create a dataframe with just the similar days for analysis
                                similar_days_df = pd.DataFrame()
                                for result in results:
                                    result_date = result["date"]
                                    day_data = df[df.index.strftime('%Y-%m-%d') == result_date]
                                    if not day_data.empty:
                                        similar_days_df = pd.concat([similar_days_df, day_data])
                                
                                if not similar_days_df.empty:
                                    # Calculate basic statistics
                                    stats = similar_days_df["Return"].describe()
                                    
                                    # Show statistics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Average Return", f"{stats['mean']:.2%}")
                                        st.metric("Maximum Return", f"{stats['max']:.2%}")
                                        st.metric("Minimum Return", f"{stats['min']:.2%}")
                                    
                                    with col2:
                                        # Additional metrics if available
                                        if "Volatility_20d" in similar_days_df.columns:
                                            vol_mean = similar_days_df["Volatility_20d"].mean()
                                            st.metric("Average Volatility", f"{vol_mean:.4f}")
                                        
                                        if "Momentum_20d" in similar_days_df.columns:
                                            mom_mean = similar_days_df["Momentum_20d"].mean()
                                            st.metric("Average Momentum", f"{mom_mean:.2%}")
                                    
                                    # Plot distribution of returns
                                    st.write("### Distribution of Returns in Similar Days")
                                    
                                    # Enhanced histogram of returns with better styling and annotations
                                    fig = px.histogram(
                                        similar_days_df,
                                        x="Return",
                                        nbins=10,
                                        title="Distribution of Returns in Similar Trading Days",
                                        labels={"Return": "Return"},
                                        color_discrete_sequence=["#2E86C1"],
                                        opacity=0.8,
                                        histnorm='percent', # Show as percentage of total
                                        text_auto=True      # Show count in each bin
                                    )
                                    
                                    # Add reference line for the selected date's return
                                    selected_return = selected_date_data["Return"].values[0]
                                    fig.add_vline(
                                        x=selected_return, 
                                        line_width=2, 
                                        line_dash="dash", 
                                        line_color="#FF5733",
                                        annotation_text=f"Selected Date Return: {selected_return:.2%}",
                                        annotation_position="top right"
                                    )
                                    
                                    # Customize layout
                                    fig.update_layout(
                                        xaxis_title="Return (%)",
                                        yaxis_title="Percentage of Similar Days",
                                        bargap=0.1,
                                        height=400,
                                        plot_bgcolor='rgba(240,240,240,0.8)',
                                        xaxis=dict(
                                            tickformat=".1%",  # Format x-axis as percentage
                                            showgrid=True,
                                            gridcolor='rgba(220,220,220,1)'
                                        ),
                                        yaxis=dict(
                                            showgrid=True,
                                            gridcolor='rgba(220,220,220,1)'
                                        )
                                    )
                                    
                                    # Format the tooltip to show percentages
                                    fig.update_traces(
                                        hovertemplate="Return: %{x:.2%}<br>Percentage: %{y:.1f}%<br>Count: %{text}"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add correlation heatmap for the similar days
                                    st.write("### Correlations Between Metrics in Similar Days")
                                    
                                    # Select numerical columns for correlation analysis
                                    corr_columns = ["Return", "Volume"]
                                    
                                    # Include technical indicators if available
                                    for col in similar_days_df.columns:
                                        if any(indicator in col for indicator in ["RSI", "Volatility", "Momentum", "MA", "MACD"]):
                                            corr_columns.append(col)
                                    
                                    # Get correlation matrix
                                    if len(corr_columns) >= 2 and len(similar_days_df) >= 2:
                                        corr_df = similar_days_df[corr_columns].copy()
                                        
                                        # Normalize volume for better correlation visualization
                                        if "Volume" in corr_df.columns:
                                            max_vol = corr_df["Volume"].max()
                                            if max_vol > 0:
                                                corr_df["Volume"] = corr_df["Volume"] / max_vol
                                        
                                        # Calculate correlation matrix
                                        corr_matrix = corr_df.corr()
                                        
                                        # Create heatmap
                                        heatmap_fig = px.imshow(
                                            corr_matrix,
                                            text_auto=True,
                                            aspect="auto",
                                            color_continuous_scale="RdBu_r",
                                            title="Correlation Matrix of Technical Indicators",
                                            labels=dict(x="Indicators", y="Indicators", color="Correlation")
                                        )
                                        
                                        # Adjust layout
                                        heatmap_fig.update_layout(
                                            height=400,
                                            margin=dict(l=10, r=10, t=50, b=50),
                                        )
                                        
                                        st.plotly_chart(heatmap_fig, use_container_width=True)
                                        
                                        # Add interpretation based on the correlations
                                        st.write("#### Interpretation of Correlations")
                                        
                                        # Example of automated interpretation
                                        if "Return" in corr_matrix.columns and "Volume" in corr_matrix.columns:
                                            return_vol_corr = corr_matrix.loc["Return", "Volume"]
                                            
                                            if abs(return_vol_corr) > 0.7:
                                                direction = "positive" if return_vol_corr > 0 else "negative"
                                                st.info(f"There is a strong {direction} correlation ({return_vol_corr:.2f}) between returns and volume in the similar trading days.")
                                            elif abs(return_vol_corr) > 0.3:
                                                direction = "positive" if return_vol_corr > 0 else "negative"
                                                st.info(f"There is a moderate {direction} correlation ({return_vol_corr:.2f}) between returns and volume in these similar trading days.")
                                            else:
                                                st.info(f"There is a weak correlation ({return_vol_corr:.2f}) between returns and volume in these similar trading days.")
                                    
                                    # Display a comparison between the selected date and the most similar day
                                    st.subheader("Comparative Analysis: Selected vs. Most Similar Day")
                                    most_similar = results[0]
                                    
                                    # Create comparison cards
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("### Selected Reference Date")
                                        st.markdown(f"**Date:** {selected_date}")
                                        
                                        # Get reference date values
                                        ref_close = selected_date_data["Close"].values[0]
                                        ref_return = selected_date_data["Return"].values[0]
                                        ref_volume = selected_date_data["Volume"].values[0]
                                        
                                        # Create a styled card
                                        st.markdown(
                                            f"""
                                            <div style="padding: 15px; border-radius: 5px; border: 1px solid #ddd; background-color: rgba(240, 242, 246, 0.8);">
                                                <p><strong>Close Price:</strong> ${ref_close:.2f}</p>
                                                <p><strong>Return:</strong> {ref_return:.2%}</p>
                                                <p><strong>Volume:</strong> {ref_volume/1e6:.2f}M</p>
                                                <p><strong>Pattern:</strong> {selected_date_data["Behavior"].values[0] if "Behavior" in selected_date_data.columns else "Unknown"}</p>
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )
                                    
                                    with col2:
                                        st.markdown("### Most Similar Trading Day")
                                        st.markdown(f"**Date:** {most_similar['date']}")
                                        
                                        # Calculate differences
                                        close_diff = ((most_similar['close'] - ref_close) / ref_close * 100)
                                        return_diff = (float(most_similar['return']) - ref_return) * 100  # already in decimal format
                                        volume_diff = ((most_similar['volume'] - ref_volume) / ref_volume * 100)
                                        
                                        # Get diff icons
                                        close_icon = "↑" if close_diff > 0 else "↓"
                                        return_icon = "↑" if return_diff > 0 else "↓"
                                        volume_icon = "↑" if volume_diff > 0 else "↓"
                                        
                                        # Create a styled card with differences highlighted
                                        st.markdown(
                                            f"""
                                            <div style="padding: 15px; border-radius: 5px; border: 1px solid #ddd; background-color: rgba(240, 242, 246, 0.8);">
                                                <p><strong>Close Price:</strong> ${most_similar['close']:.2f} <span style="color: {'green' if close_diff > 0 else 'red'}">({close_icon} {abs(close_diff):.1f}%)</span></p>
                                                <p><strong>Return:</strong> {float(most_similar['return']):.2%} <span style="color: {'green' if return_diff > 0 else 'red'}">({return_icon} {abs(return_diff):.1f}pp)</span></p>
                                                <p><strong>Volume:</strong> {float(most_similar['volume'])/1e6:.2f}M <span style="color: {'green' if volume_diff > 0 else 'red'}">({volume_icon} {abs(volume_diff):.1f}%)</span></p>
                                                <p><strong>Pattern:</strong> {most_similar['behavior']}</p>
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )
                                    
                                    # Show similarity metrics
                                    st.markdown("### Similarity Details")
                                    
                                    # Create metrics to show similarity score and distance
                                    similarity_score = 1.0 / (1.0 + most_similar['distance'])  # Convert distance to similarity (0-1)
                                    
                                    metrics_cols = st.columns(3)
                                    with metrics_cols[0]:
                                        st.metric(
                                            "Similarity Score", 
                                            f"{similarity_score:.2f}", 
                                            help="Higher is better. Normalized score from 0-1 where 1 means identical."
                                        )
                                    with metrics_cols[1]:
                                        st.metric(
                                            "Vector Distance", 
                                            f"{most_similar['distance']:.4f}",
                                            help="Lower is better. Euclidean distance in the vector space."
                                        )
                                    with metrics_cols[2]:
                                        days_apart = abs(pd.to_datetime(most_similar['date']) - pd.to_datetime(selected_date)).days
                                        st.metric(
                                            "Days Apart", 
                                            f"{days_apart} days",
                                            help="Number of calendar days between the selected date and similar day."
                                        )
                                    
                                    # Display the description (text representation)
                                    st.markdown("### Trading Day Description")
                                    st.info(most_similar["description"])
                except Exception as e:
                    st.error(f"Error during search: {e}")
                    st.exception(e)
        
        # Pattern search explanation (placeholder for future implementation)
        st.markdown("---")
        st.subheader("Future Feature: Pattern-Based Search")
        st.info("""
        In a future update, this section will be enhanced with language model-based pattern search,
        allowing you to describe the type of trading day you're looking for in natural language.
        
        Example: "Find days with high volume but small price change" or "Show me days with sharp reversals after initial losses."
        """)
        
    # Technical Dashboard Tab
    with tab5:
        st.header("Technical Indicator Dashboard")
        
        st.markdown("""
        This dashboard provides a comprehensive view of technical indicators for the selected stock.
        Monitor key metrics and use them to identify potential trading opportunities.
        """)
        
        # Date range selector for dashboard
        dashboard_date = st.date_input("Select date for technical snapshot", 
                                     value=df.index[-1].date(), 
                                     min_value=df.index[0].date(),
                                     max_value=df.index[-1].date())
        
        # Convert date to datetime for dataframe indexing
        dashboard_date_str = dashboard_date.strftime('%Y-%m-%d')
        selected_row = df[df.index.strftime('%Y-%m-%d') == dashboard_date_str]
        
        if selected_row.empty:
            st.warning(f"No data available for {dashboard_date_str}. Please select another date.")
        else:
            # Show a summary of key metrics
            st.subheader("Key Metrics")
            
            # Create metrics grid using columns
            metrics_cols = st.columns(3)
            
            with metrics_cols[0]:
                st.metric("Close", f"${selected_row['Close'].values[0]:.2f}", 
                         f"{selected_row['Return'].values[0]:.2%}")
                
                if 'RSI' in selected_row.columns:
                    rsi_value = selected_row['RSI'].values[0]
                    rsi_color = ":green" if 40 <= rsi_value <= 60 else (":red" if rsi_value > 70 or rsi_value < 30 else ":orange")
                    st.metric("RSI", f"{rsi_value:.1f}", help="Relative Strength Index - oversold (<30) or overbought (>70)")
            
            with metrics_cols[1]:
                if 'Volume' in selected_row.columns:
                    vol_value = selected_row['Volume'].values[0]
                    if 'Volume_MA_20' in selected_row.columns:
                        vol_change = (vol_value / selected_row['Volume_MA_20'].values[0] - 1) 
                        st.metric("Volume", f"{vol_value/1e6:.1f}M", f"{vol_change:.1%}")
                    else:
                        st.metric("Volume", f"{vol_value/1e6:.1f}M")
                
                if 'Volatility_20d' in selected_row.columns:
                    st.metric("Volatility (20d)", f"{selected_row['Volatility_20d'].values[0]:.4f}")
            
            with metrics_cols[2]:
                if 'Momentum_20d' in selected_row.columns:
                    mom_value = selected_row['Momentum_20d'].values[0]
                    st.metric("Momentum (20d)", f"{mom_value:.2%}")
                
                if 'ATR_14' in selected_row.columns:
                    st.metric("ATR (14d)", f"{selected_row['ATR_14'].values[0]:.2f}", 
                             help="Average True Range - measure of volatility")
            
            # Show price trend with moving averages
            st.subheader("Price Trend with Moving Averages")
            
            # Create a date range for the chart (past 6 months)
            end_date = selected_row.index[0]
            start_date = end_date - pd.Timedelta(days=180)
            mask = (df.index >= start_date) & (df.index <= end_date)
            trend_df = df[mask].copy()
            
            if len(trend_df) > 0:
                fig = go.Figure()
                
                # Add price
                fig.add_trace(go.Scatter(
                    x=trend_df.index,
                    y=trend_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#000000', width=1)
                ))
                
                # Add moving averages if available
                if 'MA_50' in trend_df.columns:
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['MA_50'],
                        mode='lines',
                        name='50-day MA',
                        line=dict(color='#2196F3', width=1.5, dash='dot')
                    ))
                
                if 'MA_200' in trend_df.columns:
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['MA_200'],
                        mode='lines',
                        name='200-day MA',
                        line=dict(color='#FF5722', width=1.5, dash='dot')
                    ))
                
                # Highlight the selected date
                fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Price with Moving Averages',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Indicator charts
            st.subheader("Technical Indicators")
            
            # Create tabs for different indicator groups
            ind_tab1, ind_tab2, ind_tab3 = st.tabs(["Momentum", "Volatility", "Volume"])
            
            with ind_tab1:
                # RSI Chart
                if 'RSI' in trend_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#673AB7', width=1.5)
                    ))
                    
                    # Add RSI reference lines (30 and 70)
                    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red")
                    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green")
                    fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="gray")
                    
                    # Highlight the selected date
                    fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                    
                    # Update layout
                    fig.update_layout(
                        title='Relative Strength Index (RSI)',
                        xaxis_title='Date',
                        yaxis_title='RSI',
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Adding interpretation
                    rsi_val = selected_row['RSI'].values[0]
                    if rsi_val > 70:
                        st.warning(f"RSI is {rsi_val:.1f}, indicating potentially overbought conditions.")
                    elif rsi_val < 30:
                        st.warning(f"RSI is {rsi_val:.1f}, indicating potentially oversold conditions.")
                    else:
                        st.info(f"RSI is {rsi_val:.1f}, in neutral territory.")
                
                # MACD Chart
                if all(col in trend_df.columns for col in ['MACD', 'MACD_Signal']):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='#2196F3', width=1.5)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['MACD_Signal'],
                        mode='lines',
                        name='Signal Line',
                        line=dict(color='#FF5722', width=1.5)
                    ))
                    
                    # Add MACD histogram
                    colors = ['red' if x < 0 else 'green' for x in trend_df['MACD_Hist']]
                    fig.add_trace(go.Bar(
                        x=trend_df.index,
                        y=trend_df['MACD_Hist'],
                        name='Histogram',
                        marker_color=colors
                    ))
                    
                    # Highlight the selected date
                    fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                    
                    # Update layout
                    fig.update_layout(
                        title='Moving Average Convergence Divergence (MACD)',
                        xaxis_title='Date',
                        yaxis_title='MACD',
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Adding interpretation
                    macd = selected_row['MACD'].values[0]
                    signal = selected_row['MACD_Signal'].values[0]
                    hist = selected_row['MACD_Hist'].values[0]
                    
                    if hist > 0 and hist > trend_df['MACD_Hist'].iloc[-2]:
                        st.info("MACD histogram is positive and increasing, suggesting bullish momentum.")
                    elif hist < 0 and hist < trend_df['MACD_Hist'].iloc[-2]:
                        st.info("MACD histogram is negative and decreasing, suggesting bearish momentum.")
                    elif macd > signal:
                        st.info("MACD is above signal line, which can indicate bullish conditions.")
                    elif macd < signal:
                        st.info("MACD is below signal line, which can indicate bearish conditions.")
            
            with ind_tab2:
                # Bollinger Bands
                if all(col in trend_df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                    fig = go.Figure()
                    
                    # Add price
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#000000', width=1)
                    ))
                    
                    # Add Bollinger Bands
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['BB_Upper'],
                        mode='lines',
                        name='Upper Band',
                        line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                        showlegend=True
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['BB_Lower'],
                        mode='lines',
                        name='Lower Band',
                        line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(173, 216, 230, 0.2)',
                        showlegend=True
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['BB_Middle'],
                        mode='lines',
                        name='Middle Band (20-day MA)',
                        line=dict(color='rgba(100, 100, 100, 0.8)', width=1, dash='dot'),
                        showlegend=True
                    ))
                    
                    # Highlight the selected date
                    fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                    
                    # Update layout
                    fig.update_layout(
                        title='Bollinger Bands',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bollinger Band Width chart
                    if 'BB_Width' in trend_df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=trend_df.index,
                            y=trend_df['BB_Width'],
                            mode='lines',
                            name='BB Width',
                            line=dict(color='#9C27B0', width=1.5)
                        ))
                        
                        # Highlight the selected date
                        fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                        
                        # Update layout
                        fig.update_layout(
                            title='Bollinger Band Width (Volatility)',
                            xaxis_title='Date',
                            yaxis_title='Width',
                            height=200,
                            margin=dict(l=0, r=0, t=30, b=0),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Adding interpretation for Bollinger Bands
                        close = selected_row['Close'].values[0]
                        upper = selected_row['BB_Upper'].values[0]
                        lower = selected_row['BB_Lower'].values[0]
                        bb_width = selected_row['BB_Width'].values[0]
                        
                        if close > upper:
                            st.warning(f"Price is above upper Bollinger Band, potentially indicating overbought conditions.")
                        elif close < lower:
                            st.warning(f"Price is below lower Bollinger Band, potentially indicating oversold conditions.")
                            
                        # Check if BB is narrowing or widening
                        recent_width = trend_df['BB_Width'].iloc[-20:].mean()
                        if bb_width < recent_width * 0.8:
                            st.info("Bollinger Bands are narrowing, suggesting decreasing volatility and potential for a bigger move soon.")
                        elif bb_width > recent_width * 1.2:
                            st.info("Bollinger Bands are widening, suggesting increasing volatility.")
                            
                # ATR Chart
                if 'ATR_14' in trend_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['ATR_14'],
                        mode='lines',
                        name='ATR (14)',
                        line=dict(color='#FF9800', width=1.5)
                    ))
                    
                    # Highlight the selected date
                    fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                    
                    # Update layout
                    fig.update_layout(
                        title='Average True Range (ATR)',
                        xaxis_title='Date',
                        yaxis_title='ATR',
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    atr = selected_row['ATR_14'].values[0]
                    avg_price = selected_row['Close'].values[0]
                    atr_pct = atr / avg_price
                    
                    st.info(f"Current ATR is ${atr:.2f}, representing {atr_pct:.2%} of the current price.")
                    
                    # Compare to historical ATR
                    hist_atr = trend_df['ATR_14'].mean()
                    hist_atr_pct = hist_atr / trend_df['Close'].mean()
                    
                    if atr_pct > hist_atr_pct * 1.2:
                        st.info("Volatility is higher than the historical average.")
                    elif atr_pct < hist_atr_pct * 0.8:
                        st.info("Volatility is lower than the historical average.")
            
            with ind_tab3:
                # Volume chart
                fig = go.Figure()
                
                # Add volume bars
                colors = ['red' if x < 0 else 'green' for x in trend_df['Return']]
                fig.add_trace(go.Bar(
                    x=trend_df.index,
                    y=trend_df['Volume'],
                    name='Volume',
                    marker_color=colors
                ))
                
                # Add volume moving average if available
                if 'Volume_MA_20' in trend_df.columns:
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['Volume_MA_20'],
                        mode='lines',
                        name='20-day MA',
                        line=dict(color='#000000', width=1.5)
                    ))
                
                # Highlight the selected date
                fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                
                # Update layout
                fig.update_layout(
                    title='Volume',
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # On-Balance Volume
                if 'OBV' in trend_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df.index,
                        y=trend_df['OBV'],
                        mode='lines',
                        name='OBV',
                        line=dict(color='#3F51B5', width=1.5)
                    ))
                    
                    # Highlight the selected date
                    fig.add_vline(x=end_date, line_width=1, line_dash="dash", line_color="green")
                    
                    # Update layout
                    fig.update_layout(
                        title='On-Balance Volume (OBV)',
                        xaxis_title='Date',
                        yaxis_title='OBV',
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume analysis
                    if 'Relative_Volume' in selected_row.columns:
                        rel_vol = selected_row['Relative_Volume'].values[0]
                        if rel_vol > 2:
                            st.warning(f"Volume is {rel_vol:.1f}x higher than the 20-day average, indicating significant interest.")
                        elif rel_vol > 1.5:
                            st.info(f"Volume is {rel_vol:.1f}x higher than the 20-day average.")
                        elif rel_vol < 0.5:
                            st.info(f"Volume is {rel_vol:.1f}x lower than the 20-day average, indicating low interest.")
    
    # Predictive Analytics Tab
    with tab6:
        st.header("Predictive Analytics")
        
        st.markdown("""
        This tab uses machine learning models to predict potential future price movements.
        These predictions are based on technical indicators and historical patterns.
        """)
        
        # Prediction method selector
        pred_method = st.radio(
            "Select prediction method:",
            ["Technical Pattern Forecast", "Similar Day Projection", "Indicator Threshold Alert"]
        )
        
        if pred_method == "Technical Pattern Forecast":
            st.subheader("Technical Pattern Forecast")
            
            # Days to forecast
            forecast_days = st.slider("Days to forecast", 1, 30, 5)
            
            # Select indicator for forecasting
            forecast_indicator = st.selectbox(
                "Select primary indicator for forecast:",
                ["Return", "Momentum_20d", "RSI", "BB_Width", "MACD", "Volatility_20d"]
            )
            
            # Make forecast button
            if st.button("Generate Forecast"):
                with st.spinner("Analyzing patterns and generating forecast..."):
                    # We'll implement a simple forecast based on historical patterns
                    # In a production system, this would use a proper ML model
                    
                    # Get the last N days of data
                    last_n_days = df.iloc[-30:].copy()
                    
                    # Calculate some basic statistics
                    if forecast_indicator in last_n_days.columns:
                        mean_val = last_n_days[forecast_indicator].mean()
                        std_val = last_n_days[forecast_indicator].std()
                        last_val = last_n_days[forecast_indicator].iloc[-1]
                        
                        # Generate a forecast using a simple mean-reversion model
                        forecast = []
                        current_val = last_val
                        
                        for i in range(forecast_days):
                            # Simple mean reversion with random noise
                            next_val = current_val + (mean_val - current_val) * 0.3 + np.random.normal(0, std_val * 0.5)
                            forecast.append(next_val)
                            current_val = next_val
                        
                        # Create forecast dates
                        last_date = df.index[-1]
                        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
                        
                        # Create a dataframe with historical and forecast data
                        hist_df = pd.DataFrame({
                            'Date': last_n_days.index,
                            'Value': last_n_days[forecast_indicator],
                            'Type': 'Historical'
                        })
                        
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Value': forecast,
                            'Type': 'Forecast'
                        })
                        
                        combined_df = pd.concat([hist_df, forecast_df])
                        
                        # Plot the forecast
                        fig = px.line(
                            combined_df,
                            x='Date',
                            y='Value',
                            color='Type',
                            title=f'{forecast_indicator} Forecast for {ticker}',
                            color_discrete_map={'Historical': '#1F77B4', 'Forecast': '#FF7F0E'}
                        )
                        
                        # Add confidence intervals around forecast
                        upper = [f + std_val * 0.5 for f in forecast]
                        lower = [f - std_val * 0.5 for f in forecast]
                        
                        fig.add_scatter(
                            x=forecast_dates,
                            y=upper,
                            fill=None,
                            mode='lines',
                            line_color='rgba(255, 127, 14, 0.3)',
                            name='Upper Bound'
                        )
                        
                        fig.add_scatter(
                            x=forecast_dates,
                            y=lower,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(255, 127, 14, 0.3)',
                            name='Lower Bound'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add cautionary note
                        st.warning("""
                        **Disclaimer**: This forecast is based on a simplified model and historical patterns. 
                        It should not be used as the sole basis for investment decisions. 
                        Past performance is not indicative of future results.
                        """)
                        
                        # Provide forecast summary
                        if forecast_indicator == "Return":
                            cumulative_return = np.prod([1 + f for f in forecast]) - 1
                            st.info(f"Projected cumulative return over {forecast_days} days: {cumulative_return:.2%}")
                        else:
                            avg_forecast = np.mean(forecast)
                            st.info(f"Average projected {forecast_indicator} over {forecast_days} days: {avg_forecast:.4f}")
                    else:
                        st.error(f"Selected indicator {forecast_indicator} not available in the data.")
        
        elif pred_method == "Similar Day Projection":
            st.subheader("Similar Day Projection")
            st.markdown("""
            This analysis finds the most similar days to today in the price history,
            then shows what typically happened after those similar days.
            """)
            
            # Days to match
            match_days = st.slider("Number of similar days to find", 3, 20, 5)
            
            # Days to project forward
            forward_days = st.slider("Days to project forward", 1, 30, 5)
            
            # Find similar days button
            if st.button("Project Based on Similar Days"):
                with st.spinner("Finding similar days and projecting outcomes..."):
                    # Use our vector database to find similar days
                    db_dir = "vector_db"
                    db_file = os.path.join(db_dir, f"{ticker}_vector_db")
                    
                    if os.path.exists(db_file):
                        # Get the most recent day's data for comparison
                        recent_day = df.iloc[-1].copy()
                        
                        # Load the vector database
                        vector_db = VectorDatabase.load(db_file)
                        
                        # Load embedding transformation data
                        embedding_data_file = os.path.join(db_dir, f"{ticker}_embedding_data.json")
                        
                        with open(embedding_data_file, "r") as f:
                            embedding_info = json.load(f)
                            
                        # Get expected features
                        expected_features = embedding_info.get("feature_names", ["Return", "Volume"])
                        
                        # Create a feature vector for the recent day
                        feature_vector = np.zeros((1, len(expected_features)), dtype=np.float32)
                        
                        # Fill in with actual values
                        for i, feature in enumerate(expected_features):
                            if feature in recent_day:
                                feature_vector[0, i] = recent_day[feature]
                            elif feature == "Volume" and "Volume" in recent_day:
                                # Normalize volume if needed
                                max_volume = df["Volume"].max()
                                feature_vector[0, i] = recent_day["Volume"] / max_volume if max_volume > 0 else 0
                        
                        # Transform feature vector
                        scaler_mean = np.array(embedding_info["scaler_mean"])
                        scaler_scale = np.array(embedding_info["scaler_scale"])
                        feature_vector_scaled = (feature_vector - scaler_mean) / scaler_scale
                        
                        # Apply PCA transformation
                        pca_components = np.array(embedding_info["pca_components"])
                        pca_mean = np.array(embedding_info["pca_mean"])
                        query_vector = np.dot(feature_vector_scaled - pca_mean, pca_components.T)
                        
                        # Search for similar days
                        results = vector_db.search(query_vector.astype(np.float32), k=match_days)
                        
                        if not results:
                            st.error("Could not find similar days in the database.")
                        else:
                            # Extract the dates of similar days
                            similar_dates = [result["date"] for result in results]
                            
                            # Find the indices of these dates in the original dataframe
                            similar_indices = []
                            for date_str in similar_dates:
                                matching_idx = df.index.astype(str).str.contains(date_str)
                                if any(matching_idx):
                                    idx = df.index[matching_idx][0]
                                    similar_indices.append(df.index.get_loc(idx))
                            
                            # Create projections based on what happened after each similar day
                            projections = []
                            
                            for idx in similar_indices:
                                # Make sure we have enough days after this index
                                if idx + forward_days < len(df):
                                    # Get the returns for the days after this day
                                    future_returns = df["Return"].iloc[idx+1:idx+forward_days+1].values
                                    projections.append(future_returns)
                            
                            if projections:
                                # Convert to numpy array and handle any missing data
                                projections = np.array(projections)
                                
                                # Calculate statistics
                                mean_proj = np.mean(projections, axis=0)
                                std_proj = np.std(projections, axis=0)
                                
                                # Create a dataframe for plotting
                                dates = [df.index[-1] + pd.Timedelta(days=i+1) for i in range(forward_days)]
                                
                                # Calculate cumulative returns
                                cum_returns = np.cumprod(1 + mean_proj) - 1
                                upper_bound = np.cumprod(1 + mean_proj + std_proj) - 1
                                lower_bound = np.cumprod(1 + mean_proj - std_proj) - 1
                                
                                plot_df = pd.DataFrame({
                                    'Date': dates,
                                    'Cumulative Return': cum_returns,
                                    'Upper Bound': upper_bound,
                                    'Lower Bound': lower_bound
                                })
                                
                                # Plot the projection
                                fig = go.Figure()
                                
                                # Add actual price line
                                last_price = df["Close"].iloc[-1]
                                
                                # Add mean projection
                                fig.add_trace(go.Scatter(
                                    x=plot_df['Date'],
                                    y=last_price * (1 + plot_df['Cumulative Return']),
                                    mode='lines+markers',
                                    name='Projected Price',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Add confidence bands
                                fig.add_trace(go.Scatter(
                                    x=plot_df['Date'],
                                    y=last_price * (1 + plot_df['Upper Bound']),
                                    mode='lines',
                                    name='Upper Bound',
                                    line=dict(color='rgba(0, 0, 255, 0.2)', width=0),
                                    showlegend=True
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=plot_df['Date'],
                                    y=last_price * (1 + plot_df['Lower Bound']),
                                    mode='lines',
                                    name='Lower Bound',
                                    line=dict(color='rgba(0, 0, 255, 0.2)', width=0),
                                    fill='tonexty',
                                    fillcolor='rgba(0, 0, 255, 0.1)',
                                    showlegend=True
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title=f'Price Projection Based on {match_days} Similar Days',
                                    xaxis_title='Date',
                                    yaxis_title='Projected Price',
                                    height=400,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Summary statistics
                                final_return = cum_returns[-1]
                                
                                if final_return > 0:
                                    st.success(f"Based on similar historical patterns, the projected return over the next {forward_days} days is {final_return:.2%}")
                                else:
                                    st.warning(f"Based on similar historical patterns, the projected return over the next {forward_days} days is {final_return:.2%}")
                                
                                # Disclaimer
                                st.warning("""
                                **Disclaimer**: This projection is based on historical patterns and is not a guarantee of future results.
                                It should be used as one of many tools for market analysis, not as the sole basis for investment decisions.
                                """)
                            else:
                                st.error("Could not generate projections from the similar days found.")
                    else:
                        st.error(f"Vector database for {ticker} not found. Please create one first.")
        
        elif pred_method == "Indicator Threshold Alert":
            st.subheader("Indicator Threshold Alert")
            st.markdown("""
            Set thresholds for key indicators to receive alerts when those thresholds are crossed.
            This helps identify potential entry and exit points based on technical analysis.
            """)
            
            # Select indicator
            alert_indicator = st.selectbox(
                "Select indicator for alert:",
                ["RSI", "BB_Width", "Volatility_20d", "MACD", "Relative_Volume"]
            )
            
            if alert_indicator in df.columns:
                # Current value
                current_val = df[alert_indicator].iloc[-1]
                
                # Calculate reasonable min/max values for the slider
                min_val = df[alert_indicator].min()
                max_val = df[alert_indicator].max()
                
                # Handle special case for RSI which has a natural 0-100 range
                if alert_indicator == "RSI":
                    min_val, max_val = 0, 100
                    
                # For threshold type
                threshold_type = st.radio("Alert when indicator:", ["Above threshold", "Below threshold"])
                
                # Set threshold
                threshold = st.slider(
                    "Set threshold value:",
                    float(min_val), float(max_val), 
                    float(current_val if threshold_type == "Above threshold" else min_val + (max_val - min_val) * 0.3)
                )
                
                # Show current value
                st.metric(
                    f"Current {alert_indicator} value", 
                    f"{current_val:.4f}", 
                    f"{current_val - threshold:.4f}"
                )
                
                # Check if threshold is crossed
                is_above = current_val > threshold
                
                if (threshold_type == "Above threshold" and is_above) or (threshold_type == "Below threshold" and not is_above):
                    st.success(f"🚨 Alert: {alert_indicator} is {'above' if is_above else 'below'} the threshold of {threshold:.4f}")
                else:
                    st.info(f"No alert: {alert_indicator} is {'below' if is_above else 'above'} the threshold of {threshold:.4f}")
                
                # Historical view of indicator with threshold
                st.subheader(f"Historical View of {alert_indicator}")
                
                fig = go.Figure()
                
                # Add indicator line
                fig.add_trace(go.Scatter(
                    x=df.index[-100:],
                    y=df[alert_indicator].iloc[-100:],
                    mode='lines',
                    name=alert_indicator,
                    line=dict(color='blue', width=1.5)
                ))
                
                # Add threshold line
                fig.add_hline(y=threshold, line_width=1, line_dash="dash", 
                             line_color="red", annotation_text=f"Threshold: {threshold:.4f}")
                
                # Update layout
                fig.update_layout(
                    title=f'{alert_indicator} with Alert Threshold',
                    xaxis_title='Date',
                    yaxis_title=alert_indicator,
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show historical alerts
                st.subheader("Historical Alerts")
                
                # Create a signal when threshold is crossed
                if threshold_type == "Above threshold":
                    signals = df[alert_indicator] > threshold
                else:
                    signals = df[alert_indicator] < threshold
                
                # Find days when signals occurred
                signal_days = df[signals].index
                
                if len(signal_days) > 0:
                    # Create a dataframe with signal days and subsequent returns
                    signal_returns = []
                    
                    for day in signal_days[-30:]:  # Limit to last 30 signals
                        idx = df.index.get_loc(day)
                        if idx + 5 < len(df):  # Make sure we have 5 days of returns after signal
                            next_5d_return = np.prod(1 + df["Return"].iloc[idx+1:idx+6]) - 1
                            signal_returns.append({
                                "Date": day,
                                "Indicator Value": df[alert_indicator].loc[day],
                                "Next Day Return": df["Return"].iloc[idx+1],
                                "Next 5D Return": next_5d_return
                            })
                    
                    if signal_returns:
                        signal_df = pd.DataFrame(signal_returns)
                        
                        # Format returns as percentages
                        signal_df["Next Day Return"] = signal_df["Next Day Return"].apply(lambda x: f"{x:.2%}")
                        signal_df["Next 5D Return"] = signal_df["Next 5D Return"].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(signal_df, use_container_width=True)
                        
                        # Calculate effectiveness
                        success_rate = sum(1 for r in signal_returns if isinstance(r["Next 5D Return"], str) and np.sign(float(r["Next 5D Return"].rstrip("%")) / 100) > 0 or not isinstance(r["Next 5D Return"], str) and np.sign(r["Next 5D Return"]) > 0) / len(signal_returns)
                        
                        st.metric("Success Rate (5D Returns)", f"{success_rate:.1%}")
                    else:
                        st.info("No signal days with enough subsequent data to analyze returns.")
                else:
                    st.info("No historical instances found where the threshold was crossed.")
            else:
                st.error(f"Selected indicator {alert_indicator} is not available in the data.")
    
    # Pattern Backtesting Tab
    with tab7:
        st.header("Pattern Backtesting")
        
        st.markdown("""
        Backtest trading strategies based on technical patterns and indicator combinations.
        This analysis helps evaluate how different strategies would have performed historically.
        """)
        
        # Pattern selection
        pattern_type = st.selectbox(
            "Select pattern to backtest:",
            ["RSI Reversal", "Bollinger Band Breakout", "Moving Average Crossover", "MACD Signal", "Custom Indicator Combination"]
        )
        
        # Timeframe options
        st.subheader("Backtest Configuration")
        cols = st.columns(3)
        
        with cols[0]:
            entry_days = st.number_input("Entry delay (days)", 0, 10, 1, 
                                       help="Number of days after pattern to enter position")
        
        with cols[1]:
            exit_days = st.number_input("Exit after (days)", 1, 90, 10, 
                                      help="Number of days to hold position")
        
        with cols[2]:
            stop_loss = st.number_input("Stop loss %", 0.0, 20.0, 5.0) / 100
        
        # Pattern-specific parameters
        if pattern_type == "RSI Reversal":
            st.subheader("RSI Reversal Parameters")
            
            rsi_cols = st.columns(2)
            with rsi_cols[0]:
                rsi_buy_threshold = st.slider("Buy when RSI below", 10, 40, 30, 
                                             help="Enter position when RSI falls below this value")
            
            with rsi_cols[1]:
                rsi_sell_threshold = st.slider("Sell when RSI above", 60, 90, 70,
                                              help="Enter position when RSI rises above this value")
            
            # Define the pattern function
            def rsi_reversal_pattern(row):
                # Bullish pattern (oversold)
                if 'RSI' in row and row['RSI'] < rsi_buy_threshold:
                    return True
                return False
            
            pattern_function = rsi_reversal_pattern
            
        elif pattern_type == "Bollinger Band Breakout":
            st.subheader("Bollinger Band Breakout Parameters")
            
            bb_direction = st.radio("Breakout direction", ["Upper Band (Bullish)", "Lower Band (Bullish)"])
            lookback = st.slider("Confirmation period (days)", 1, 10, 2, 
                               help="Number of days price should stay near the band before breakout")
            
            # Define the pattern function
            def bb_breakout_pattern(row):
                # Skip if we don't have all columns
                if not all(c in row for c in ['Close', 'BB_Upper', 'BB_Lower', 'BB_Middle']):
                    return False
                
                if bb_direction == "Upper Band (Bullish)" and row['Close'] > row['BB_Upper']:
                    # Price crossed above upper band
                    return True
                elif bb_direction == "Lower Band (Bullish)" and row['Close'] < row['BB_Lower']:
                    # Price crossed below lower band
                    return True
                return False
            
            pattern_function = bb_breakout_pattern
            
        elif pattern_type == "Moving Average Crossover":
            st.subheader("Moving Average Crossover Parameters")
            
            ma_cols = st.columns(3)
            
            with ma_cols[0]:
                fast_ma = st.selectbox("Fast MA", ["5-day", "10-day", "20-day", "50-day"], index=0)
            
            with ma_cols[1]:
                slow_ma = st.selectbox("Slow MA", ["20-day", "50-day", "100-day", "200-day"], index=1)
            
            with ma_cols[2]:
                cross_direction = st.radio("Cross direction", ["Bullish (Fast crosses above Slow)", "Bearish (Fast crosses below Slow)"])
            
            # Define the pattern function
            def ma_crossover_pattern(row):
                # Map selected MAs to column names
                ma_mapping = {
                    "5-day": "MA_5",
                    "10-day": "MA_10",
                    "20-day": "MA_20",
                    "50-day": "MA_50",
                    "100-day": "MA_100",
                    "200-day": "MA_200"
                }
                
                fast_col = ma_mapping.get(fast_ma)
                slow_col = ma_mapping.get(slow_ma)
                
                # Skip if we don't have both columns
                if not (fast_col in row.index and slow_col in row.index):
                    return False
                
                # Get previous values - this is approximate since we don't access previous rows
                prev_idx = row.name - pd.Timedelta(days=1)
                if prev_idx in df.index:
                    prev_fast = df.loc[prev_idx, fast_col] if fast_col in df.columns else None
                    prev_slow = df.loc[prev_idx, slow_col] if slow_col in df.columns else None
                    curr_fast = row[fast_col]
                    curr_slow = row[slow_col]
                    
                    if prev_fast is not None and prev_slow is not None:
                        if cross_direction == "Bullish (Fast crosses above Slow)":
                            # Bullish crossover
                            return prev_fast < prev_slow and curr_fast > curr_slow
                        else:
                            # Bearish crossover
                            return prev_fast > prev_slow and curr_fast < curr_slow
                
                return False
            
            pattern_function = ma_crossover_pattern
            
        elif pattern_type == "MACD Signal":
            st.subheader("MACD Signal Parameters")
            
            macd_signal_type = st.radio(
                "MACD Signal Type",
                ["Bullish Crossover (MACD crosses above Signal)", 
                 "Bearish Crossover (MACD crosses below Signal)",
                 "Bullish Divergence (Price makes lower low, MACD higher low)",
                 "Bearish Divergence (Price makes higher high, MACD lower high)"]
            )
            
            # Define the pattern function
            def macd_signal_pattern(row):
                # Skip if we don't have MACD columns
                if not all(c in row for c in ['MACD', 'MACD_Signal', 'MACD_Hist']):
                    return False
                
                idx = df.index.get_loc(row.name)
                if idx > 0:  # Ensure we have previous data
                    curr_macd = row['MACD']
                    curr_signal = row['MACD_Signal']
                    prev_macd = df.iloc[idx-1]['MACD']
                    prev_signal = df.iloc[idx-1]['MACD_Signal']
                    
                    if macd_signal_type == "Bullish Crossover (MACD crosses above Signal)":
                        # Bullish crossover
                        return prev_macd < prev_signal and curr_macd > curr_signal
                    elif macd_signal_type == "Bearish Crossover (MACD crosses below Signal)":
                        # Bearish crossover
                        return prev_macd > prev_signal and curr_macd < curr_signal
                    
                    # For divergences, we need more historical data
                    if idx > 20:  # At least 20 days of history
                        # Check for local extremes in price and MACD
                        if macd_signal_type == "Bullish Divergence (Price makes lower low, MACD higher low)":
                            # Find recent low points in price and MACD
                            price_window = df['Close'].iloc[idx-20:idx+1]
                            macd_window = df['MACD'].iloc[idx-20:idx+1]
                            
                            if len(price_window) > 5 and len(macd_window) > 5:
                                # Check if current price is a local minimum
                                if row['Close'] == price_window.min():
                                    # Find the previous local minimum in the window
                                    prev_min_idx = price_window.idxmin()
                                    if prev_min_idx != row.name:  # Different day
                                        # Check for bullish divergence
                                        prev_min_price = df.loc[prev_min_idx, 'Close']
                                        prev_min_macd = df.loc[prev_min_idx, 'MACD']
                                        
                                        if row['Close'] < prev_min_price and row['MACD'] > prev_min_macd:
                                            return True
                        
                        elif macd_signal_type == "Bearish Divergence (Price makes higher high, MACD lower high)":
                            # Find recent high points in price and MACD
                            price_window = df['Close'].iloc[idx-20:idx+1]
                            macd_window = df['MACD'].iloc[idx-20:idx+1]
                            
                            if len(price_window) > 5 and len(macd_window) > 5:
                                # Check if current price is a local maximum
                                if row['Close'] == price_window.max():
                                    # Find the previous local maximum in the window
                                    prev_max_idx = price_window.idxmax()
                                    if prev_max_idx != row.name:  # Different day
                                        # Check for bearish divergence
                                        prev_max_price = df.loc[prev_max_idx, 'Close']
                                        prev_max_macd = df.loc[prev_max_idx, 'MACD']
                                        
                                        if row['Close'] > prev_max_price and row['MACD'] < prev_max_macd:
                                            return True
                
                return False
            
            pattern_function = macd_signal_pattern
            
        elif pattern_type == "Custom Indicator Combination":
            st.subheader("Custom Indicator Combination")
            
            # Let user select indicators and conditions
            custom_cols = st.columns(2)
            
            available_indicators = [
                "RSI", "MACD", "MACD_Signal", "Volatility_20d", "Momentum_20d", 
                "BB_Width", "Relative_Volume", "Return"
            ]
            
            with custom_cols[0]:
                indicator1 = st.selectbox("Indicator 1", available_indicators, index=0)
                condition1 = st.selectbox("Condition 1", ["Above", "Below", "Increasing", "Decreasing"], index=0)
                threshold1 = st.number_input("Threshold 1", value=30.0, step=1.0)
            
            with custom_cols[1]:
                indicator2 = st.selectbox("Indicator 2", available_indicators, index=1)
                condition2 = st.selectbox("Condition 2", ["Above", "Below", "Increasing", "Decreasing"], index=0)
                threshold2 = st.number_input("Threshold 2", value=0.0, step=0.01)
            
            logical_operator = st.radio("Combine conditions with:", ["AND", "OR"])
            
            # Define the pattern function
            def custom_pattern(row):
                result1, result2 = False, False
                
                # Check if indicators exist
                if indicator1 not in row:
                    return False
                
                if indicator2 not in row:
                    return False
                
                # Evaluate first condition
                if condition1 == "Above":
                    result1 = row[indicator1] > threshold1
                elif condition1 == "Below":
                    result1 = row[indicator1] < threshold1
                elif condition1 == "Increasing":
                    idx = df.index.get_loc(row.name)
                    if idx > 0:
                        prev_val = df.iloc[idx-1][indicator1]
                        result1 = row[indicator1] > prev_val * (1 + threshold1/100)
                elif condition1 == "Decreasing":
                    idx = df.index.get_loc(row.name)
                    if idx > 0:
                        prev_val = df.iloc[idx-1][indicator1]
                        result1 = row[indicator1] < prev_val * (1 - threshold1/100)
                
                # Evaluate second condition
                if condition2 == "Above":
                    result2 = row[indicator2] > threshold2
                elif condition2 == "Below":
                    result2 = row[indicator2] < threshold2
                elif condition2 == "Increasing":
                    idx = df.index.get_loc(row.name)
                    if idx > 0:
                        prev_val = df.iloc[idx-1][indicator2]
                        result2 = row[indicator2] > prev_val * (1 + threshold2/100)
                elif condition2 == "Decreasing":
                    idx = df.index.get_loc(row.name)
                    if idx > 0:
                        prev_val = df.iloc[idx-1][indicator2]
                        result2 = row[indicator2] < prev_val * (1 - threshold2/100)
                
                # Combine results based on logical operator
                if logical_operator == "AND":
                    return result1 and result2
                else:  # OR
                    return result1 or result2
                
            pattern_function = custom_pattern
        
        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest simulation..."):
                # Create a backtester instance
                backtester = PatternBacktester(df)
                
                # Run the backtest
                results = backtester.backtest_pattern_strategy(
                    pattern_function=pattern_function,
                    entry_days=entry_days,
                    exit_days=exit_days,
                    stop_loss_pct=stop_loss
                )
                
                if results.empty:
                    st.warning("No trading signals were generated for this pattern and configuration.")
                else:
                    # Get performance statistics
                    stats = backtester.get_performance_stats()
                    
                    # Display summary statistics
                    st.subheader("Backtest Summary")
                    
                    # Use columns for key metrics
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric("Total Trades", f"{stats['Total Trades']}")
                        
                    with metric_cols[1]:
                        st.metric("Win Rate", f"{stats['Win Rate']:.1%}")
                        
                    with metric_cols[2]:
                        st.metric("Avg Return", f"{stats['Average Return']:.2%}")
                        
                    with metric_cols[3]:
                        st.metric("Cumulative Return", f"{stats['Cumulative Return']:.2%}")
                    
                    # Display results table
                    st.subheader("Detailed Trade Results")
                    
                    # Format the results for display
                    display_results = results.copy()
                    display_results['Entry Date'] = display_results['Entry Date'].dt.strftime('%Y-%m-%d')
                    display_results['Exit Date'] = display_results['Exit Date'].dt.strftime('%Y-%m-%d')
                    display_results['Return'] = display_results['Return'].apply(lambda x: f"{x:.2%}")
                    display_results['Cum Return'] = display_results['Cum Return'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(display_results, use_container_width=True)
                    
                    # Equity curve
                    st.subheader("Equity Curve")
                    
                    # Create equity curve
                    equity_curve = (1 + results['Return']).cumprod()
                    
                    # Plot equity curve
                    fig = px.line(
                        x=results['Exit Date'], 
                        y=equity_curve,
                        labels={"x": "Date", "y": "Portfolio Value (Starting = 1)"},
                        title="Strategy Equity Curve"
                    )
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Drawdowns
                    st.subheader("Drawdown Analysis")
                    
                    # Calculate drawdowns
                    rolling_max = np.maximum.accumulate(equity_curve)
                    drawdowns = (equity_curve / rolling_max) - 1
                    
                    # Plot drawdowns
                    fig = px.area(
                        x=results['Exit Date'],
                        y=drawdowns,
                        labels={"x": "Date", "y": "Drawdown"},
                        title="Strategy Drawdowns",
                        color_discrete_sequence=["red"]
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Maximum drawdown
                    max_drawdown = drawdowns.min()
                    st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
                    
                    # Trade distribution
                    st.subheader("Trade Return Distribution")
                    
                    # Convert returns to numeric values for histogram
                    numeric_returns = results['Return'].astype(float)
                    
                    # Create histogram
                    fig = px.histogram(
                        numeric_returns, 
                        nbins=20,
                        labels={"value": "Return", "count": "Number of Trades"},
                        title="Distribution of Trade Returns",
                        color_discrete_sequence=["skyblue"]
                    )
                    
                    # Add average return line
                    fig.add_vline(
                        x=numeric_returns.mean(), 
                        line_width=2, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Avg: {numeric_returns.mean():.2%}"
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display drawdown and recovery statistics
                    recovery_periods = []
                    in_drawdown = False
                    drawdown_start = 0
                    
                    for i in range(1, len(equity_curve)):
                        if not in_drawdown and equity_curve[i] < equity_curve[i-1]:
                            in_drawdown = True
                            drawdown_start = i
                        elif in_drawdown and equity_curve[i] >= equity_curve[drawdown_start-1]:
                            recovery_periods.append(i - drawdown_start)
                            in_drawdown = False
                    
                    if recovery_periods:
                        avg_recovery = sum(recovery_periods) / len(recovery_periods)
                        st.metric("Average Recovery Time (Trades)", f"{avg_recovery:.1f} trades")
                    
                    # Disclaimer
                    st.warning("""
                    **Disclaimer**: Past performance is not indicative of future results. This backtest is based on historical 
                    data and does not account for transaction costs, slippage, or other market factors that would affect actual trading results.
                    """)
    
    # Pattern Groups Tab
    with tab8:
        st.header("30-Day Pattern Groups")
        
        st.markdown("""
        This tab shows similar market patterns grouped together based on 30-day return-volume characteristics.
        Each group represents a distinct market behavior pattern that has occurred multiple times.
        """)
        
        # Check if we have a vector database with pattern groups
        db_dir = "vector_db"
        db_file = os.path.join(db_dir, f"{ticker}_vector_db")
        
        if os.path.exists(f"{db_file}_index") and os.path.exists(f"{db_file}_data.pkl"):
            # Load the vector database
            vector_db = VectorDatabase.load(db_file)
            
            # Check if the database has pattern groups
            if vector_db.pattern_groups and 'n_groups' in vector_db.pattern_groups:
                n_pattern_groups = vector_db.pattern_groups.get('n_groups', 0)
                
                if n_pattern_groups > 0:
                    st.subheader(f"Found {n_pattern_groups} Distinct Pattern Groups")
                    
                    # Get representative patterns for each group
                    representatives = vector_db.get_representative_patterns()
                    
                    if representatives:
                        # Explore different groups
                        selected_group = st.selectbox(
                            "Select a pattern group to explore:",
                            options=range(n_pattern_groups),
                            format_func=lambda x: f"Pattern Group {x+1}"
                        )
                        
                        # Get all patterns in the selected group
                        group_patterns = vector_db.get_pattern_group(selected_group)
                        
                        if group_patterns:
                            st.success(f"Found {len(group_patterns)} occurrences of pattern group {selected_group+1}")
                            
                            # Find representative for this group
                            rep = next((r for r in representatives if r.get('group_id') == selected_group), None)
                            
                            if rep:
                                rep_date = rep.get('date', '')
                                
                                # Show visual of example pattern
                                st.subheader(f"Representative Pattern (from {rep_date})")
                                
                                # Get the 30-day window for this date
                                try:
                                    rep_idx = df.index.get_loc(pd.to_datetime(rep_date))
                                    if rep_idx >= 30:
                                        # Extract 30-day window
                                        window_start = rep_idx - 30
                                        window_end = rep_idx
                                        
                                        # Get price data
                                        price_window = df['Close'].iloc[window_start:window_end+1]
                                        normalized_price = price_window / price_window.iloc[0]
                                        
                                        # Get volume data
                                        volume_window = df['Volume'].iloc[window_start:window_end+1]
                                        normalized_volume = volume_window / volume_window.mean() if volume_window.mean() > 0 else volume_window
                                        
                                        # Create a dataframe for plotting
                                        window_df = pd.DataFrame({
                                            'Date': price_window.index,
                                            'Normalized Price': normalized_price.values,
                                            'Normalized Volume': normalized_volume.values
                                        })
                                        
                                        # Plot the pattern
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Price chart
                                            fig = px.line(
                                                window_df, 
                                                x='Date', 
                                                y='Normalized Price',
                                                title='30-Day Price Pattern (Normalized)'
                                            )
                                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                        with col2:
                                            # Volume chart
                                            fig = px.bar(
                                                window_df, 
                                                x='Date', 
                                                y='Normalized Volume',
                                                title='30-Day Volume Pattern (Normalized)'
                                            )
                                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                        # Pattern characteristics
                                        st.subheader("Pattern Characteristics")
                                        
                                        # Calculate key statistics
                                        daily_returns = normalized_price.pct_change().iloc[1:]
                                        
                                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                                        with stats_col1:
                                            final_return = normalized_price.iloc[-1] - 1
                                            st.metric("30-Day Return", f"{final_return:.2%}")
                                            
                                            # Calculate trend strength using linear regression
                                            days = np.arange(len(normalized_price))
                                            slope, _, _, _, _ = stats.linregress(days, normalized_price.values)
                                            st.metric("Trend Strength", f"{slope:.4f}", 
                                                     help="Slope of linear regression line - positive indicates uptrend, negative indicates downtrend")
                                        
                                        with stats_col2:
                                            volatility = daily_returns.std()
                                            st.metric("Volatility", f"{volatility:.4f}")
                                            
                                            # Calculate max drawdown
                                            cum_max = np.maximum.accumulate(normalized_price)
                                            drawdown = (normalized_price / cum_max) - 1
                                            max_dd = drawdown.min()
                                            st.metric("Max Drawdown", f"{max_dd:.2%}")
                                            
                                        with stats_col3:
                                            # Volume characteristics
                                            vol_trend = rep.get('Volume_Trend', 0)
                                            st.metric("Volume Trend", f"{vol_trend:.4f}")
                                            
                                            vol_spikes = rep.get('Volume_Spikes', 0)
                                            st.metric("Volume Spikes", f"{vol_spikes}", 
                                                     help="Number of days with volume > 2x average")
                                except Exception as e:
                                    st.error(f"Error displaying pattern: {e}")
                            
                            # Show all occurrences of this pattern
                            st.subheader("All Occurrences of This Pattern")
                            
                            # Convert to dataframe for display
                            patterns_df = pd.DataFrame(group_patterns)
                            
                            # Convert dates to datetime
                            if 'date' in patterns_df.columns:
                                patterns_df['date'] = pd.to_datetime(patterns_df['date'])
                                patterns_df = patterns_df.sort_values('date', ascending=False)
                            
                            # Show in a data table
                            if not patterns_df.empty:
                                # Format columns for display
                                display_df = patterns_df.copy()
                                
                                # Select and rename columns for display
                                cols_to_display = {
                                    'date': 'Date',
                                    'return': 'Return',
                                    'pattern_group': 'Pattern Group'
                                }
                                
                                # Add additional columns if they exist
                                for col in ['volatility', 'momentum', 'similarity']:
                                    if col in display_df.columns:
                                        cols_to_display[col] = col.capitalize()
                                
                                # Select and rename columns
                                display_df = display_df[[c for c in cols_to_display.keys() if c in display_df.columns]]
                                display_df.columns = [cols_to_display[c] for c in display_df.columns]
                                
                                # Format numeric columns
                                if 'Return' in display_df.columns:
                                    display_df['Return'] = display_df['Return'].apply(lambda x: f"{x:.2%}")
                                
                                if 'Volatility' in display_df.columns:
                                    display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.4f}")
                                    
                                if 'Momentum' in display_df.columns:
                                    display_df['Momentum'] = display_df['Momentum'].apply(lambda x: f"{x:.2%}")
                                    
                                if 'Similarity' in display_df.columns:
                                    display_df['Similarity'] = display_df['Similarity'].apply(lambda x: f"{x:.1f}%")
                                
                                # Display table
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Plot occurrences over time
                                if 'Date' in display_df.columns:
                                    st.subheader("Pattern Occurrences Over Time")
                                    
                                    # Create a timeline of occurrences
                                    fig = go.Figure()
                                    
                                    # Add markers for each occurrence
                                    fig.add_trace(go.Scatter(
                                        x=patterns_df['date'],
                                        y=[1] * len(patterns_df),
                                        mode='markers',
                                        marker=dict(
                                            symbol='circle',
                                            size=10,
                                            color='blue'
                                        ),
                                        name='Pattern Occurrence'
                                    ))
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title='Timeline of Pattern Occurrences',
                                        height=200,
                                        margin=dict(l=0, r=0, t=30, b=0),
                                        yaxis=dict(
                                            showticklabels=False,
                                            title='',
                                            zeroline=False
                                        ),
                                        xaxis=dict(title='Date')
                                    )
                                    
                                    # Display the chart
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Calculate and display the average time between occurrences
                                    if len(patterns_df) > 1:
                                        dates = sorted(patterns_df['date'])
                                        time_diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                                        avg_days = sum(time_diffs) / len(time_diffs)
                                        
                                        st.info(f"This pattern occurs approximately every {avg_days:.1f} days on average.")
                                        
                                # Analyze subsequent returns after this pattern
                                st.subheader("What Usually Happens After This Pattern?")
                                
                                if 'date' in patterns_df.columns:
                                    # Calculate subsequent returns for different time periods
                                    subsequent_returns = []
                                    
                                    for _, row in patterns_df.iterrows():
                                        pattern_date = pd.to_datetime(row['date'])
                                        try:
                                            # Find index of pattern date
                                            idx = df.index.get_loc(pattern_date)
                                            
                                            # Calculate forward returns if we have enough data
                                            if idx + 20 < len(df):
                                                next_day = df['Return'].iloc[idx+1] if idx+1 < len(df) else np.nan
                                                next_5d = np.prod(1 + df['Return'].iloc[idx+1:idx+6]).item() - 1 if idx+6 < len(df) else np.nan
                                                next_10d = np.prod(1 + df['Return'].iloc[idx+1:idx+11]).item() - 1 if idx+11 < len(df) else np.nan
                                                next_20d = np.prod(1 + df['Return'].iloc[idx+1:idx+21]).item() - 1 if idx+21 < len(df) else np.nan
                                                
                                                subsequent_returns.append({
                                                    'date': pattern_date,
                                                    'next_day': next_day,
                                                    'next_5d': next_5d,
                                                    'next_10d': next_10d,
                                                    'next_20d': next_20d
                                                })
                                        except:
                                            # Skip if date not found in the dataframe
                                            pass
                                    
                                    if subsequent_returns:
                                        # Convert to dataframe
                                        returns_df = pd.DataFrame(subsequent_returns)
                                        
                                        # Calculate statistics
                                        returns_stats = {
                                            'mean': returns_df[['next_day', 'next_5d', 'next_10d', 'next_20d']].mean(),
                                            'median': returns_df[['next_day', 'next_5d', 'next_10d', 'next_20d']].median(),
                                            'positive': (returns_df[['next_day', 'next_5d', 'next_10d', 'next_20d']] > 0).mean()
                                        }
                                        
                                        # Create a dataframe for display
                                        stats_df = pd.DataFrame({
                                            'Avg Return': returns_stats['mean'],
                                            'Median Return': returns_stats['median'],
                                            'Win Rate': returns_stats['positive']
                                        })
                                        
                                        # Format values
                                        for col in ['Avg Return', 'Median Return']:
                                            stats_df[col] = stats_df[col].apply(lambda x: f"{x:.2%}")
                                        
                                        stats_df['Win Rate'] = stats_df['Win Rate'].apply(lambda x: f"{x:.1%}")
                                        
                                        # Rename index
                                        stats_df.index = ['1 Day', '5 Days', '10 Days', '20 Days']
                                        
                                        # Display statistics
                                        st.dataframe(stats_df, use_container_width=True)
                                        
                                        # Plot distribution of returns
                                        st.subheader("Distribution of Returns After Pattern")
                                        
                                        # Create a dataframe for plotting
                                        return_periods = ['next_day', 'next_5d', 'next_10d', 'next_20d']
                                        period_names = ['1 Day', '5 Days', '10 Days', '20 Days']
                                        
                                        # Select which period to plot
                                        selected_period = st.selectbox(
                                            "Select return period to view:",
                                            options=range(len(return_periods)),
                                            format_func=lambda i: period_names[i]
                                        )
                                        
                                        period_col = return_periods[selected_period]
                                        period_name = period_names[selected_period]
                                        
                                        # Create histogram
                                        fig = px.histogram(
                                            returns_df,
                                            x=period_col,
                                            nbins=15,
                                            title=f"Distribution of {period_name} Returns After Pattern",
                                            labels={period_col: f"{period_name} Return"},
                                            color_discrete_sequence=['skyblue']
                                        )
                                        
                                        # Add vertical line for average
                                        fig.add_vline(
                                            x=returns_df[period_col].mean(),
                                            line_dash="dash",
                                            line_color="red",
                                            annotation_text=f"Mean: {returns_df[period_col].mean():.2%}"
                                        )
                                        
                                        # Display the chart
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        avg_return = returns_df[period_col].mean()
                                        win_rate = (returns_df[period_col] > 0).mean()
                                        
                                        if avg_return > 0 and win_rate > 0.5:
                                            st.success(f"This pattern tends to be bullish over the next {period_name.lower()}, with a {win_rate:.1%} win rate and {avg_return:.2%} average return.")
                                        elif avg_return < 0 and win_rate < 0.5:
                                            st.warning(f"This pattern tends to be bearish over the next {period_name.lower()}, with a {win_rate:.1%} win rate and {avg_return:.2%} average return.")
                                        else:
                                            st.info(f"This pattern shows mixed performance over the next {period_name.lower()}, with a {win_rate:.1%} win rate and {avg_return:.2%} average return.")
                        else:
                            st.warning(f"No patterns found in group {selected_group+1}")
                    else:
                        st.warning("No representative patterns found in the vector database.")
                else:
                    st.warning("No pattern groups found in the vector database.")
            else:
                st.warning("The vector database doesn't contain pattern groups. Try rebuilding the database.")
        else:
            st.warning(f"No vector database found for {ticker}. Please create one first by going to the Vector Database tab.")
            
            # Show a guide on how to create a vector database
            st.info("""
            To create a vector database:
            1. Go to the "Vector Database" tab
            2. The database will be automatically created
            3. Come back to this tab to explore pattern groups
            """)
            
            # Add a button to go to the Vector Database tab
            if st.button("Go to Vector Database Tab"):
                st.experimental_set_query_params(tab="vector_db")

else:
    st.warning("Please enter a valid ticker symbol to begin analysis.")
    
    # Display some example tickers
    st.info("""
    Try these popular ticker symbols:
    - AAPL (Apple)
    - MSFT (Microsoft)
    - AMZN (Amazon)
    - GOOGL (Alphabet/Google)
    - TSLA (Tesla)
    - SPY (S&P 500 ETF)
    """)

# Add footer
st.markdown("---")
st.markdown("""
**Stock Price-Volume Behavior Explorer** | 
Data provided by Yahoo Finance through yfinance | 
Analysis uses KMeans clustering and PCA dimensionality reduction
""")

def main():
    """Entry point for the application when used as a module."""
    # This function is used as an entry point when the app is installed as a package
    pass

if __name__ == "__main__":
    # This code only runs when the file is executed directly, not when imported
    main()
