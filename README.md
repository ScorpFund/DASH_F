# Market Pattern Explorer

A sophisticated stock analysis tool that leverages machine learning for advanced stock pattern clustering and visualization. This application provides in-depth analysis of stock price-volume behaviors with vector database similarity search, pattern grouping, backtesting, and enhanced exploratory data analysis.

![Market Pattern Explorer](./assets/screenshot.png)

## Features

- **Advanced Pattern Analysis**: Discover hidden relationships in market behavior through machine learning clustering
- **Vector Database Search**: Find similar trading days based on various technical indicators
- **Pattern Groups**: Identify and analyze recurring 30-day price-volume patterns in the market
- **Pattern Backtesting**: Test trading strategies based on technical patterns with detailed performance metrics
- **Price Alerts**: Set up technical indicator thresholds for potential trading signals
- **Date-Based Search**: Input any trading date to find similar patterns throughout history
- **Interactive Visualizations**: Explore patterns with PCA projections, correlation analysis, and 3D clustering
- **Professional UI**: Sleek Apple-inspired design with interactive elements and smooth animations
- **Technical Indicators**: Over 15 indicators including RSI, MACD, Bollinger Bands, and volatility measures

## Installation

### Option 1: Clone from GitHub (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/yourusername/market-pattern-explorer.git
cd market-pattern-explorer
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r github_requirements.txt
```

3. Initialize the project structure:
```bash
python init_project.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

### Option 2: Install as a Package

You can also install the Market Pattern Explorer as a Python package:

```bash
pip install git+https://github.com/yourusername/market-pattern-explorer.git
```

Then run the app:

```bash
streamlit run -m market_pattern_explorer.app
```

### Option 3: Deploy on Streamlit Cloud

This repository is compatible with [Streamlit Cloud](https://streamlit.io/cloud). Simply point your Streamlit Cloud deployment to this GitHub repository to deploy the app online.

## Usage

1. Enter a stock ticker symbol (default examples are provided)
2. Select a time period for analysis
3. Choose clustering parameters (algorithm, features)
4. Explore the different visualization tabs:
   - **3D Visualization**: Explore trading days in 3D space with price-volume-behavior clustering
   - **Cluster Analysis**: Analyze the characteristics of each market behavior cluster
   - **Data Table**: View detailed data of all trading days with technical indicators
   - **Vector Database**: Find similar trading days to any selected date using vector similarity search
   - **Technical Dashboard**: Comprehensive view of technical indicators with interpretations
   - **Predictive Analytics**: Forecast potential future price movements based on similar patterns
   - **Pattern Backtesting**: Test trading strategies based on technical patterns and indicators
   - **Pattern Groups**: Discover and analyze recurring 30-day price-volume patterns

### Example Data

The repository includes example embedding data for several stocks that you can use immediately:
- Apple (AAPL)
- Google (GOOGL)
- NVIDIA (NVDA)
- Tata Motors (TATAMOTORS.NS)
- Tata Consultancy Services (TCS.NS)

These example files will allow you to explore the vector database functionality without having to generate data from scratch.

## Requirements

- Python 3.8+
- streamlit
- yfinance
- pandas
- numpy
- scikit-learn
- plotly
- faiss-cpu
- umap-learn

## Project Structure

```
market-pattern-explorer/
├── app.py                   # Main Streamlit application
├── init_project.py          # Project initialization script
├── github_requirements.txt  # Python dependencies
├── setup.py                 # Python package setup script
├── MANIFEST.in              # Includes non-Python files in the package
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── .streamlit/              # Streamlit configuration
│   ├── config.toml          # App configuration
│   └── style.css            # Custom CSS styling
├── vector_db/               # Vector database storage
│   ├── .gitkeep             # Ensures directory is tracked by Git
│   └── *_embedding_data.json # Example data files
├── assets/                  # Images and static files
│   ├── screenshot.png       # Screenshot for README preview
│   └── SCREENSHOT_INSTRUCTIONS.md # Instructions for adding screenshot
└── .gitignore               # Git ignore file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.