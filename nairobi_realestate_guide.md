# Nairobi Real Estate Valuation Model — Complete Build & Execution Guide

---

## PREREQUISITES — Set Up First

### Tools to Install

| Tool | Why | How to Get |
|------|-----|------------|
| Python 3.10+ | Everything runs on this | python.org/downloads |
| VS Code | Your editor | Already installed |
| Git | Version control | git-scm.com |
| pip | Package manager | Comes with Python |
| virtualenv | Isolate dependencies | `pip install virtualenv` |

### Accounts to Create

| Platform | Why | URL |
|----------|-----|-----|
| GitHub | Host your code and portfolio | github.com |
| Streamlit Community Cloud | Free deployment | share.streamlit.io |
| Google Cloud (optional) | Geocoding API | console.cloud.google.com |

---

## PROJECT STRUCTURE

Create this folder structure before writing a single line of code:

```
nairobi-realestate/
│
├── data/
│   ├── raw/                  # Raw scraped data (never touch after saving)
│   ├── processed/            # Cleaned data
│   └── external/             # Geocoding results, reference data
│
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_modeling.ipynb
│
├── src/
│   ├── scraper.py
│   ├── cleaner.py
│   ├── features.py
│   └── model.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── final_model.pkl       # Saved trained model
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## PHASE 1 — ENVIRONMENT SETUP

### Step 1: Initialize the project

```bash
mkdir nairobi-realestate
cd nairobi-realestate
git init
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

### Step 2: Install all dependencies at once

```bash
pip install requests beautifulsoup4 selenium scrapy \
            pandas numpy \
            scikit-learn statsmodels \
            matplotlib seaborn plotly \
            geopy \
            streamlit \
            joblib \
            jupyter \
            lxml
```

### Step 3: Save your dependencies

```bash
pip freeze > requirements.txt
```

### Step 4: Create .gitignore

```
venv/
__pycache__/
*.pyc
data/raw/
*.pkl
.env
.DS_Store
```

> **Why exclude data/raw from git?** Raw scraped data can be large and contains no code. Store it locally only.

---

## PHASE 2 — DATA SCRAPING (Week 1)

### Understanding the Target

BuyRentKenya (buyrentkenya.com) lists properties with:
- Title, price, location
- Bedrooms, bathrooms, size (sq ft or sq m)
- Amenities and description
- Property type

### Step 1: Inspect the website first

Before writing code:
1. Open buyrentkenya.com in Chrome
2. Go to listings (e.g., apartments for sale in Nairobi)
3. Right-click → Inspect → Network tab
4. Scroll through listings and watch what HTTP requests fire
5. Check if the site loads data via JSON API (XHR requests) — if yes, hitting the API directly is faster and more stable than scraping HTML

### Step 2: Write the scraper

**File: `src/scraper.py`**

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BASE_URL = "https://www.buyrentkenya.com/property-for-sale/nairobi"

def get_listings_page(page_number):
    """Fetch one page of listings."""
    url = f"{BASE_URL}?page={page_number}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    
    if response.status_code != 200:
        print(f"Failed page {page_number}: {response.status_code}")
        return None
    
    return BeautifulSoup(response.content, "lxml")

def parse_listing_card(card):
    """Extract data from a single listing card."""
    data = {}
    
    try:
        # These selectors WILL need adjustment after you inspect the actual HTML
        data['title'] = card.find('h2', class_='listing-title').get_text(strip=True)
        data['price'] = card.find('span', class_='price').get_text(strip=True)
        data['location'] = card.find('span', class_='location').get_text(strip=True)
        data['bedrooms'] = card.find('span', class_='beds').get_text(strip=True)
        data['bathrooms'] = card.find('span', class_='baths').get_text(strip=True)
        data['size'] = card.find('span', class_='size').get_text(strip=True)
        data['property_type'] = card.find('span', class_='property-type').get_text(strip=True)
        data['listing_url'] = card.find('a')['href']
        data['scraped_at'] = datetime.now().isoformat()
    except AttributeError as e:
        print(f"Parse error: {e}")
        return None
    
    return data

def scrape_all_listings(max_pages=50):
    """Scrape multiple pages and return DataFrame."""
    all_listings = []
    
    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}...")
        soup = get_listings_page(page)
        
        if soup is None:
            break
        
        # Adjust selector to match actual HTML structure
        cards = soup.find_all('div', class_='listing-card')
        
        if not cards:
            print(f"No listings found on page {page}. Stopping.")
            break
        
        for card in cards:
            listing = parse_listing_card(card)
            if listing:
                all_listings.append(listing)
        
        # Polite scraping: wait between requests
        time.sleep(random.uniform(2, 4))
    
    return pd.DataFrame(all_listings)

if __name__ == "__main__":
    df = scrape_all_listings(max_pages=50)
    df.to_csv("data/raw/listings_raw.csv", index=False)
    print(f"Saved {len(df)} listings to data/raw/listings_raw.csv")
```

### Step 3: Run the scraper and verify

```bash
cd nairobi-realestate
source venv/bin/activate
python src/scraper.py
```

Then check the output:
```python
import pandas as pd
df = pd.read_csv("data/raw/listings_raw.csv")
print(df.shape)          # How many rows and columns
print(df.head())         # First 5 rows
print(df.isnull().sum()) # Count missing values per column
```

### Step 4: Handle scraping failures

If BuyRentKenya blocks you:
- Add longer delays (`time.sleep(random.uniform(5, 10))`)
- Rotate user agents (use `fake_useragent` library)
- Use Selenium if the site requires JavaScript rendering
- **Fallback option:** Download Kaggle's Nairobi housing dataset as backup

**Fallback dataset:**
Search "Nairobi house prices" on kaggle.com. Download, save to `data/raw/`, and proceed.

### Testing the scraper

- Run on 2-3 pages first before doing 50
- Print the soup object and check if the HTML matches what you see in DevTools
- If class names don't match, open the actual HTML and find the correct selectors

---

## PHASE 3 — DATA CLEANING (Week 2, Part 1)

**File: `src/cleaner.py`**

```python
import pandas as pd
import numpy as np
import re

def load_raw_data(filepath="data/raw/listings_raw.csv"):
    return pd.read_csv(filepath)

def clean_price(price_str):
    """Convert 'KES 12,500,000' to 12500000.0"""
    if pd.isna(price_str):
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(cleaned)
    except:
        return np.nan

def clean_size(size_str):
    """Convert '150 sq ft' or '150 sqm' to float in sq ft."""
    if pd.isna(size_str):
        return np.nan
    
    size_str = str(size_str).lower()
    number = re.findall(r'[\d.]+', size_str)
    
    if not number:
        return np.nan
    
    value = float(number[0])
    
    # Convert sqm to sqft if needed
    if 'sqm' in size_str or 'm²' in size_str or 'sq m' in size_str:
        value = value * 10.7639
    
    return value

def clean_bedrooms(bed_str):
    """Extract integer bedroom count."""
    if pd.isna(bed_str):
        return np.nan
    match = re.findall(r'\d+', str(bed_str))
    return int(match[0]) if match else np.nan

def standardize_location(location_str):
    """Normalize location names."""
    if pd.isna(location_str):
        return np.nan
    
    location = str(location_str).strip().title()
    
    # Map common variations
    location_map = {
        'Westlands': 'Westlands',
        'West Lands': 'Westlands',
        'Kilimani': 'Kilimani',
        'Karen': 'Karen',
        'Lavington': 'Lavington',
        'Kileleshwa': 'Kileleshwa',
        'Parklands': 'Parklands',
        'Runda': 'Runda',
        'Muthaiga': 'Muthaiga',
        'Thika Road': 'Thika Road',
        'Ngong Road': 'Ngong Road',
        'Langata': 'Langata',
        'Lang\'ata': 'Langata',
        'South C': 'South C',
        'South B': 'South B',
        'Eastleigh': 'Eastleigh',
        'Ruaka': 'Ruaka',
        'Ruiru': 'Ruiru',
    }
    
    for key, value in location_map.items():
        if key.lower() in location.lower():
            return value
    
    return location

def remove_outliers(df, column, lower_pct=0.01, upper_pct=0.99):
    """Remove extreme outliers using percentiles."""
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]

def clean_data(df):
    print(f"Starting with {len(df)} rows")
    
    # Clean each column
    df['price_kes'] = df['price'].apply(clean_price)
    df['size_sqft'] = df['size'].apply(clean_size)
    df['bedrooms'] = df['bedrooms'].apply(clean_bedrooms)
    df['location_clean'] = df['location'].apply(standardize_location)
    
    # Drop rows with missing price (target variable — non-negotiable)
    df = df.dropna(subset=['price_kes'])
    print(f"After dropping missing prices: {len(df)} rows")
    
    # Drop rows with price = 0 or negative
    df = df[df['price_kes'] > 0]
    
    # Remove extreme price outliers
    df = remove_outliers(df, 'price_kes')
    print(f"After removing price outliers: {len(df)} rows")
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['title', 'price_kes', 'location_clean'])
    print(f"After removing duplicates: {len(df)} rows")
    
    # Log-transform price (handles right skew)
    df['log_price'] = np.log(df['price_kes'])
    
    return df

if __name__ == "__main__":
    df = load_raw_data()
    df_clean = clean_data(df)
    df_clean.to_csv("data/processed/listings_clean.csv", index=False)
    print(f"\nSaved clean data: {len(df_clean)} rows")
    print(df_clean.dtypes)
```

### Test cleaning:

```bash
python src/cleaner.py
```

Check output:
```python
df = pd.read_csv("data/processed/listings_clean.csv")
print(df[['price_kes', 'size_sqft', 'bedrooms', 'location_clean']].describe())
print(df['location_clean'].value_counts().head(20))
```

---

## PHASE 4 — EDA (Exploratory Data Analysis) (Week 2, Part 2)

Do this in a Jupyter notebook: `notebooks/03_eda.ipynb`

```bash
jupyter notebook
```

Key plots to generate:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("data/processed/listings_clean.csv")

# 1. Price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['price_kes'], bins=50)
axes[0].set_title('Price Distribution (Raw)')
axes[1].hist(df['log_price'], bins=50)
axes[1].set_title('Price Distribution (Log-Transformed)')
plt.savefig('data/external/price_distribution.png')
plt.show()

# 2. Median price by neighborhood
neighborhood_prices = df.groupby('location_clean')['price_kes'].median().sort_values(ascending=False)
plt.figure(figsize=(14, 6))
neighborhood_prices.head(20).plot(kind='bar')
plt.title('Median Property Price by Neighborhood')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/external/price_by_neighborhood.png')
plt.show()

# 3. Price vs bedrooms
plt.figure(figsize=(8, 5))
df.boxplot(column='price_kes', by='bedrooms')
plt.title('Price by Bedrooms')
plt.savefig('data/external/price_vs_bedrooms.png')
plt.show()

# 4. Price vs size (scatter)
plt.figure(figsize=(8, 5))
plt.scatter(df['size_sqft'], df['price_kes'], alpha=0.4)
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (KES)')
plt.title('Price vs Size')
plt.savefig('data/external/price_vs_size.png')
plt.show()

# 5. Correlation matrix
numeric_cols = ['price_kes', 'size_sqft', 'bedrooms', 'bathrooms']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('data/external/correlation_matrix.png')
plt.show()
```

**What to look for:**
- Is price right-skewed? (Yes — always use log_price as your target)
- Which neighborhoods command highest prices?
- Does size correlate with price? (It should)
- Are there outliers distorting the scatter?

---

## PHASE 5 — FEATURE ENGINEERING (Week 3)

**File: `src/features.py`**

```python
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

# Nairobi CBD coordinates
CBD_COORDS = (-1.2864, 36.8172)

# Key SGR/infrastructure coordinates
INFRASTRUCTURE = {
    'Nairobi CBD': (-1.2864, 36.8172),
    'JKIA Airport': (-1.3192, 36.9275),
    'Westgate Mall': (-1.2634, 36.8026),
    'Two Rivers Mall': (-1.2093, 36.8126),
    'Thika Road (General)': (-1.2107, 36.8879),
}

def geocode_locations(locations, cache_file="data/external/geocode_cache.csv"):
    """
    Geocode location names to coordinates.
    Uses a cache file to avoid re-geocoding (saves API calls and time).
    """
    # Load existing cache if available
    try:
        cache = pd.read_csv(cache_file, index_col='location')
        cache_dict = cache.to_dict('index')
    except FileNotFoundError:
        cache_dict = {}
    
    geolocator = Nominatim(user_agent="nairobi_realestate_model")
    results = {}
    
    unique_locations = set(locations.dropna().unique())
    
    for loc in unique_locations:
        if loc in cache_dict:
            results[loc] = (cache_dict[loc]['lat'], cache_dict[loc]['lon'])
            continue
        
        try:
            query = f"{loc}, Nairobi, Kenya"
            location = geolocator.geocode(query, timeout=10)
            if location:
                results[loc] = (location.latitude, location.longitude)
                cache_dict[loc] = {'lat': location.latitude, 'lon': location.longitude}
            else:
                results[loc] = (np.nan, np.nan)
            
            time.sleep(1)  # Nominatim rate limit
        except Exception as e:
            print(f"Geocoding failed for {loc}: {e}")
            results[loc] = (np.nan, np.nan)
    
    # Save updated cache
    cache_df = pd.DataFrame.from_dict(cache_dict, orient='index')
    cache_df.index.name = 'location'
    cache_df.to_csv(cache_file)
    
    return results

def calculate_distance_to_cbd(lat, lon):
    """Distance from property to Nairobi CBD in km."""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    return geodesic((lat, lon), CBD_COORDS).km

def assign_neighborhood_tier(neighborhood):
    """
    Classify neighborhoods into tiers based on market knowledge.
    Tier 1 = Premium, Tier 2 = Mid-range, Tier 3 = Affordable
    """
    tiers = {
        1: ['Karen', 'Runda', 'Muthaiga', 'Kitisuru', 'Gigiri', 'Lavington', 'Westlands'],
        2: ['Kilimani', 'Kileleshwa', 'Parklands', 'Spring Valley', 'Loresho', 'Ridgeways'],
        3: ['Kasarani', 'Ruaka', 'Thika Road', 'Langata', 'South C', 'South B', 
            'Eastleigh', 'Umoja', 'Donholm', 'Komarock']
    }
    
    if pd.isna(neighborhood):
        return 2  # Default to mid-tier
    
    for tier, areas in tiers.items():
        if any(area.lower() in str(neighborhood).lower() for area in areas):
            return tier
    
    return 2  # Default

def engineer_features(df):
    print("Starting feature engineering...")
    
    # 1. Geocode locations
    print("Geocoding neighborhoods...")
    geocode_results = geocode_locations(df['location_clean'])
    
    df['lat'] = df['location_clean'].map(lambda x: geocode_results.get(x, (np.nan, np.nan))[0])
    df['lon'] = df['location_clean'].map(lambda x: geocode_results.get(x, (np.nan, np.nan))[1])
    
    # 2. Distance to CBD
    df['dist_to_cbd_km'] = df.apply(
        lambda row: calculate_distance_to_cbd(row['lat'], row['lon']), axis=1
    )
    
    # 3. Neighborhood tier
    df['neighborhood_tier'] = df['location_clean'].apply(assign_neighborhood_tier)
    
    # 4. Price per sq ft
    df['price_per_sqft'] = df['price_kes'] / df['size_sqft']
    
    # 5. Bedroom-to-bathroom ratio
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
    
    # 6. Size categories
    df['size_category'] = pd.cut(
        df['size_sqft'],
        bins=[0, 500, 1000, 2000, 5000, float('inf')],
        labels=['Studio/Small', 'Medium', 'Large', 'Very Large', 'Mansion']
    )
    
    # 7. One-hot encode property type and neighborhood tier
    df = pd.get_dummies(df, columns=['property_type', 'size_category'], drop_first=True)
    
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/listings_clean.csv")
    df_features = engineer_features(df)
    df_features.to_csv("data/processed/listings_features.csv", index=False)
    print("Saved to data/processed/listings_features.csv")
```

---

## PHASE 6 — MODELING (Week 4)

**File: `src/model.py`**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

FEATURES = [
    'size_sqft', 'bedrooms', 'bathrooms',
    'dist_to_cbd_km', 'neighborhood_tier',
    'bed_bath_ratio'
    # Add one-hot encoded columns dynamically
]

TARGET = 'log_price'  # Always model on log scale

def load_data(filepath="data/processed/listings_features.csv"):
    df = pd.read_csv(filepath)
    
    # Get all one-hot encoded columns
    ohe_cols = [col for col in df.columns if col.startswith('property_type_') 
                or col.startswith('size_category_')]
    
    feature_cols = FEATURES + ohe_cols
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, TARGET]
    
    return X, y, feature_cols

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate and print model performance metrics."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Convert log-scale RMSE to interpretable percentage error
    rmse_pct = (np.exp(rmse) - 1) * 100
    
    print(f"\n--- {model_name} ---")
    print(f"R²:        {r2:.4f}")
    print(f"RMSE (log):{rmse:.4f}")
    print(f"RMSE (%):  {rmse_pct:.1f}%")
    print(f"MAE (log): {mae:.4f}")
    
    return {'model': model_name, 'r2': r2, 'rmse': rmse, 'rmse_pct': rmse_pct}

def run_statsmodels_ols(X_train, y_train):
    """
    Run OLS via statsmodels for academic-grade output.
    This gives you p-values, confidence intervals, VIF.
    Use this for your report/dissertation writeup.
    """
    X_with_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_with_const).fit()
    print(model.summary())
    return model

def plot_residuals(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].scatter(y_pred, residuals, alpha=0.4)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name} — Residual Plot')
    
    axes[1].hist(residuals, bins=30)
    axes[1].set_title(f'{model_name} — Residual Distribution')
    
    plt.tight_layout()
    plt.savefig(f'data/external/residuals_{model_name.replace(" ", "_")}.png')
    plt.show()

def train_all_models():
    X, y, feature_cols = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    results = []
    
    # 1. OLS (statsmodels) — for academic output
    print("\n=== STATSMODELS OLS (Academic Output) ===")
    ols_sm = run_statsmodels_ols(X_train, y_train)
    
    # 2. sklearn OLS — for sklearn pipeline
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    results.append(evaluate_model(ols, X_test, y_test, "OLS Linear Regression"))
    plot_residuals(ols, X_test, y_test, "OLS")
    
    # 3. Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate_model(ridge, X_test, y_test, "Ridge Regression"))
    
    # 4. Lasso
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    results.append(evaluate_model(lasso, X_test, y_test, "Lasso Regression"))
    
    # 5. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))
    plot_residuals(rf, X_test, y_test, "Random_Forest")
    
    # Feature importance (Random Forest)
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances.sort_values(ascending=False).head(15).plot(kind='barh')
    plt.title('Feature Importance — Random Forest')
    plt.tight_layout()
    plt.savefig('data/external/feature_importance.png')
    plt.show()
    
    # 6. Compare models
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    print("\n=== MODEL COMPARISON ===")
    print(results_df)
    
    # Save best model
    best_model_name = results_df.iloc[0]['model']
    print(f"\nBest model: {best_model_name}")
    
    # Save Random Forest as final (usually wins on this type of data)
    joblib.dump(rf, 'models/final_model.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print("Saved model to models/final_model.pkl")
    
    return rf, feature_cols

if __name__ == "__main__":
    train_all_models()
```

### Run modeling:

```bash
python src/model.py
```

---

## PHASE 7 — STREAMLIT APP (Week 5)

**File: `app/streamlit_app.py`**

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load('models/final_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/listings_features.csv')

def predict_price(model, feature_cols, inputs):
    """Build feature vector and predict."""
    X = pd.DataFrame([inputs])
    
    # Add missing columns (one-hot encoded) as 0
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    
    X = X[feature_cols]
    
    log_pred = model.predict(X)[0]
    price = np.exp(log_pred)
    
    # Return price range (±15%)
    return price * 0.85, price, price * 1.15

def main():
    st.set_page_config(page_title="Nairobi Property Valuation", layout="wide")
    
    st.title("🏠 Nairobi Real Estate Valuation Model")
    st.markdown("Estimate property prices in Nairobi using machine learning and hedonic pricing methodology.")
    
    model, feature_cols = load_model()
    df = load_data()
    
    # Sidebar inputs
    st.sidebar.header("Property Details")
    
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 8, 2)
    size_sqft = st.sidebar.number_input("Size (sq ft)", min_value=100, max_value=20000, value=1200)
    
    neighborhood_tier = st.sidebar.selectbox(
        "Neighborhood Tier",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Premium (Karen, Runda, Westlands)", 
                                2: "Mid-range (Kilimani, Parklands)", 
                                3: "Affordable (Thika Road, Langata)"}[x]
    )
    
    dist_to_cbd = st.sidebar.slider("Distance to CBD (km)", 0.5, 30.0, 5.0, step=0.5)
    
    # Prediction
    inputs = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'size_sqft': size_sqft,
        'neighborhood_tier': neighborhood_tier,
        'dist_to_cbd_km': dist_to_cbd,
        'bed_bath_ratio': bedrooms / max(bathrooms, 1)
    }
    
    low, mid, high = predict_price(model, feature_cols, inputs)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Lower Estimate", f"KES {low:,.0f}")
    with col2:
        st.metric("Best Estimate", f"KES {mid:,.0f}", delta="Most likely")
    with col3:
        st.metric("Upper Estimate", f"KES {high:,.0f}")
    
    st.markdown("---")
    
    # Market analysis
    st.header("Market Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price by neighborhood
        neighborhood_data = df.groupby('location_clean')['price_kes'].median().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=neighborhood_data.values,
            y=neighborhood_data.index,
            orientation='h',
            title='Median Price by Neighborhood (Top 15)',
            labels={'x': 'Median Price (KES)', 'y': 'Neighborhood'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price vs size scatter
        sample = df.sample(min(500, len(df)))
        fig = px.scatter(
            sample,
            x='size_sqft',
            y='price_kes',
            color='neighborhood_tier',
            title='Price vs Size',
            labels={'size_sqft': 'Size (sq ft)', 'price_kes': 'Price (KES)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Dataset stats
    st.header("Dataset")
    st.write(f"Model trained on **{len(df):,} listings** scraped from BuyRentKenya")
    
    if st.checkbox("Show raw data sample"):
        st.dataframe(df[['location_clean', 'bedrooms', 'bathrooms', 'size_sqft', 'price_kes']].head(50))

if __name__ == "__main__":
    main()
```

### Test locally:

```bash
cd nairobi-realestate
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`

---

## PHASE 8 — DEPLOYMENT (Week 6)

### Step 1: Push to GitHub

```bash
# In your project root
git add .
git commit -m "Initial commit: Nairobi real estate valuation model"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/nairobi-realestate.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `nairobi-realestate`
5. Set main file: `app/streamlit_app.py`
6. Click Deploy

Your app will be live at: `https://YOUR_USERNAME-nairobi-realestate.streamlit.app`

### Step 3: Write a strong README.md

```markdown
# Nairobi Real Estate Valuation Model

A hedonic pricing model that estimates property values across Nairobi neighborhoods 
using machine learning.

## Live Demo
[Launch App](YOUR_STREAMLIT_URL)

## What It Does
- Scrapes property listings from BuyRentKenya
- Cleans and engineers features (location geocoding, distance to CBD, neighborhood tiers)
- Trains OLS, Ridge, Lasso, and Random Forest models
- Deploys an interactive price estimator via Streamlit

## Key Results
- Best model: Random Forest (R² = X.XX, RMSE = X%)
- Training data: ~XXX listings across XX Nairobi neighborhoods

## Tech Stack
Python | pandas | scikit-learn | statsmodels | geopy | Streamlit

## Methodology
Hedonic pricing: property value = f(structural attributes, location attributes)
Log-linear OLS as baseline for academic interpretability.

## How to Run
```bash
pip install -r requirements.txt
python src/scraper.py
python src/cleaner.py
python src/features.py
python src/model.py
streamlit run app/streamlit_app.py
```
```

---

## TESTING STRATEGY

At every phase, run these checks before moving to the next:

| Phase | Test |
|-------|------|
| Scraper | `df.shape` returns 500+ rows; no columns are 100% null |
| Cleaning | `df.isnull().sum()` shows <10% missing per key column |
| EDA | Price distribution is right-skewed (log_price is normal) |
| Features | `dist_to_cbd_km` has realistic values (0.5–25km range) |
| Model | R² > 0.60 on test set (below this, revisit features) |
| Streamlit | All sliders work, predictions are in realistic KES ranges |

---

## RESOURCES

| Resource | URL | What For |
|----------|-----|----------|
| BeautifulSoup Docs | crummy.com/software/BeautifulSoup/bs4/doc | Scraping reference |
| pandas Docs | pandas.pydata.org/docs | Data manipulation |
| scikit-learn User Guide | scikit-learn.org/stable/user_guide | Modeling |
| statsmodels Docs | statsmodels.org | OLS with p-values |
| geopy Docs | geopy.readthedocs.io | Geocoding |
| Streamlit Docs | docs.streamlit.io | App building |
| Streamlit Deployment | share.streamlit.io | Free hosting |
| Kaggle Nairobi Housing | kaggle.com (search "Nairobi house prices") | Fallback dataset |
| Hedonic Pricing Paper | Search "hedonic pricing real estate methodology" on Google Scholar | Academic grounding |

---

## COMMON ERRORS AND FIXES

| Error | Cause | Fix |
|-------|-------|-----|
| Scraper returns empty DataFrame | CSS selectors don't match site HTML | Open DevTools, re-inspect actual class names |
| 403 Forbidden | Site blocked your requests | Add delays, rotate User-Agent |
| Geocoding returns NaN | Nominatim can't resolve location string | Append ", Nairobi, Kenya" to query |
| R² below 0.4 | Too few features or too much noise | Add more location features, check for data leakage |
| Streamlit crash on model load | Model file path wrong | Use absolute paths with `os.path` |
| Memory error on large dataset | DataFrame too large | Sample 2000 rows for modeling if needed |

---

## WHAT SUCCESS LOOKS LIKE

- GitHub repo with clean commit history (not one giant commit)
- Live Streamlit app accessible via URL
- R² above 0.65 on test set
- README that explains what you built, why, and what you found
- At least one interesting finding (e.g., "distance to CBD explains X% of variance after controlling for size")

That last point is your differentiator. Anyone can run a Random Forest. Not everyone can interpret what the model says about the Nairobi market.

