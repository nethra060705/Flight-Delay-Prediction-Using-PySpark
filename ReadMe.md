# Flight Delay Prediction Using PySpark

A machine learning project for predicting flight arrival delays using Apache Spark and PySpark, trained on the U.S. Department of Transportation 2007 airline dataset.

## Overview

This project implements a distributed machine learning pipeline to predict flight arrival delays with a Mean Absolute Error (MAE) of approximately 8 minutes. The solution leverages PySpark's MLlib for scalable data processing and model training, incorporating advanced feature engineering techniques to capture temporal patterns and aircraft characteristics.

## Dataset

The project uses two primary datasets:

- **U.S. Department of Transportation 2007 Airline Dataset**: Contains 29 variables describing commercial flight activity, including scheduled/actual times, delays, flight numbers, and route information.
- **Aircraft Dataset (`plane-data.csv`)**: Enriches flight data with aircraft metadata including type, manufacturer, model, and year of manufacture, merged via the `TailNum` identifier.

### Forbidden Variables

To prevent data leakage and ensure realistic predictions, the following post-event variables are excluded:
- `ArrTime`
- `ActualElapsedTime`
- `AirTime`
- `TaxiIn`
- `Diverted`
- `CarrierDelay`
- `WeatherDelay`
- `NASDelay`
- `SecurityDelay`
- `LateAircraftDelay`

## Data Preprocessing

### Cleaning Pipeline

- **Null Handling**: Rows with missing target variable (`ArrDelay`) are removed
- **Duplicate Removal**: Unique flight identifiers created using `Month`, `DayOfWeek`, `FlightNum`, and other descriptors to eliminate duplicates
- **Time Parsing**: Numeric HHMM format converted to datetime, then transformed into cyclic features using sine/cosine functions to preserve temporal continuity
- **Scaling**: `RobustScaler` applied to numerical features to minimize outlier impact
- **Categorical Encoding**: `StringIndexer` followed by `OneHotEncoder` for all categorical variables

### Merge Statistics

After merging with aircraft metadata, approximately 12.6% of rows were dropped due to missing tail numbers.

![6.png](/img/6.png)

## Feature Engineering

### Custom Features

1. **Time of Day Binning**: Flight times categorized into periods (morning, afternoon, evening, night)
   - `DepTime_TOD`
   - `CRSDepTime_TOD`
   - `CRSArrTime_TOD`

2. **Weekend Indicator**: Binary feature derived from `DayOfWeek` to capture weekday/weekend patterns

3. **Time Between Departures**: Categorical feature estimating flight spacing
   - `NOT_ENOUGH` (≤30 minutes)
   - `BARELY_ENOUGH` (30-60 minutes)
   - `ENOUGH` (60-120 minutes)
   - `MORE_THAN_ENOUGH` (>120 minutes)

### Cyclical Feature Transformation

Temporal features converted to polar coordinates using sine/cosine encoding:
- `DayofMonth`, `Month`, `DayOfWeek`
- `DepTime`, `CRSDepTime`, `CRSArrTime`

## Feature Selection

A Decision Tree Regressor was used for initial feature importance assessment. Top predictors identified:

1. `DepDelay` (Departure Delay)
2. `TaxiOut` (Taxi Out Time)
3. `DepTime` (Actual Departure Time)
4. `CRSDepTime` (Scheduled Departure Time)
5. `CRSDepTime_minutes_sine` and `cosine` (Cyclical time features)

Correlation analysis confirmed strong relationships between scheduled/actual times and departure/arrival delays.

![1.png](/img/1.png)

## Machine Learning Models

### Model Comparison

**Linear Regression (Baseline)**
- MAE: ~8.6 minutes
- RMSE: ~11.8 minutes

**Decision Tree Regressor (Tuned)**
- MAE: 8.07 minutes
- RMSE: 12.87 minutes
- Improved after feature selection and hyperparameter tuning

### Hyperparameter Tuning

3-fold cross-validation with grid search:
- `maxDepth`: [5, 10, 15]
- `maxBins`: [20, 40, 60]

**Best Configuration**: `maxDepth=15`, `maxBins=60`

![16.png](/img/16.png)

## Performance Summary

The final model achieves:
- **MAE**: 8.07 minutes
- **RMSE**: 12.87 minutes

The model demonstrates a strong balance between complexity and interpretability, with departure delay and taxi-out time emerging as the strongest predictors.

## Project Structure

```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.sh                          # Environment setup script (macOS)
├── spark_submit_script.sh            # Spark job submission script (Unix)
├── spark_submit_script.bat           # Spark job submission script (Windows)
├── conf/
│   ├── log4j.properties              # Logging configuration
│   └── spark-defaults.conf           # Spark configuration
├── img/                              # Visualization images
│   ├── 1.png                         # Correlation matrix
│   ├── 6.png                         # Data merge statistics
│   └── 16.png                        # Hyperparameter tuning results
├── notebook/
│   └── Model.ipynb                   # Jupyter notebook with EDA and modeling
├── output_dir_large/                 # Predictions for large dataset
├── output_full/                      # Full dataset predictions
├── output_full_t10/                  # Predictions with threshold=10
├── resources/
│   ├── large_flights_500k.csv        # Training dataset (500k rows)
│   ├── large_flights_test.csv        # Test dataset
│   └── ...
├── src/
│   └── main/
│       ├── main.py                   # Entry point
│       ├── helper_methods.py         # Spark utilities and model training
│       ├── dataset_utils.py          # Data preparation and transformation
│       ├── custom_features.py        # Feature engineering functions
│       └── dataset/
│           ├── airports.csv
│           ├── carriers.csv
│           ├── plane-data.csv
│           └── variable-descriptions.csv
└── tools/                            # Utility scripts
    ├── decompress_bz2.py
    ├── generate_flights.py
    ├── generate_report_figures_from_predictions.py
    └── generate_report_figures.py
```

## Requirements

- Python 3.9+
- Java 17+
- Apache Spark 3.5.0+
- Dependencies listed in `requirements.txt`:
  - `numpy~=1.26.4`
  - `pandas~=2.2.3`
  - `seaborn~=0.13.2`
  - `pyspark~=3.5.0`

## Installation

### macOS Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd predicting-flight-delay-pyspark-master
   ```

2. **Run the setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   The script will:
   - Install `pyenv` if not present
   - Install Python 3.10.15
   - Create a virtual environment
   - Install dependencies

3. **Activate the environment**
   ```bash
   source .venv310/bin/activate
   ```

### Manual Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Pipeline

**Using Spark Submit (Recommended)**
```bash
./spark_submit_script.sh [input_file] [output_dir]
```

**Direct Python Execution**
```bash
python src/main/main.py <input_file> <output_dir>
```

**Example**
```bash
python src/main/main.py resources/large_flights_500k.csv output_predictions/
```

### Additional Options

**No-Spark Mode** (Pandas-based smoke test)
```bash
python src/main/main.py resources/large_flights_500k.csv output/ --no-spark
```

**Count-Only Mode** (Display dataset sizes without training)
```bash
python src/main/main.py resources/large_flights_500k.csv output/ --count-only
```

**With Test File**
```bash
python src/main/main.py resources/large_flights_500k.csv output/ --test-file resources/large_flights_test.csv
```

### Jupyter Notebook

Explore the data and models interactively:
```bash
jupyter notebook notebook/Model.ipynb
```

## Output

The pipeline generates two CSV files in the specified output directory:
- `predictions.csv`: Predictions on the validation set
- `test_predictions.csv`: Predictions on the test set (if provided)

Each prediction includes:
- Original features
- `prediction`: Predicted arrival delay (minutes)
- `predicted_label`: Categorical label (`early`, `on time`, `delayed`)

## Key Findings

- **Departure delay** and **taxi-out time** are the strongest predictors of arrival delay
- **Time-based features** transformed into cyclic representations significantly improve model interpretability
- Integrating **aircraft metadata** adds valuable signal despite reducing sample size by ~12.6%
- **Tree-based models** excel at feature selection, while **linear regression** initially performs better
- The final model achieves a practical balance between complexity and interpretability with **MAE ~8 minutes**

## Visualizations

The `img/` directory contains key visualizations:

- **Correlation Matrix** (`1.png`): Relationships between numeric features
- **Merge Statistics** (`6.png`): Impact of aircraft data integration
- **Hyperparameter Tuning** (`16.png`): Cross-validation results

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Dataset provided by the U.S. Department of Transportation.
