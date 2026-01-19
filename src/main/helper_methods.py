import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, RobustScaler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window

target_col = "ArrDelay"
important_features_post_dt_testing = ['DepDelay', 'TaxiOut', 'CRSDepTime_minutes_cosine', 'DepTime', 'CRSDepTime',
                                      'CRSDepTime_minutes_sine']
categorical_columns_plane = ['type', 'manufacturer', 'model', 'aircraft_type', 'engine_type', 'year_plane']
new_features_names = ['DepTime_TOD', 'CRSDepTime_TOD', 'CRSArrTime_TOD', 'Weekend', 'TimeBetweenDepartures']
total_categorical_features = categorical_columns_plane + new_features_names


def get_data_without_forbidden_variables_sampled(data, sample_fraction=0.3):
    forbidden_variables = ['ArrTime',
                           'ActualElapsedTime',
                           'AirTime',
                           'TaxiIn',
                           'Diverted',
                           'CarrierDelay',
                           'WeatherDelay',
                           'NASDelay',
                           'SecurityDelay',
                           'LateAircraftDelay']

    full_data = data.drop(*forbidden_variables)

    # If full_data is empty, return both as empty DataFrames
    try:
        total_rows = full_data.count()
    except Exception:
        total_rows = 0

    if total_rows == 0:
        return full_data, full_data

    sampled_data = full_data.sample(fraction=sample_fraction, seed=42)

    # If sampled is empty (small datasets), fall back to full_data
    try:
        sampled_rows = sampled_data.count()
    except Exception:
        sampled_rows = 0

    if sampled_rows == 0:
        sampled_data = full_data

    return sampled_data, full_data


def get_numeric_columns(flights_data):
    numeric_fields = [column for column in flights_data.columns if
                      ('integer' in str(flights_data.select(column).schema).lower()) and (
                              flights_data.select(column).distinct().count() > 30)]
    return numeric_fields


def get_data_distribution(flights_data):
    numeric_fields = get_numeric_columns(flights_data)
    quantiles = {
        fld: [flights_data.where(f.col(fld).isNull()).count()] +
             [flights_data.corr(fld, 'ArrDelay')] +
        flights_data.approxQuantile(fld, [0.05, 0.25, 0.5, 0.75, 0.95], 0) for fld in numeric_fields}

    dfq = pd.DataFrame(quantiles)
    dfq['summary'] = np.array(
        ['nulls', 'corr', 'quantile_05', 'quantile_25', 'quantile_50', 'quantile_75', 'quantile_95'])

    cols = dfq.columns.tolist()
    cols = [cols[-1]] + cols[1:-1]
    finalDf = dfq[cols]
    return finalDf


def get_correlation_matrix_arrival_delay(flights_data):
    numeric_column_names = get_numeric_columns(flights_data)
    data_to_correlate = flights_data.select(numeric_column_names).sample(0.1).toPandas()
    plt.figure()
    corr = data_to_correlate.corr()

    ax = sb.heatmap(corr, annot=True)
    plt.title("Correlation matrix for numeric column names", fontsize=20)
    plt.show()


def get_numeric_and_categorical_columns(df):
    numeric_column_names = get_numeric_columns(df)
    numeric_column_names = numeric_column_names[1:]
    all_columns = df.columns
    categorical_columns = [col for col in all_columns if col not in numeric_column_names]
    return numeric_column_names, categorical_columns


def get_stacked_bar_for_quantiles(distribution_df_for_plot):
    distribution_df_for_plot = distribution_df_for_plot.drop([0, 1])

    # Extracting the selected and other columns
    selected_columns_delay = ['ArrDelay', 'DepDelay']
    selected_columns_dep_delay = ['DepDelay']
    selected_columns_dep_time = ['DepTime', 'CRSDepTime', 'CRSArrTime']
    selected_columns_distance = ['Distance']
    selected_columns_taxi_out = ['TaxiOut']

    # Rename the Quantile labels
    quantile_labels = ['Quantile 05', 'Quantile 25', 'Quantile 50', 'Quantile 75', 'Quantile 95']

    # Set custom colors for each quantile
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantile_labels)))

    # Plotting for 'ArrDelay'
    ax1 = distribution_df_for_plot[selected_columns_delay].transpose().plot(kind='bar', stacked=True, color=colors,
                                                                            figsize=(10, 6))
    ax1.set_ylabel('Quantile Values')
    ax1.set_xlabel('Columns')
    ax1.set_title('Quantile Distribution for Delay')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.legend(quantile_labels, title='Quantiles', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

    # Plotting for 'DepTime', 'CRSDepTime', and 'CRSArrTime'
    ax3 = distribution_df_for_plot[selected_columns_dep_time].transpose().plot(kind='bar', stacked=True, color=colors,
                                                                               figsize=(10, 6))
    ax3.set_ylabel('Quantile Values')
    ax3.set_xlabel('Columns')
    ax3.set_title('Quantile Distribution for DepTime, CRSDepTime, and CRSArrTime')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.legend(quantile_labels, title='Quantiles', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def get_relevant_categorical_variables(flights_data):
    relevant_categorical_variables = [c for c in flights_data.columns if
                                      (flights_data.select(c).distinct().count() <= 30) \
                                      and ('Cancelled' not in c) and (
                                              flights_data.select(c).distinct().count() > 1)] + ['ArrDelay']
    return relevant_categorical_variables


def plot_mean_delay_for_relevant_categorical_variables(relevant_categorical_values, data):
    fig = plt.figure(figsize=(15, 12))
    fig.subplots_adjust(hspace=1.1, wspace=0.2)
    p = 1

    for fld in relevant_categorical_values[:-1]:
        ax = fig.add_subplot(5, 2, p)

        # Aggregate to calculate mean for each group
        gdf = data.groupby(fld).agg(f.mean('ArrDelay').alias('MeanArrDelay'))

        # Check condition and plot if necessary
        if gdf.selectExpr('max(MeanArrDelay) / min(MeanArrDelay)').first()[0] > 1.3:
            sb.barplot(x=fld, y='MeanArrDelay', data=gdf.toPandas(), ax=ax)
            plt.xticks(rotation=70)
            p += 1

    plt.suptitle('Mean Delay for Categorical variables', fontsize=21)
    plt.show()


def get_duplicate_flights(some_data):
    window_spec = Window.partitionBy("unique_id").orderBy("unique_id")
    duplicate_rows = (
        some_data
        .withColumn("row_number", f.row_number().over(window_spec))
        .filter("row_number > 1")
        .drop("row_number")
    )
    return duplicate_rows


def extract_feature_importance(feature_imp, dataset, features_col):  # from Predicting Flight Delay project
    # Extract feature indices and importance scores
    feature_indices = range(len(feature_imp))
    feature_scores = feature_imp.toArray()

    # Create a DataFrame with feature indices and scores
    varlist = pd.DataFrame({'idx': feature_indices, 'score': feature_scores})

    # Merge with metadata to get feature names
    for i in dataset.schema[features_col].metadata["ml_attr"]["attrs"]:
        varlist = pd.merge(varlist, pd.DataFrame(dataset.schema[features_col].metadata["ml_attr"]["attrs"][i]),
                           how='left', left_on='idx', right_on='idx')

    return varlist.sort_values('score', ascending=False)


def convert_table_to_readable(feature_table, numeric_columns):
    feature_table['suffix'] = feature_table['name_x'].apply(
        lambda x: int(x.split('_')[-1]) if isinstance(x, str) and x.startswith('scaledFeatures_') else None)
    valid_rows = feature_table['suffix'].notna()
    feature_table['suffix'] = feature_table['suffix'].fillna(-1).astype(int)
    feature_table.loc[valid_rows, 'numeric_column'] = feature_table.loc[valid_rows, 'suffix'].apply(
        lambda x: numeric_columns[x] if 0 <= x < len(numeric_columns) else None)
    feature_table['numeric_column'].fillna(feature_table['name_x'], inplace=True)
    feature_table['numeric_column'].fillna(feature_table['name_y'], inplace=True)

    # Drop unnecessary columns and reset indices
    feature_table.drop(['suffix', 'name_x', 'idx', "name_y"], axis=1, inplace=True)
    feature_table.reset_index(drop=True, inplace=True)

    return feature_table


def get_plane_dataset(spark, plane_data_path):
    plane_data = (spark.read
                  .option("header", "true")
                  .option("inferSchema", True)
                  .option("nullValue", "NA")
                  .csv(plane_data_path))
    return plane_data


def initialize_spark_session():
    import sys
    print("Initializing Spark session...")

    # Warn if Python version may be unsupported by the installed PySpark / Spark build
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 11:
        print(f"Warning: You are running Python {py_ver.major}.{py_ver.minor}. PySpark is best supported on Python 3.7-3.10;"
              " consider using Python 3.9 or 3.10 if you hit JVM/py4j errors.")

    # Use modest local memory defaults so SparkSession starts on typical developer machines
    try:
        spark = (SparkSession.builder
                 .appName('FlightDelayPredictionApp')
                 .config("spark.executor.memory", "2g")
                 .config("spark.driver.memory", "1g")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        print("Spark session initialized!")
        return spark
    except Exception as e:
        # Provide an actionable error message and re-raise
        print("Failed to initialize Spark session. This often happens when PySpark and the local JVM/Spark"
              " are incompatible or when Python version is unsupported. Error:")
        print(str(e))
        raise


def create_pipeline():
    stages = []

    for feature in total_categorical_features:
        indexer = StringIndexer(inputCol=feature, outputCol=feature + '_index')
        indexer.setHandleInvalid("keep")

        one_hot_encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[feature + '_ONEHOT'])
        stages += [indexer, one_hot_encoder]

    # Scale numerical features
    percent_assembler = VectorAssembler(inputCols=important_features_post_dt_testing, outputCol="COMBINED_vec")
    percent_assembler.setHandleInvalid("skip")

    scaler = RobustScaler(inputCol="COMBINED_vec", outputCol="scaledFeatures",
                          withScaling=True, withCentering=False,
                          lower=0.25, upper=0.75)
    stages += [percent_assembler, scaler]

    # Feature assembler
    assembler_inputs = [feature + "_ONEHOT" for feature in total_categorical_features] + ['scaledFeatures']

    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    stages += [assembler]

    pipeline = Pipeline(stages=stages)
    return pipeline


def train_model(data, pipeline_model, output_path):
    transformed_data = pipeline_model.transform(data)
    train_ratio = 0.9
    validation_ratio = 1 - train_ratio

    # Be defensive for small datasets: if transformed_data is small, avoid sampling which can produce empty splits
    try:
        total_rows = transformed_data.count()
    except Exception:
        total_rows = 0

    if total_rows == 0:
        print("No rows after pipeline transform; skipping model training.")
        return None, None

    if total_rows < 50:
        # use the full transformed data to split deterministically when dataset is small
        train_data, validation_data = transformed_data.randomSplit([train_ratio, validation_ratio], seed=42)
    else:
        train_data, validation_data = transformed_data.sample(0.5, seed=42).randomSplit([train_ratio, validation_ratio], seed=42)
    dt = DecisionTreeRegressor(labelCol=target_col, featuresCol="features", maxDepth=15, maxBins=60, seed=42)

    try:
        n = train_data.count()
        print(f"Train data count: {n}")
        print("Train data sample (top 5):")
        train_data.show(5, truncate=False)
        print("Train data columns: ", train_data.columns)
    except:
        pass

    # If train_data is empty after splits, fall back to using transformed_data as both train and validation
    try:
        n = train_data.count()
    except Exception:
        n = 0

    if n == 0:
        print("Train split is empty; falling back to using the full transformed dataset for training.")
        train_data = transformed_data
        validation_data = transformed_data

    # Ensure we have rows to fit on
    try:
        fit_rows = train_data.select('features', target_col).na.drop().count()
    except Exception:
        fit_rows = 0

    if fit_rows == 0:
        # Can't train a model; produce constant predictions using mean ArrDelay if available
        try:
            mean_delay = transformed_data.agg(f.avg(target_col)).first()[0]
            if mean_delay is None:
                mean_delay = 0.0
        except Exception:
            mean_delay = 0.0
        predictions = validation_data.withColumn('prediction', f.lit(float(mean_delay)))
        print(f"Unable to train model due to insufficient rows; returning constant prediction = {mean_delay}")
        return predictions, None

    dt_model = dt.fit(train_data.select(['features', target_col]))
    predictions = dt_model.transform(validation_data)
    return predictions, dt_model


def show_results(predictions):
    evaluator_mae = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")
    evaluator_rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
    # Defensive: if predictions DataFrame is empty or lacks the prediction column, skip evaluation
    try:
        if predictions is None:
            print("No predictions to evaluate (predictions is None).")
            return
        # ensure prediction and label columns exist
        cols = predictions.columns
        if target_col not in cols or "prediction" not in cols:
            print(f"Predictions DataFrame missing required columns: {target_col} or 'prediction'. Columns: {cols}")
            return
        count = predictions.count()
        if count == 0:
            print("Predictions DataFrame is empty; skipping evaluation.")
            return

        mae_dt = evaluator_mae.evaluate(predictions)
        print(f"Mean Absolute Error (MAE) on validation data (Decision Tree): {mae_dt}")
        rmse_dt = evaluator_rmse.evaluate(predictions)
        print(f"Root Mean Squared Error (RMSE) on validation data (Decision Tree): {rmse_dt}")
    except Exception as e:
        print("Failed to evaluate predictions:", e)


def load_data(spark, input_file):
    reader = spark.read.option("header", "true").option("inferSchema", True).option("nullValue", "NA")

    # If the input file is bzip2 compressed, set compression option; otherwise read normally
    if isinstance(input_file, str) and input_file.lower().endswith('.bz2'):
        reader = reader.option("compression", "bzip2")

    data = reader.csv(input_file)
    return data
