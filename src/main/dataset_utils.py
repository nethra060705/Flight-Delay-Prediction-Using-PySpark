import math

from .custom_features import add_flight_time_window, add_weekend_indicator, add_enough_time_estimation
from .helper_methods import important_features_post_dt_testing, total_categorical_features, target_col

from pyspark.sql.functions import col
from pyspark.sql import functions as f
from pyspark.sql.window import Window


def get_percentage_of_flights_without_tailnum(plane_data, original_data):
    flights_tailnums = original_data.select("TailNum").distinct()
    plane_tailnums = plane_data.select("tailnum").distinct()
    missing_tailnums = flights_tailnums.subtract(plane_tailnums)
    missing_tailnums_list = [row.TailNum for row in missing_tailnums.collect()]

    flights_data_with_missing = original_data.filter(col("TailNum").isin(missing_tailnums_list))

    total_rows = original_data.count()
    missing_rows = flights_data_with_missing.count()
    percentage_missing = (missing_rows / total_rows) * 100

    print(f"Percentage of rows with missing tail numbers: {percentage_missing:.2f}%")


def add_new_custom_features(data):
    data = add_flight_time_window(data)
    data = add_weekend_indicator(data)
    data = add_enough_time_estimation(data)
    return data


def extend_original_dataset(original_dataset, plane_data):
    columns_to_drop = ["issue_date", "status"]
    plane_data_almost_clean = plane_data.drop(*columns_to_drop)

    threshold_missing_values = 2  # Drop rows with more than 2 missing values
    plane_data_almost_clean = plane_data_almost_clean.na.drop(thresh=(6 - threshold_missing_values))

    # just to see the size difference expected
    get_percentage_of_flights_without_tailnum(plane_data_almost_clean, original_dataset)

    # need to rename this column to be more specific
    plane_data_almost_clean = plane_data_almost_clean.withColumnRenamed("year", "year_plane")

    # merge datasets on tailnum==TailNum
    flights_data_extended_nonencoded = original_dataset.join(plane_data_almost_clean,
                                                             original_dataset.TailNum == plane_data_almost_clean.tailnum,
                                                             "inner")
    flights_data_extended_nonencoded = flights_data_extended_nonencoded.drop("tailnum")

    return flights_data_extended_nonencoded


def polar_coordinates(column):
    # Calculate angle in radians
    window_spec = Window().rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    max_value = f.max(column).over(window_spec)
    angle = 2.0 * math.pi * (f.col(column) - 1) / max_value + (math.pi / 2.0)

    # Create new columns for polar coordinates
    x_col = f.cos(angle).alias(column + "_polar_x")
    y_col = f.sin(angle).alias(column + "_polar_y")

    return x_col, y_col


def transform_cyclical_features_month_week(data):
    cyclical_columns = ["DayofMonth", "Month", "DayOfWeek"]
    transformed_data = data
    for column in cyclical_columns:
        x_col, y_col = polar_coordinates(column)
        transformed_data = transformed_data.withColumn(column + "_polar_x", x_col) \
            .withColumn(column + "_polar_y", y_col)
    return transformed_data


def transform_cyclical_features_localtime(data, time_column):
    # Keep rows where time column is present (non-null)
    data = data.filter(col(time_column).isNotNull())

    # Robust extraction: strip non-digits, handle empty strings, then compute hours/minutes
    cleaned = f.regexp_replace(f.col(time_column).cast('string'), '[^0-9]', '')
    as_int = f.when(cleaned == '', None).otherwise(cleaned.cast('int'))
    hours = (as_int / 100).cast('int')
    minutes = (as_int % 100).cast('int')

    # Convert to minutes since midnight (null-safe)
    minutes_since_midnight = f.when(as_int.isNotNull(), hours * 60 + minutes).otherwise(None)

    # Encode minutes and hours cyclically
    minutes_cosine = f.when(minutes_since_midnight.isNotNull(),
                            f.cos(2.0 * math.pi * minutes_since_midnight / 1440)).otherwise(0).alias(
        time_column + "_minutes_cosine")
    minutes_sine = f.when(minutes_since_midnight.isNotNull(),
                          f.sin(2.0 * math.pi * minutes_since_midnight / 1440)).otherwise(0).alias(
        time_column + "_minutes_sine")
    hours_cosine = f.when(hours.isNotNull(),
                          f.cos(2.0 * math.pi * hours / 24)).otherwise(0).alias(time_column + "_hours_cosine")
    hours_sine = f.when(hours.isNotNull(),
                        f.sin(2.0 * math.pi * hours / 24)).otherwise(0).alias(time_column + "_hours_sine")

    # Create new columns with the results
    return data.withColumn(time_column + "_minutes_cosine", minutes_cosine) \
        .withColumn(time_column + "_minutes_sine", minutes_sine) \
        .withColumn(time_column + "_hours_cosine", hours_cosine) \
        .withColumn(time_column + "_hours_sine", hours_sine)


def transform_cyclical_features_localtime_2(data):
    time_columns = ["DepTime", "CRSDepTime", "CRSArrTime"]
    cyclical_data_localtime = data
    for column in time_columns:
        cyclical_data_localtime = transform_cyclical_features_localtime(cyclical_data_localtime, column)

    return cyclical_data_localtime


# filter out rows with ArrDelay that are null and flights are known to be cancelled
def clean_data(some_data):
    some_data_cleaned = some_data.drop("Year", "CancellationCode")  # , "TailNum", "TaxiOut")
    some_data_cleaned = some_data_cleaned.filter((f.col("ArrDelay").isNotNull()) & (f.col("Cancelled") == 0))
    some_data_cleaned = some_data_cleaned.filter((f.col("Distance").isNotNull()))
    some_data_cleaned = some_data_cleaned.drop("Cancelled")
    some_data_cleaned = some_data_cleaned.dropDuplicates(["unique_id"])
    return some_data_cleaned


def append_unique_id(some_data):
    some_data_appended = some_data.withColumn(
        "unique_id",
        f.concat_ws("_", "Month", "DayofMonth", "DayOfWeek", "FlightNum", "Origin", "CRSDepTime", "Cancelled")
    )
    return some_data_appended


def prepare_data(data, plane_data):
    data = append_unique_id(data)
    data = clean_data(data)
    data = transform_cyclical_features_localtime_2(data)
    data = transform_cyclical_features_month_week(data)
    data = extend_original_dataset(data, plane_data)
    data = add_new_custom_features(data)
    data = data.select(important_features_post_dt_testing + total_categorical_features + [target_col])

    return data
