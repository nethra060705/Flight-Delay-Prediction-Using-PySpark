from pyspark.sql import functions as f
from pyspark.sql.functions import col
from pyspark.sql.functions import when, expr, length, unix_timestamp


# Create UDF to categorize the time of day
def categorize_time(hour):
    morning_interval = [5, 6, 7, 8, 9, 10, 11]
    afternoon_interval = [12, 13, 14, 15, 16, 17, 18]
    evening_interval = [19, 20, 21, 22, 23]
    night_interval = [0, 1, 2, 3, 4]

    if hour in morning_interval:
        return "morning"
    elif hour in afternoon_interval:
        return "afternoon"
    elif hour in evening_interval:
        return "evening"
    elif hour in night_interval:
        return "night"
    else:
        return "unknown"


# Feature 1: Flight Time Window: Create a feature that represents the time of day or time window of the flight. You
# can use DepTime or CRSDepTime to categorize flights into morning, afternoon, evening, etc.
def add_flight_time_window(data):
    new_features_data = data
    # Robustly extract hour from time-like columns (handles integers, strings like '730', '0730', or '07:30')
    def hour_from_time_col(c):
        cleaned = f.regexp_replace(f.col(c).cast('string'), '[^0-9]', '')
        as_int = f.when(cleaned == '', None).otherwise(cleaned.cast('int'))
        hour = (as_int / 100).cast('int')
        return hour

    categorize_time_udf = f.udf(categorize_time)

    new_features_data = new_features_data.withColumn('DepHour', hour_from_time_col('DepTime'))
    new_features_data = new_features_data.withColumn('CRSDepHour', hour_from_time_col('CRSDepTime'))
    new_features_data = new_features_data.withColumn('CRSArrHour', hour_from_time_col('CRSArrTime'))

    new_features_data = new_features_data.withColumn("DepTime_TOD", categorize_time_udf(col("DepHour")))
    new_features_data = new_features_data.withColumn("CRSDepTime_TOD", categorize_time_udf(col("CRSDepHour")))
    new_features_data = new_features_data.withColumn("CRSArrTime_TOD", categorize_time_udf(col("CRSArrHour")))

    new_features_data = new_features_data.drop('DepHour', 'CRSDepHour', 'CRSArrHour')
    return new_features_data


# Feature 2: Day of Week and Weekend Indicator: Combine information from DayOfWeek to create a binary indicator for
# the weekend. This can be helpful, as weekends might have different patterns of delays compared to weekdays.
def add_weekend_indicator(data):
    new_features_data = data
    new_features_data = new_features_data.withColumn("Weekend",
                                                     when(col("DayOfWeek").isin([5, 6, 7]), "Weekend").otherwise(
                                                         "Weekday"))
    return new_features_data


# Feature 3: Scheduled Departure and Arrival Difference: Calculate the time difference between scheduled departure (
# CRSDepTime) and scheduled arrival (CRSArrTime). This might capture the total planned flight time.
def add_enough_time_estimation(data):
    new_features_data = data
    # Convert scheduled times to minutes-since-midnight robustly (strip non-digits, handle nulls)
    def minutes_since_midnight_expr(c):
        cleaned = f.regexp_replace(f.col(c).cast('string'), '[^0-9]', '')
        as_int = f.when(cleaned == '', None).otherwise(cleaned.cast('int'))
        hours = (as_int / 100).cast('int')
        minutes = (as_int % 100).cast('int')
        return (hours * 60 + minutes)

    dep_minutes = minutes_since_midnight_expr('CRSDepTime')
    arr_minutes = minutes_since_midnight_expr('CRSArrTime')

    new_features_data = new_features_data.withColumn('DepMinutes', dep_minutes)
    new_features_data = new_features_data.withColumn('ArrMinutes', arr_minutes)

    new_features_data = new_features_data.withColumn('TimeBetweenDepartures', (f.col('ArrMinutes') - f.col('DepMinutes')))

    # If TimeBetweenDepartures is null or negative, set to a sensible default (e.g., null) before categorizing
    new_features_data = new_features_data.withColumn('TimeBetweenDepartures', f.when(f.col('TimeBetweenDepartures').isNull(), None).otherwise(f.col('TimeBetweenDepartures')))

    new_features_data = new_features_data.withColumn("TimeBetweenDepartures", 
                                                     when(col("TimeBetweenDepartures") <= 30, "NOT_ENOUGH")
                                                     .when((col("TimeBetweenDepartures") > 30) & (col("TimeBetweenDepartures") <= 60), "BARELY_ENOUGH")
                                                     .when((col("TimeBetweenDepartures") > 60) & (col("TimeBetweenDepartures") <= 120), "ENOUGH")
                                                     .otherwise("MORE_THAN_ENOUGH"))

    new_features_data = new_features_data.drop('DepMinutes', 'ArrMinutes')
    return new_features_data
