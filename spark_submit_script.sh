#!/bin/bash

## Will change depending on where you have Spark installed
SPARK_HOME="C:\tools\spark\spark-3.5.6-bin-hadoop3\bin"

conda activate predicting-flight-delay-pyspark
set PYSPARK_PYTHON=M:\action_recognition\conda\envs\predicting-flight-delay-pyspark\python.exe
set PYSPARK_DRIVER_PYTHON=M:\action_recognition\conda\envs\predicting-flight-delay-pyspark\python.exe

APP_NAME="FlightDelayPredictionApp"
APP_SCRIPT="src\main\main.py"

INPUT_FILE="resources\2007.csv.bz2"
OUTPUT_DIR="output\\"

if [ "$#" -ge 1 ]; then
  INPUT_FILE="$1"
  echo "Using the paths: Input: $INPUT_FILE, Output (default): $OUTPUT_DIR"
else
  echo "Using default paths: Input: $INPUT_FILE, Output: $OUTPUT_DIR"
fi

$SPARK_HOME/bin/spark-submit \
  --master local[*] \
  --name $APP_NAME \
  --conf spark.logConf=true \
  --conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:$SPARK_HOME/conf/log4j.properties" \
  $APP_SCRIPT "$INPUT_FILE" $OUTPUT_DIR
