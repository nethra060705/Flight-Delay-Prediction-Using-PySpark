@echo off

REM  Set the path to your Spark installation
set SPARK_HOME=C:\tools\spark\spark-3.5.6-bin-hadoop3

REM Set Python paths for PySpark (should be pyspark-bd-project for me and predicting-flight-delay-pyspark for you)
call conda activate pyspark-bd-project
set PYSPARK_PYTHON=M:\action_recognition\conda\envs\pyspark-bd-project\python.exe
set PYSPARK_DRIVER_PYTHON=M:\action_recognition\conda\envs\pyspark-bd-project\python.exe


REM   Application name and script
set APP_NAME=FlightDelayPredictionApp
set APP_SCRIPT=src\main\main.py

REM Default input and output paths
set INPUT_FILE=resources\2007.csv.bz2
set OUTPUT_DIR=output\

REM Check if an input file is provided as an argument
if "%~1" NEQ "" (
    set INPUT_FILE=%~1
    echo "Using the paths: Input: %INPUT_FILE%, Output (default): %OUTPUT_DIR%"
) else (
    echo "Using default paths: Input: %INPUT_FILE%, Output: %OUTPUT_DIR%"
)

REM Run the Spark application
"%SPARK_HOME%\bin\spark-submit" ^
  --master local[*] ^
  --name %APP_NAME% ^
  --conf spark.logConf=true ^
  --conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:%SPARK_HOME%\conf\log4j.properties" ^
  %APP_SCRIPT% "%INPUT_FILE%" "%OUTPUT_DIR%"