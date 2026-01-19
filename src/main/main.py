import os
import sys
import argparse

from src.main.helper_methods import initialize_spark_session, load_data, get_data_without_forbidden_variables_sampled, \
    get_plane_dataset, create_pipeline, train_model, show_results
from src.main.dataset_utils import prepare_data
import pandas as pd


def validate_args(input_path, output_dir):
    # Validate input_path
    if not input_path or not os.path.isfile(input_path):
        print(f"Error: input file '{input_path}' not found. Please provide the path to the real flights CSV (for example: resources/2007.csv.bz2).")
        sys.exit(2)

    # Validate output_dir
    if not output_dir:
        print(f"Error: output directory not provided. Please provide an output directory path to store predictions and logs.")
        sys.exit(2)
    # Create output dir if it does not exist
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error: could not create output directory '{output_dir}': {e}")
            sys.exit(2)

    return input_path, output_dir

def main(input_file, output_path, no_spark=False, test_file=None, count_only=False):
    # If user requested no-spark mode, run a small pandas-based smoke test instead
    if no_spark:
        print("Running in --no-spark mode: using pandas for a lightweight smoke test")
        df = pd.read_csv(input_file)
        print("Sample data loaded (pandas):")
        print(df.head())
        print("Summary statistics:")
        print(df.describe(include='all'))
        # Basic aggregation: mean arrival delay
        if 'ArrDelay' in df.columns:
            print(f"Mean ArrDelay: {df['ArrDelay'].dropna().mean()}")
        print("--no-spark smoke test finished")
        return

    # Initialize Spark session
    spark = initialize_spark_session()

    print(f"Current working directory: {os.getcwd()}")
    plane_data_path = "src/main/dataset/plane-data.csv"
    plane_data_path = os.path.join(os.path.dirname(__file__), "dataset", "plane-data.csv")
    print(f"Plane data path: {plane_data_path}")
    # Load data
    print("Loading data...")
    data = load_data(spark, input_file)
    sampled_data, full_data = get_data_without_forbidden_variables_sampled(data)

    # Clean, transform and extend data
    print("Cleaning, transforming and extending data...")
    plane_data = get_plane_dataset(spark, plane_data_path)
    # Use the full dataset (not the sampled subset) for training so large_flights_500k is used as requested
    data = prepare_data(full_data, plane_data)

    # Report sizes for visibility
    try:
        raw_count = full_data.count()
        prepared_count = data.count()
        print(f"Raw input rows: {raw_count}")
        print(f"Rows after prepare_data (ready for pipeline): {prepared_count}")
    except Exception:
        pass

    # If user only wants counts, exit after printing sizes
    if count_only:
        print("--count-only requested; exiting after showing counts.")
        spark.stop()
        return

    # Create pipeline and train model
    print("Creating pipeline and training model...")
    pipeline = create_pipeline()
    pipeline_model = pipeline.fit(data)

    print("Pipeline and model created! Training and evaluating...")
    # train_model now returns (validation_predictions, trained_model)
    predictions, dt_model = train_model(data, pipeline_model, output_path)

    # Add categorical predicted label based on numeric prediction
    # Assumption: prediction is minutes of arrival delay (positive = delayed, negative = early).
    # We map predictions to labels with small tolerance to avoid noise (thresholds set to +/-10 minutes):
    #   prediction <= -10 -> 'early'
    #   -10 < prediction < 10 -> 'on time'
    #   prediction >= 10 -> 'delayed'
    try:
        from pyspark.sql import functions as F
        if predictions is not None and 'prediction' in predictions.columns:
            predictions = predictions.withColumn(
                'predicted_label',
                F.when(F.col('prediction') >= 10, F.lit('delayed'))
                 .when(F.col('prediction') <= -10, F.lit('early'))
                 .otherwise(F.lit('on time'))
            )
            # If actual ArrDelay exists, also add actual_label for comparison
            if 'ArrDelay' in predictions.columns:
                predictions = predictions.withColumn(
                    'actual_label',
                    F.when(F.col('ArrDelay') >= 10, F.lit('delayed'))
                     .when(F.col('ArrDelay') <= -10, F.lit('early'))
                     .otherwise(F.lit('on time'))
                )
    except Exception:
        # if anything fails adding labels, continue without them
        pass

    # Persist predictions to output_path
    preds_out = os.path.join(output_path, 'predictions.parquet')
    try:
        predictions.write.mode('overwrite').parquet(preds_out)
        print(f"Predictions written to: {preds_out}")
    except Exception as e:
        print(f"Warning: failed to write predictions to {preds_out}: {e}")

    # Also write a single CSV with header for easy consumption by other tools
    # CSVs can't contain complex types (VectorUDT, structs with arrays), so select only primitive columns
    try:
        from pyspark.sql.types import IntegerType, LongType, DoubleType, FloatType, StringType, BooleanType, ShortType

        primitive_types = (IntegerType, LongType, DoubleType, FloatType, StringType, BooleanType, ShortType)
        safe_cols = [f.name for f in predictions.schema.fields if isinstance(f.dataType, primitive_types)]

        # Ensure 'prediction' and target column are present
        if 'prediction' not in safe_cols and 'prediction' in predictions.columns:
            safe_cols.insert(0, 'prediction')
        if 'ArrDelay' in predictions.columns and 'ArrDelay' not in safe_cols:
            safe_cols.insert(0, 'ArrDelay')
        # Ensure label columns are included when present
        if 'predicted_label' in predictions.columns and 'predicted_label' not in safe_cols:
            safe_cols.insert(0, 'predicted_label')
        if 'actual_label' in predictions.columns and 'actual_label' not in safe_cols:
            safe_cols.insert(0, 'actual_label')

        if not safe_cols:
            print('No CSV-safe columns found in predictions; skipping CSV export')
        else:
            csv_out_dir = os.path.join(output_path, 'predictions_csv')
            csv_out_file = os.path.join(output_path, 'predictions.csv')
            # coalesce and write
            predictions.select(*[c for c in safe_cols if c in predictions.columns]).coalesce(1).write.mode('overwrite').option('header', 'true').csv(csv_out_dir)

            # move the generated part-*.csv to a single predictions.csv
            import glob
            import shutil
            files = glob.glob(os.path.join(csv_out_dir, 'part-*.csv'))
            if files:
                shutil.move(files[0], csv_out_file)
                try:
                    shutil.rmtree(csv_out_dir)
                except Exception:
                    pass
                print(f"Predictions CSV written to: {csv_out_file}")
            else:
                print(f"CSV write succeeded but no part files found in {csv_out_dir}")
    except Exception as e:
        print(f"Warning: failed to write CSV predictions: {e}")

    print("Model trained! Results: ")
    show_results(predictions)

    # If a separate test file is provided, run the pipeline and model on it and save predictions
    if test_file:
        if not os.path.isfile(test_file):
            print(f"Warning: test file '{test_file}' not found; skipping test-time predictions.")
        else:
            try:
                print(f"Loading test data from: {test_file}")
                test_raw = load_data(spark, test_file)
                print("Preparing test data (same transforms as training)...")
                test_prepared = prepare_data(test_raw, plane_data)

                # Apply the same pipeline model to create feature columns and then the trained DT model
                test_transformed = pipeline_model.transform(test_prepared)
                test_predictions = dt_model.transform(test_transformed)

                # Add categorical labels to test_predictions too
                try:
                    from pyspark.sql import functions as F
                    if test_predictions is not None and 'prediction' in test_predictions.columns:
                        test_predictions = test_predictions.withColumn(
                            'predicted_label',
                            F.when(F.col('prediction') >= 10, F.lit('delayed'))
                             .when(F.col('prediction') <= -10, F.lit('early'))
                             .otherwise(F.lit('on time'))
                        )
                        if 'ArrDelay' in test_predictions.columns:
                            test_predictions = test_predictions.withColumn(
                                'actual_label',
                                F.when(F.col('ArrDelay') >= 10, F.lit('delayed'))
                                 .when(F.col('ArrDelay') <= -10, F.lit('early'))
                                 .otherwise(F.lit('on time'))
                            )
                except Exception:
                    pass

                test_out = os.path.join(output_path, 'test_predictions.parquet')
                try:
                    test_predictions.write.mode('overwrite').parquet(test_out)
                    print(f"Test predictions written to: {test_out}")
                except Exception as e:
                    print(f"Warning: failed to write test predictions to {test_out}: {e}")

                # Export test predictions to a single CSV (primitive columns only)
                try:
                    from pyspark.sql.types import IntegerType, LongType, DoubleType, FloatType, StringType, BooleanType, ShortType

                    primitive_types = (IntegerType, LongType, DoubleType, FloatType, StringType, BooleanType, ShortType)
                    safe_cols = [f.name for f in test_predictions.schema.fields if isinstance(f.dataType, primitive_types)]

                    # Ensure prediction column is included
                    if 'prediction' not in safe_cols and 'prediction' in test_predictions.columns:
                        safe_cols.insert(0, 'prediction')
                    # Ensure label columns are included when present
                    if 'predicted_label' in test_predictions.columns and 'predicted_label' not in safe_cols:
                        safe_cols.insert(0, 'predicted_label')
                    if 'actual_label' in test_predictions.columns and 'actual_label' not in safe_cols:
                        safe_cols.insert(0, 'actual_label')

                    if not safe_cols:
                        print('No CSV-safe columns found in test predictions; skipping CSV export')
                    else:
                        csv_out_dir = os.path.join(output_path, 'test_predictions_csv')
                        csv_out_file = os.path.join(output_path, 'test_predictions.csv')
                        test_predictions.select(*[c for c in safe_cols if c in test_predictions.columns]).coalesce(1).write.mode('overwrite').option('header', 'true').csv(csv_out_dir)

                        import glob
                        import shutil
                        files = glob.glob(os.path.join(csv_out_dir, 'part-*.csv'))
                        if files:
                            shutil.move(files[0], csv_out_file)
                            try:
                                shutil.rmtree(csv_out_dir)
                            except Exception:
                                pass
                            print(f"Test predictions CSV written to: {csv_out_file}")
                        else:
                            print(f"CSV write succeeded but no part files found in {csv_out_dir}")
                except Exception as e:
                    print(f"Warning: failed to write CSV test predictions: {e}")

                # If the test set contains labels, show evaluation metrics; otherwise just show a preview
                if 'ArrDelay' in test_predictions.columns:
                    print("Evaluation on test set:")
                    show_results(test_predictions)
                else:
                    print("Test predictions sample:")
                    try:
                        test_predictions.select('prediction').show(10)
                    except Exception:
                        test_predictions.show(5)
            except Exception as e:
                print(f"Warning: failed to produce test-time predictions: {e}")
    print("Done!")
    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='?', default=None)
    parser.add_argument('output_dir', nargs='?', default=None)
    parser.add_argument('--test-file', dest='test_file', default=None, help='Optional CSV (or .bz2) test file to run predictions on after training')
    parser.add_argument('--no-spark', action='store_true', help='Run a pandas-based smoke test without starting Spark')
    parser.add_argument('--count-only', action='store_true', help='Only load and prepare data, print counts, then exit')
    args = parser.parse_args()

    input_file, output_dir = validate_args(args.input_file, args.output_dir)
    main(input_file, output_dir, no_spark=args.no_spark, test_file=args.test_file, count_only=args.count_only)
