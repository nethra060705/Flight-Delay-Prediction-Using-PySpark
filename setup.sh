#!/usr/bin/env bash
set -euo pipefail

# setup.sh
# A helper script for macOS to create a reproducible Python 3.10 venv using pyenv,
# install dependencies, and run a smoke test on the included sample_flights.csv.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_VERSION="3.10.15"
VENV_DIR="$REPO_ROOT/.venv310"
SAMPLE_CSV="$REPO_ROOT/resources/sample_flights.csv"

echo "Repository root: $REPO_ROOT"

# Install pyenv if not present (Homebrew)
if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found. Installing via Homebrew..."
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to install pyenv. Please install Homebrew first: https://brew.sh/"
    exit 1
  fi
  brew update
  brew install pyenv
fi

# Ensure pyenv is loaded in this shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

# Install Python version if missing
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
  echo "Installing Python ${PYTHON_VERSION} via pyenv (this may take several minutes)..."
  pyenv install ${PYTHON_VERSION}
fi

# Use the installed Python for creating venv
PY_EXEC=$(pyenv root)/versions/${PYTHON_VERSION}/bin/python
if [ ! -x "$PY_EXEC" ]; then
  echo "Python executable not found at $PY_EXEC"
  exit 1
fi

# Create venv
if [ -d "$VENV_DIR" ]; then
  echo "Virtualenv $VENV_DIR already exists. Reusing it."
else
  echo "Creating virtualenv with Python ${PYTHON_VERSION}..."
  "$PY_EXEC" -m venv "$VENV_DIR"
fi

# Activate venv and install requirements
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found; installing minimal dependencies..."
  pip install pyspark pandas numpy matplotlib seaborn
fi

# Ensure pyspark installed
if ! python - <<'PY' 2>/dev/null
import importlib
try:
    importlib.import_module('pyspark')
    print('pyspark_ok')
except Exception as e:
    print('pyspark_missing')
PY
then
  echo "Failed to import pyspark inside venv. Please check the installation."
  exit 1
fi

# Run smoke test
if [ -f "$SAMPLE_CSV" ]; then
  echo "Running smoke test with sample CSV..."
  python -m src.main.main "$SAMPLE_CSV" "$REPO_ROOT/output_dir"
else
  echo "Sample CSV not found at $SAMPLE_CSV."
  exit 1
fi

echo "Setup + smoke test completed. If SparkSession failed due to Python/JVM mismatch, consider using the created venv with Python ${PYTHON_VERSION} or adjusting Spark/PySpark versions."
