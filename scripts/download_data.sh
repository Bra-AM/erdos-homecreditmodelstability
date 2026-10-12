#!/usr/bin/env bash
# Download the competition data into data/ (requires a Kaggle API token and
# acceptance of the competition rules; see data/README.md).
set -euo pipefail

cd "$(dirname "$0")/.."
kaggle competitions download -c home-credit-credit-risk-model-stability -p data
unzip -q -o data/home-credit-credit-risk-model-stability.zip -d data
rm data/home-credit-credit-risk-model-stability.zip
echo "Data ready in data/csv_files/"
