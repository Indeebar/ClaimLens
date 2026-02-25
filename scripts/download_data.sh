#!/bin/bash
set -e
mkdir -p data/raw/damage_images
mkdir -p data/raw/claims_tabular

echo '--- Downloading car damage image datasets ---'
kaggle datasets download -d anujms/car-damage-detection -p data/raw/damage_images --unzip

echo '--- Downloading insurance claims tabular data ---'
kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p data/raw/claims_tabular --unzip

echo '--- All datasets downloaded successfully ---'
