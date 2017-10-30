#!/usr/bin/env bash

cp -r . ../project1-bak
./clear.sh

cd data
rm *.csv
unzip test.csv.zip
unzip train.csv.zip
rm solutions.csv
rm *.zip

cd ..
rm -rf support_scripts
rm deliver.sh

rm -rf scripts/__pycache__
rm -rf data/__MACOSX/
find . -name \*.pyc -delete
find . -name \*.DS_Store -delete

zip -r -X "../project1.zip" *
cd ..
rm -rf project1
mv project1-bak project1