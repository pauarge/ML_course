#!/usr/bin/env bash

cp -r . ../project2-bak
./clear.sh

rm deliver.sh

rm -rf src/__pycache__
rm -rf src_old/__pycache__
rm -rf data/__MACOSX/
find . -name \*.pyc -delete
find . -name \*.DS_Store -delete

zip -r -X "../project2.zip" *
cd ..
rm -rf project2
mv project1-bak project2