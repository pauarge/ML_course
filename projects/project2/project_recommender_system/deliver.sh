#!/usr/bin/env bash

cp -r . ../project_recommender_system-bak
./clear.sh

rm deliver.sh

rm -rf src/__pycache__
rm -rf src_old/__pycache__
rm -rf data/__MACOSX/
find . -name \*.pyc -delete
find . -name \*.DS_Store -delete

zip -r -X "../project_recommender_system.zip" *
cd ..
rm -rf project_recommender_system
mv project_recommender_system-bak project_recommender_system
