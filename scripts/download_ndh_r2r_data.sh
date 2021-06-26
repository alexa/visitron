#!/bin/sh

# NDH data (https://github.com/mmurray/cvdn/blob/master/tasks/NDH/data/download.sh)

wget https://cvdn.dev/dataset/NDH/train_val/train.json -P srv/task_data/NDH/data/
wget https://cvdn.dev/dataset/NDH/train_val/val_seen.json -P srv/task_data/NDH/data/
wget https://cvdn.dev/dataset/NDH/train_val/val_unseen.json -P srv/task_data/NDH/data/
wget https://cvdn.dev/dataset/NDH/test_cleaned/test_cleaned.json -P srv/task_data/NDH/data/ -O srv/task_data/NDH/data/test.json

# R2R data (https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/data/download.sh)

wget https://www.dropbox.com/s/hh5qec8o5urcztn/R2R_train.json -P srv/task_data/R2R/data/
wget https://www.dropbox.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json -P srv/task_data/R2R/data/
wget https://www.dropbox.com/s/p6hlckr70a07wka/R2R_val_unseen.json -P srv/task_data/R2R/data/
wget https://www.dropbox.com/s/w4pnbwqamwzdwd1/R2R_test.json -P srv/task_data/R2R/data/

