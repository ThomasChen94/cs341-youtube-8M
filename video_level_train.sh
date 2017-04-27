MODEL_DIR=/home/cs341/model 
python train.py --train_data_pattern='/home/cs341/data/video_level/train/train*.tfrecord' --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model
