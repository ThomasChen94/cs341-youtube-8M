MODEL_DIR=/home/cs341/model

python inference.py --output_file=$MODEL_DIR/video_level_logistic_model/predictions.csv --input_data_pattern='/home/cs341/data/video_level/test/test*.tfrecord' --train_dir=$MODEL_DIR/video_level_logistic_model
