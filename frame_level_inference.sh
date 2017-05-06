MODEL_DIR=/home/cs341/model

python inference.py --input_data_pattern='/home/cs341/data/frame_level/test/test*.tfrecord' \
--frame_features=True   \
--feature_sizes="1024" --batch_size=1024 \
--train_dir=$MODEL_DIR/frame_level_bidirect_lstm_model \
--output_file=$MODEL_DIR/frame_level_bidirect_lstm_model/predictions.csv \
--feature_names="rgb"
