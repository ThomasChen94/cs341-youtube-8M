MODEL_DIR=/Users/ThomasChen/Documents/study_in_Stanford/CS341/model
python train.py --train_data_pattern='/Users/ThomasChen/Documents/study_in_Stanford/CS341/data/frame_level/train/train*.tfrecord' \
--frame_features=True --model=BiLstmModel \
--feature_names="rgb" \
--feature_sizes="1024" --batch_size=64 \
--train_dir=$MODEL_DIR/frame_level_bidirect_lstm_model \
--base_learning_rate=0.008
