MODEL_DIR=/home/cs341/model
python train.py --train_data_pattern='/home/cs341/data/frame_level/train/train*.tfrecord' \
--frame_features=True --model=LstmModel \
--feature_names="rgb" \
--feature_sizes="1024" --batch_size=128 \
--train_dir=$MODEL_DIR/frame_level_lstm_model_2 \
--base_learning_rate=0.001
