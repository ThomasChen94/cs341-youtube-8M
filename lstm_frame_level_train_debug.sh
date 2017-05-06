MODEL_DIR=/home/cs341/model
python -m pdb train.py --train_data_pattern='/home/cs341/data/frame_level/train/train*.tfrecord' \
--frame_features=True --model=LstmModel \
--feature_names="rgb" \
--feature_sizes="1024" --batch_size=5096 \
--train_dir=$MODEL_DIR/frame_level_lstm_model \
--base_learning_rate=0.00001
