MODEL_DIR=/home/cs341/model
python -m pdb train.py --train_data_pattern='/home/cs341/data/frame_level/train/train*.tfrecord' \
--frame_features=True --model=BiLstmModel \
--feature_names="rgb" \
--feature_sizes="1024" --batch_size=256 \
--train_dir=$MODEL_DIR/frame_level_bidirect_lstm_model_test2 \
--base_learning_rate=0.0001
