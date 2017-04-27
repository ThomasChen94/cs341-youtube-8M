MODEL_DIR=/home/cs341/model
python eval.py --eval_data_pattern='/home/cs341/data/frame_level/validate/validate*.tfrecord' \
--frame_features=True --model=LstmModel --feature_names="rgb" \
--feature_sizes="1024" --batch_size=64 \
--train_dir=$MODEL_DIR/frame_level_lstm_model --run_once=True \
