MODEL_DIR=/home/cs341/model
python train.py --train_data_pattern='/home/cs341/data/frame_level/train/train*.tfrecord' \
--frame_features=True --model=RankModel \
--feature_names="rgb" \
--feature_sizes="1024" --batch_size=64 \
--train_dir=$MODEL_DIR/frame_level_rank_model_only_shuffle \
--base_learning_rate=0.01

