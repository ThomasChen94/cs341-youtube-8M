MODEL_DIR=/home/cs341/model
python eval.py --eval_data_pattern='/home/cs341/data/video_level/validate/validate*.tfrecord'  --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model --run_once=True
