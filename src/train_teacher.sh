python main.py --data_dir ../data \
               --output_dir ../model_saved/teacher_model \
               --teacher_bert_config_file ../model_config/teacher_bert_config \
               --teacher_init_checkpoint ../pretrained_models/Roberta-large/bert_model.ckpt \
               --training_mode training_teacher \
               --batch_size 8 \
               --learning_rate 2e-5 \
               --num_train_epochs 3