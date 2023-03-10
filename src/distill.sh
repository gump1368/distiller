python main.py --data_dir ../data \
               --output_dir ../model_saved/student_model \
               --teacher_bert_config_file ../model_config/teacher_bert_config \
               --teacher_init_checkpoint ../model_saved/teacher_model/bert_model-12000.ckpt \
               --student_bert_config_file ../model_config/student_bert_config \
               --training_mode distill \
               --temperature 2.0 \
               --alpha_ce 0.33 \
               --alpha_kl 0.33 \
               --alpha_cos 0.33 \
               --batch_size 8 \
               --learning_rate 2e-5 \
               --num_train_epochs 3 \
               --do_train=True