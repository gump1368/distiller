python main.py --data_dir ../data \
               --output_dir ../model_saved/student_model \
               --student_bert_config_file ../model_config/teacher_bert_config \
               --training_mode eval_student \
               --batch_size 8 \
               --do_train=False \
               --do_eval=True