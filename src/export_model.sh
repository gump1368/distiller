python main.py --data_dir ../data \
               --output_dir ../model_saved/student_model \
               --student_bert_config_file ../model_config/student_bert_config.json \
               --training_mode export \
               --export_dir ../model_saved \
               --do_export=True \
               --do_train=False \
               --do_eval=False
