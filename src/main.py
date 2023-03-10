#! -*- coding: utf-8 -*-
"""
@Author: Gump
@Create Time: 20230214
@Info: main
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import tensorflow as tf
from bert import tokenization
from bert.modeling import BertConfig

from processing import DataProcess, input_fn_builder
from model import model_fn_builder
from utils import serving_input_fn


os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 配置显卡

flags = tf.flags
args = flags.FLAGS

flags.DEFINE_string(
    'data_dir', '../data/', 'The input data dir. Should be tfrecoder file'
)
flags.DEFINE_string(
    'output_dir', '../model_saved',
    'The output dir where the teacher model checkpoints will be written'
)

# teacher config
flags.DEFINE_string(
    'teacher_bert_config_file', 'pretrained_models/Roberta-large/student_bert_config.json',
    'The config json file for teacher model'
)

flags.DEFINE_string(
    'teacher_init_checkpoint', 'pretrained_models/Roberta-large/bert_model.ckpt',
    'init checkpoint for teacher model'
)

# student config
flags.DEFINE_string(
    'student_bert_config_file', None, 'The config json file for student model'
)
flags.DEFINE_string(
    'student_init_checkpoint', None,
    'init checkpoint for student model'
)
flags.DEFINE_float(
    'temperature', 2.0, 'temperature for softmax'
)
flags.DEFINE_float(
    'alpha_ce', 0.33, 'weight for cross entropy loss'
)
flags.DEFINE_float(
    'alpha_kl', 0.33, 'weight for cross KL loss'
)
flags.DEFINE_float(
    'alpha_cos', 0.33, 'weight for cosine loss'
)

# public flags
flags.DEFINE_string(
    'vocab_file', '../pretrained_models/Roberta-large/vocab.txt', 'BERT vocabulary file'
)
flags.DEFINE_integer(
    'max_seq_length', 128, 'The maximum length of input sequence'
)
flags.DEFINE_bool(
    'do_train', True, 'Whether to run training'
)
flags.DEFINE_bool(
    'do_eval', True, 'Whether to run eval on dev data'
)
flags.DEFINE_bool(
    'do_predict', True, 'Whether to run predict on predict data'
)
flags.DEFINE_bool(
    'do_export', True, 'Whether to export model'
)
flags.DEFINE_string(
    'export_dir', './', 'where to save .pb model file'
)
flags.DEFINE_string(
    'training_mode', 'distill', 'select from [`training_teacher`, `distill`, `eval_student_model`, `export`]'
)

flags.DEFINE_integer(
    'batch_size', 8, 'train batch size'
)
flags.DEFINE_float(
    'learning_rate', 2e-5, 'The initial learning rate for adam'
)
flags.DEFINE_float(
    'num_train_epochs', 3, 'Total number of training epochs'
)
flags.DEFINE_float(
    'warmup_proportion', 0.1, 'Proportion of training to perform linear learning rate warmup for'
)
flags.DEFINE_integer(
    'save_checkpoints_steps', 4000, 'how often to save model checkpoint'
)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # 数据加载
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    processor = DataProcess(tokenizer=tokenizer, args=args)

    num_train_steps, num_warmup_steps, examples = None, None, -1
    origin_train_file = os.path.join(args.data_dir, 'origin', 'train.csv')
    tf_record_file = os.path.join(args.data_dir, 'tf_record', 'train.tfrecord')
    save_checkpoints_steps = args.save_checkpoints_steps
    if args.do_train:
        examples = processor.read_file(origin_train_file)
        processor.write_tf_record_data(output_file=tf_record_file, examples=examples)

        tf.logging.info('####train data length#######', len(examples))
        num_train_steps = int(len(examples) / args.batch_size * args.num_train_epochs)
        save_checkpoints_steps = int(len(examples) / args.batch_size)
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    # estimator配置
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(model_dir=args.output_dir,
                                        save_checkpoints_steps=save_checkpoints_steps
                                        ).replace(session_config=sess_config)

    teacher_bert_config = BertConfig.from_json_file(args.teacher_bert_config_file)
    if args.student_bert_config_file is not None:
        student_bert_config = BertConfig.from_json_file(args.student_bert_config_file)
    else:
        student_bert_config = None
    model_fn = model_fn_builder(teacher_bert_config=teacher_bert_config,
                                student_bert_config=student_bert_config,
                                teacher_init_checkpoint=args.teacher_init_checkpoint,
                                student_init_checkpoint=args.student_init_checkpoint)

    params = {'learning_rate': args.learning_rate, 'num_warmup_steps': num_warmup_steps,
              'batch_size': args.batch_size, 'num_train_steps': num_train_steps,
              'training_mode': args.training_mode, 'alpha_ce': args.alpha_ce, 'alpha_kl': args.alpha_kl,
              'alpha_cos': args.alpha_cos, 'temperature': args.temperature}

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=params)

    if args.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(examples))
        tf.logging.info("  Batch size = %d", args.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = input_fn_builder(input_file=tf_record_file, is_training=True, seq_length=args.max_seq_length)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if args.do_eval:
        dev_file = os.path.join(args.data_dir, 'origin', 'eval.csv')
        record_eval_file = os.path.join(args.data_dir, 'tf_record', 'eval.tfrecord')
        eval_examples = processor.read_file(dev_file)
        processor.write_tf_record_data(record_eval_file, eval_examples)
        eval_steps = int(len(eval_examples) // args.batch_size)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Num steps = %d", eval_steps)

        eval_input_fn = input_fn_builder(input_file=record_eval_file, is_training=False, seq_length=args.max_seq_length)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)  # checkpoint_path=''
        tf.logging.info("eval precision: %.2f", result['eval_accuracy'])
        tf.logging.info("teacher eval precision: %.2f", result['teacher_accuracy'])  # None for training teacher model
        tf.logging.info("eval loss: %.2f", result['eval_loss'])

    if args.do_predict:
        dev_file = os.path.join(args.data_dir, 'origin', 'eval.csv')
        record_eval_file = os.path.join(args.data_dir, 'tf_record', 'eval.tfrecord')
        eval_examples = processor.read_file(dev_file)
        processor.write_tf_record_data(record_eval_file, eval_examples)
        eval_input_fn = input_fn_builder(input_file=record_eval_file, is_training=False, seq_length=args.max_seq_length)
        checkpoint = '../model_saved/student_model/model.ckpt-12000'
        result = estimator.predict(input_fn=eval_input_fn, checkpoint_path=checkpoint)
        pred_labels, pred_scores = [], []
        for res in result:
            probabilities = res['probabilities']
            prob_label = probabilities.argmax(0)
            prob_score = probabilities[prob_label]
            pred_labels.append(prob_label)
            pred_scores.append(prob_score)

        df = pd.read_csv(dev_file, sep='\t')
        df['pred_labels'] = pred_labels
        df['pred_scores'] = pred_scores
        df.to_csv('test_pred.csv', encoding='utf_8_sig', index=False)

    # export model
    if args.do_export:
        # select best model
        dev_file = os.path.join(args.data_dir, 'origin', 'eval.csv')
        record_eval_file = os.path.join(args.data_dir, 'tf_record', 'eval.tfrecord')
        eval_examples = processor.read_file(dev_file)
        processor.write_tf_record_data(record_eval_file, eval_examples)
        eval_steps = int(len(eval_examples) // args.batch_size)
        eval_input_fn = input_fn_builder(input_file=record_eval_file, is_training=False, seq_length=args.max_seq_length)
        best_checkpoint = (0.0, '')
        for file in os.listdir(args.output_dir):
            if file.endswith('index'):
                checkpoint = os.path.join(args.output_dir, file[:-6])
                result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=checkpoint)
                tf.logging.info("eval precision: %.2f", result['eval_accuracy'])
                tf.logging.info("teacher eval precision: %.2f", result['teacher_accuracy'])
                tf.logging.info("eval loss: %.2f", result['eval_loss'])
                if result['eval_accuracy'] > best_checkpoint[0]:
                    best_checkpoint = (result['eval_accuracy'], checkpoint)

        tf.logging.info('save best checkpoint {}, eval score: {}'.format(best_checkpoint[1], best_checkpoint[0]))
        estimator.export_saved_model(args.export_dir, serving_input_fn, checkpoint_path=best_checkpoint[1])


if __name__ == '__main__':
    tf.app.run()
