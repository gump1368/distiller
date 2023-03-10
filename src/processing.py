#! -*- coding: utf-8 -*-
"""
@Author: Gump
@Crate Info: 20230211
@Info: 数据预处理
"""
import csv
import tensorflow as tf


class DataProcess(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_length = args.max_seq_length

    @classmethod
    def read_file(cls, file_path):
        examples = []
        with open(file_path, 'r', encoding='utf_8_sig') as f:
            rows = csv.reader(f, delimiter='\t')
            for row in rows:
                examples.append(row)
        return examples

    def _create_example(self, text_a, text_b, label):
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)

        # 超过max length，截断较长文本
        length_a = len(tokens_a)
        length_b = len(tokens_b)
        if length_a + length_b > self.max_length - 3:  # [CLS], [SEP]
            if length_a > length_b:
                tokens_a = tokens_a[:self.max_length - 3 - length_b]
            else:
                tokens_b = tokens_b[:self.max_length - 3 - length_a]

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        masks_id = [1] * len(tokens)
        segments_id = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        assert len(segments_id) == len(tokens_id)

        # padding
        padding = [0] * (self.max_length - len(tokens_id))
        tokens_id += padding
        masks_id += padding
        segments_id += padding

        return {
            'input_ids': tokens_id,
            'token_type_ids': segments_id,
            'input_mask': masks_id,
            'labels': [int(label)]
        }

    def write_tf_record_data(self, output_file, examples):
        """tf格式数据"""
        int_feature = lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x))
        writer = tf.python_io.TFRecordWriter(output_file)
        for example in examples:
            text_a, text_b, label = example
            feature = self._create_example(text_a, text_b, label)
            feature = {key: int_feature(value) for key, value in feature.items()}
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())


def input_fn_builder(input_file, is_training, seq_length, drop_remainder=True):
    """构造数据输入"""
    def _parse_record(example):
        features = {
            'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
            'token_type_ids': tf.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
            'labels': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(example, features)
        return parsed_example

    def input_fn(params):
        print(params)
        batch_size = params['batch_size']
        parsed_data = tf.data.TFRecordDataset(input_file)

        if is_training:
            parsed_data = parsed_data.repeat()
            parsed_data = parsed_data.shuffle(buffer_size=100)

        parsed_data = parsed_data.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _parse_record(record),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return parsed_data

    return input_fn


if __name__ == '__main__':
    pass
