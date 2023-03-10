#! -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('sample.csv')
train_df = df.sample(frac=0.8)
print(len(train_df))
train_df.to_csv('train.csv', index=False, encoding='utf_8_sig')
eval_index = list(set(df.index).difference(set(train_df.index)))
eval_df = df.loc[eval_index]
print(len(eval_df))
eval_df.to_csv('eval.csv', index=False, encoding='utf_8_sig')
