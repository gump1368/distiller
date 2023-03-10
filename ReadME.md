# Bert Distill
将24层large-bert蒸馏到12层，使得在句子相似性任务中的准确率下降不超过三个点。

参考论文：[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

预训练模型：[RoBERTa-large-pair大句子对模型](https://github.com/CLUEbenchmark/CLUEPretrainedModels)

tensorflow版本：`tensorflow-gpu==1.14.0`

## **模型简介**
模型分为两部分：训练教师模型、蒸馏学生模型 

训练教师模型部分，以large-bert作为预训练模型，接一个2分类全连接层，使用交叉熵损失进行训练。
训练结果精度达到0.88  

蒸馏部分，修改教师模型的bert-config.json文件中的`num_hidden_layers`为12。

使用教师模型的部分参数来初始化学生模型（这里选取了前12层参数），经teacher model初始化的模型会比随机初始化的模型更快收敛，
效果也更好 

训练过程中结合三种loss：KL loss、cos loss、CE loss 
KL loss: 度量教师模型的结果与学生模型的结果的概率分布的差异
cos loss: 度量教师模型与学生模型的输出层的相似度差异性
CE loss: 学生模型的交叉熵损失

最终学生模型的验证精度达到0.87，符合论文结果。

## **训练过程**
1、使用large-bert作为预训练模型训练teacher model
```shell
sh ./train_teacher.sh
```
2、蒸馏student模型
```shell
sh ./distill.sh
```
注：此步骤只保存学生模型的checkpoint，故无法进行评估。

3、评估学生模型
加载上个步骤保存的学生模型的参数进行评估
```shell
sh ./eval_student_model.sh
```
3、导出pb模型文件
```shell
sh ./export_model.sh
```
