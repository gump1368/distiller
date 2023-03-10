#! -*- coding: utf-8 -*-
"""
@Author: Gump
@Create Time: 20230213
@Info: build bert model
"""
import tensorflow as tf
from bert.modeling import BertModel, get_assignment_map_from_checkpoint
from bert import optimization
from loss import cos_loss_batch, KL_loss, cross_entropy_loss


def model(bert_config, input_ids, input_mask=None, token_type_ids=None, labels=None, is_training=False, num_class=2,
          scope='teacher'):
    """bert model"""
    # scope: `teacher` for build teacher model, `student` for build student model
    with tf.variable_scope(scope):
        bert_model = BertModel(config=bert_config,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=token_type_ids,
                               is_training=is_training)
        output_layer = bert_model.get_pooled_output()
        hidden_size = output_layer.shape[-1]

        # 全连接层
        fc_weights = tf.get_variable("fc_weights", shape=[hidden_size, num_class],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

        fc_bias = tf.get_variable('fc_bias', shape=[num_class], initializer=tf.zeros_initializer())

        drop_output_layer = tf.nn.dropout(output_layer, keep_prob=0.9) if is_training else output_layer
        logits = tf.matmul(drop_output_layer, fc_weights)
        logits = tf.nn.bias_add(logits, fc_bias)
        prob = tf.nn.softmax(logits, axis=-1)
        log_prob = tf.nn.log_softmax(logits, axis=-1)

        # 使用交叉熵计算loss
        per_example_loss, loss = None, None
        if labels is not None:
            per_example_loss, loss = cross_entropy_loss(log_prob, labels, num_class)
    return logits, prob, per_example_loss, loss, output_layer


def distiller(teacher_bert_config, student_bert_config, input_ids, input_mask=None, token_type_ids=None, labels=None,
              is_training=False, **kwargs):
    # 24 layers
    t_logits, t_prob, t_per_example_loss, t_loss, t_hidden_state = model(bert_config=teacher_bert_config,
                                                                         input_ids=input_ids,
                                                                         input_mask=input_mask,
                                                                         token_type_ids=token_type_ids,
                                                                         labels=labels,
                                                                         is_training=False,  # distill for False
                                                                         scope='teacher')
    # 12 layers
    s_logits, s_prob, s_per_example_loss, s_loss, s_hidden_state = model(bert_config=student_bert_config,
                                                                         input_ids=input_ids,
                                                                         input_mask=input_mask,
                                                                         token_type_ids=token_type_ids,
                                                                         labels=labels,
                                                                         is_training=is_training,
                                                                         scope='student')

    kl_loss = KL_loss(tf.nn.log_softmax(s_logits / kwargs['temperature'], axis=-1),
                      tf.nn.log_softmax(t_logits / kwargs['temperature'], axis=-1)) * kwargs['temperature'] ** 2

    cos_loss = cos_loss_batch(t_hidden_state, s_hidden_state, labels)

    loss = kwargs['alpha_ce'] * s_loss + kwargs['alpha_kl'] * kl_loss + kwargs['alpha_cos'] * cos_loss

    return s_logits, s_prob, s_per_example_loss, loss


def model_fn_builder(teacher_bert_config, student_bert_config, teacher_init_checkpoint, student_init_checkpoint=None):
    def model_fn(features, labels, mode, params):

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        token_type_ids = features['token_type_ids']
        labels = features['labels']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if params['training_mode'] == 'training_teacher':
            # training for teacher
            logits, prob, per_example_loss, loss, _ = model(bert_config=teacher_bert_config,
                                                            input_ids=input_ids,
                                                            input_mask=input_mask,
                                                            token_type_ids=token_type_ids,
                                                            labels=labels,
                                                            is_training=is_training,
                                                            scope='teacher')
        elif params['training_mode'] == 'distill':
            # distill student model
            logits, prob, per_example_loss, loss = distiller(teacher_bert_config=teacher_bert_config,
                                                             student_bert_config=student_bert_config,
                                                             input_ids=input_ids,
                                                             input_mask=input_mask,
                                                             token_type_ids=token_type_ids,
                                                             labels=labels,
                                                             is_training=is_training,
                                                             **params)
        else:
            # eval or export student model
            logits, prob, per_example_loss, loss, _ = model(bert_config=student_bert_config,
                                                            input_ids=input_ids,
                                                            input_mask=input_mask,
                                                            token_type_ids=token_type_ids,
                                                            labels=labels,
                                                            is_training=is_training,
                                                            scope='student')

        initialized_variable_names = {}
        train_vars = tf.trainable_variables()
        # init teacher model variables
        if teacher_init_checkpoint is not None:
            teacher_vars = tf.trainable_variables('teacher')
            (teacher_assignment_map, teacher_initialized_variable_names) = \
                get_assignment_map_from_checkpoint(teacher_vars, teacher_init_checkpoint, mode=params['training_mode'])
            initialized_variable_names.update(teacher_initialized_variable_names)
            tf.train.init_from_checkpoint(teacher_init_checkpoint, teacher_assignment_map)

        # init student model variables
        scaffold_fn = None
        if student_init_checkpoint is not None:
            student_vars = tf.trainable_variables('student')
            if student_init_checkpoint:
                (student_assignment_map, student_initialized_variable_names) = \
                    get_assignment_map_from_checkpoint(student_vars, student_init_checkpoint)

                tf.train.init_from_checkpoint(student_init_checkpoint, student_assignment_map)

        # init student model from teacher model
        # we select 0~11 layers from teacher model to init student model layers
        if params['training_mode'] == 'distill':
            tf.logging.info('%%%%%Init student parameters from teacher model')
            student_vars = tf.trainable_variables('student')
            (student_assignment_map, student_initialized_variable_names) = \
                get_assignment_map_from_checkpoint(student_vars, teacher_init_checkpoint, mode='init_from_teacher')

            tf.train.init_from_checkpoint(teacher_init_checkpoint, student_assignment_map)

            # only save student model variables
            global_variables = tf.global_variables()
            saved_variables = [variable for variable in global_variables if not variable.name.startswith('teacher')]
            saver = tf.train.Saver(saved_variables)
            scaffold_fn = tf.train.Scaffold(saver=saver)

            initialized_variable_names.update(student_initialized_variable_names)

        tf.logging.info("**** Trainable Variables ****")
        for var in train_vars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # when training_mode is distill, we only calculate student model gradients
            train_op = optimization.create_optimizer(
                loss, params['learning_rate'], params['num_train_steps'], params['num_warmup_steps'],
                mode=params['training_mode'])

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold=scaffold_fn
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn():
                predictions = tf.argmax(prob, axis=-1)
                accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
                eval_loss = tf.metrics.mean(values=per_example_loss)

                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": eval_loss
                }

            eval_metrics = metric_fn()
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=loss,
                                                     eval_metric_ops=eval_metrics)
        # predict
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions={'probabilities': prob})

        return output_spec

    return model_fn
