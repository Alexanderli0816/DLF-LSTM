# !/usr/bin/python3.6
# -*- coding: UTF-8 -*-
from __future__ import print_function
import sys
module_path = r'/home/alex/桌面/Python'
sys.path.append(module_path)
import time
import Project.Tools.namespace_utils as namespace_utils
from Project.deep_learning.data_processing.convert_score_to_bt_format import convert2format
import tensorflow as tf
from Project.Tools.DataStream import DataStream
from Project.deep_learning.Model_graph import ModelGraph
from Tools.Back_Test.back_test import BT_STOCK_DAY
import os
import warnings
import datetime
import calendar
warnings.filterwarnings('ignore')


def get_eval_dates(start_date, end_date):
    def _get_date_to_slice(date, num=3):
        year, month = date.year, date.month
        if month + num - 1 > 12:
            end_num = calendar.monthrange(int(year + 1), int(month + num - 13))[1]
            end_date = datetime.datetime(year + 1, month + num - 13, end_num)
        else:
            end_num = calendar.monthrange(int(year), int(month + num - 1))[1]
            end_date = datetime.datetime(year, month + num - 1, end_num)
        if month + num <= 12:
            start_date = datetime.datetime(year, month + num, 1)
        else:
            start_date = datetime.datetime(year + 1, month + num - 12, 1)
        return start_date, end_date

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    _, date_end_flag = _get_date_to_slice(start_date)
    result = []
    flag = 1
    while date_end_flag <= end_date and flag == 1:
        if date_end_flag == end_date:
            flag = 0
            next_start = None
        else:
            next_start, date_end_flag = _get_date_to_slice(start_date)

        result.append((str(start_date.date()), str(date_end_flag.date())))
        start_date = next_start
        date_end_flag = date_end_flag + datetime.timedelta(days=1)
    return result


def judgement_classifier(predictions, target, class_num):
    result = dict()
    for i in range(class_num):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(len(predictions)):
            if predictions[j] == target[j] == i:
                TP += 1
            elif predictions[j] != i and target[j] != i:
                TN += 1
            elif predictions[j] == i and target[j] != i:
                FP += 1
            elif predictions[j] != i and target[j] == i:
                FN += 1
            else:
                raise ValueError('Unknown predictions%s' % str(predictions[i]))
            result[i] = [TP, TN, FP, FN]
    return result


def judgement_rate(TP_SUM, FP_SUM, TN_SUM, FN_SUM, class_num):
    if TP_SUM + FP_SUM != 0:
        precision_rate = TP_SUM / (TP_SUM + FP_SUM) * 100
    else:
        precision_rate = 0
    if TP_SUM + FN_SUM != 0:
        recall_rate = TP_SUM / (TP_SUM + FN_SUM) * 100
    else:
        recall_rate = 0
    if precision_rate + recall_rate != 0:
        F1score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
    else:
        F1score = 0
    print(
        'Results class %d: precision_rate %.2f |recall_rate %.2f |F1score %.2f ========== TP_SUM: %d| FP_SUM: %d| '
        'TN_SUM: %d| FN_SUM: %d' % (
            class_num, precision_rate, recall_rate, F1score, TP_SUM, FP_SUM, TN_SUM, FN_SUM))
    return precision_rate, recall_rate, F1score


def get_file_name_list(path):
    file_name_list = []
    for root, dirs, files in os.walk(path):
        file_name_list.extend(list(map(lambda x: root + '/' + x, files)))
    file_name_list.sort()
    return file_name_list


def enrich_options(options):
    if "in_format" not in options.__dict__.keys():
        options.__dict__["in_format"] = 'tsv'
    return options


def evaluation(sess, valid_graph, devDataStream, options, out_path=None, mode='eval'):
    if mode == 'eval' and out_path is None: raise ValueError('In eval mode, out_path cannot be None or empty!')
    sharpe = None
    # initial statement
    header = 0
    if out_path is not None:
        if not os.path.exists(out_path):
            pass
        else:
            header = 1

    total = 0
    total_loss = 0
    total_loss_0 = 0
    total_loss_1 = 0
    correct = 0
    TP_SUM_0 = 0
    FP_SUM_0 = 0
    TN_SUM_0 = 0
    FN_SUM_0 = 0
    TP_SUM_1 = 0
    FP_SUM_1 = 0
    TN_SUM_1 = 0
    FN_SUM_1 = 0
    TP_SUM_2 = 0
    FP_SUM_2 = 0
    TN_SUM_2 = 0
    FN_SUM_2 = 0

    # generate results
    num_batch = devDataStream.get_num_batch()
    count_writing = 0
    for batch_index in range(num_batch):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch)
        [cur_correct, probs, predictions, output, loss_value, loss_classifier, loss_fit] = sess.run(
            [valid_graph.eval_correct, valid_graph.final_output_0, valid_graph.predictions, valid_graph.final_output_1,
             valid_graph.loss, valid_graph.loss_0, valid_graph.loss_1], feed_dict=feed_dict)

        correct += cur_correct
        total_loss += loss_value
        total_loss_0 += loss_classifier
        total_loss_1 += loss_fit
        target = cur_batch.label_truth_0
        result = judgement_classifier(predictions, target, class_num=3)
        TP_SUM_0 += result[0][0]
        FP_SUM_0 += result[0][2]
        TN_SUM_0 += result[0][1]
        FN_SUM_0 += result[0][3]
        TP_SUM_1 += result[1][0]
        FP_SUM_1 += result[1][2]
        TN_SUM_1 += result[1][1]
        FN_SUM_1 += result[1][3]

        TP_SUM_2 += result[2][0]
        FP_SUM_2 += result[2][2]
        TN_SUM_2 += result[2][1]
        FN_SUM_2 += result[2][3]

        if out_path is not None:
            file = open(out_path, 'a+', encoding='utf-8')
            if mode == 'trading':
                if header == 0:
                    file.write('ticker,score1,score2,score3,score4\n')
                    header = 1
                for i in range(cur_batch.batch_size):
                    ticker = cur_batch.ticker[i]
                    count_writing += 1
                    if count_writing % 1000 == 0:
                        print('Writing Instances %4d' % count_writing)
                    file.write(str(ticker) + ',' + str(probs[i][0]) + ',' + str(probs[i][1]) + ',' + str(probs[i][-1])
                               + ',' + str(output[i][0]) + '\n')
                    # file.write(str(0) + ',' + str(0) + ',' + str(0) + ',' + str(
                    #     output[i][0]) + '\n')
                file.close()
            else:
                count = 0
                if header == 0:
                    file.write('Timestamp,ticker,score1,score2,score3,score4\n')
                    header = 1
                for i in range(cur_batch.batch_size):
                    Datetime, ticker, = cur_batch.datetime[i], cur_batch.ticker[i]
                    if mode == 'eval' and count % 10000 == 0 and count != 0:
                        print('Writing Instances | Processing Line: %s' % (str(count)))
                    file.write(str(Datetime) + ',' + str(ticker) + ',' + str(probs[i][0]) + ',' + str(probs[i][1]) +
                               ',' + str(probs[i][-1]) + ',' + str(output[i][0]) + '\n')
                    count += 1
                file.close()

    if mode == 'eval' or mode == 'trading':
        return None
    else:
        # summarize results
        accuracy = correct / float(total) * 100
        print("Accuracy: %.2f" % accuracy)
        print('Dev loss = %.4f' % (total_loss / num_batch))
        print('Dev loss_classifier = %.4f' % (total_loss_0 / num_batch))
        print('Dev loss_fit = %.4f' % (total_loss_1 / num_batch))
        judgement_rate(TP_SUM_0, FP_SUM_0, TN_SUM_0, FN_SUM_0, 0)
        judgement_rate(TP_SUM_1, FP_SUM_1, TN_SUM_1, FN_SUM_1, 1)
        judgement_rate(TP_SUM_2, FP_SUM_2, TN_SUM_2, FN_SUM_2, 2)
        if out_path is not None:
            convert2format(out_path, options.output_file, mode='single')
            configs = namespace_utils.load_namespace(options.bt_config_path)
            configs.score_path = options.output_file
            start_date = '2019-10-01'
            end_date = '2020-03-31'
            bt = BT_STOCK_DAY(configs, start_date=start_date, end_date=end_date)
            bt.run()
            print('Back test summary:')
            _, df_sum = bt.evaluate(
                evalRange=tuple(get_eval_dates(start_date=start_date, end_date=end_date)))
            sharpe = df_sum['sharpe'].values[-1]

        return total_loss / num_batch, total_loss_0 / num_batch, total_loss_1 / num_batch, sharpe


def train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, options):
    # training
    best_sharpe = -100
    train_step = None
    train_summaries = None

    path_prefix = options.model_dir + "/LSTM_{}".format(options.suffix)
    tensorboard_log = os.path.join(options.model_dir, options.tensorboard_suffix)
    if not os.path.exists(tensorboard_log): os.makedirs(tensorboard_log)
    summary_writer = tf.summary.FileWriter(tensorboard_log, sess.graph)
    count = 0
    best_path_in_epoch = path_prefix + '.best.model'
    for epoch in range(options.max_epochs):
        loop_total = 0
        for loop in range(options.loops_for_each_reading):
            total_loop_count = loop_total * options.loops_for_each_reading + loop
            print('Train in epoch %d - %d' % (epoch, total_loop_count))
            num_batch = trainDataStream.get_num_batch()
            start_time = time.time()
            total_loss = 0
            total_loss_0 = 0
            total_loss_1 = 0
            for batch_index in range(num_batch):  # for each batch
                cur_batch = trainDataStream.get_batch(batch_index)
                # print(cur_batch.vector)
                feed_dict = train_graph.create_feed_dict(cur_batch)
                _, loss_value, loss_value_0, loss_value_1, train_summaries, train_step = sess.run(
                    [train_graph.train_op, train_graph.loss, train_graph.loss_0, train_graph.loss_1,
                     train_graph.summaries, train_graph.global_step],
                    feed_dict=feed_dict)

                total_loss += loss_value
                total_loss_0 += loss_value_0
                total_loss_1 += loss_value_1
                if batch_index % 1000 == 0:
                    print('batch{} is done!'.format(batch_index))
                    sys.stdout.flush()

            print()
            duration = time.time() - start_time
            loss_this = total_loss / num_batch
            print('Train set performance：')
            print('loss = %.4f (%.3f sec)' % (loss_this, duration))
            print('loss_classifier = %.4f' % (total_loss_0 / num_batch))
            print('loss_fit = %.4f\n' % (total_loss_1 / num_batch))

            print('Validation set performance：')
            start_time = time.time()
            output_path = os.path.join(options.model_dir, 'output', 'score.csv')
            output_folder = os.path.dirname(output_path)
            if not os.path.exists(output_folder): os.makedirs(output_folder)
            output_path = str(output_path).split('.')[0] + '_' + str('{:0>2}'.format(epoch)) + '_' + \
                          '{:0>2}'.format(total_loop_count) + '.csv'
            loss_all_dev, loss_classifier_dev, loss_fit_dev, sharpe = \
                evaluation(sess, valid_graph, devDataStream, options, out_path=output_path, mode='train')
            duration = time.time() - start_time
            print('Evaluation time: %.3f sec' % duration)
            trainDataStream.shuffle()
            print()

            # summary log
            if options.model != 'equal_weight_linear':
                with tf.variable_scope("Loss"):
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag='Loss_all', simple_value=loss_this),
                        tf.Summary.Value(tag='Loss_classifier', simple_value=total_loss_0 / num_batch),
                        tf.Summary.Value(tag='Loss_fit', simple_value=total_loss_1 / num_batch),
                        tf.Summary.Value(tag='Loss_all_dev', simple_value=loss_all_dev),
                        tf.Summary.Value(tag='Loss_classifier_dev', simple_value=loss_classifier_dev),
                        tf.Summary.Value(tag='Loss_all_dev', simple_value=loss_fit_dev)
                    ])
                summary_writer.add_summary(summary, train_step)
                summary_writer.add_summary(train_summaries, train_step)
                summary_writer.flush()

                count += 1
                if count % int(options.save_model_every_n_iters) == 0 and count > 1:
                    temp = path_prefix + '.Training_benchmark_' + str(epoch) + '_' + str(total_loop_count)
                    saver.save(sess, temp)
                if sharpe >= best_sharpe:
                    best_sharpe = sharpe
                    saver.save(sess, best_path_in_epoch)
                trainDataStream.shuffle()

        loop_total += 1
        if options.read_data_for_each_loop:
            path_list = get_file_name_list(options.train_path)
            path = path_list[loop_total % len(path_list)]
            print('Build Training DataStream ... | %s' % path)
            trainDataStream.instances = None
            trainDataStream.read_instances(path, header=False, sep=',', mode='new', method='feed')
            trainDataStream.make_batches()
        print()


def main():
    devices = '0'  # str(random.randint(1, 8))
    config_path = r'/home/alex/桌面/Python/Project/deep_learning/configs/load_mmap.config'
    print('Loading the configuration from ' + config_path)
    FLAGS = namespace_utils.load_namespace(config_path)
    FLAGS = enrich_options(FLAGS)

    print('=' * 80)
    print('Training begins at %s...' % str(datetime.datetime.now()))
    print('=' * 80)
    print('Train path: %s' % FLAGS.train_path)
    print('Dev path: %s' % FLAGS.dev_path)
    print('Model path: %s' % FLAGS.model_dir)
    print('Score path: %s' % FLAGS.score_folder)
    print('GPU device: %s' % devices)
    print('=' * 80)

    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    path_prefix = log_dir + "/LSTM_{}".format(FLAGS.suffix)
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    best_path = path_prefix + '.best.model'
    has_pre_trained_model = False
    if os.path.exists(best_path + ".index"):
        has_pre_trained_model = True
        print('Loading from a pre-trained model ...')
    print()

    data_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    path = get_file_name_list(data_path)[0]
    trainDataStream = DataStream(isShuffle=True, isLoop=False, isSort=False, options=FLAGS)
    print('Build Training DataStream ... | %s' % path)
    trainDataStream.read_instances(path, header=False, sep=',', mode='new', method='feed')
    trainDataStream.make_batches()
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of features in trainDataStream: {}'.format(trainDataStream.vector_dim))
    print()

    print('Build Devaluation DataStream ... ')
    devDataStream = DataStream(isShuffle=False, isLoop=False, isSort=False, options=FLAGS)
    devDataStream.read_instances(dev_path, header=False, sep=',', mode='new', method='feed', set_type='dev')
    devDataStream.make_batches()
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    print()

    tf.reset_default_graph()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    with tf.Graph().as_default():
        train_graph = ModelGraph(is_training=True, options=FLAGS)

        valid_graph = ModelGraph(is_training=False, options=FLAGS)

        vars_ = {}
        for var in tf.global_variables():
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        if FLAGS.show_model_analysis is True:
            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        initializer = tf.global_variables_initializer()
        config = tf.ConfigProto(inter_op_parallelism_threads=0, intra_op_parallelism_threads=0, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        sess.run(initializer)
        print()

        if has_pre_trained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")
            print()

        # training
        train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, FLAGS)


if __name__ == '__main__':
    main()
