import util
from aux import preprocessData, next_batch

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os


#
#
# X_train1 = mnist.train.images[mnist.train.labels < 5]
# y_train1 = mnist.train.labels[mnist.train.labels < 5]
# X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
# y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]
# X_test1 = mnist.test.images[mnist.test.labels < 5]
# y_test1 = mnist.test.labels[mnist.test.labels < 5]


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epoch', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, '')
flags.DEFINE_integer('unit_epoch', 5, 'unit_epoch 마다 evaluation 값을 나타낸다. validation 값의 inference 라고 표현해도 되나? '
                                      '또한 조기 종료를 위한 loss 값 비교를 진행한다. ')
flags.DEFINE_string('input_path', 'data',
                    '훈련할 데이터들이 있는 path ')

flags.DEFINE_string('temp_path', "model_temp/",
                    'i 번째 마다 저장되는 checkpoint의 경로')

flags.DEFINE_string('output_path', "model_final/",
                    '학습을 다하고 최종적으로 나온 checkpoint의 경로')

flags.DEFINE_string('board_log_path', "board_log",
                    'tensorboard에 보여줄 로그들의 경로')


# hidden layer 당 뉴런의 수, input의 fan_out 부터 output의 fan_in 까지
dnn_hidden_neuron_num = [100, 100, 100, 100, 100]
# 최종적으로 몇개의 클래스로 분류할지.
dnn_output_num = 5

# 조기종료를 위한 best loss 값 저장, minimize 로 학습이 진행되니 inf 값이 들어간다.
best_loss = np.inf
# 조기 종료 횟수 마지노선  : best loss 가 10 * unit_epoch 동안
terminated_limit = 10

########################################################################################################################
#0. init, saver 만들기

# tensorflow graph init
tf.reset_default_graph()
# tensorboard init
tf.summary.merge_all()


# saver 관련 param
checkpoint_path = "./" + FLAGS.temp_path+"mnist_model.ckpt"


checkpoint_epoch_path = checkpoint_path + ".epoch"
final_checkpoint_path = "./" + FLAGS.output_path + "final.ckpt"


#1. 원하는 Input Data(training, valid, test) 를 받기


tf.logging.set_verbosity(tf.logging.ERROR)  # deprecated 경고 메세지를 출력하지 않기 위해
mnist = input_data.read_data_sets("data")
tf.logging.set_verbosity(tf.logging.INFO)


processedData = preprocessData(mnist, (lambda x: x < 5))

Train = processedData[0]
Validation = processedData[1]
Test = processedData[2]

X_train = Train[0]
y_train = Train[1]

X = tf.placeholder(tf.float32, shape = (None, input_features), name = "X")
y = tf.placeholder(tf.int64, shape = (None), name="y")


#2. 원하는 뉴럴 네트워크 만들기



he_init = tf.contrib.layers.variance_scaling_initializer()


with tf.name_scope("model"):
    inputs = X
    for idx, fan_out in enumerate(dnn_hidden_neuron_num):
        inputs= tf.layers.dense(inputs=inputs, units = fan_out,
                                  activation= tf.nn.elu,
                                  kernel_initializer=he_init,
                                  name=("hidden" + str(idx)))

    logits = tf.layers.dense(inputs=inputs,
                             units= dnn_output_num,
                            name = "logits")
    y_proba = tf.nn.softmax(logits, name="y_proba")


with tf.name_scope("loss"):
    xentropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels= y)
    loss= tf.reduce_mean(xentropy, name= "loss")
    loss_str = tf.summary.scalar(name="log_loss", tensor=loss)


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)



with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1) # (M, 1) vector가 생성
    accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)




########################################################################################################################

#3. 뉴럴 네트워크를 세션에서 학습하기


# 정의한 graph의 모든 값들을 초기화


with tf.name_scope("init"):

    # tensorflow value initializer
    init = tf.global_variables_initializer()

    # tensorboard file handler
    file_writer = tf.summary.FileWriter(FLAGS.board_log_path, graph=tf.get_default_graph())

    # saver handler

    saver = tf.train.Saver(max_to_keep=100000)






with tf.Session() as sess:



    if os.path.isfile(checkpoint_epoch_path):
        # epoch 값을 가져와야 된다.
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print(" 이전 훈련이 중지되었습니다. 에포크 {}에서 시작합니다".format(start_epoch))
        # 이전까지 했던 session의 상태를 가져온다.
        saver.restore(sess, checkpoint_path+"-"+str(start_epoch-1))
    else:
        start_epoch = 0

        sess.run(init)



    batch_cnt = len(Train[0]) // FLAGS.batch_size

    for epoch in range(start_epoch, FLAGS.epoch):
        # Train에서 batch 만큼 가져와
        for batch_idx in range(batch_cnt):
            X_batch, y_batch = next_batch(Train, epoch, batch_idx, FLAGS.batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})



        accuracy_val, loss_val, accuracy_summary_val, loss_summary_val = \
            sess.run([accuracy, loss, accuracy_summary, loss_str],
             feed_dict={X: Validation[0], y: Validation[1]})

        file_writer.add_summary(summary=accuracy_summary_val, global_step=epoch)
        file_writer.add_summary(summary=loss_summary_val, global_step=epoch)
        # stdout 출력

        if epoch % FLAGS.unit_epoch == 0:
            print("에포크: " , epoch, " loss val : ", loss_val,
                  "accuracy_val: ", accuracy_val)


            # temp model 저장
            # 현재까지 진행된 epoch도 저장
            saver.save(sess= sess, save_path= checkpoint_path, global_step= epoch)
            with open(file=checkpoint_epoch_path, mode= "wb") as epoch_F:
                epoch_F.write(b"%d" %(epoch+1))


            # 조기 종료 구현
            if best_loss > loss_val:
                best_loss = loss_val
                saver.save(sess = sess, save_path = final_checkpoint_path)
            else:
                terminated_limit -= 1
                if not terminated_limit:
                    print(" 조기 종료 ")
                    break





os.remove(checkpoint_epoch_path)



# #4. 결과로 만들어진 뉴럴 네트워크로 test set 으로 prediction 한 결과 만들기



with tf.Session() as sess:
    saver.restore(sess, final_checkpoint_path)
    acc_test = accuracy.eval(feed_dict={X:Test[0], y: Test[1]})
    print("최종 : ", acc_test)


