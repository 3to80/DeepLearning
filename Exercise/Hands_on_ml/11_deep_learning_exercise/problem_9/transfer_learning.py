import tensorflow as tf
import os
import numpy as np





from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)  # deprecated 경고 메세지를 출력하지 않기 위해
mnist = input_data.read_data_sets("data")
tf.logging.set_verbosity(tf.logging.INFO)
X_train2_full = mnist.train.images[mnist.train.labels >= 5]
y_train2_full = mnist.train.labels[mnist.train.labels >= 5] - 5
X_valid2_full = mnist.validation.images[mnist.validation.labels >= 5]
y_valid2_full = mnist.validation.labels[mnist.validation.labels >= 5] - 5
X_test2 = mnist.test.images[mnist.test.labels >= 5]
y_test2 = mnist.test.labels[mnist.test.labels >= 5] - 5


def strafied_sample_instances(X, y, n=100):
    Xs, ys = [], []

    for label in np.unique(y):
        # target label가지는 idx Set 구함
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)

    return np.concatenate(Xs), np.concatenate(ys)

X_train2, y_train2 = strafied_sample_instances(X_train2_full, y_train2_full, n=100)
X_valid2, y_valid2 = strafied_sample_instances(X_valid2_full, y_valid2_full, n=30)



best_model_path = "/home/jeunghwankim/DeepLearning/DeepLearning/Exercise/Hands_on_ml/11_deep_learning_exercise/problem_8/hptuning_trainier/best_hyper_parameter_model/model"
best_model_graph = "/home/jeunghwankim/DeepLearning/DeepLearning/Exercise/Hands_on_ml/11_deep_learning_exercise/problem_8/hptuning_trainier/best_hyper_parameter_model/model.meta"

tf.reset_default_graph()

restore_saver = tf.train.import_meta_graph(best_model_graph)





X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
loss = tf.get_default_graph().get_tensor_by_name("loss/loss:0")
## 이전에 softmax 나타내는 tensor, 이거만 바꾸기 위해 지정
y_proba = tf.get_default_graph().get_tensor_by_name("dnn/Y_proba:0")

logits = y_proba.op.inputs[0]
print("######## 전체를 가져다 사용한 경우 ########")
with tf.Session() as sess:
    restore_saver.restore(sess, best_model_path)
    acc_test,loss_test = sess.run([accuracy, loss], feed_dict={X: X_test2, y: y_test2})
    # acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("최종 테스트 정확도: {:.4f}% 최종 손실 값 : {:.4f}".format(acc_test * 100, loss_test))


print("######## 최상위 층만 학습 다시 시킨 경우 ########")




# # 중요한 연산은 인스턴스 변수로 저장하여 참조하기 쉽게 합니다.
# self._X, self._y = X, y
# self._Y_proba, self._loss = Y_proba, loss
# self._training_op, self._accuracy = training_op, accuracy
# self._loss_str = loss_summary
# self._acc_str = accuracy_summary
# self._init, self._saver = init, saver
#
#
learning_rate = 0.01

for name in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(name)

# soft max 입력층 선택
output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
# 이 층만 다시 학습
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam2")
training_op = optimizer.minimize(loss, var_list=output_layer_vars)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
# five_frozen_saver = tf.train.Saver()
#
# import time
#
n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty
import time

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "/home/jeunghwankim/DeepLearning/DeepLearning/Exercise/Hands_on_ml/11_deep_learning_exercise/problem_8/hptuning_trainier/best_hyper_parameter_model/model")

    # 학습을 다시할 곳만 initializing
    for var in output_layer_vars:
        var.initializer.run()

    t0 = time.time()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("조기 종료!")
                break
        print("{}\t검증 세트 손실: {:.6f}\t최선의 손실: {:.6f}\t정확도: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))




print("######## 위에서 2개 층 학습 다시 시킨 경우 ########")

# <tf.Variable 'hidden3/kernel:0' shape=(120, 120) dtype=float32_ref>
# <tf.Variable 'hidden3/bias:0' shape=(120,) dtype=float32_ref>
# <tf.Variable 'batch_normalization_2/gamma:0' shape=(120,) dtype=float32_ref>
# <tf.Variable 'batch_normalization_2/beta:0' shape=(120,) dtype=float32_ref>
# <tf.Variable 'hidden4/kernel:0' shape=(120, 120) dtype=float32_ref>
# <tf.Variable 'hidden4/bias:0' shape=(120,) dtype=float32_ref>
# <tf.Variable 'batch_normalization_3/gamma:0' shape=(120,) dtype=float32_ref>
# <tf.Variable 'batch_normalization_3/beta:0' shape=(120,) dtype=float32_ref>
# <tf.Variable 'logits/kernel:0' shape=(120, 5) dtype=float32_ref>
# <tf.Variable 'logits/bias:0' shape=(5,) dtype=float32_ref>


#  최상위 2개 층
tr_smax= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
tr_bat3= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="batch_normalization_3")
tr_h4= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden4")


# # 이 층만 다시 학습
re_training_li = [tr_smax, tr_bat3, tr_h4]
optimizer3= tf.train.AdamOptimizer(learning_rate, name="Adam3")
training_op3 = optimizer.minimize(loss, var_list=re_training_li)

init = tf.global_variables_initializer()




n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty
import time

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "/home/jeunghwankim/DeepLearning/DeepLearning/Exercise/Hands_on_ml/11_deep_learning_exercise/problem_8/hptuning_trainier/best_hyper_parameter_model/model")


    # 학습을 다시할 곳만 initializing

    for retrain in re_training_li:
        for var in retrain:
            var.initializer.run();

    t0 = time.time()





    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op3, feed_dict={X: X_batch, y: y_batch})

        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("조기 종료!")
                break
        print("{}\t검증 세트 손실: {:.6f}\t최선의 손실: {:.6f}\t정확도: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))


###### 동결 층 캐싱 ########






n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty
import time

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "/home/jeunghwankim/DeepLearning/DeepLearning/Exercise/Hands_on_ml/11_deep_learning_exercise/problem_8/hptuning_trainier/best_hyper_parameter_model/model")


    # 학습을 다시할 곳만 initializing

    for retrain in re_training_li:
        for var in retrain:
            var.initializer.run();

    t0 = time.time()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op3, feed_dict={X: X_batch, y: y_batch})

        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("조기 종료!")
                break
        print("{}\t검증 세트 손실: {:.6f}\t최선의 손실: {:.6f}\t정확도: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

