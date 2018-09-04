from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

# from utils import model_dir

def log_dir(output_path, prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = prefix + "run-" + now
    return "{}/{}/".format(output_path, name)


def train_dir(output_path):
  return os.path.join(output_path, 'train')


def eval_dir(output_path):
  return os.path.join(output_path, 'eval')


def model_dir(output_path):
  return os.path.join(output_path, 'model')


# from model import DNNClassifier



from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import numpy as np
import tensorflow as tf
# from utils import log_dir




he_init = tf.contrib.layers.variance_scaling_initializer()
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, n_neurons=100, optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, batch_size=20, activation=tf.nn.elu, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None):
        """모든 하이퍼파파미터를 저장하는 것으로 DNNClassifier를 초기화합니다."""
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        """배치 정규화와 드롭아웃 기능을 넣어 은닉층을 구성합니다."""
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.n_neurons,
                                     kernel_initializer=self.initializer,
                                     name="hidden%d" % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,
                                                       training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs


    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None


        with tf.name_scope("dnn"):
            dnn_outputs = self._dnn(X)
            logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
            Y_proba = tf.nn.softmax(logits, name="Y_proba")

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,                                                                  logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")
            loss_summary = tf.summary.scalar("loss_summary", loss)


        with tf.name_scope("training"):
            optimizer = self.optimizer_class(learning_rate=self.learning_rate)
            training_op = optimizer.minimize(loss)


        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # 중요한 연산은 인스턴스 변수로 저장하여 참조하기 쉽게 합니다.
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._loss_str = loss_summary
        self._acc_str = accuracy_summary
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """모든 변수 값을 가져옵니다 (조기 종료를 위해 사용하며 디스크에 저장하는 것보다 빠릅니다)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """모든 변수를 주어진 값으로 설정합니다 (조기 종료를 위해 사용하며 디스크에 저장하는 것보다 빠릅니다)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        """훈련 세트에 모델을 훈련시킵니다. X_valid와 y_valid가 주어지면 조기 종료를 적용합니다."""
        self.close_session()
        tf.summary.merge_all()


        # 훈련 세트로부터 n_inputs와 n_outputs를 구합니다.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        # 레이블 벡터를 정렬된 클래스 인덱스 벡터로 변환합니다.
        # 0부터 n_outputs - 1까지의 정수를 담고 있게 됩니다.
        # 예를 들어, y가 [8, 8, 9, 5, 7, 6, 6, 6]이면
        # 정렬된 클래스 레이블(self.classes_)은 [5, 6, 7, 8, 9]가 되고
        # 레이블 벡터는 [3, 3, 4, 0, 2, 1, 1, 1]로 변환됩니다.
        self.class_to_index_ = {label: index
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # 배치 정규화를 위한 추가 연산
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # 조기 종료를 위해
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        # 훈련

        file_writer = tf.summary.FileWriter(log_dir("board_log"), graph=self._graph)

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val, loss_str, acc_str = sess.run([self._loss, self._accuracy, self._loss_str, self._acc_str],
                                                 feed_dict={self._X: X_valid,
                                                            self._y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\t검증 세트 손실: {:.6f}\t최선의 손실: {:.6f}\t정확도: {:.2f}%".format(
                        epoch, loss_val, best_loss, acc_val * 100))
                    file_writer.add_summary(summary=acc_str, global_step=epoch)
                    file_writer.add_summary(summary=loss_str, global_step=epoch)
                    if checks_without_progress > max_checks_without_progress:
                        print("조기 종료!")
                        break

                else:
                    loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                     feed_dict={self._X: X_batch,
                                                                self._y: y_batch})
                    print("{}\t마지막 훈련 배치 손실: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_train, acc_train * 100))
            # 조기 종료를 사용하면 이전의 최상의 모델로 되돌립니다.

            if best_params:
                self._restore_model_params(best_params)


            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("%s 객체가 아직 훈련되지 않았습니다" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)



tf.logging.set_verbosity(tf.logging.ERROR)  # deprecated 경고 메세지를 출력하지 않기 위해
mnist = input_data.read_data_sets("data")
tf.logging.set_verbosity(tf.logging.INFO)

#
#
X_train1 = mnist.train.images[mnist.train.labels < 5]
y_train1 = mnist.train.labels[mnist.train.labels < 5]

X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]

X_test1 = mnist.test.images[mnist.test.labels < 5]
y_test1 = mnist.test.labels[mnist.test.labels < 5]


# GC결과
# [CV] dropout_rate=0.25, n_hidden_layers=4, n_neurons=120, batch_size=100, activation=<function selu at 0x7f9be53e3b90>, optimizer_class=<functools.partial object at 0x7f9bb7733520>, total= 2.0min
# [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95, use_nesterov=True)]

from functools import partial

dnn_clf = DNNClassifier(random_state=42, n_neurons = 120, n_hidden_layers=4, dropout_rate= 0.25,
                        batch_norm_momentum= 0.99, batch_size=100, activation=tf.nn.selu,
                        optimizer_class=partial(tf.train.MomentumOptimizer, momentum=0.95, use_nesterov=True))

dnn_clf.fit(X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)


from sklearn.metrics import accuracy_score
y_pred = dnn_clf.predict(X_test1)
print(accuracy_score(y_test1, y_pred))

dnn_clf.save(model_dir("test"))


from sklearn.model_selection import RandomizedSearchCV

# def leaky_relu(alpha=0.01):
#     def parametrized_leaky_relu(z, name=None):
#         return tf.maximum(alpha * z, z, name=name)
#     return parametrized_leaky_relu
#



## 너무 시간 오래 걸려서 구글 클라우드 이용#########3
#
#
# from functools import partial
# param_distribs = {
#     "n_neurons": [70, 90, 110],
#     "batch_size": [10, 50, 100],
#     "dropout_rate": [0.25, 0.5, 0.75],
#     "activation": [tf.nn.relu, tf.nn.selu, tf.nn.elu, leaky_relu(alpha=0.1)],
#     "n_hidden_layers": [3, 4, 5, 6, 7],
#     "optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95, use_nesterov=True)],
# }
# #
# rnd_search = RandomizedSearchCV(
#     DNNClassifier(random_state=42), param_distribs,
#     n_iter= 100 , random_state=42, verbose=2)
#
# fit_params={"X_valid": X_valid1, "y_valid": y_valid1, "n_epochs": 500}
# rnd_search.fit(X_train1, y_train1, **fit_params)
#
# print("BEST PARAM : ", rnd_search.best_params_)

print(" 최종 결과 ")
y_pred = dnn_clf.predict(X_test1)
print(accuracy_score(y_test1, y_pred))

dnn_clf.save(model_dir("best_hyper_parameter_model"))






