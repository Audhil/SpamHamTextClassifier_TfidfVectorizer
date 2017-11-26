"""
demo @ https://github.com/Audhil/tensorflow_cookbook/blob/master/07_Natural_Language_Processing/03_Implementing_tf_idf/03_implementing_tf_idf.py
"""
import tensorflow as tf
import csv
import numpy as np
import os
import string
import requests
import io
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from apptokenizer import tokenizer

# from tensorflow.python.tools import freeze_graph
# from tensorflow.python.tools import optimize_for_inference_lib

batch_size = 200
max_features = 1000
EPOCHS = 30000
learning_rate = 0.0015
display_step = 100

out_dir = 'out/'
# log_dir = 'logs/'
input_node_name = 'MODEL_INPUTT'
output_node_name = 'MODEL_OUTPUTT'

prediction_node_name = 'MODEL_PREDICTION'

MODEL_NAME = 'spam_ham_text_classifier'
tf_idf_file = 'tfidf.pickle'

# Check if data was downloaded, otherwise download it and save for future use
save_file_name = os.getcwd() + '/temp_spam_data.csv'
if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]

    # And write to csv
    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]

# Relabel 'spam' as 1, 'ham' as 0
target = [1. if x == 'spam' else 0. for x in target]

# Normalize text
# Lower case
texts = [x.lower() for x in texts]

# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# def export_model(input_node_names, output_node_name):
#     """
#     exporting model to be used in Android app
#     :param input_node_names:
#     :param output_node_name:
#     :return:
#     """
#     freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
#                               'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
#                               "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")
#
#     input_graph_def = tf.GraphDef()
#     with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
#         input_graph_def.ParseFromString(f.read())
#
#     output_graph_def = optimize_for_inference_lib.optimize_for_inference(
#         input_graph_def, input_node_names, [output_node_name],
#         tf.float32.as_datatype_enum)
#
#     with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
#         f.write(output_graph_def.SerializeToString())
#
#     print("---graph saved!")
#     print('---all done!')


# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

# dumping tfidf vectorizer
if not os.path.isdir(out_dir):
    print('---making a output dir')
    os.mkdir(out_dir)

# tut @ https://stackoverflow.com/questions/32764991/how-do-i-store-a-tfidfvectorizer-for-future-use-in-scikit-learn
with open(out_dir + tf_idf_file, "wb") as file:
    print('---dumping vectorizer with pickle')
    pickle.dump(tfidf, file)

# Split up data set into train/test
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8 * sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))

# texts
texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]

# labels
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Initialize placeholders
x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32, name=input_node_name)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(x_data, A), b, name=output_node_name)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output), name=prediction_node_name)
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# tf.summary.scalar("loss", loss)
# tf.summary.scalar("accuracy", accuracy)
# merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

# Intitialize Variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # tf.train.write_graph(sess.graph_def, 'out', MODEL_NAME + '.pbtxt', True)
    # op to write logs to Tensorboard
    # summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    print('---training started')
    for index in range(EPOCHS):
        rand_index = np.random.choice(texts_train.shape[0], size=batch_size)

        rand_x = texts_train[rand_index].todense()
        rand_y = np.transpose([target_train[rand_index]])

        # _, summary = sess.run([train_step, merged_summary_op], feed_dict={x_data: rand_x, y_target: rand_y})
        # summary_writer.add_summary(summary, index)

        _ = sess.run(train_step, feed_dict={x_data: rand_x,
                                            y_target: rand_y})

        if (index + 1) % display_step == 0:
            print('Accuracy on step %d is %g' % (index + 1, sess.run(accuracy, feed_dict={x_data: rand_x,
                                                                                          y_target: rand_y})))
    # saver.save(sess, out_dir + MODEL_NAME + '.chkp')
    saver.save(sess, out_dir + MODEL_NAME + '_model.ckpt')

    # How well do we perform on held-out test data?
    print("final accuracy on test set: %s" % str(sess.run(accuracy,
                                                          feed_dict={x_data: texts_test.todense(),
                                                                     y_target: np.transpose([target_test])})))

    print('---training ended')
    # export_model([input_node_name], output_node_name)
