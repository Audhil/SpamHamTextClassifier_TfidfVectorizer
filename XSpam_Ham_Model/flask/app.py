from flask import Flask, request, jsonify
import pickle
import os
import string
import tensorflow as tf
from apptokenizer import tokenizer

app = Flask(__name__)


# prediction happens here
def vectorize_user_input(user_input_text):
    if not os.path.exists(out_dir + tf_idf_file):
        return 'Pickle file not found'

    with open(out_dir + tf_idf_file, "rb") as file:
        print('---get vectorizer from pickle')
        vectorizer = pickle.load(file)
        user_input_text = user_input_text.lower()
        user_input_text = ''.join(c for c in user_input_text if c not in string.punctuation)  # remove punctuations
        user_input_text = ''.join(c for c in user_input_text if c not in '0123456789')  # remove digits
        user_input_text = ' '.join(user_input_text.split())  # trim extra spaces
        user_input_text = user_input_text.split()
        sparse_texts = vectorizer.transform(user_input_text)
        return sparse_texts[0]


@app.route('/api/get_text_prediction', methods=['POST'])
def get_text_prediction():
    """
    predicts requested text whether it is ham or spam
    :return: json
    """
    json = request.get_json()
    print(json)
    if len(json['text']) == 0:
        return jsonify({'error': 'invalid input'})

    print('---user_data before processing :: ', json['text'])
    user_input_text = vectorize_user_input(json['text'])
    print('---user_data after vectorized user_input_text.shape :: ', user_input_text.shape)
    print('---user_data after vectorized user_input_text.todense() :: ', user_input_text.todense())

    # load the model
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(out_dir + MODEL_NAME + '_model.ckpt'))
            saver.restore(sess, out_dir + MODEL_NAME + '_model.ckpt')

            # Get the placeholders from the graph by name
            input_ = graph.get_operation_by_name(input_node_name).outputs[0]
            output_ = graph.get_operation_by_name(output_node_name).outputs[0]
            prediction_ = graph.get_operation_by_name(prediction_node_name).outputs[0]

            # Make the prediction
            # value = sess.run(prediction_, feed_dict={input_: user_input_text.todense()})
            value = sess.run(output_, feed_dict={input_: user_input_text.todense()})
            print('---TTT :: value :: ', value)
            print('---TTT :: len(value) :: ', len(value))
            print('---TTT :: value[0] :: ', value[0])

    return jsonify({'you sent this': 'thank you!'})


@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "Test!!!!"


@app.route('/users/<user>')
def hello_user(user):
    """
    this serves as a demo purpose
    :param user:
    :return: str
    """
    return "Hello %s!" % user


if __name__ == '__main__':
    out_dir = '/Users/mohammed-2284/Documents/ZTF_projects/XSpam_Ham_Model/out/'
    tf_idf_file = 'tfidf.pickle'
    MODEL_NAME = 'spam_ham_text_classifier'
    model_ckpt = '_model.ckpt'
    input_node_name = 'MODEL_INPUTT'
    output_node_name = 'MODEL_OUTPUTT'
    prediction_node_name = 'MODEL_PREDICTION'
    app.run(host='0.0.0.0', port=5000)
