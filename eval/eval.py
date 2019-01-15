# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.contrib import learn


#self.news_title_tokens = []
#self.news_content_tokens = []
#self.items = []
SAVE_PATH = './model'
MODEL_NAME = 'model-2600'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)
tf.reset_default_graph()

with tf.Session() as sess:
    # import the saved graph
    saver = tf.train.import_meta_graph('./model/model-2600' + '.meta')
    # get the graph for this session
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    # get the tensors that we need
    headline = graph.get_tensor_by_name('input_x_head:0')
    body = graph.get_tensor_by_name('input_x_body:0')
    dropout_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
    prediction = graph.get_tensor_by_name('output/predictions:0')
    head_embedding = graph.get_tensor_by_name('Variable:0')
    body_embedding = graph.get_tensor_by_name('Variable_1:0')

    model_headline_input = tf.saved_model.utils.build_tensor_info(headline)
    model_body_input = tf.saved_model.utils.build_tensor_info(body)
    model_dropout_prob = tf.saved_model.utils.build_tensor_info(dropout_prob)
    model_prediction = tf.saved_model.utils.build_tensor_info(prediction)
    model_head_embedding = tf.saved_model.utils.build_tensor_info(head_embedding)
    model_body_embedding = tf.saved_model.utils.build_tensor_info(body_embedding)

# build signature definition
    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
            inputs ={'headline_inputs': model_headline_input,
                     'body_inputs': model_body_input,
                     'dropout_prob' : model_dropout_prob,
                     'head_embedding' : model_head_embedding,
                     'body_embedding' : model_body_embedding},
            outputs={'prediction' : model_prediction},
            method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })
    # Save the model so we can serve it with a model server :)
    builder.save()


#self.vocab_processor_head = learn.preprocessing.VocabularyProcessor(max_document_length=128)
#self.vocab_processor_body = learn.preprocessing.VocabularyProcessor(max_document_length=1280)

#self.news_title_tokens.append(self.tokenize(item['news_title']))
#self.news_content_tokens.append(self.tokenize(item['news_content']))

        
#x_head = np.array(list(self.vocab_processor_head.fit_transform(self.news_title_tokens)))
#x_body = np.array(list(self.vocab_processor_body.fit_transform(self.news_content_tokens)))

            

