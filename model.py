import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions, agent_history_length, resized_width, resized_height, name_scope):
  with tf.device("/cpu:0"):
    with tf.name_scope(name_scope):
        # state = tf.placeholder(tf.float32, [None, agent_history_length, resized_width, resized_height], name="state")
        state = tf.placeholder(tf.float32, [None, 1024], name="state")
        inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
        inputs = Input(shape=(1, 1, 1024,))
        # model = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu', padding='same', data_format='channels_first')(inputs)
        # model = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same', data_format='channels_first')(model)
        #model = Conv2D(filter=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(model)
        # model = Flatten()(inputs)
        model = Dense(1024, activation='relu')(inputs)
        q_values = Dense(num_actions)(model)
        
        #UserWarning: Update your `Model` call to the Keras 2 API: 
        # `Model(outputs=Tensor("de..., inputs=Tensor("in..
        m = Model(inputs=inputs, outputs=q_values)
        
  return state, m
