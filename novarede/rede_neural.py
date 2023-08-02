import tensorflow as tf
import os
class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_actions)
        self.num_actions = num_actions
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(q_network=self, optimizer=self.optimizer)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))


    def initialize_model(self, state):
        self(tf.expand_dims(state, axis=0))

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def predict(self, state):
        return self(tf.expand_dims(state, axis=0))

    def save_model(self, model_path):
        self.save_weights(model_path)

    def load_model(self, model_path):
        self.load_weights(model_path)