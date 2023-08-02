import tensorflow as tf

# Carregar o modelo
q_network = tf.keras.models.load_model('desisto2.h5')

# Imprimir o resumo do modelo
print(q_network.summary())