from novarede.labirinto import LabirintoAmbiente
from novarede.agente import Agente
from novarede.rede_neural import QNetwork
import tensorflow as tf
import numpy as np
import os
import time
# Crie um diretório para salvar os checkpoints

def carregar_modelo(model_path):
    num_actions = 4
    q_network = QNetwork(num_actions)
    checkpoint = tf.train.Checkpoint(q_network=q_network)
    checkpoint.restore(tf.train.latest_checkpoint(model_path))
    return q_network
def verificar_modelo_treinado(model_path):
    return os.path.exists(model_path)

def listar_checkpoints(model_save_dir):
    checkpoints = []
    for filename in os.listdir(model_save_dir):
        if filename.startswith("modelo_episodio_"):
            checkpoints.append(os.path.join(model_save_dir, filename))
    return checkpoints

def train_q_learning(q_network, checkpoint) :
    num_actions = 4
    q_network = QNetwork(num_actions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_function = tf.keras.losses.MeanSquaredError()
    num_episodes = 100
    discount_factor = 0.9
    max_steps_per_episode = 100
    ambiente = LabirintoAmbiente()
    model_save_dir = os.path.join(checkpoint_dir, "ckpt")
    checkpoint.save(file_prefix='ckpt')
    for episode in range(num_episodes):
        state = ambiente.agente_pos
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Obter a ação do agente a partir do estado atual usando a rede neural
            q_values = q_network(tf.expand_dims(state, axis=0))
            action = np.argmax(q_values[0])

            # Verificar se a ação é válida
            if ambiente.actions[action] not in ambiente.actions:
                continue

            # Realizar a ação no ambiente e obter a próxima observação e recompensa
            next_state, reward, done = ambiente.step(ambiente.actions[action])

            # Obter o valor Q esperado para a ação tomada usando a rede neural
            q_values_next = q_network(tf.expand_dims(next_state, axis=0))
            expected_q_value = reward + discount_factor * np.max(q_values_next)

            # Calcular a perda
            with tf.GradientTape() as tape:
                q_values = q_network(tf.expand_dims(state, axis=0))
                q_value = tf.gather(q_values[0], action)
                loss = loss_function(tf.expand_dims(expected_q_value, axis=0), tf.expand_dims(q_value, axis=0))

            # Atualizar os parâmetros da rede neural
            gradients = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

            state = next_state
            steps += 1

            # Imprimir o labirinto e a posição do agente após cada movimento
            ambiente.imprimir_labirinto()
            time.sleep(0.1)  # Pausa para visualização

        # Verificar se o agente completou o labirinto
        if done:
            print(f"Episódio {episode + 1}: Agente completou o labirinto em {steps} passos.")
            model_save_path = os.path.join(model_save_dir, f"modelo_episodio_{episode + 1}")
            checkpoint.save(file_prefix=model_save_path)
            print("Modelo salvo:", model_save_path)

            # Carregar todos os checkpoints salvos
            checkpoints = listar_checkpoints(model_save_dir)
            for checkpoint_path in checkpoints:
                q_network = carregar_modelo(checkpoint_path)
                # Testar o modelo carregado no ambiente de teste
                testar_modelo(q_network)


if __name__ == "__main__":
    checkpoint_dir = './training_checkpoints'
    num_actions = 4
    checkpoint = tf.train.Checkpoint(q_network=QNetwork(num_actions))  # Crie o objeto Checkpoin
    if verificar_modelo_treinado(checkpoint_dir):
        print('Carregando modelo treinado...')
        q_network = carregar_modelo(model_path)
        train_q_learning(q_network, checkpoint)  # Passe o modelo carregado para a função de treinamento

    else:
        print('Criando novo modelo...')
        q_network = QNetwork(num_actions)
        train_q_learning(q_network, checkpoint)  # Passe
