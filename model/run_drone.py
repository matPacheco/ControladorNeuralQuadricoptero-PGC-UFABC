import gym_pybullet_drones
import gymnasium as gym
import numpy as np
import random
from keras import layers
import keras
import pandas as pd


env = gym.make("GPS-Distance-v0", rng=random.Random(42), gui=True)
action_space = env.action_space.shape[1]
observation_space = 14

min_neurons = 5  # Número mínimo de neurônios em uma camada oculta
max_neurons = 20  # Número máximo de neurônios em uma camada oculta

def build_model():
    # Leitura do melhor indivíduo
    df = pd.read_csv("best_individual.csv")
    # best_individual = df.iloc[49]["individual"]
    best_individual = df.tail(1)["individual"]
    best_individual = best_individual.replace("[", "").replace("]", "").split(", ")

    topology = best_individual[-11:]
    topology = [int(n) for n in topology]
    weights_vector = best_individual[:-11]
    weights_vector = np.array([float(n) for n in weights_vector])

    # Criação da rede
    model = keras.models.Sequential()
    model.add(layers.Input((observation_space,)))  # Definir a entrada do modelo

    # Camadas Ocultas
    num_layers = topology[0]
    i = 1
    for _ in range(num_layers):
        # Garantir que está dentro do intervalo desejado
        num_neurons = int(np.clip(topology[i], min_neurons, max_neurons))
        model.add(layers.Dense(num_neurons, activation='relu'))
        i += 1
    model.add(layers.Dense(action_space, activation='tanh'))  # Ações

    # Setar os parâmetros da rede neural
    new_weights = []
    idx = 0
    for layer in model.layers:
        for w in layer.get_weights():
            size = np.prod(w.shape)
            # Extraímos exatamente 'size' elementos do vetor
            new_w = weights_vector[idx: idx + size].reshape(w.shape)
            new_weights.append(new_w)
            idx += size
    model.set_weights(new_weights)
    return model

def run_env(model):
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0
    while not done:
        obs = np.concatenate((
            obs[0].flatten(), 
            obs[1].flatten(), 
            obs[2].flatten()), axis=0)
        obs = obs.reshape(1, -1)  # Garante (1, 14)
        action = model.predict(obs, verbose=0)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    return reward,


model = build_model()
run_env(model)