import gym_pybullet_drones

import gymnasium as gym
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

import tensorflow as tf
import keras
from keras import layers
from keras import backend as K

import os
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-p", "--population", default=500, type=int)
parser.add_argument("-g", "--generations", default=100, type=int)
parser.add_argument("-c", "--checkpoint", action="store_true")
parser.add_argument("-w", "--workers", default=multiprocessing.cpu_count()//2, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativa a GPU
NUM_PROCESSES = args.workers
print("Número de processos:", NUM_PROCESSES)

# Ambiente do Gym
RNG = random.Random(42)
env = gym.make("GPS-Distance-v0", rng=np.pi)
observation_space = 14
action_space = env.action_space.shape[1]
print("Action Space:", action_space)

# Parâmetros para limitar a evolução morfológica
min_neurons = 5  # Número mínimo de neurônios em uma camada oculta
max_neurons = 20  # Número máximo de neurônios em uma camada oculta

min_layers = 1
max_layers = 10

# Função para inicializar cada processo worker
def init_worker():
    global model_cache
    model_cache = {}
    # Limpa qualquer estado residual do TF
    tf.keras.backend.clear_session()

# Função para obter a quantidade de parâmetros em determinado modelo
def get_num_weights(individual):
    num_layers = individual[0]

    i = 1
    previous_layer = observation_space
    num_weights = 0
    # Camadas ocultas
    for _ in range(num_layers):
        current_layer = int(np.clip(individual[i], min_neurons, max_neurons))
        num_weights += (previous_layer * current_layer) + current_layer
        
        previous_layer = current_layer
        i += 1

    # Camada de saída
    current_layer = action_space
    num_weights += (previous_layer * current_layer) + current_layer
    return num_weights

def get_max_weights_per_layer():
    return (max_neurons * max_neurons) + max_neurons

# Função para construir uma rede com um número variável de neurônios na camada oculta
def build_model(individual):
    num_layers = individual[0]
    key = tuple(individual[:11])  # Topologia como chave do cache

    if key not in model_cache:
        model = keras.models.Sequential()
        model.add(layers.Input((observation_space,)))  # Definir a entrada do modelo

        # Camadas Ocultas
        num_layers = individual[0]
        i = 1
        for _ in range(num_layers):
            # Garantir que está dentro do intervalo desejado
            num_neurons = int(np.clip(individual[i], min_neurons, max_neurons))
            model.add(layers.Dense(num_neurons, activation='relu'))
            i += 1
        model.add(layers.Dense(action_space, activation='tanh'))  # Ações
        model_cache[key] = model

    return keras.models.clone_model(model_cache[key])


# Função para obter os pesos e topologia (número de neurônios) da rede
def model_weights_to_vector(model):
    weights = model.get_weights()
    weight_vector = np.concatenate([w.flatten() for w in weights])
    return weight_vector

def vector_to_model_weights(model, weight_vector):
    new_weights = []
    idx = 0
    for layer in model.layers:
        for w in layer.get_weights():
            size = np.prod(w.shape)
            # Extraímos exatamente 'size' elementos do vetor
            new_w = weight_vector[idx: idx + size].reshape(w.shape)
            new_weights.append(new_w)
            idx += size
    model.set_weights(new_weights)


# Função de fitness que também considera a estrutura da rede
def evaluate(individual):
    num_layers = int(individual[0])  # O primeiro gene representa o número de camadas
    weight_vector = np.array(individual[11:])
    model = build_model(individual)
    vector_to_model_weights(model, weight_vector)

    obs, _ = env.reset()
    done = False
    while not done:
        obs = np.concatenate((
            obs[0].flatten(), 
            obs[1].flatten(), 
            obs[2].flatten()), axis=0)
        obs = obs.reshape(1, -1)  # Garante (1, 14)
        action = model.predict(obs, verbose=0)
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    if reward >= 50:
        for _ in range(15):
            obs = np.concatenate((
                obs[0].flatten(), 
                obs[1].flatten(), 
                obs[2].flatten()), axis=0)
            obs = obs.reshape(1, -1)  # Garante (1, 14)
            action = model.predict(obs, verbose=0)
            obs, reward, done, truncated, _ = env.step(action)

    return reward,

# Definir como inicializar indivíduos
def init_individual():
    num_layers = int(np.random.randint(min_layers, max_layers))
    num_neurons = np.random.randint(min_neurons, max_neurons, size=(max_layers,))

    num_weights_per_layer = get_max_weights_per_layer()
    num_max_weights = num_weights_per_layer * (max_layers+1)  # Considerar a camada de saída
    weights = np.random.randn(num_max_weights).astype(np.float64).tolist()

    return [num_layers] + list(num_neurons) + weights

# Mutação morfológica e de pesos
def mutate(individual):
    if np.random.rand() < 0.1:  # 10% de chance de mutação morfológica na quantidade de camadas
        choice = int(np.random.choice([-1, 1]))
        individual[0] = int(np.clip(individual[0] + choice, min_layers, max_layers))

    num_layers = individual[0]
 
    # TODO: dá pra melhorar
    if np.random.rand() < 0.1: # 10% de chance de mutação morfológica na quantidade de neurônios em uma camada
        i_layer = np.random.randint(min_layers, max_layers)
        individual[i_layer] = int(np.clip(individual[i_layer] + np.random.randint(-3, 4), min_neurons, max_neurons))

    # Mutar os pesos normalmente
    individual[num_layers+1:] = [w + np.random.normal(0, 0.1) for w in individual[num_layers+1:]]

    return individual,

# Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Registro de operações genéticas
toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# População inicial
population = toolbox.population(n=50)

# Executar algoritmo genético
num_generations = 50
hall_of_fame = tools.HallOfFame(1)

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_num_layers = tools.Statistics(key=lambda ind: ind[0])
stats_num_neurons = tools.Statistics(key=lambda ind: int(np.mean(ind[1:11])))
mstats = tools.MultiStatistics(fitness=stats_fit, num_layers=stats_num_layers, num_neurons_per_layer=stats_num_neurons)

mstats.register("avg", lambda values: np.round(np.mean(values), 1))
mstats.register("std", lambda values: np.round(np.std(values), 1))
mstats.register("min", lambda values: np.round(np.min(values), 1))
mstats.register("max", lambda values: np.round(np.max(values), 1))

logbook = tools.Logbook()
logbook.header = ["gen", "fitness", "layers"]  # Cabeçalho para MultiStatistics
logbook.chapters["fitness"].header = ["média", "melhor", "pior"]
logbook.chapters["layers"].header = ["média_camadas"]

pool = ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=init_worker)
toolbox.register("map", pool.map)

# CXPB = crossing probability
# MUTPB = mutating probability
final_pop, logbook = algorithms.eaSimple(
    population, 
    toolbox, 
    cxpb=0.7, 
    mutpb=0.2, 
    ngen=num_generations,
    stats=mstats, 
    halloffame=hall_of_fame, 
    verbose=True
)

df = pd.DataFrame(logbook)
df.to_csv("primeiro_algoritmo.csv")

# Melhor solução encontrada
best_individual = hall_of_fame[0]
best_num_layers = int(best_individual[0])
best_num_neurons_per_layer = best_individual[1:11]
best_weights = best_individual[11:]

print(f"Melhor número de camadas: {best_num_layers}")
print("Melhor número de neurônios por camada:", best_num_neurons_per_layer)
print(f"Fitness da melhor solução: {best_individual.fitness.values[0]}")

# Testar o modelo treinado
best_model = build_model(best_individual)
vector_to_model_weights(best_model, np.array(best_weights))

obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    obs_processed = np.concatenate([obs[0].flatten(), obs[1].flatten(), obs[2].flatten()])
    obs_processed = obs_processed.reshape(1, -1).astype(np.float32)
    
    action = best_model.predict(obs_processed, verbose=0)[0]
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    env.render()
env.close()

print(f"Recompensa total: {total_reward}")
