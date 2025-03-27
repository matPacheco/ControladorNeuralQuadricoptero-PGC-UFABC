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


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativa a GPU
NUM_PROCESSES = multiprocessing.cpu_count()
NUM_PROCESSES = 6
print("Número de processos:", NUM_PROCESSES)

# Ambiente do Gym
rng = random.Random(42)
env = gym.make("GPS-Distance-v0", rng=rng)
observation_space = 14
action_space = env.action_space.shape[1]
print("Action Space:", action_space)

# Parâmetros para limitar a evolução morfológica
min_neurons = 5  # Número mínimo de neurônios em uma camada oculta
max_neurons = 20  # Número máximo de neurônios em uma camada oculta

min_layers = 1
max_layers = 10

# # Função para inicializar cada processo worker
# def init_worker():
#     global model_cache
#     model_cache = {}
#     # Limpa qualquer estado residual do TF
#     tf.keras.backend.clear_session()
global model_cache
model_cache = {}
# Função para construir uma rede com um número variável de neurônios na camada oculta
def build_model(topology):
    key = tuple(topology)  # Topologia como chave do cache

    if key not in model_cache:
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

def mutate_topology(individual, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:  # 10% de chance de mutação na quantidade de camadas
        choice = int(np.random.choice([-1, 1]))
        individual[0] = int(np.clip(individual[0] + choice, min_layers, max_layers))

    num_layers = individual[0]

    # TODO: dá pra melhorar
    if np.random.rand() < mutation_rate: # 10% de chance de mutação na quantidade de neurônios em uma camada
        i_layer = np.random.randint(min_layers, max_layers)
        individual[i_layer] = int(np.clip(individual[i_layer] + np.random.randint(-3, 4), min_neurons, max_neurons))

    return individual,

def evaluate(topology, weights):
    model = build_model(topology)
    weights_vector = np.array(weights)
    vector_to_model_weights(model, weights_vector)

    total_reward = 0
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
        total_reward += reward
        if done or truncated:
            break
    return reward,

# Função para inicializar indivíduos de topologia
def init_topology():
    num_layers = int(np.random.randint(min_layers, max_layers))
    num_neurons = np.random.randint(min_neurons, max_neurons, size=(max_layers,))

    return [num_layers] + list(num_neurons)

def get_max_weights_per_layer():
    return (max_neurons * max_neurons) + max_neurons

# Função para inicializar indivíduos de pesos
def init_weights():
    num_weights_per_layer = get_max_weights_per_layer()
    num_max_weights = num_weights_per_layer * (max_layers+1)  # Considerar a camada de saída
    weights = np.random.randn(num_max_weights).astype(np.float64).tolist()

    return weights

# Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Topology", list, fitness=creator.FitnessMax) 
creator.create("Weights", list, fitness=creator.FitnessMax)

toolbox_topology = base.Toolbox()
toolbox_topology.register("individual", tools.initIterate, creator.Topology, init_topology)
toolbox_topology.register("population", tools.initRepeat, list, toolbox_topology.individual)
toolbox_topology.register("evaluate", evaluate)
toolbox_topology.register("mate", tools.cxUniform, indpb=0.5)
toolbox_topology.register("mutate", mutate_topology)
toolbox_topology.register("select", tools.selBest)
# toolbox_topology.register("select", tools.selTournament, tournsize=3)

toolbox_weights = base.Toolbox()
toolbox_weights.register("individual", tools.initIterate, creator.Weights, init_weights)
toolbox_weights.register("population", tools.initRepeat, list, toolbox_weights.individual)
toolbox_weights.register("evaluate", evaluate)
toolbox_weights.register("mate", tools.cxUniform, indpb=0.5)
# toolbox_weights.register("mutate", mutate_weights, indpb=0.1)
toolbox_weights.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
# toolbox_weights.register("select", tools.selTournament, tournsize=3)
toolbox_weights.register("select", tools.selBest)

# # Estatísticas
# stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
# stats_num_layers = tools.Statistics(key=lambda ind: ind[0])
# stats_num_neurons = tools.Statistics(key=lambda ind: int(np.mean(ind[1:11])))
# mstats_topology = tools.MultiStatistics(fitness=stats_fit, num_layers=stats_num_layers, num_neurons_per_layer=stats_num_neurons)

# mstats_topology.register("avg", lambda values: np.round(np.mean(values), 1))
# mstats_topology.register("std", lambda values: np.round(np.std(values), 1))
# mstats_topology.register("min", lambda values: np.round(np.min(values), 1))
# mstats_topology.register("max", lambda values: np.round(np.max(values), 1))

# # Logbook Topologia
# logbook_topology = tools.Logbook()
# logbook_topology.header = ["gen", "fitness", "layers", "neurons"]
# logbook_topology.chapters["fitness"].header = ["avg", "min", "max"]
# logbook_topology.chapters["layers"].header = ["avg", "min", "max"]
# logbook_topology.chapters["neurons"].header = ["avg", "min", "max"]

# # Estatísticas de pesos
# stats_fit_weights = tools.Statistics(key=lambda ind: ind.fitness.values[0])
# mstats_weights = tools.MultiStatistics(fitness=stats_fit_weights)
# mstats_weights["fitness"].register("avg", np.mean)
# mstats_weights["fitness"].register("min", np.min)
# mstats_weights["fitness"].register("max", np.max)

# logbook_weights = tools.Logbook()
# logbook_weights.header = ["gen", "fitness"]
# logbook_weights.chapters["fitness"].header = ["avg", "min", "max"]

# Configurar estatísticas
stats_topology = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats_topology.register("média", np.mean)
stats_topology.register("melhor", max)
stats_topology.register("pior", min)

stats_weights = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats_weights.register("média", np.mean)
stats_weights.register("melhor", max)
stats_weights.register("pior", min)

# Configurar logbooks
logbook_topology = tools.Logbook()
logbook_topology.header = ["geração", "média", "melhor"]

logbook_weights = tools.Logbook()
logbook_weights.header = ["geração", "média", "melhor"]

num_generations = 50
topology_pop = toolbox_topology.population(n=50)
weights_pop = toolbox_weights.population(n=50)

halloffame_topology = tools.HallOfFame(5, similar=np.array_equal)
halloffame_weights = tools.HallOfFame(5, similar=np.array_equal)

for gen in range(num_generations):
    print(f"Geração {gen}")	
    print("Evolução topológica:")
    # Avaliar topologias usando os melhores pesos
    for topology in topology_pop:
        try:
            best_weight = halloffame_weights[0]
        except IndexError:  # Primeira geração
            best_weight = weights_pop[0]
        topology.fitness.values = toolbox_topology.evaluate(topology, best_weight)
    # Registrar estatísticas no logbook
    record_topology = stats_topology.compile(topology_pop)
    logbook_topology.record(gen=gen, **record_topology)  # Adiciona a geração atual
    
    # Imprimir estatísticas da geração
    print(logbook_topology.stream)  # Saída formatada

    print("-"*64)
    print("Evolução de pesos")
    # Avaliar pesos usando as melhores topologias
    for weights in weights_pop:
        try:
            best_topology = halloffame_topology[0]
        except IndexError:  # Primeira geração
            best_topology = topology_pop[0]
        weights.fitness.values = toolbox_weights.evaluate(best_topology, weights)

    # Registrar estatísticas no logbook
    record_weights = stats_weights.compile(weights_pop)
    logbook_weights.record(gen=gen, **record_weights)  # Adiciona a geração atual
    
    # Imprimir estatísticas da geração
    print(logbook_weights.stream)  # Saída formatada

    # Evoluir topologias
    offspring_topology = algorithms.varAnd(topology_pop, toolbox_topology, cxpb=0.7, mutpb=0.2)
    topology_pop[:] = offspring_topology
    
    # Evoluir pesos
    offspring_weights = algorithms.varAnd(weights_pop, toolbox_weights, cxpb=0.7, mutpb=0.2)
    weights_pop[:] = offspring_weights
    
    # Atualizar Hall of Fame
    halloffame_topology.update(topology_pop)
    halloffame_weights.update(weights_pop)

    print("MELHOR INDIVÍDUO:")
    print("Fitness:", halloffame_weights[0].fitness.values)
    print("-"*128)

df_topology = pd.DataFrame(logbook_topology)
df_topology.to_csv("topologia.csv", index=False)

df_weights = pd.DataFrame(logbook_weights)
df_weights.to_csv("pesos.csv", index=False)