import gym_pybullet_drones

import gymnasium as gym
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

import tensorflow as tf
import keras
from keras import layers
from keras import backend as K

import pickle

import os
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial, reduce
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-p", "--population", default=500, type=int)
parser.add_argument("-g", "--generations", default=100, type=int)
parser.add_argument("-e", "--elitism", default=0.1, type=float)
parser.add_argument("-m", "--microgens", default=3, type=int)
parser.add_argument("-c", "--checkpoint", action="store_true")
parser.add_argument("-w", "--workers", default=multiprocessing.cpu_count()//2, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativa a GPU
NUM_PROCESSES = args.workers
print("Número de processos:", NUM_PROCESSES)

# Ambiente do Gym
RNG = random.Random(42)
env_base = gym.make("GPS-Distance-v0", rng=0.)
observation_space = 14
action_space = env_base.action_space.shape[1]
print("Action Space:", action_space)

# Parâmetros para limitar a evolução morfológica
min_neurons = 5  # Número mínimo de neurônios em uma camada oculta
max_neurons = 20  # Número máximo de neurônios em uma camada oculta

min_layers = 1
max_layers = 10

mutation_choice = reduce(lambda r, e: 2*r + e, [[-k, k] for k in range(1, 5)], [])

def init_worker():
    tf.keras.backend.clear_session()
    global model_cache
    model_cache = {}

pool = ProcessPoolExecutor(
    max_workers=NUM_PROCESSES, 
    initializer=init_worker  # Inicializa cada worker
)

def build_model(topology):
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
    return model
    

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

def mutate_topology(individual, mutation_rate=0.3):
    if np.random.rand() < mutation_rate:  # 10% de chance de mutação na quantidade de camadas
        choice = int(np.random.choice(mutation_choice))
        individual[0] = int(np.clip(individual[0] + choice, min_layers, max_layers))

    num_layers = individual[0]

    # TODO: dá pra melhorar
    if np.random.rand() < mutation_rate: # 10% de chance de mutação na quantidade de neurônios em uma camada
        i_layer = np.random.randint(min_layers, max_layers)
        individual[i_layer] = int(np.clip(individual[i_layer] + np.random.randint(-3, 4), min_neurons, max_neurons))

    return individual,

def evaluate(topology, weights, print_info=False, rng=0.0):
    model = build_model(topology)
    weights_vector = np.array(weights)
    vector_to_model_weights(model, weights_vector)
    
    env = gym.make("GPS-Distance-v0", rng=rng)
    obs, _ = env.reset()
    done = False
    printed = True
    while not done:
        obs = np.concatenate((
            obs[0].flatten(), 
            obs[1].flatten(), 
            obs[2].flatten()), axis=0)
        obs = obs.reshape(1, -1)  # Garante (1, 14)
        action = model.predict(obs, verbose=0)
        obs, reward, done, truncated, info = env.step(action)
        if not printed:
            print("Target Position:")
            print(info["target position"])
            printed= True
        if done or truncated:
            break
    if print_info:
        print("######## Info ########")
        print("Reward:", reward)
        print(info)
    env.close()
    return reward,

def evaluate_weights(weights, topology):  # Nova função com ordem invertida
    return evaluate(topology, weights)  # Chama a função original com a ordem correta

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

def set_evolution(checkpoint=False):
    if checkpoint:
        with open("checkpoint.pkl", "rb") as cp_file:
            cp = pickle.load(cp_file)

        topology_pop = cp["topology_pop"]
        weights_pop = cp["weights_pop"]

        start_gen = cp["generation"]

        halloffame_topology = cp["halloffame_topology"]
        halloffame_weights = cp["halloffame_weights"]

        logbook_topology = cp["logbook_topology"]
        logbook_weights = cp["logbook_weights"]

        random.setstate(cp["rndstate"])

        global RNG
        RNG = cp["rng_obj"]
        print("Checkpoint carregado com sucesso.")
    else:
        topology_pop = toolbox_topology.population(n=n_population)
        weights_pop = toolbox_weights.population(n=n_population)

        start_gen = 0

        halloffame_topology = tools.HallOfFame(5, similar=np.array_equal)
        halloffame_weights = tools.HallOfFame(5, similar=np.array_equal)
        
        logbook_topology = tools.Logbook()
        logbook_topology.header = ["gen", "fitness", "num_layers", "num_neurons_per_layer"]
        logbook_topology.chapters["fitness"].header = ["média", "maior", "menor", "std"]
        logbook_topology.chapters["num_layers"].header = ["média", "maior", "menor", "std"]
        logbook_topology.chapters["num_neurons_per_layer"].header = ["média", "maior", "menor", "std"]

        logbook_weights = tools.Logbook()
        logbook_weights.header = ["gen", "média", "melhor", "pior"]

    return topology_pop, weights_pop, start_gen, halloffame_topology, halloffame_weights, logbook_topology, logbook_weights

# Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Topology", list, fitness=creator.FitnessMax) 
creator.create("Weights", list, fitness=creator.FitnessMax)

toolbox_topology = base.Toolbox()
toolbox_topology.register("individual", tools.initIterate, creator.Topology, init_topology)
toolbox_topology.register("population", tools.initRepeat, list, toolbox_topology.individual)
toolbox_topology.register("evaluate", evaluate)
toolbox_topology.register("mate", tools.cxUniform, indpb=0.3)
toolbox_topology.register("mutate", mutate_topology)
# toolbox_topology.register("select", tools.selBest)
toolbox_topology.register("select", tools.selTournament, tournsize=3)
toolbox_topology.register("map", pool.map)

toolbox_weights = base.Toolbox()
toolbox_weights.register("individual", tools.initIterate, creator.Weights, init_weights)
toolbox_weights.register("population", tools.initRepeat, list, toolbox_weights.individual)
# toolbox_weights.register("evaluate", evaluate_weights)
toolbox_weights.register("evaluate", evaluate)
toolbox_weights.register("mate", tools.cxUniform, indpb=0.7)
# toolbox_weights.register("mutate", mutate_weights, indpb=0.1)
toolbox_weights.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
# toolbox_weights.register("select", tools.selTournament, tournsize=3)
toolbox_weights.register("select", tools.selBest)
toolbox_weights.register("map", pool.map)

# Configurar estatísticas
fitness_topology = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats_num_neurons = tools.Statistics(key=lambda ind: int(np.mean(ind[1:11])))
stats_num_layers = tools.Statistics(key=lambda ind: ind[0])
stats_topology = tools.MultiStatistics(
    fitness=fitness_topology, 
    num_layers=stats_num_layers, 
    num_neurons_per_layer=stats_num_neurons
)

stats_topology.register("média", np.mean)
stats_topology.register("maior", np.max)
stats_topology.register("menor", np.min)
stats_topology.register("std", lambda values: np.round(np.std(values), 1))

stats_weights = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats_weights.register("média", np.mean)
stats_weights.register("melhor", max)
stats_weights.register("pior", min)
stats_topology.register("std", lambda values: np.round(np.std(values), 2))

n_population = args.population
elitism_rate = args.elitism
n_elite = int(n_population * elitism_rate)

num_generations = args.generations

topology_pop, weights_pop, start_gen, halloffame_topology, halloffame_weights, logbook_topology, logbook_weights = set_evolution(checkpoint=args.checkpoint)

df = pd.DataFrame()
micro_gens = args.microgens
for gen in range(start_gen, num_generations):
    print(f"Geração {gen}")	
    print("Evolução topológica:")

    rng = RNG.random()

    best_weight = weights_pop[0] if gen == 0 else halloffame_weights[0]
    evaluate_with_weight = partial(toolbox_topology.evaluate, weights=best_weight, rng=rng)
    for _ in range(micro_gens):
        completed = False
        for i in range(5):
            try:
                fitnesses = toolbox_topology.map(evaluate_with_weight, topology_pop)
                completed = True
                break
            except BrokenProcessPool:
                print("Erro no pool, tentando novamente...")

                pool.shutdown(wait=True)  # Encerra o pool atual
                # Cria um novo pool e atualiza a referência nos workers do toolbox
                pool = ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=init_worker)
                toolbox_topology.register("map", pool.map)
                toolbox_weights.register("map", pool.map)

        if not completed:
            raise RuntimeError("Falha ao avaliar em paralelo após várias tentativas.")

            
        # Atribuir fitness
        for topology, fit in zip(topology_pop, fitnesses):
            topology.fitness.values = fit

        # Registrar estatísticas no logbook
        record_topology = stats_topology.compile(topology_pop)
        logbook_topology.record(gen=gen, **record_topology)  # Adiciona a geração atual
    
        # Atualizar Hall of Fame
        halloffame_topology.update(topology_pop)

        # Imprimir estatísticas da geração
        print(logbook_topology.stream)  # Saída formatada

        # Selecionar elite
        elite_topology = tools.selBest(topology_pop, k=n_elite)
        
        # Selecionar pais
        parents = toolbox_topology.select(topology_pop, len(topology_pop) - n_elite)
        
        # Gerar filhos
        offspring_topology = algorithms.varAnd(parents, toolbox_topology, cxpb=0.7, mutpb=0.2)
        
        # Nova população = filhos + elite
        topology_pop[:] = offspring_topology + elite_topology


    
    print("-"*64)
    print("Evolução de pesos")
    best_topology = topology_pop[0] if gen == 0 else halloffame_topology[0]
    # Avaliar pesos em paralelo
    evaluate_with_topology = partial(toolbox_weights.evaluate, best_topology, rng=rng)

    completed = False
    for _ in range(micro_gens):
        for i in range(5):
            try:
                fitnesses = toolbox_weights.map(evaluate_with_topology, weights_pop)
                completed = True
                break
            except BrokenProcessPool:
                print("Erro no pool, reinicializando o ProcessPoolExecutor...")

                pool.shutdown(wait=True)  # Encerra o pool atual
                # Cria um novo pool e atualiza a referência nos workers do toolbox
                pool = ProcessPoolExecutor(max_workers=NUM_PROCESSES, initializer=init_worker)
                toolbox_topology.register("map", pool.map)
                toolbox_weights.register("map", pool.map)

        if not completed:
            raise RuntimeError("Falha ao avaliar em paralelo após várias tentativas.")
                
        # Atribuir fitness
        for weights, fit in zip(weights_pop, fitnesses):
            weights.fitness.values = fit
    
        # Registrar estatísticas no logbook
        record_weights = stats_weights.compile(weights_pop)
        logbook_weights.record(gen=gen, **record_weights)  # Adiciona a geração atual
        
        # Imprimir estatísticas da geração
        print(logbook_weights.stream)  # Saída formatada

        # Atualizar Hall of Fame
        halloffame_weights.update(weights_pop)

        # Selecionar elite
        elite_weights = tools.selBest(weights_pop, k=n_elite)
        
        # Selecionar pais
        parents = toolbox_weights.select(weights_pop, len(weights_pop) - n_elite)
        
        # Gerar filhos
        offspring_weights = algorithms.varAnd(parents, toolbox_weights, cxpb=0.7, mutpb=0.2)
        
        # Nova população = filhos + elite
        weights_pop[:] = offspring_weights + elite_weights


    print("MELHOR INDIVÍDUO:")
    print("Fitness:", halloffame_weights[0].fitness.values)
    evaluate(halloffame_topology[0], halloffame_weights[0], print_info=True, rng=rng)
    print("-"*128)

    best_individual = halloffame_topology[0] + halloffame_weights[0]

    new_data = {
        "index": [gen],
        "individual": [best_individual]
    }

    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df])
    df.to_csv("best_individual.csv", index=False)

    df_topology = pd.DataFrame(logbook_topology)
    df_topology.to_csv("topologia.csv", index=False)

    df_weights = pd.DataFrame(logbook_weights)
    df_weights.to_csv("pesos.csv", index=False)

    cp = dict(topology_pop=topology_pop, weights_pop=weights_pop, generation=gen, 
              halloffame_topology=halloffame_topology, halloffame_weights=halloffame_weights, 
              logbook_topology=logbook_topology, logbook_weights=logbook_weights, rndstate=random.getstate(),
              rng_obj=RNG)

    with open("checkpoint.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)