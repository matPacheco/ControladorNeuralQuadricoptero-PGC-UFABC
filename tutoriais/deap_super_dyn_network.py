import gymnasium as gym
import numpy as np
from deap import base, creator, tools, algorithms

import tensorflow as tf
import keras
from keras import layers

# Ambiente do Gym
env = gym.make("LunarLander-v2", render_mode=None)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Parâmetros para limitar a evolução morfológica
min_neurons = 5  # Número mínimo de neurônios em uma camada oculta
max_neurons = 20  # Número máximo de neurônios em uma camada oculta

min_layers = 1
max_layers = 10

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
    model.add(layers.Dense(action_space, activation='softmax'))  # Ações
    return model

# Função para obter os pesos e topologia (número de neurônios) da rede
def model_weights_to_vector(model):
    weights = model.get_weights()
    weight_vector = np.concatenate([w.flatten() for w in weights])
    return weight_vector

def vector_to_model_weights(model, weight_vector):
    max_neurons_per_layer = get_max_weights_per_layer()

    idx_layer = 0
    new_weights = []
    for layer in model.layers:
        shapes = [w.shape for w in layer.get_weights()]
        new_weights_layer = weight_vector[idx_layer:idx_layer + max_neurons_per_layer]

        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            new_weights.append(new_weights_layer[idx:idx + size].reshape(shape))
            idx += size

        idx_layer += max_neurons_per_layer

    model.set_weights(new_weights)

# Função de fitness que também considera a estrutura da rede
def evaluate(individual):
    num_layers = int(individual[0])  # O primeiro gene representa o número de camadas
    weight_vector = np.array(individual[11:])
    model = build_model(individual)
    vector_to_model_weights(model, weight_vector)
    
    total_reward = 0
    obs, _ = env.reset()
    done = False
    while not done:
        obs = obs.reshape(1, -1)  # Reformatar observação
        action_prob = model.predict(obs, verbose=0)
        action = np.argmax(action_prob)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    return total_reward,

# Definir como inicializar indivíduos
def init_individual():
    num_layers = int(np.random.randint(min_layers, max_layers))
    num_neurons = np.random.randint(min_neurons, max_neurons, size=(max_layers,))

    num_weights_per_layer = get_max_weights_per_layer()
    num_max_weights = num_weights_per_layer * (max_layers+1)  # Considerar a camada de saída
    weights = np.random.randn(num_max_weights)

    return [num_layers] + list(num_neurons) + list(weights)

# Mutação morfológica e de pesos
def mutate(individual):
    # print("mutou")
    # print(len(individual))
    # print("mutou")
    # print("primeiro teste")
    # print(len(individual))
    # print(individual)
    # num_layers = individual[0]
    # model = build_model(individual)
    # weight_vector = np.array(individual[num_layers+1:])
    # vector_to_model_weights(model, weight_vector)

    if np.random.rand() < 0.1:  # 10% de chance de mutação morfológica na quantidade de camadas
        choice = int(np.random.choice([-1, 1]))
        individual[0] = int(np.clip(individual[0] + choice, min_layers, max_layers))

    num_layers = individual[0]
 
    # TODO: dá pra melhorar
    if np.random.rand() < 0.1: # 10% de chance de mutação morfológica na quantidade de neurônios em uma camada
        i_layer = np.random.randint(min_layers, max_layers)
        # print("i_layer")
        # print(individual[i_layer])
        individual[i_layer] = int(np.clip(individual[i_layer] + np.random.randint(-3, 4), min_neurons, max_neurons))
        # print(individual[i_layer])


    # Mutar os pesos normalmente
    individual[num_layers+1:] = [w + np.random.normal(0, 0.1) for w in individual[num_layers+1:]]
    
    # print("segundo teste")
    # print(len(individual))
    # print(individual)
    # model = build_model(individual)
    # weight_vector = np.array(individual[num_layers+1:])
    # vector_to_model_weights(model, weight_vector)
    # print("-"*64)
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
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("min", np.min)
# stats.register("max", np.max)

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_num_layers = tools.Statistics(key=lambda ind: ind[0])
stats_num_neurons = tools.Statistics(key=lambda ind: int(np.mean(ind[1:11])))
mstats = tools.MultiStatistics(fitness=stats_fit, num_layers=stats_num_layers, num_neurons_per_layer=stats_num_neurons)

mstats.register("avg", lambda values: np.round(np.mean(values), 1))
mstats.register("std", lambda values: np.round(np.std(values), 1))
mstats.register("min", lambda values: np.round(np.min(values), 1))
mstats.register("max", lambda values: np.round(np.max(values), 1))


# CXPB = crossing probability
# MUTPB = mutating probability
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations,
                    stats=mstats, halloffame=hall_of_fame, verbose=True)

# Melhor solução encontrada
best_individual = hall_of_fame[0]
best_num_layers = int(best_individual[0])
best_weights = best_individual[1:]

print(f"Melhor número de camadas: {best_num_layers}")
print(f"Fitness da melhor solução: {best_individual.fitness.values[0]}")

# Testar o modelo treinado
best_model = build_model(best_individual)
vector_to_model_weights(best_model, np.array(best_weights))

obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    obs = obs.reshape(1, -1)
    action_prob = best_model.predict(obs, verbose=0)
    action = np.argmax(action_prob)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    env.render()
env.close()

print(f"Recompensa total: {total_reward}")
