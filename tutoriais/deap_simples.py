import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from deap import base, creator, tools, algorithms

# Ambiente do Gym
env = gym.make("LunarLander-v2", render_mode=None)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Parâmetros para limitar a evolução morfológica
min_neurons = 5  # Número mínimo de neurônios em uma camada oculta
max_neurons = 20  # Número máximo de neurônios em uma camada oculta

# Função para construir uma rede com um número variável de neurônios na camada oculta
def build_model(num_neurons):
    if num_neurons < min_neurons:
        num_neurons = min_neurons
    if num_neurons > max_neurons:
        num_neurons = max_neurons
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=observation_space, activation='relu'))
    model.add(Dense(action_space, activation='softmax'))  # Ações
    return model

# Função para obter os pesos e topologia (número de neurônios) da rede
def model_weights_to_vector(model):
    weights = model.get_weights()
    weight_vector = np.concatenate([w.flatten() for w in weights])
    return weight_vector

def vector_to_model_weights(model, weight_vector):
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(weight_vector[idx:idx + size].reshape(shape))
        idx += size
    model.set_weights(new_weights)

# Função de fitness que também considera a estrutura da rede
def evaluate(individual):
    num_neurons = int(individual[0])  # O primeiro gene representa o número de neurônios
    weight_vector = np.array(individual[1:])  # Os demais genes representam os pesos
    model = build_model(num_neurons)
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

# Configuração do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Definir como inicializar indivíduos
def init_individual():
    # num_neurons = np.random.randint(min_neurons, max_neurons)
    num_neurons = 10000
    model = build_model(num_neurons)
    weight_vector = model_weights_to_vector(model)
    return [num_neurons] + list(weight_vector)

# Registro de operações genéticas
toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)

# Mutação morfológica e de pesos
def mutate(individual):
    if np.random.rand() < 0.1:  # 10% de chance de mutação morfológica
        individual[0] = int(np.clip(individual[0] + np.random.randint(-3, 4), min_neurons, max_neurons))
    # Mutar os pesos normalmente
    individual[1:] = [w + np.random.normal(0, 0.1) for w in individual[1:]]
    return individual,

toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# População inicial
population = toolbox.population(n=50)

# Executar algoritmo genético
num_generations = 50
hall_of_fame = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# CXPB = crossing probability
# MUTPB = mutating probability
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations,
                    stats=stats, halloffame=hall_of_fame, verbose=True)

# Melhor solução encontrada
best_individual = hall_of_fame[0]
best_num_neurons = int(best_individual[0])
best_weights = best_individual[1:]

print(f"Melhor número de neurônios: {best_num_neurons}")
print(f"Fitness da melhor solução: {best_individual.fitness.values[0]}")

# Testar o modelo treinado
best_model = build_model(best_num_neurons)
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
