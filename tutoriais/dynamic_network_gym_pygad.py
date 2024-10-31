import gymnasium as gym
import numpy as np
import pygad
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

env = gym.make("CartPole-v1", render_mode=None)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Parâmetros para limitar a evolução morfológica
min_neurons = 5  # Número mínimo de neurônios em uma camada oculta
max_neurons = 20  # Número máximo de neurônios em uma camada oculta

# Função para construir uma rede com um número variável de neurônios na camada oculta
def build_model(num_neurons):
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
def fitness_function(ga_instance, solution, _):
    num_neurons = int(solution[0])  # O primeiro gene representa o número de neurônios
    weight_vector = solution[1:]  # Os demais genes representam os pesos
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
    return total_reward

# Definir o tamanho dinâmico da rede e número de genes (inclui neurônios e pesos)
def calculate_num_genes(num_neurons):
    model = build_model(num_neurons)
    num_genes = len(model_weights_to_vector(model)) + 1  # +1 para armazenar o número de neurônios
    return num_genes

def mutation_function(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        if np.random.rand() < 0.1:  # 10% de chance de mutação morfológica
            # Mutar o número de neurônios com limite
            offspring[i, 0] = int(np.clip(offspring[i, 0] + np.random.randint(-3, 4), min_neurons, max_neurons))
        # Mutar os pesos normalmente
        offspring[i, 1:] += np.random.normal(0, 0.1, offspring[i, 1:].shape)
    return offspring


# Callback para cada geração
def on_generation(ga_instance):
    best_solution_fitness = ga_instance.best_solution()[1]
    print(f"Geração {ga_instance.generations_completed} | Melhor recompensa: {best_solution_fitness}")

# Parâmetros do AG
num_neurons_initial = np.random.randint(min_neurons, max_neurons)
num_genes = calculate_num_genes(num_neurons_initial)

ga_instance = pygad.GA(num_generations=50,
                       num_parents_mating=20,
                       fitness_func=fitness_function,
                       sol_per_pop=50,
                       num_genes=num_genes,
                       mutation_percent_genes=5,
                       mutation_type=mutation_function,
                       on_generation=on_generation)

ga_instance.run()

# Melhor solução encontrada
best_solution, best_solution_fitness, _ = ga_instance.best_solution()
best_num_neurons = int(best_solution[0])
best_weights = best_solution[1:]

# Aplicar a solução para a rede
best_model = build_model(best_num_neurons)
vector_to_model_weights(best_model, best_weights)

print(f"Melhor número de neurônios: {best_num_neurons}")
print(f"Fitness da melhor solução: {best_solution_fitness}")

# Testar a rede neural treinada
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
