import gymnasium as gym
import numpy as np
import pygad
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
import time  # Para adicionar atraso

# Criação do ambiente CartPole-v1
env = gym.make("CartPole-v1", render_mode=None)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Definir a rede neural em Keras
def build_model():
    model = Sequential()
    model.add(Input(shape=(observation_space,)))  # Definir a entrada do modelo
    model.add(Dense(10, activation='relu'))
    model.add(Dense(action_space, activation='softmax'))  # Probabilidade para cada ação
    return model

model = build_model()
model.summary()

# Converter a rede neural para um vetor de pesos
def model_weights_to_vector(model):
    weights = model.get_weights()
    weight_vector = np.concatenate([w.flatten() for w in weights])
    return weight_vector

# Definir uma função para converter um vetor de pesos de volta para a rede neural
def vector_to_model_weights(model, weight_vector):
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(weight_vector[idx:idx+size].reshape(shape))
        idx += size
    model.set_weights(new_weights)

# Função de fitness atualizada com três parâmetros
def fitness_function(ga_instance, solution, solution_idx):
    vector_to_model_weights(model, solution)  # Aplicar pesos na rede neural
    total_reward = 0
    obs, _ = env.reset()
    done = False
    while not done:
        obs = obs.reshape(1, -1)  # Reformatar a observação para input da rede
        action_prob = model.predict(obs, verbose=0)
        action = np.argmax(action_prob)  # Escolher ação com maior probabilidade
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    return total_reward

# Callback para printar a melhor recompensa a cada geração
def on_generation(ga_instance):
    best_solution_fitness = ga_instance.best_solution()[1]
    print(f"Geração {ga_instance.generations_completed} | Melhor recompensa: {best_solution_fitness}")
    print(ga_instance.best_solution())
    
# Parâmetros do Algoritmo Genético
sol_per_pop = 50  # Tamanho da população
num_generations = 50
num_parents_mating = 20
num_genes = len(model_weights_to_vector(model))  # Número de pesos da rede neural

# Executar o algoritmo genético
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       mutation_percent_genes=10,
                       on_generation=on_generation)

ga_instance.run()

# Melhor solução encontrada
best_solution, best_solution_fitness, _ = ga_instance.best_solution()
print(f"Melhor solução: {best_solution}")
print(f"Fitness da melhor solução: {best_solution_fitness}")

# Aplicar os melhores pesos na rede neural
vector_to_model_weights(model, best_solution)

# Testar a melhor rede treinada
obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    obs = obs.reshape(1, -1)
    action_prob = model.predict(obs, verbose=0)
    action = np.argmax(action_prob)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    env.render()  # Renderizar o ambiente
env.close()

print(f"Recompensa total após treinamento: {total_reward}")
