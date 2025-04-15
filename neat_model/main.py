import random
import os

import neat
from neat.genome import DefaultGenome
import visualize

import numpy as np

import gym_pybullet_drones
import gymnasium as gym


RNG = random.Random(42)


class DroneGenome(DefaultGenome):
    def __init__(self, key):
        super().__init__(key)

    def configure_new(self, config):
        """Inicializa o genoma e define 'tanh' para nós de saída."""
        super().configure_new(config)
        
        # Nós de saída têm IDs de 0 a (num_outputs - 1)
        output_ids = range(4)
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if node_id in output_ids:  # Verifica se o ID é de saída
                node.activation = 'tanh'

    def mutate(self, config):
        """Personaliza a mutação para ignorar a função de ativação dos nós de saída."""
        super().mutate(config)

        # Nós de saída têm IDs de 0 a (num_outputs - 1)
        output_ids = range(4)
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if node_id in output_ids:  # Verifica se o ID é de saída
                node.activation = 'tanh'

def eval_genome(genome, config):
    env = gym.make("GPS-Distance-v0", rng=RNG)

    obs, _ = env.reset()
    done = False

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    while not done:
        obs = np.concatenate((
            obs[0].flatten(), 
            obs[1].flatten(), 
            obs[2].flatten()), axis=0)
        # obs = obs.reshape(1, -1)  # Garante (1, 14)
        action = np.array(net.activate(obs.tolist())).reshape(1, 4)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break

    genome.fitness = reward
    env.close()


def run(config_file):
    # Load configuration.
    config = neat.Config(DroneGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    population = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    parallel_evaluator = neat.ParallelEvaluator(4, eval_genome)

    # Run for up to 300 generations.
    winner = population.run(parallel_evaluator.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # TODO: Visualize the winning genome.

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)