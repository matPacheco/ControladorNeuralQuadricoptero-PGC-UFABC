import random
import os
import pickle
import multiprocessing
from argparse import ArgumentParser

import visualize

import neat
from neat.genome import DefaultGenome
from neat.reporting import BaseReporter

import numpy as np

import gym_pybullet_drones
import gymnasium as gym


RNG = random.Random(42)

parser = ArgumentParser()
parser.add_argument("-c", "--checkpoint", type=str, default=False)
parser.add_argument("-w", "--workers", default=multiprocessing.cpu_count()//2, type=int)
args = parser.parse_args()

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

    env.close()

    return reward

class PeriodicStatsSaver(BaseReporter):
    """Classe para salvar um arquivo de estatísticas a cada 5 gerações"""
    def __init__(self, stats_obj, interval=5, filename='checkpoints/stats.pkl'):
        self.stats = stats_obj
        self.interval = interval
        self.filename = filename
        self.current_generation = 0  # Adicionamos um contador interno

        
    def post_evaluate(self, config, population, species, best_genome):
        if self.current_generation % self.interval == 0:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.stats, f)

    def start_generation(self, generation):
        """Atualiza a geração atual"""
        self.current_generation = generation


def run(config_file, checkpoint_path=False):
    # Load configuration.
    config = neat.Config(DroneGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    if checkpoint_path:
        population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        # Create the population, which is the top-level object for a NEAT run.
        population = neat.Population(config)

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    stats_saver = PeriodicStatsSaver(stats, interval=5)
    population.add_reporter(stats_saver)

    population.add_reporter(
        neat.Checkpointer(5, filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-')))

    parallel_evaluator = neat.ParallelEvaluator(args.workers, eval_genome)

    # Run for up to 300 generations.
    winner = population.run(parallel_evaluator.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    pickle.dump(winner, open('checkpoints/winner.pkl', 'wb'))

    with open('checkpoints/neat-stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # TODO: Visualize the winning genome.


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path, checkpoint_path=args.checkpoint)
