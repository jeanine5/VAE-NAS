"""
This contains the functions ncessary for running the genetic algorithm, NSGA-II. Here in this file, we have the
functions for binary tournament selection, crossover, mutation, and generating offspring. You can see how these
functions are used in the NSGA_II class in EcoNAS/EA/NSGA.py.
"""

import random

from vae import VAE, VAEArchitectures


def binary_tournament_selection(population: list[VAEArchitectures], tournament_size=2):
    """
    Perform binary tournament selection on the population
    :param population: list of NeuralArchitecture objects
    :param tournament_size: size of the tournament, default is 2 for binary tournament selection
    :return: the NeuralArchitecture object with the best fitness value
    """

    # select 2 parent models from population P
    selected_parents = random.sample(population, tournament_size)

    # determine two parents w/ best fitness val
    return min(selected_parents, key=lambda arch: arch.nondominated_rank)


def crossover(parent1: VAEArchitectures, parent2: VAEArchitectures, crossover_rate: float):
    """
    Perform crossover on two NeuralArchitecture objects
    :param parent1: NeuralArchitecture object
    :param parent2: NeuralArchitecture object
    :param crossover_rate: probability of crossover
    :return:
    """
    ...


def mutate(offspring: VAEArchitectures, mutation_factor: float):
    """
    Perform mutation on a NeuralArchitecture object
    :param offspring: NeuralArchitecture object
    :param mutation_factor: probability of mutation
    :return: mutated NeuralArchitecture object
    """
    ...


def generate_offspring(population: list[VAEArchitectures], crossover_rate: float, mutation_rate: float):
    """
    Generate offspring population using binary tournament selection, crossover, and mutation
    :param population: list of NeuralArchitecture objects
    :param crossover_rate: probability of crossover
    :param mutation_rate: probability of mutation
    :param train_loader: Data loader for the training dataset
    :param test_loader: Data loader for the testing dataset
    :param epoch: Current epoch of training
    :return: list of NeuralArchitecture objects
    """
    offspring_pop = []

    for _ in range(len(population)):
        parent_1 = binary_tournament_selection(population)
        parent_2 = binary_tournament_selection(population)

        offspring = crossover(parent_1, parent_2, crossover_rate)

        mutated_offspring = mutate(offspring, mutation_rate)

        # predicted_performance = regression_trainer.predict_performance(mutated_offspring)
        # mutated_offspring.objectives = {
        #     'accuracy': predicted_performance[0],
        #     'introspectability': predicted_performance[1],
        #     'flops': predicted_performance[2]
        # }

        offspring_pop.append(mutated_offspring)

    return offspring_pop
