"""
This contains the functions ncessary for running the genetic algorithm, NSGA-II. Here in this file, we have the
functions for binary tournament selection, crossover, mutation, and generating offspring. You can see how these
functions are used in the NSGA_II class in EcoNAS/EA/NSGA.py.
"""

import random

import torch

from vae import VAEArchitecture


def binary_tournament_selection(population: list[VAEArchitecture], tournament_size=2):
    """
    Perform binary tournament selection on the population
    :param population: list of VAEArchitecture objects
    :param tournament_size: size of the tournament, default is 2 for binary tournament selection
    :return: the VAEArchitecture object with the best fitness value
    """

    # select 2 parent models from population P
    selected_parents = random.sample(population, tournament_size)

    # determine two parents w/ best fitness val
    return min(selected_parents, key=lambda arch: arch.nondominated_rank)


def crossover(parent1: VAEArchitecture, parent2: VAEArchitecture, crossover_rate: float):
    """
    Perform crossover on two NeuralArchitecture objects
    :param parent1: VAEArchitecture object
    :param parent2: VAEArchitecture object
    :param crossover_rate: probability of crossover
    :return:
    """
    offspring = parent1.clone()

    # Perform crossover for encoder and decoder weights
    for offspring_param, parent2_param in zip(offspring.model.parameters(), parent2.model.parameters()):
        if torch.rand(1) < crossover_rate:
            # Take weights from parent2 with probability crossover_rate
            offspring_param.data = parent2_param.data.clone()

    return offspring


def mutate(offspring: VAEArchitecture, mutation_factor: float):
    """
    Perform mutation on a VAEArchitecture object. Randomly modify the model's parameters.
    :param offspring: VAEArchitecture object
    :param mutation_factor: probability of mutation
    :return: mutated NeuralArchitecture object
    """
    mutated_offspring = offspring.clone()

    # Mutate encoder and decoder weights
    for param in mutated_offspring.model.parameters():
        if torch.rand(1) < mutation_factor:
            param.data += torch.randn_like(param.data) * 0.1  # Adding Gaussian noise

    return mutated_offspring


def generate_offspring(population: list[VAEArchitecture], crossover_rate: float, mutation_rate: float):
    """
    Generate offspring population using binary tournament selection, crossover, and mutation
    :param population: list of VAEArchitecture objects
    :param crossover_rate: probability of crossover
    :param mutation_rate: probability of mutation
    :param train_loader: Data loader for the training dataset
    :param test_loader: Data loader for the testing dataset
    :param epoch: Current epoch of training
    :return: list of VAEArchitecture objects
    """
    offspring_pop = []

    for _ in range(len(population)):
        parent_1 = binary_tournament_selection(population)
        parent_2 = binary_tournament_selection(population)

        offspring = crossover(parent_1, parent_2, crossover_rate)

        mutated_offspring = mutate(offspring, mutation_rate)

        offspring_pop.append(mutated_offspring)

    return offspring_pop
