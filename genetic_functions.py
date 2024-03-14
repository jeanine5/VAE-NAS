"""
This contains the functions ncessary for running the genetic algorithm, NSGA-II. Here in this file, we have the
functions for binary tournament selection, crossover, mutation, and generating offspring. You can see how these
functions are used in the NSGA_II class in EcoNAS/EA/NSGA.py.
"""

import random


def binary_tournament_selection(population: list[NeuralArchitecture], tournament_size=2):
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


def crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture, crossover_rate: float):
    """
    Perform crossover on two NeuralArchitecture objects
    :param parent1: NeuralArchitecture object
    :param parent2: NeuralArchitecture object
    :param crossover_rate: probability of crossover
    :return:
    """
    if random.uniform(0, 1) < crossover_rate:
        return two_point_crossover(parent1, parent2)
    else:
        return one_point_crossover(parent1, parent2)


def one_point_crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """
    Perform one point crossover on two NeuralArchitecture objects
    :param parent1: NeuralArchitecture object
    :param parent2: NeuralArchitecture object
    :return: offspring NeuralArchitecture object
    """
    # Randomly choose a crossover point based on the number of hidden layers
    crossover_point = random.randint(1, min(len(parent1.hidden_sizes), len(parent2.hidden_sizes)))
    offspring_hidden_sizes = (
            parent1.hidden_sizes[:crossover_point] + parent2.hidden_sizes[crossover_point:]
    )

    offspring = NeuralArchitecture(
        parent1.input_size,
        parent1.output_size,
        offspring_hidden_sizes
    )

    return offspring


def two_point_crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """
    Perform two point crossover on two NeuralArchitecture objects
    :param parent1: NeuralArchitecture object
    :param parent2: NeuralArchitecture object
    :return: offspring NeuralArchitecture object
    """
    len_parent1 = len(parent1.hidden_sizes)
    len_parent2 = len(parent2.hidden_sizes)

    if len_parent1 <= 2 or len_parent2 <= 2:
        return parent1 if len_parent1 > 0 else parent2

    # randomly choose a crossover point based on the number of hidden layers
    crossover_points = sorted(random.sample(range(1, min(len_parent1, len_parent2)), 2))

    child_hidden_sizes = (
            parent1.hidden_sizes[:crossover_points[0]] +
            parent2.hidden_sizes[crossover_points[0]:crossover_points[1]] +
            parent1.hidden_sizes[crossover_points[1]:]
    )

    offspring = NeuralArchitecture(
        parent1.input_size,
        parent1.output_size,
        child_hidden_sizes
    )

    return offspring


def mutate(offspring: NeuralArchitecture, mutation_factor: float):
    """
    Perform mutation on a NeuralArchitecture object
    :param offspring: NeuralArchitecture object
    :param mutation_factor: probability of mutation
    :return: mutated NeuralArchitecture object
    """
    if random.uniform(0, 1) < mutation_factor:
        mutated_offspring = mutate_add_remove_hidden_layer(offspring)
    else:
        mutated_offspring = mutate_random_hidden_sizes(offspring)

    return mutated_offspring


def mutate_random_hidden_sizes(architecture: NeuralArchitecture):
    """
    Modify the hidden sizes of a NeuralArchitecture object
    :param architecture: NeuralArchitecture object
    :return: mutated NeuralArchitecture object
    """
    mutated_architecture = architecture.clone()
    max_hidden_size = max(mutated_architecture.hidden_sizes)
    for i in range(len(mutated_architecture.hidden_sizes)):
        mutated_architecture.hidden_sizes[i] = random.randint(10, max_hidden_size)
    return mutated_architecture


def mutate_add_remove_hidden_layer(architecture: NeuralArchitecture):
    """
    Add or remove a hidden layer from a NeuralArchitecture object
    :param architecture: NeuralArchitecture object
    :return: mutated NeuralArchitecture object
    """
    mutated_architecture = architecture.clone()
    if len(mutated_architecture.hidden_sizes) > 1:
        index_to_remove = random.randint(0, len(mutated_architecture.hidden_sizes) - 1)
        del mutated_architecture.hidden_sizes[index_to_remove]
    else:
        mutated_architecture.hidden_sizes.append(random.randint(10, max(mutated_architecture.hidden_sizes)))
    return mutated_architecture


def generate_offspring(population: list[NeuralArchitecture], crossover_rate: float, mutation_rate: float,
                       regression_trainer):
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

        predicted_performance = regression_trainer.predict_performance(mutated_offspring)
        mutated_offspring.objectives = {
            'accuracy': predicted_performance[0],
            'introspectability': predicted_performance[1],
            'flops': predicted_performance[2]
        }

        offspring_pop.append(mutated_offspring)

    return offspring_pop


