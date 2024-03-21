"""
This file contains the implementation of the NSGA-2 algorithm. It is a multi-objective evolutionary algorithm
that is used for the evolutionary search of neural network architectures. The algorithm is implemented in the
NSGA_II class. The algorithm is used in EcoNAS/Training/CIFAR.py.
"""
import csv

from matplotlib import pyplot as plt

from genetic_functions import *
from pareto_functions import *
import functools
import numpy as np

from regression_predictor import RregressionPredictor


class NSGA_II:
    def __init__(self, population_size, generations, crossover_factor, mutation_factor):
        self.population_size = population_size
        self.generations = generations
        self.crossover_factor = crossover_factor
        self.mutation_factor = mutation_factor

    def initial_population(self):
        """
        Initialize the population pool with random deep neural architectures
        :param max_hidden_layers: Maximum number of hidden layers
        :param max_hidden_size: Maximum number of hidden units per layer
        :return: Returns a list of NeuralArchitecture objects
        """
        archs = []
        for _ in range(self.population_size):
            mid_layer = random.randint(10, 50)
            latent_dim = random.randint(3, 10)
            # TODO
            arch = VAEArchitectures(mid_layer, latent_dim)

            archs.append(arch)

        return archs

    def evolve(self, train_loader, mid_layer, latent_dim):
        """
        The NSGA-2 algorithm. It evolves the population for a given number of generations, however
        there is quite a bit of excessive training going on here.
        :param hidden_layers:
        :param hidden_size:
        :param data_name:
        :return: List of the best performing NeuralArchitecture objects of size of at most population_size
        """
        regression_trainer = RregressionPredictor('Training_CSV.csv')

        regression_trainer.train_models()
        regression_trainer.evaluate_models()

        evolutionary_path = []

        # step 1: generate initial population
        archs = self.initial_population()

        # step 2 : evaluate the objective functions for each arch
        for a in archs:
            a.train(train_loader, mid_layer, latent_dim)
            results = store_results(a.latent_dim, a.mid_layer, a.objectives['loss'], a.objectives['OOD'])
            store_results_to_csv(results, 'Training_CSV')

        # step 3: set the non-dominated ranks for the population and sort the architectures by rank
        set_non_dominated(archs)  # fitness vals
        archs.sort(key=lambda arch: arch.nondominated_rank)

        # step 4: create an offspring population Q0 of size N
        offspring_pop = generate_offspring(archs, self.crossover_factor, self.mutation_factor)

        for offspring in offspring_pop:
            offspring.train(train_loader, mid_layer, latent_dim)
            results = store_results(offspring.latent_dim, offspring.mid_layer, offspring.objectives['loss'],
                                    offspring.objectives['OOD'])
            store_results_to_csv(results, 'Training_CSV')

        # step 5: start algorithm's counter -> MAIN LOOP
        for generation in range(self.generations):
            print(f'Generation: {generation}')
            # step 6: combine parent and offspring population
            combined_population = archs + offspring_pop  # of size 2N
            set_non_dominated(combined_population)

            population_by_objectives = [[ind.objectives['loss'], ind.objectives['OOD']] for ind in combined_population]

            generation_metrics = []
            # for ind in archs:
            #     # Convert objectives to a dictionary for clarity
            #     objectives_dict = {
            #         'loss': ind.objectives['loss'],
            #         'OOD': ind.objectives['OOD']
            #     }
            #     generation_metrics.append({'objectives': objectives_dict})
            #
            # evolutionary_path.append(generation_metrics)

            # step 7:
            non_dom_fronts = fast_non_dominating_sort(population_by_objectives)

            # step 8: initialize new parent list and non-dominated front counter
            archs, i = [], 0

            # step 9: calculate crowding-distance in Fi until the parent population is filled
            while len(archs) + len(non_dom_fronts[i]) <= self.population_size:
                corresponding_archs = get_corr_archs(non_dom_fronts[i], combined_population)
                # calculated crowding-distance
                crowding_metric = crowding_distance_assignment(population_by_objectives, non_dom_fronts[i])
                for j in range(len(corresponding_archs)):
                    corresponding_archs[j].crowding_distance = crowding_metric[j]
                archs += corresponding_archs
                i += 1

            # step 8: sort front by crowding comparison operator
            last_front_archs = get_corr_archs(non_dom_fronts[i], combined_population)
            last_front_archs.sort(key=functools.cmp_to_key(crowded_comparison_operator), reverse=True)

            archs = archs + last_front_archs[1: self.population_size - len(archs)]

            offspring_pop = generate_offspring(archs, self.crossover_factor, self.mutation_factor)
            for offspring in offspring_pop:
                predicted_performance = regression_trainer.predict_performance(offspring)
                offspring.objectives = {
                    'loss': predicted_performance[0],
                    'ood': predicted_performance[1]
                }

            # TODO: sort by some characteristic
            offspring_pop.sort(key=functools.cmp_to_key(set_non_dominated), reverse=True)
            # train best N/2
            num_to_train = len(offspring_pop) // 2
            for i in range(num_to_train):
                offspring_pop[i].train(train_loader, mid_layer, latent_dim)

        plot_evolutionary_path(evolutionary_path, 'loss')
        plot_evolutionary_path(evolutionary_path, 'OOD')

        return archs


def plot_evolutionary_path(evolutionary_path, metric_name):
    plt.figure(figsize=(10, 6))

    for i, generation_metrics in enumerate(evolutionary_path):
        if i % 4 == 0:
            # Extract the metric values for each architecture in the current generation
            metric_values = [arch['objectives'][metric_name] for arch in generation_metrics]

            # Plot the metric values for each architecture in the current generation
            plt.plot(metric_values, label=f'Generation {i + 1}')

    plt.xlabel('Architecture Index')
    plt.ylabel(metric_name)
    plt.title(f'Evolutionary Path of NSGA-2 on dataset')
    plt.legend()
    plt.show()


def store_results(latent_dim, mid_layer, loss, ood):
    result = {
        'latent_dim': latent_dim,
        'mid_layer': mid_layer,
        'loss': loss,
        'OOD': ood
    }

    return result


def store_results_to_csv(results, filename):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['latent_dim', 'mid_layer', 'loss', 'OOD']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        csvfile.seek(0, 2)
        is_empty = csvfile.tell() == 0

        if is_empty:
            writer.writeheader()

        for result in results:
            writer.writerow(result)
