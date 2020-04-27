import copy
import random
import pygame

from PIL import Image

# Parameters of GA
population_size = 500
maximum_generation_number = 100
selection_factor = 0.2
mutation_rate = 0.1

# Input image file
image_name = 'image.jpg'
original_image = Image.open(image_name)
size_x = original_image.size[0]
size_y = original_image.size[1]
maximum_drawing_iteration_number = 8

# Initiate canvas here
pygame.init()
Art_Progress_Display_Window = pygame.display.set_mode((size_x, size_y))
pygame.display.set_caption('Digital Canvas')


# GA individual variable definition
class Individual:
    def __init__(self, design_variables):
        self.design_variables = design_variables
        self.fitness = -1


# Calculate the cost function value
def cost(value, target_values):
    cost_value = 0
    for _ in range(len(target_values)):
        cost_value += (value[_] - target_values[_]) ** 2
    return cost_value


# Initial variables are populated here
def init_population(len_variables, init_variables):
    population = []
    if init_variables:
        population.append(Individual(init_variables))
    for individual in range(population_size - len(population)):
        design_variables = []
        for _ in range(len_variables):
            design_variables.append(random.randint(0, 255))
        population.append(Individual(design_variables))
    return population


# This is the main function
def run_ga():
    surface = pygame.display.get_surface()
    i_count = 0
    n_sections = [1]

    while n_sections[len(n_sections) - 1] < size_x:
        n_sections.append(n_sections[len(n_sections) - 1] * 2)

    for n_iter in range(0, len(n_sections)):
        opt_count = 0
        n_section = n_sections[n_iter]
        print('Iteration: ' + str(i_count))

        temp_surface = pygame.display.get_surface()
        temp_surface.fill((0, 0, 0))

        for n in range(n_section):
            for m in range(n_section):
                section_len_x = float(size_x / n_section)
                section_len_y = float(size_y / n_section)
                sum_r_original = 0
                sum_g_original = 0
                sum_b_original = 0
                count = 0

                # Determine the sum of each color channel value of original image within the selected area
                for i in range(int(section_len_x)):
                    for j in range(int(section_len_y)):
                        sum_r_original += \
                            original_image.getpixel((int(n * section_len_x + i), int(m * section_len_y + j)))[0]
                        sum_g_original += \
                            original_image.getpixel((int(n * section_len_x + i), int(m * section_len_y + j)))[1]
                        sum_b_original += \
                            original_image.getpixel((int(n * section_len_x + i), int(m * section_len_y + j)))[2]
                        count += 1

                # Determine average rgb values added with random variation
                avg_r_original = int(sum_r_original / count) + random.randrange(-min(int(sum_r_original / count), 30),
                                                                                min(256 - int(sum_r_original / count),
                                                                                    30))
                avg_g_original = int(sum_g_original / count) + random.randrange(-min(int(sum_g_original / count), 30),
                                                                                min(256 - int(sum_g_original / count),
                                                                                    30))
                avg_b_original = int(sum_b_original / count) + random.randrange(-min(int(sum_b_original / count), 30),
                                                                                min(256 - int(sum_b_original / count),
                                                                                    30))

                # Get the current rgb values in the canvas as initial values for the GA
                init_variables = [0, 0, 0]
                init_variables[0] = surface.get_at((int(n * section_len_x), int(m * section_len_y)))[0]
                init_variables[1] = surface.get_at((int(n * section_len_x), int(m * section_len_y)))[1]
                init_variables[2] = surface.get_at((int(n * section_len_x), int(m * section_len_y)))[2]

                # Start GA using the initial rgb values
                target_values = [avg_r_original, avg_g_original, avg_b_original]
                optimal_individual = ga(target_values, init_variables)

                r = optimal_individual.design_variables[0]
                g = optimal_individual.design_variables[1]
                b = optimal_individual.design_variables[2]

                # Calculate random value
                random_val = int(
                    random.uniform(-int(int(2 * section_len_x / 3) / 2), + int(int(2 * section_len_x / 3) / 2)))

                # Draw circle
                pygame.draw.circle(temp_surface, (r, g, b),
                                   (int(n * section_len_x) + int(section_len_x / 2) + random_val,
                                    int(m * section_len_y) + int(section_len_y / 2) + random_val),
                                   int(2 * section_len_x / 3) + random_val)
                # Update display
                pygame.display.update()
                opt_count += 1
        surface = temp_surface
        # Save current canvas
        pygame.image.save(Art_Progress_Display_Window, str(i_count) + ".png")
        i_count += 1
        if i_count > maximum_drawing_iteration_number:
            break


# Genetic algorithm function
def ga(target_values, init_variables):
    n_variables = len(init_variables)
    population = init_population(n_variables, init_variables)
    recent_best_individual = []
    for generation in range(maximum_generation_number):
        population = get_generation_fitness(population, target_values)
        best_individual = copy.deepcopy(get_best_individual(population))
        if recent_best_individual:
            if best_individual.fitness <= recent_best_individual.fitness:
                recent_best_individual = copy.deepcopy(get_best_individual(population))
            else:
                population = update_population(population, recent_best_individual)
                recent_best_individual = copy.deepcopy(get_best_individual(population))
        else:
            recent_best_individual = copy.deepcopy(get_best_individual(population))
        population = selection(population)
        population = crossover(population)
        population = mutation(population)
        # print('Generation: ' + str(generation) + ' Best Fitness: ' + str(recent_best_individual.fitness))
        if recent_best_individual.fitness == 0:
            return recent_best_individual
    return recent_best_individual


# Update the fitness value of each individuals of entire population in a generation
def get_generation_fitness(population, target_values):
    for _ in range(len(population)):
        population[_].fitness = cost(population[_].design_variables, target_values)
    return population


# Selection operation of GA
def selection(population):
    population = sorted(population, key=lambda individual: individual.fitness, reverse=False)
    population = population[:int(selection_factor * len(population))]
    return population


# Crossover operation of GA
def crossover(population):
    offspring = []
    for _ in range((population_size - len(population)) // 2):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        split = random.random()
        child1 = get_child(parent1, parent2, split)
        child2 = get_child(parent1, parent2, (1.0 - split))
        offspring.append(child1)
        offspring.append(child2)
    population.extend(offspring)
    return population


# Generate a child from two parents
def get_child(parent1, parent2, split):
    design_variables = []
    for i in range(len(parent1.design_variables)):
        design_variables.append(
            int(parent1.design_variables[i] * split) + int(parent2.design_variables[i] * (1.0 - split)))
    child = Individual(design_variables)
    return child


# Mutation operation of GA
def mutation(population):
    for individual in population:
        design_variables = individual.design_variables
        for _ in range(len(design_variables)):
            if random.uniform(0.0, 1.0) <= mutation_rate:
                individual.design_variables[_] = random.randint(0, 255)
    return population


# Get the best individual of a population in GA
def get_best_individual(population):
    population = sorted(population, key=lambda individual: individual.fitness, reverse=False)
    return population[0]


# Update the population to confirm having the previous best in case current best is not better than the previous one
def update_population(population, recent_best_individual):
    population = sorted(population, key=lambda individual: individual.fitness, reverse=False)
    if population[0].fitness > recent_best_individual.fitness:
        population = elements_replaced(population, recent_best_individual, [len(population) - 1])
        population = sorted(population, key=lambda individual: individual.fitness, reverse=False)
    return population


# Individual replacement in a population
def elements_replaced(lst, new_element, indices):
    return [new_element if i in indices else element for i, element in enumerate(lst)]


if __name__ == '__main__':
    run_ga()
