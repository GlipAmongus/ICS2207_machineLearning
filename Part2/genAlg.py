import random
import copy
import matplotlib.pyplot as plt

# Calculate Distance between any 2 keys
def calculate_distance(key1_pos, key2_pos):
    # key1_pos and key2_pos are tuples of the form (row, col)
    col1, row1 = key1_pos
    col2, row2 = key2_pos
    
    cost = 0
    # Case 1: Same row (left to right or right to left)
    if row1 == row2:
        cost = abs(col1-col2)
        
    # Case 2: Top row and Middle row
    if row1 == 0 and row2 == 1:
        # Pythagorean theorem
        h = ((1.032)**2 - 1)**0.5
        cost = ((1 + (h+(col2-col1))**2))**0.5
    elif row2 == 0 and row1 == 1:
        h = ((1.032)**2 - 1)**0.5
        cost = ((1 + (h+(col1-col2))**2))**0.5
           
    # Case 3: Top row and Bottom row
    if row1 == 0 and row2 == 2:
        h = ((2.138)**2 - 4)**0.5
        cost = ((4 + (h+(col2-col1))**2))**0.5
    elif row2 == 0 and row1 == 2:
        h = ((2.138)**2 - 4)**0.5
        cost = ((4 + (h+(col1-col2))**2))**0.5
        
    # Case 4: Middle row and Bottom row
    if row1 == 1 and row2 == 2:
        h = ((1.118)**2 - 1)**0.5
        cost = ((1 + (h+(col2-col1))**2))**0.5
    elif row2 == 1 and row1 == 2:
        h = ((1.118)**2 - 1)**0.5
        cost = ((1 + (h+(col1-col2))**2))**0.5
    return cost
    

# Function to compute the fitness of a keyboard layout
def fitness(keyboard, corpus):
    # Compute the dist done by all fingers for given corpus
    score = 0
    
    # Track finger positions
    finger_positions = {"f1":(0,1),"f2":(1,1),"f3":(2,1),"f4":(3,1),"f5":(0,3),"f7":(6,1),"f8":(7,1),"f9":(8,1),"f10":(9,1)}
    
    # Enumerate coords of keys 
    key_positions = {}
    for row_idx, row in enumerate(keyboard):
        for col_idx, key in enumerate(row):
            key_positions[key] = (col_idx, row_idx)
            
    # Iterate through letters
    for letter in corpus:
        # Determine the new position of the current letter on the keyboard
        new_letter_pos = key_positions[letter]
        
        # Identify which finger is responsible for pressing this key
        responsible_finger = finger_space[new_letter_pos]
        
        # Get the current position of the responsible finger
        current_finger_pos = finger_positions[responsible_finger]
        
        # Calculate the distance the finger needs to move to press the current key
        score +=  calculate_distance(new_letter_pos, current_finger_pos)
        
        # Update the finger's position to the new letter's position
        finger_positions[responsible_finger] = new_letter_pos
        
    return score


# Generate initial random population 
def init_population(pop_size):
    population = []
    base_keys = [
        'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
        'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',
        'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '?',
    ]
    
    for _ in range(pop_size):
        # Shuffle the flattened list of keys
        shuffled_keys = random.sample(base_keys, len(base_keys))
        
        # Reshape it back into a 2D layout
        keyboard = [
            shuffled_keys[:10],  # First row
            shuffled_keys[10:20],  # Second row
            shuffled_keys[20:30],  # Third row
            [" "]  # Spacebar row
        ]
        
        population.append(keyboard)
    
    return population


# Create the next generation of layouts
def new_generation(weighted_pop, pop_size, mate_prob, mate_type, mate_cut, muta_prob, muta_num, muta_str, elit_rate):
    """
    Create the next generation of layouts.

    Args:
        weighted_pop: List of tuples [(chromosome, fitness), ...].
        pop_size: Total population size.
        mate_prob: Probability of crossover.
        mate_type: Type of crossover to perform.
        muta_prob: Probability of mutation.
        muta_num: Number of mutations to apply.
        muta_str: Mutation strength
        elit_rate: Proportion of top performers to carry over.
        mate_cut: Range for selecting parents

    Returns:
        List of new generation chromosomes.
    """
    
    new_gen = []

    # Sort the population by fitness (ascending order)
    sorted_population = sorted(weighted_pop, key=lambda x: x[1])  # Use tuple indexing

    # Extract only the chromosomes
    chromosomes_only = [chromosome for chromosome, _ in sorted_population]
    
    # Elitism
    for i in range(int(pop_size * elit_rate)):
        new_gen.append(chromosomes_only[i])
        
    # Crossover
    for _ in range(int(pop_size * (1-elit_rate))):
        # Select parents from above cutoff point
        p1 = random.choice(chromosomes_only[:int(pop_size * mate_cut)])
        
        # Decide if crossover should happen based on  muta_prob
        if random.random() < mate_prob:
            p2 = random.choice(chromosomes_only[:int(pop_size * mate_cut)])
        
            # Ensure unique parents
            while p1 == p2:
                p2 = random.choice(chromosomes_only[:int(pop_size * mate_cut)])
                
            # Perform crossover
            child = mate(p1, p2, mate_type)
        else:
            # No crossover, child is a clone
            child = copy.deepcopy(p1)
            
        # Mutation Operation
        if random.random() < muta_prob:
            for _ in range(muta_num):
                if random.random() < muta_str:
                    row_id1 = random.randint(0, 2)
                    col_id1 = random.randint(0, 9)
                    row_id2 = random.randint(0, 2)
                    col_id2 = random.randint(0, 9)
                    allele = child[row_id1][col_id1]
                    child[row_id1][col_id1] = child[row_id2][col_id2]
                    child[row_id2][col_id2] = allele
        
        new_gen.append(child)

    return new_gen


# Combine two keyboards together
def mate(board1, board2, mate_type):
    
    child = []
    # Flatten board2 into a single list of keys
    board2_keys = [key for row in board2 for key in row]
    
    if mate_type == 1:
        split = random.randint(0, 9)
        child = [
            board1[0][:split],
            board1[1][:split],
            board1[2][:split]
        ]
        
        # Fill the remaining slots in child with keys from board2
        for key in board2_keys:
            if any(key in row for row in child):  # Skip if key is already in child
                continue
        
            # Append key to the first row with space left (less than 10 keys)
            for row in child:
                if len(row) < 10:
                    row.append(key)
                    break
    else:
        split1 = random.randint(0, 8)
        split2 = random.randint(split1 + 1, 9) 
        child = [
            board1[0][:split1],
            board1[1][:split1],
            board1[2][:split1]
        ]
        
        # Fill the remaining slots in child with keys from board2
        for key in board2_keys:
            if any(key in row for row in child):  # Skip if key is already in child
                continue
            
            for row in child:
                if len(row) < split2 + 1:
                    row.append(key)
                    break
          
        board1_keys = [key for row in board1 for key in row]
        for key in board1_keys:
            if any(key in row for row in child):  # Skip if key is already in child
                continue
            # Append key to the first row with space left (less than 10 keys)
            for row in child:
                if len(row) < 10:
                    row.append(key)
                    break

    child.append([" "])
    return child


def get_int_input():
    while True:
        inp = input()
        if inp.isdigit() and int(inp) >= 0:
            return int(inp)
        else:
            print("Invalid input. Please enter an integer.")

def get_float_input():
    while True:
        inp = input()
        try:
            value = float(inp)
            if 0 <= value <= 1:
                return value
            else:
                print("Invalid input. Please enter a float between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a valid float.")

def main():
    # --------------------- Menu ------------------------
    # Generations
    print("Enter number of generations: ")
    generations = get_int_input()
    
    # Population
    print("Enter population size: ") 
    population_size = get_int_input()
    
    # Cross-over
    crossover_type = 0
    while crossover_type not in [1, 2]:
        print("Enter number of crossover split points: ")
        crossover_type = get_int_input()            
    print("Enter crossover rate: ")
    crossover_rate = get_float_input()
    print("Enter crossover cutoff: ")
    crossover_cutoff = get_float_input()
    
    # Mutation 
    print("Enter mutation rate: ")
    mutation_rate = get_float_input()
    print("Enter mutation scale: ")
    mutations = get_int_input()
    print("Enter mutation strength: ")
    mutation_strength = get_float_input()
    
    # Elitism
    print("Enter elitism rate: ")
    elitism_rate = get_float_input()
    # ---------------------------------------------------
    
    # ---- Load Corpus and Calculate Initial Fitness ----
    with open('corpus.txt', 'r') as file:
        corpus = file.read()

    # Fitness calculation for QWERTY and Dvorak keyboards
    qwerty_fit = fitness(qwerty, corpus)
    azerty_fit = fitness(azerty, corpus)
    print(f"QWERTY keyboard fitness: {qwerty_fit}")
    print(f"AZERTY keyboard fitness: {azerty_fit}")

    # ---- Initialize Population ----
    population = init_population(population_size)
    weighted_population = [(chromosome, fitness(chromosome, corpus)) for chromosome in population]

    total = sum(fitness for _, fitness in weighted_population)
    average_fitness = total / len(population)
    print(f"Generation: 0\nAverage Fitness: {average_fitness}")

    # ---- Initialize Lists for Plotting ----
    average_fitness_list = []
    best_fitness_list = []
    best_chromosome = []

    # ---- Start the Genetic Algorithm ----
    for generation in range(generations):
        # Generate new population
        population = new_generation(
            weighted_population,
            population_size,
            crossover_rate,
            crossover_type,
            crossover_cutoff,
            mutation_rate,
            mutations,
            mutation_strength,
            elitism_rate,
        )
        weighted_population = [(chromosome, fitness(chromosome, corpus)) for chromosome in population]
                
        # Calculate average and best fitness
        total = sum(fitness for _, fitness in weighted_population)
        average_fitness = total / len(population)
        best_chromosome, best_fitness = min(weighted_population, key=lambda x: x[1])
        
        # Append fitness values to the lists for plotting
        average_fitness_list.append(average_fitness)
        best_fitness_list.append(best_fitness)
        
        # Output results for the current generation
        print(f"Generation: {generation+1}")
        print(f"Average Fitness: {average_fitness}")
        print(f"Best Fitness: {best_fitness}")
        print("Best Keyboard layout: ")
        for row in best_chromosome:
            print(row)
    
    keyboard_text = "\n".join([" | ".join(row) for row in best_chromosome])
    # ---- Plotting ----
    # Hyperparameters
    parameters = [
        "Generations",
        "Population Size",
        "Crossover Rate",
        "Crossover Type",
        "Crossover Cutoff",
        "Mutation Rate",
        "Mutation Scale",
        "Mutation Strength",
        "Elitism Rate"
    ]

    values = [
        generations,
        population_size,
        crossover_rate,
        crossover_type,
        crossover_cutoff,
        mutation_rate,
        mutations,
        mutation_strength,
        elitism_rate
    ]
    
    # ---- Plotting ----
    # Create subplots with two columns: one for the plot, one for the table
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Plot the fitness trends in the left subplot
    ax[0].plot(range(1, generations + 1), average_fitness_list, label="Average Fitness", color="blue")
    ax[0].plot(range(1, generations + 1), best_fitness_list, label="Best Fitness", color="red")
    
    # Mark Qwerty and Dvorak
    ax[0].axhline(y=qwerty_fit, color="black", linestyle="--", label="QWERTY fitness")
    ax[0].axhline(y=azerty_fit, color="black", linestyle="--", label="AZERTY fitness")
    
    # Labels, legend, and grid
    ax[0].set_title("Fitness Trends Over Generations")
    ax[0].set_xlabel("Generation")
    ax[0].set_ylabel("Fitness Value")
    ax[0].legend()
    ax[0].grid()

    # Add the table
    ax[1].axis("off")  # Turn off the axes for the table
    table_data = [[param, value] for param, value in zip(parameters, values)]
    table = ax[1].table(cellText=table_data, colLabels=["Parameter", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])  # Adjust column widths

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)  # Styling for the text box
    # Place the text box with the best keyboard layout
    fig.text(
        0.72, 0.25,  # Adjust the position (x, y) of the text box as needed
        f"Best Keyboard Layout:\n\n{keyboard_text}",
        fontsize=12,
        bbox=props,
        verticalalignment="top"
    )
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # Define the fingers' allowed positions
    finger_space = {
        (0, 0): "f1",
        (0, 1): "f1",
        (0, 2): "f1",
        (1, 0): "f2",
        (1, 1): "f2",
        (1, 2): "f2",
        (2, 0): "f3",
        (2, 1): "f3",
        (2, 2): "f3",
        (3, 0): "f4",
        (3, 1): "f4",
        (3, 2): "f4",
        (4, 0): "f4",
        (4, 1): "f4",
        (4, 2): "f4",
        (0, 3): "f5",
        (5, 0): "f7",
        (5, 1): "f7",
        (5, 2): "f7",
        (6, 0): "f7",
        (6, 1): "f7",
        (6, 2): "f7",
        (7, 0): "f8",
        (7, 1): "f8",
        (7, 2): "f8",
        (8, 0): "f9",
        (8, 1): "f9",
        (8, 2): "f9",
        (9, 0): "f10",
        (9, 1): "f10",
        (9, 2): "f10"
    }
    qwerty = [
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '?',],
        [" "]
    ]
    azerty = [
        ['a', 'z', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['q', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm'],
        ['w', 'x', 'c', 'v', 'b', 'n', ',', ';', '.', '?'],
        [" "]
    ]
    main()