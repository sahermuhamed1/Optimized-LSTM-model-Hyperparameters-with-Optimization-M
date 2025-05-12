#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random


# In[3]:


POP_SIZE = 6
CHROMOSOME_LENGTH = 5  # For x in [0, 31]
GENERATIONS = 10
MUTATION_RATE = 0.1


# In[4]:


# Fitness function
def fitness(x):
    return x**2


# In[5]:


# Binary to integer
def decode(chromosome):
    return int(chromosome, 2)


# In[6]:


# Selection using roulette wheel
def roulette_wheel(pop, fitnesses):
    total = sum(fitnesses)
    probs = [f / total for f in fitnesses]
    selected = random.choices(pop, weights=probs, k=2)
    return selected


# In[7]:


# Crossover (single point)
def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return (parent1[:point] + parent2[point:], parent2[:point] + parent1[point:])


# In[8]:


# Mutation
def mutate(chromosome):
    mutated = ''
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            mutated += '0' if bit == '1' else '1'
        else:
            mutated += bit
    return mutated


# In[9]:


# Initial population
population = [''.join(random.choices('01', k=CHROMOSOME_LENGTH)) for _ in range(POP_SIZE)]
#population = ['010101', '111000', '000110', '101011']


# In[11]:


# GA main loop
for generation in range(GENERATIONS):
    decoded = [decode(ind) for ind in population]   #[23, 21, 4, 17, 4, 19]
    print(decoded)
    fitnesses = [fitness(x) for x in decoded]
    
    print(f"Gen {generation}:")
    for ind, val, fit in zip(population, decoded, fitnesses):   #zip() function combines multiple lists element by element.
        print(f"  {ind} -> x={val}, f(x)={fit}")
    
    new_population = []
    while len(new_population) < POP_SIZE:
        parent1, parent2 = roulette_wheel(population, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([mutate(child1), mutate(child2)])
    
    population = new_population


# In[10]:


# Final result
best = max(population, key=lambda ind: fitness(decode(ind)))
print(f"\nBest solution: {best} (x={decode(best)}, f(x)={fitness(decode(best))})")


# In[ ]:





# In[ ]:




