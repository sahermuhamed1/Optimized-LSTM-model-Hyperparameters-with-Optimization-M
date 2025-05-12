#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# Objective function
def f(x, y):
    return x**2 + y**2


# In[12]:




# Particle Swarm Optimization (Maximization)
def pso(n_particles=10, epochs=15, w=0.7, c1=2, c2=1.5, fitness_function=None):
    # Define dimension ranges for each hyperparameter
    dim_ranges = [
        (64, 512),    # lstm1_units
        (32, 256),    # lstm2_units
        (16, 64),     # dense_units
        (0.1, 0.5),   # dropout1
        (0.1, 0.5),   # dropout2
        (0.0001, 0.01) # learning_rate
    ]
    
    n_dimensions = len(dim_ranges)
    
    # Initialize particles
    positions = np.zeros((n_particles, n_dimensions))
    velocities = np.zeros((n_particles, n_dimensions))
    
    # Initialize each dimension within its range
    for i in range(n_dimensions):
        positions[:, i] = np.random.uniform(dim_ranges[i][0], dim_ranges[i][1], n_particles)
        velocities[:, i] = np.random.uniform(-1, 1, n_particles)
    
    # Personal best
    pbest = positions.copy()
    pbest_values = np.array([fitness_function(pos) for pos in pbest])
    
    # Global best
    gbest = pbest[np.argmax(pbest_values)]
    gbest_value = np.max(pbest_values)
    
    for epoch in range(epochs):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Update velocity
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - positions[i])
                + c2 * r2 * (gbest - positions[i])
            )
            
            # Update position
            positions[i] += velocities[i]
            
            # Clip positions to valid ranges
            for j in range(n_dimensions):
                positions[i, j] = np.clip(positions[i, j], dim_ranges[j][0], dim_ranges[j][1])
            
            # Evaluate fitness
            fitness_value = fitness_function(positions[i])
            if fitness_value > pbest_values[i]:
                pbest[i] = positions[i]
                pbest_values[i] = fitness_value
        
        # Update global best
        if np.max(pbest_values) > gbest_value:
            gbest = pbest[np.argmax(pbest_values)]
            gbest_value = np.max(pbest_values)
        
        print(f"Epoch {epoch}: Best fitness = {gbest_value:.6f}")
    
    return gbest, gbest_value

# Run the PSO optimizer (Maximization)
best_pos, best_val = pso()
print("\nFinal Result:")
print(f"Maximum at x = {best_pos[0]:.6f}, y = {best_pos[1]:.6f}")
print(f"Function value = {best_val:.6f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




