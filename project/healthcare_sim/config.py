import numpy as np
import random

NUM_PATIENTS = 10
NUM_PATHWAYS = 10
NUM_ACTIONS = 10
NUM_STEPS = 365
BASE_CAPACITY = 10
AGE_THRESHOLD = 60
PROBABILITY_OF_DISEASE = 0.075

# --- Ideal clinical values ---
IDEAL_CLINICAL_VALUES = {
    'bp': 120,
    'glucose': 90,
    'bmi': 22,
    'oxygen': 98,
    'mental_health': 80,
}

INPUT_ACTIONS = ['a0', 'a1']  # Two standard input actions
OUTPUT_ACTIONS = 'a9'       # Standard output action

np.random.seed(42)
random.seed(42)

# --- Q-learning parameters ---
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate
