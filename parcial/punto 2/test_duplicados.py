import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Total number of rows (calculated from Diabetes_binary counts: 218334 + 35346)
N = 253680

# Probability distributions for each column based on provided data
# Each inner dictionary maps a value to its probability (proportion)
column_probabilities = {
    "Diabetes_binary": {0: 0.8607, 1: 0.1393},
    "HighBP": {0: 0.571, 1: 0.429},
    "HighChol": {0: 0.5759, 1: 0.4241},
    "CholCheck": {1: 0.9627, 0: 0.0373},
    "BMI": {
        27: 0.0970, 26: 0.0811, 24: 0.0771, 25: 0.0676, 28: 0.0652,
        23: 0.0615, 29: 0.0587, 30: 0.0574, 22: 0.0538, 31: 0.0484,
        32: 0.0413, 21: 0.0388, 33: 0.0353, 34: 0.0283, 20: 0.0249,
        35: 0.0220, 36: 0.0183, 37: 0.0163, 19: 0.0156, 38: 0.0134,
        39: 0.0115, 40: 0.0089, 18: 0.0071, 41: 0.0065, 42: 0.0065,
        43: 0.0059, 44: 0.0041, 45: 0.0032, 17: 0.0031, 46: 0.0030,
        47: 0.0025, 48: 0.0019, 49: 0.0016, 50: 0.0015, 16: 0.0014,
        51: 0.0010, 53: 0.0009, 52: 0.0008, 55: 0.0007, 15: 0.0005,
        54: 0.0004, 56: 0.0004, 57: 0.0003, 58: 0.0003, 79: 0.0003,
        60: 0.0002, 87: 0.0002, 77: 0.0002, 59: 0.0002, 75: 0.0002,
        71: 0.0002, 81: 0.0002, 73: 0.0002, 84: 0.0002, 62: 0.0002,
        14: 0.0002, 82: 0.0001, 61: 0.0001, 63: 0.0001, 92: 0.0001,
        89: 0.0001, 64: 0.0001, 13: 0.0001, 65: 0.0001, 74: 0.0001,
        70: 0.0001, 67: 0.0001, 68: 0.0001, 72: 0.0001, 66: 0.0001,
        95: 0.0000, 69: 0.0000, 98: 0.0000, 12: 0.0000, 76: 0.0000,
        88: 0.0000, 80: 0.0000, 83: 0.0000, 96: 0.0000, 85: 0.0000,
        86: 0.0000, 91: 0.0000, 90: 0.0000, 78: 0.0000
    },
    "Smoker": {0: 0.5568, 1: 0.4432},
    "Stroke": {0: 0.9594, 1: 0.0406},
    "HeartDiseaseorAttack": {0: 0.9058, 1: 0.0942},
    "PhysActivity": {1: 0.7565, 0: 0.2435},
    "Fruits": {1: 0.6343, 0: 0.3657},
    "Veggies": {1: 0.8114, 0: 0.1886},
    "HvyAlcoholConsump": {0: 0.9438, 1: 0.0562},
    "AnyHealthcare": {1: 0.9511, 0: 0.0489},
    "NoDocbcCost": {0: 0.9158, 1: 0.0842},
    "GenHlth": {2: 0.3512, 3: 0.2982, 1: 0.1786, 4: 0.1244, 5: 0.0476},
    "MentHlth": {
        0: 0.6925, 2: 0.0515, 30: 0.0477, 5: 0.0356, 1: 0.0337,
        3: 0.0291, 10: 0.0251, 15: 0.0217, 4: 0.0149, 20: 0.0133,
        7: 0.0122, 25: 0.0047, 14: 0.0046, 6: 0.0039, 8: 0.0025,
        12: 0.0016, 28: 0.0013, 21: 0.0009, 29: 0.0006, 18: 0.0004,
        9: 0.0004, 16: 0.0003, 27: 0.0003, 22: 0.0002, 17: 0.0002,
        26: 0.0002, 11: 0.0002, 13: 0.0002, 23: 0.0001, 24: 0.0001,
        19: 0.0001
    },
    "PhysHlth": {
        0: 0.6309, 30: 0.0765, 2: 0.0582, 1: 0.0449, 3: 0.0335,
        5: 0.0300, 10: 0.0221, 15: 0.0194, 4: 0.0179, 7: 0.0179,
        20: 0.0129, 14: 0.0102, 25: 0.0053, 6: 0.0052, 8: 0.0032,
        21: 0.0026, 12: 0.0023, 28: 0.0021, 29: 0.0008, 9: 0.0007,
        18: 0.0006, 16: 0.0004, 27: 0.0004, 17: 0.0004, 24: 0.0003,
        22: 0.0003, 26: 0.0003, 13: 0.0003, 11: 0.0002, 23: 0.0002,
        19: 0.0001
    },
    "DiffWalk": {0: 0.8318, 1: 0.1682},
    "Sex": {0: 0.5597, 1: 0.4403},
    "Age": {
        9: 0.1310, 10: 0.1269, 8: 0.1215, 7: 0.1037, 11: 0.0928,
        6: 0.0781, 13: 0.0684, 5: 0.0637, 12: 0.0630, 4: 0.0545,
        3: 0.0438, 2: 0.0300, 1: 0.0225
    },
    "Education": {
        6: 0.4231, 5: 0.2756, 4: 0.2474, 3: 0.0374, 2: 0.0159, 1: 0.0007
    },
    "Income": {
        8: 0.3563, 7: 0.1704, 6: 0.1438, 5: 0.1020, 4: 0.0794,
        3: 0.0630, 2: 0.0464, 1: 0.0387
    }
}

# --- Calculation ---

# Calculate the probability that two random rows match on a single column k.
# P(match_k) = Sum over all values v [P(column_k = v)^2]
prob_match_per_column = {}
for col_name, probs in column_probabilities.items():
    prob_match_col = sum(p**2 for p in probs.values())
    prob_match_per_column[col_name] = prob_match_col
    # print(f"P(match) for {col_name}: {prob_match_col:.4e}") # Optional: print per-column match probability

# Calculate the probability that two random rows are identical across all columns
# P(identical) = Product over all columns k [P(match_k)]
prob_two_rows_identical = 1.0
for prob_match in prob_match_per_column.values():
    prob_two_rows_identical *= prob_match

print(f"Probability of two random rows being identical: {prob_two_rows_identical:.4e}")

# Calculate the total number of distinct pairs of rows
# num_pairs = N * (N - 1) / 2
num_pairs = N * (N - 1) // 2 # Use integer division for large numbers

print(f"Total number of row pairs: {num_pairs}")

# Calculate the expected number of identical pairs
expected_duplicate_pairs = num_pairs * prob_two_rows_identical

print(f"\nExpected number of duplicate pairs in the dataset: {expected_duplicate_pairs:.4f}")

# This represents the expected count of pairs (row A, row B) such that row A is identical to row B.
# The number of *unique* rows that have duplicates is harder to calculate directly but related to this value.


