"""
Generate experiment CSV files with customizable trial structure.

This script generates experiment files with:
- Three experimental blocks with different rotation angles
- Configurable number of trials per block
- First trial of each block has Instruction=1, rest have Instruction=0
- Targets randomly selected from upper semicircle positions
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# CONFIGURATION - Customize these parameters
# =============================================================================

# Block configuration: list of (num_trials, rotation_angle, first_trial_instruction) tuples
BLOCK_CONFIG = [
    (30, 0, 1),   # Block 1: 20 trials, 0° rotation, R
    (30, 0, 2),   # Block 2: 20 trials, 0° rotation, L
    (100, 45, 3),  # Block 3: 20 trials, 45° rotation, L
    (100, 45, 1),  # Block 4: 20 trials, 45° rotation, R
    (50, 0, 4),   # Block 5: 20 trials, 0° rotation, R
    (50, 0, 2),   # Block 6: 20 trials, 0° rotation, L
]

# uncomment for no adapt version
# BLOCK_CONFIG = [
#     (30, 0, 1),   # Block 1: 20 trials, 0° rotation, R
#     (30, 0, 2),   # Block 2: 20 trials, 0° rotation, L
#     (100, 0, 3),   # Block 3: 20 trials, 0° rotation, L
#     (100, 45, 1),  # Block 4: 20 trials, 45° rotation, R
#     (50, 0, 4),   # Block 5: 20 trials, 0° rotation, R
#     (50, 0, 2),   # Block 6: 20 trials, 0° rotation, L
# ]

# Target positions on the circle (non-cardinal positions)
# Upper semicircle only (Y >= 0)
UPPER_SEMICIRCLE_TARGETS = [
    (-0.831491579260158, 0.344415089128581),   # ~157.5°
    (-0.344415089128581, 0.831491579260158),   # ~112.5°
    (0.344415089128581, 0.831491579260158),    # ~67.5°
    (0.831491579260158, 0.344415089128581),    # ~22.5°
]

# Random seed for reproducibility (set to None for different results each time)
RANDOM_SEED = 33

# Output filename
OUTPUT_FILENAME = "experiment_adapt_full.csv"

# =============================================================================
# Script implementation
# =============================================================================


def generate_experiment(block_config, targets, random_seed=None):
    """
    Generate experiment trial structure.

    Parameters:
    -----------
    block_config : list of tuples
        Each tuple is (num_trials, rotation_angle, first_trial_instruction)
    targets : list of tuples
        Each tuple is (TargetX, TargetY) coordinate
    random_seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame : Generated experiment data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    trials = []
    trial_number = 1

    for block_idx, (num_trials, rotation_angle, first_trial_instruction) in enumerate(block_config, 1):
        print(f"Generating Block {block_idx}: {num_trials} trials, {rotation_angle}° rotation, first trial instruction={first_trial_instruction}")

        for trial_in_block in range(num_trials):
            # First trial of each block has specified instruction, rest have Instruction=0
            instruction = first_trial_instruction if trial_in_block == 0 else 0

            # Randomly select target from upper semicircle
            target_x, target_y = targets[np.random.randint(0, len(targets))]

            trial = {
                'Trial': trial_number,
                'TargetX': target_x,
                'TargetY': target_y,
                'AimX': 0,
                'AimY': 0,
                'Angle1': rotation_angle,
                'Angle2': 0,
                'Layer1': 1,
                'Layer2': 0,
                'Instruction': instruction
            }

            trials.append(trial)
            trial_number += 1

    df = pd.DataFrame(trials)
    return df


def main():
    """Main function to generate and save experiment file."""
    print("=" * 60)
    print("EXPERIMENT GENERATOR")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Number of blocks: {len(BLOCK_CONFIG)}")
    for i, (num_trials, angle, first_instr) in enumerate(BLOCK_CONFIG, 1):
        print(f"    Block {i}: {num_trials} trials at {angle}° rotation, first trial instruction={first_instr}")
    print(f"  Target positions: {len(UPPER_SEMICIRCLE_TARGETS)} (upper semicircle)")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Output file: {OUTPUT_FILENAME}")

    # Generate experiment
    print(f"\n{'-' * 60}")
    print("Generating experiment...")
    print(f"{'-' * 60}")

    df = generate_experiment(BLOCK_CONFIG, UPPER_SEMICIRCLE_TARGETS, RANDOM_SEED)

    # Calculate total trials
    total_trials = len(df)
    print(f"\nTotal trials generated: {total_trials}")

    # Show summary statistics
    print(f"\nSummary:")
    print(f"  Trials per rotation angle:")
    for angle, count in df['Angle1'].value_counts().sort_index().items():
        print(f"    {angle}°: {count} trials")
    print(f"  Instruction distribution:")
    for instr, count in df['Instruction'].value_counts().sort_index().items():
        print(f"    Instruction {instr}: {count} trials")

    # Save to CSV
    output_path = Path(__file__).parent / OUTPUT_FILENAME
    df.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"SUCCESS: Experiment saved to {output_path}")
    print(f"{'=' * 60}")

    # Show first few trials
    print(f"\nFirst 10 trials:")
    print(df.head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    main()
