"""
Convert adapt experiment to no-adapt experiment.

This script takes an adaptation experiment CSV and creates a no-adapt version
by changing the rotation angle in block 3 from 45° to 0°, while keeping all
other trial information (target positions, instructions, etc.) identical.
"""

import pandas as pd
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file (adapt version)
INPUT_FILENAME = "experiment_adapt_full.csv"

# Output file (no-adapt version)
OUTPUT_FILENAME = "experiment_no_adapt_full.csv"

# Block configuration for identifying which block to modify
# This should match the ADAPT configuration used to generate the input file
ADAPT_BLOCK_CONFIG = [
    (30, 0, 1),    # Block 1: 30 trials, 0° rotation
    (30, 0, 2),    # Block 2: 30 trials, 0° rotation
    (100, 45, 3),  # Block 3: 100 trials, 45° rotation -> will change to 0°
    (100, 45, 1),  # Block 4: 100 trials, 45° rotation
    (50, 0, 4),    # Block 5: 50 trials, 0° rotation
    (50, 0, 2),    # Block 6: 50 trials, 0° rotation
]

# Which block(s) to modify (1-indexed)
# Block 3 is the adaptation block we want to change to no-adapt
BLOCKS_TO_MODIFY = [3]

# =============================================================================
# Script implementation
# =============================================================================


def convert_adapt_to_no_adapt(input_path, output_path, block_config, blocks_to_modify):
    """
    Convert adapt experiment to no-adapt by changing rotation angles.

    Parameters:
    -----------
    input_path : Path
        Path to input CSV file (adapt version)
    output_path : Path
        Path to output CSV file (no-adapt version)
    block_config : list of tuples
        Block configuration (num_trials, rotation_angle, first_trial_instruction)
    blocks_to_modify : list of int
        1-indexed block numbers to modify (change rotation to 0°)
    """
    print("=" * 60)
    print("ADAPT TO NO-ADAPT CONVERTER")
    print("=" * 60)

    # Read the input file
    print(f"\nReading input file: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total trials: {len(df)}")

    # Create a copy for modification
    df_no_adapt = df.copy()

    # Calculate trial ranges for each block
    trial_ranges = []
    cumulative_trials = 0
    for block_idx, (num_trials, rotation, instruction) in enumerate(block_config, 1):
        start_trial = cumulative_trials + 1
        end_trial = cumulative_trials + num_trials
        trial_ranges.append({
            'block': block_idx,
            'start': start_trial,
            'end': end_trial,
            'num_trials': num_trials,
            'original_rotation': rotation,
            'instruction': instruction
        })
        cumulative_trials += num_trials

    # Display block structure
    print(f"\nBlock structure:")
    for block_info in trial_ranges:
        print(f"  Block {block_info['block']}: Trials {block_info['start']}-{block_info['end']} "
              f"({block_info['num_trials']} trials), Angle={block_info['original_rotation']}°")

    # Modify specified blocks
    print(f"\nModifying blocks: {blocks_to_modify}")
    total_modified = 0

    for block_num in blocks_to_modify:
        if block_num < 1 or block_num > len(trial_ranges):
            print(f"  WARNING: Block {block_num} out of range, skipping")
            continue

        block_info = trial_ranges[block_num - 1]
        start_trial = block_info['start']
        end_trial = block_info['end']
        original_rotation = block_info['original_rotation']

        # Find trials in this block and change rotation to 0
        mask = (df_no_adapt['Trial'] >= start_trial) & (df_no_adapt['Trial'] <= end_trial)
        num_modified = mask.sum()

        print(f"  Block {block_num} (Trials {start_trial}-{end_trial}):")
        print(f"    Original rotation: {original_rotation}°")
        print(f"    New rotation: 0°")
        print(f"    Trials modified: {num_modified}")

        # Change Angle1 to 0 for this block
        df_no_adapt.loc[mask, 'Angle1'] = 0
        total_modified += num_modified

    # Verify changes
    print(f"\n{'-' * 60}")
    print("Verification:")
    print(f"  Total trials modified: {total_modified}")
    print(f"\n  Rotation distribution in original file:")
    for angle, count in df['Angle1'].value_counts().sort_index().items():
        print(f"    {angle}°: {count} trials")
    print(f"\n  Rotation distribution in no-adapt file:")
    for angle, count in df_no_adapt['Angle1'].value_counts().sort_index().items():
        print(f"    {angle}°: {count} trials")

    # Check that everything else is identical
    print(f"\n{'-' * 60}")
    print("Checking that all other columns are identical...")
    cols_to_check = ['Trial', 'TargetX', 'TargetY', 'AimX', 'AimY',
                     'Angle2', 'Layer1', 'Layer2', 'Instruction']
    all_identical = True
    for col in cols_to_check:
        if col in df.columns and col in df_no_adapt.columns:
            if not df[col].equals(df_no_adapt[col]):
                print(f"  WARNING: Column '{col}' differs!")
                all_identical = False

    if all_identical:
        print("  ✓ All other columns are identical")

    # Save the no-adapt version
    df_no_adapt.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"SUCCESS: No-adapt experiment saved to {output_path}")
    print(f"{'=' * 60}")

    # Show sample of modified block
    if blocks_to_modify:
        block_num = blocks_to_modify[0]
        block_info = trial_ranges[block_num - 1]
        start_trial = block_info['start']

        print(f"\nSample from modified Block {block_num} (first 5 trials):")
        sample_mask = (df_no_adapt['Trial'] >= start_trial) & (df_no_adapt['Trial'] < start_trial + 5)
        print(df_no_adapt[sample_mask].to_string(index=False))

    return df_no_adapt


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_FILENAME
    output_path = script_dir / OUTPUT_FILENAME

    # Check if input file exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    # Convert
    df_no_adapt = convert_adapt_to_no_adapt(
        input_path,
        output_path,
        ADAPT_BLOCK_CONFIG,
        BLOCKS_TO_MODIFY
    )


if __name__ == "__main__":
    main()
