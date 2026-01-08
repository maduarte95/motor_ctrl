"""
Analyze trial structure from control_experiment_full.csv

This script extracts information about:
- Number of trials with each instruction value
- Number of trials with each rotation (Angle1)
- Number of unique target coordinates (TargetX, TargetY)
- Consecutive trial blocks and their progression patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_angle_from_coords(x, y):
    """Convert (x, y) coordinates to angle in degrees (0-360)."""
    angle = np.arctan2(y, x) * 180 / np.pi
    # Convert to 0-360 range
    if angle < 0:
        angle += 360
    return angle


def analyze_block_progression(df, block_info):
    """
    Analyze how targets progress within each block.
    Determines if movement is clockwise, counterclockwise, or random.
    """
    print("\n10. TARGET PROGRESSION WITHIN BLOCKS")
    print("-" * 60)

    # Add angle column to dataframe
    df['TargetAngle'] = df.apply(lambda row: get_angle_from_coords(row['TargetX'], row['TargetY']), axis=1)

    progression_results = []

    for _, block_row in block_info.iterrows():
        block_num = block_row['Block']
        block_trials = df[df['Block'] == block_num]

        condition_str = f"Angle1={block_row['Angle1']}, Instr={block_row['Instruction']}"

        if len(block_trials) < 2:
            progression_results.append({
                'Block': block_num,
                'Length': len(block_trials),
                'Progression': 'N/A (single trial)',
                'AngleChanges': [],
                'Condition': condition_str,
                'Angle1': block_row['Angle1'],
                'Instruction': block_row['Instruction']
            })
            continue

        # Calculate angle changes between consecutive trials
        angles = block_trials['TargetAngle'].values
        angle_changes = []

        for i in range(len(angles) - 1):
            diff = angles[i + 1] - angles[i]
            # Normalize to -180 to 180 range
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            angle_changes.append(diff)

        # Determine progression type
        if len(angle_changes) == 0:
            progression_type = 'N/A'
        else:
            positive_changes = sum(1 for x in angle_changes if x > 0)
            negative_changes = sum(1 for x in angle_changes if x < 0)
            zero_changes = sum(1 for x in angle_changes if x == 0)

            # Determine predominant direction
            if positive_changes > negative_changes * 1.5:
                progression_type = 'Counterclockwise'
            elif negative_changes > positive_changes * 1.5:
                progression_type = 'Clockwise'
            elif zero_changes == len(angle_changes):
                progression_type = 'Same target'
            else:
                progression_type = 'Mixed/Random'

        progression_results.append({
            'Block': block_num,
            'Length': len(block_trials),
            'Progression': progression_type,
            'AngleChanges': angle_changes,
            'Condition': condition_str,
            'Angle1': block_row['Angle1'],
            'Instruction': block_row['Instruction']
        })

    # Create summary
    progression_df = pd.DataFrame(progression_results)

    # Show first 20 blocks
    print("\nProgression pattern for first 50 blocks:")
    for row in progression_results[:50]:
        print(f"  Block {row['Block']} (Length={row['Length']}, {row['Condition']}): {row['Progression']}")

    # Summary statistics
    print("\n11. PROGRESSION PATTERN SUMMARY")
    print("-" * 60)

    progression_counts = progression_df['Progression'].value_counts()
    print("\nOverall progression patterns:")
    for prog_type, count in progression_counts.items():
        pct = count / len(progression_df) * 100
        print(f"  {prog_type}: {count} blocks ({pct:.1f}%)")

    # Break down by condition
    print("\nProgression patterns by condition:")
    for angle in sorted(df['Angle1'].unique()):
        for instruction in sorted(df['Instruction'].unique()):
            condition_mask = progression_df['Condition'] == f"Angle1={angle}, Instr={instruction}"
            condition_progs = progression_df[condition_mask]
            if len(condition_progs) > 0:
                print(f"\n  Angle1={angle}°, Instruction={instruction}:")
                cond_counts = condition_progs['Progression'].value_counts()
                for prog_type, count in cond_counts.items():
                    pct = count / len(condition_progs) * 100
                    print(f"    {prog_type}: {count} blocks ({pct:.1f}%)")

    return progression_df


def analyze_trial_structure(csv_path):
    """
    Analyze the trial structure from the experiment CSV file.

    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file containing experiment data

    Returns:
    --------
    dict : Dictionary containing analysis results
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    print(f"Total number of trials: {len(df)}")
    print(f"\nColumns in dataset: {list(df.columns)}")
    print("=" * 60)

    # Analyze instruction distribution
    print("\n1. INSTRUCTION DISTRIBUTION")
    print("-" * 60)
    instruction_counts = df['Instruction'].value_counts().sort_index()
    print(instruction_counts)
    print(f"\nTotal trials per instruction:")
    for instruction, count in instruction_counts.items():
        print(f"  Instruction {instruction}: {count} trials ({count/len(df)*100:.1f}%)")

    # Analyze rotation (Angle1) distribution
    print("\n2. ROTATION (ANGLE1) DISTRIBUTION")
    print("-" * 60)
    angle1_counts = df['Angle1'].value_counts().sort_index()
    print(angle1_counts)
    print(f"\nTotal trials per rotation:")
    for angle, count in angle1_counts.items():
        print(f"  Angle1 = {angle}°: {count} trials ({count/len(df)*100:.1f}%)")

    # Cross-tabulation: Instruction x Angle1
    print("\n3. TRIALS BY INSTRUCTION AND ROTATION")
    print("-" * 60)
    cross_tab = pd.crosstab(df['Angle1'], df['Instruction'], margins=True)
    print(cross_tab)

    # Analyze unique target coordinates
    print("\n4. UNIQUE TARGET COORDINATES")
    print("-" * 60)
    unique_targets = df[['TargetX', 'TargetY']].drop_duplicates()
    num_unique_targets = len(unique_targets)
    print(f"Number of unique target coordinates: {num_unique_targets}")
    print(f"\nUnique target coordinates:")
    print(unique_targets.sort_values(['TargetX', 'TargetY']).to_string(index=False))

    # Count occurrences of each target
    print("\n5. TARGET COORDINATE FREQUENCIES")
    print("-" * 60)
    target_counts = df.groupby(['TargetX', 'TargetY']).size().reset_index(name='count')
    target_counts = target_counts.sort_values('count', ascending=False)
    print(f"Most common targets:")
    print(target_counts.to_string(index=False))

    # Analyze consecutive trial blocks
    print("\n6. CONSECUTIVE TRIAL BLOCKS")
    print("-" * 60)

    # Create a column that combines Angle1 and Instruction
    df['Condition'] = df['Angle1'].astype(str) + '_' + df['Instruction'].astype(str)

    # Find consecutive blocks
    df['Block'] = (df['Condition'] != df['Condition'].shift()).cumsum()

    # Calculate block lengths
    block_info = df.groupby('Block').agg({
        'Condition': 'first',
        'Angle1': 'first',
        'Instruction': 'first',
        'Trial': ['first', 'last', 'count']
    }).reset_index()

    block_info.columns = ['Block', 'Condition', 'Angle1', 'Instruction', 'Start_Trial', 'End_Trial', 'Length']

    print(f"\nTotal number of blocks: {len(block_info)}")
    print(f"\nBlock structure (first 50 blocks):")
    print(block_info.head(50).to_string(index=False))

    # Analyze block lengths by condition
    print("\n7. BLOCK LENGTH STATISTICS BY CONDITION")
    print("-" * 60)
    block_length_stats = block_info.groupby(['Angle1', 'Instruction'])['Length'].describe()
    print(block_length_stats)

    # Distribution of block lengths
    print("\n8. DISTRIBUTION OF CONSECUTIVE TRIAL COUNTS")
    print("-" * 60)
    for angle in sorted(df['Angle1'].unique()):
        for instruction in sorted(df['Instruction'].unique()):
            condition_blocks = block_info[(block_info['Angle1'] == angle) &
                                         (block_info['Instruction'] == instruction)]
            if len(condition_blocks) > 0:
                length_counts = condition_blocks['Length'].value_counts().sort_index()
                print(f"\nAngle1={angle}°, Instruction={instruction}:")
                print(f"  Number of blocks: {len(condition_blocks)}")
                print(f"  Block lengths distribution:")
                for length, count in length_counts.items():
                    print(f"    {length} consecutive trials: {count} blocks")

    # Analyze target progression within blocks
    progression_df = analyze_block_progression(df, block_info)

    # Summary statistics
    print("\n12. SUMMARY")
    print("=" * 60)
    results = {
        'total_trials': len(df),
        'instructions': instruction_counts.to_dict(),
        'rotations': angle1_counts.to_dict(),
        'num_unique_targets': num_unique_targets,
        'unique_targets': unique_targets.values.tolist(),
        'num_blocks': len(block_info),
        'block_info': block_info.to_dict('records'),
        'progression_summary': progression_df['Progression'].value_counts().to_dict()
    }

    print(f"Total trials: {results['total_trials']}")
    print(f"Number of instruction types: {len(results['instructions'])}")
    print(f"Number of rotation values: {len(results['rotations'])}")
    print(f"Number of unique target positions: {results['num_unique_targets']}")
    print(f"Total number of condition blocks: {results['num_blocks']}")
    print(f"Average block length: {block_info['Length'].mean():.2f} trials")
    print(f"Block length range: {block_info['Length'].min()}-{block_info['Length'].max()} trials")

    return results


if __name__ == "__main__":
    # Path to the CSV file (relative to this script)
    csv_path = Path(__file__).parent / "experiment_adapt_short.csv" #"control_experiment_full.csv", "double_cursor_full.csv"

    # Run analysis
    results = analyze_trial_structure(csv_path)
