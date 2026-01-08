"""
Plot superimposed trajectories for a representative trial from each block.

For each block (1-6) and each group (1-2), we plot all participants' trajectories
for a representative trial on a single figure. This creates 6 x 2 = 12 figures total.

The plots reproduce the experimental screen with all 4 targets shown at their correct positions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# All target positions from the experiment (4 non-cardinal positions)
TARGET_POSITIONS = [
    (-0.831492, 0.344415),  # Upper left
    (0.831492, 0.344415),   # Upper right
    (0.344415, 0.831492),   # Top middle-right
    (-0.344415, 0.831492),  # Top middle-left
]

# Define block structure
BLOCK_STRUCTURE = {
    "group_1": {
        1: {"start": 0, "end": 29, "hand": "right", "condition": "baseline"},
        2: {"start": 30, "end": 59, "hand": "left", "condition": "baseline"},
        3: {"start": 60, "end": 159, "hand": "left", "condition": "baseline extended"},
        4: {"start": 16, "end": 259, "hand": "left", "condition": "45° perturbation"},
        5: {"start": 260, "end": 309, "hand": "right", "condition": "aftereffects"},
        6: {"start": 310, "end": 359, "hand": "left", "condition": "aftereffects"},
    },
    "group_2": {
        1: {"start": 0, "end": 29, "hand": "right", "condition": "baseline"},
        2: {"start": 30, "end": 59, "hand": "left", "condition": "baseline"},
        3: {"start": 60, "end": 159, "hand": "right", "condition": "45° perturbation"},
        4: {"start": 160, "end": 259, "hand": "left", "condition": "45° perturbation"},
        5: {"start": 260, "end": 309, "hand": "right", "condition": "aftereffects"},
        6: {"start": 310, "end": 359, "hand": "left", "condition": "aftereffects"},
    },
}


def load_trial_data(participant_dir: Path, trial_num: int) -> pd.DataFrame:
    """Load cursor trajectory data for a specific trial."""
    trial_file = participant_dir / f"cursor1_trial{trial_num}.txt"

    if not trial_file.exists():
        return None

    # Read CSV with x, y, timestamp columns
    df = pd.read_csv(trial_file, header=None, names=["x", "y", "timestamp"])
    return df


def get_representative_trial(block_start: int, block_end: int, trial_index: str = "middle") -> int:
    """
    Get a trial number from a block.

    Args:
        block_start: First trial in block
        block_end: Last trial in block
        trial_index: Which trial to select - "first", "middle", "last", or a specific trial number

    Returns:
        Trial number
    """
    if trial_index == "first":
        return block_start
    elif trial_index == "last":
        return block_end
    elif trial_index == "middle":
        return (block_start + block_end) // 2
    else:
        # Assume it's a specific trial number
        try:
            trial_num = int(trial_index)
            if block_start <= trial_num <= block_end:
                return trial_num
            else:
                print(f"Warning: trial {trial_num} not in block range [{block_start}, {block_end}], using middle trial")
                return (block_start + block_end) // 2
        except ValueError:
            print(f"Warning: invalid trial_index '{trial_index}', using middle trial")
            return (block_start + block_end) // 2


def plot_block_trajectories(group_name: str, block_num: int, output_dir: Path, trial_index: str = "middle"):
    """
    Plot superimposed trajectories for all participants in a group for a specific block.

    Args:
        group_name: "group_1" or "group_2"
        block_num: Block number (1-6)
        output_dir: Directory to save the figure
        trial_index: Which trial to plot - "first", "middle", "last", or a specific trial number
    """
    data_dir = Path("data") / group_name
    participants = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    block_info = BLOCK_STRUCTURE[group_name][block_num]
    trial_num = get_representative_trial(block_info["start"], block_info["end"], trial_index)

    # Create figure with black background (like experimental screen)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot all target positions (white circles like in the experiment)
    for target_x, target_y in TARGET_POSITIONS:
        ax.plot(target_x, target_y, 'o', color='white', markersize=15,
                markeredgecolor='white', markerfacecolor='white', zorder=5)

    # Load trial config to get the actual target for this trial
    config_file = (
        Path("experiments") / f"experiment_{'adapt' if group_name == 'group_2' else 'no_adapt'}_full.csv"
    )
    config_df = pd.read_csv(config_file)

    # Trial numbers in config are 1-indexed, convert to 0-indexed
    trial_config = config_df[config_df["Trial"] == trial_num + 1].iloc[0]
    active_target_x = trial_config["TargetX"]
    active_target_y = trial_config["TargetY"]

    # Highlight the active target for this trial with a larger circle
    ax.plot(active_target_x, active_target_y, 'o', color='yellow', markersize=20,
            markeredgecolor='yellow', alpha=0.5, zorder=6)

    # Plot each participant's trajectory (no legend with names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))

    for i, participant_dir in enumerate(participants):
        # Load trial data
        df = load_trial_data(participant_dir, trial_num)

        if df is not None and len(df) > 0:
            ax.plot(
                df["x"],
                df["y"],
                color=colors[i],
                alpha=0.7,
                linewidth=2,
            )

    # Add starting point marker at origin
    ax.plot(0, 0, 'o', color='red', markersize=12, zorder=10)

    # Format plot - consistent axes for all plots
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("X Position", fontsize=14, color='white')
    ax.set_ylabel("Y Position", fontsize=14, color='white')
    ax.set_title(
        f"{group_name.replace('_', ' ').title()} - Block {block_num}\n"
        f"{block_info['hand'].capitalize()} hand - {block_info['condition'].capitalize()}\n"
        f"Trial {trial_num}",
        fontsize=16,
        fontweight="bold",
        color='white'
    )
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_aspect("equal")
    ax.tick_params(colors='white', which='both')

    # Make spines white
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{group_name}_block{block_num}_trajectories.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor='black')
    plt.close()

    print(f"Saved: {output_file}")


def main():
    """Generate all trajectory plots."""
    import sys

    # Check if trial_index is provided as command line argument
    trial_index = "middle"
    if len(sys.argv) > 1:
        trial_index = sys.argv[1]
        print(f"Using trial index: {trial_index}")

    output_dir = Path("data") / "trajectory_plots"

    # Plot for each group and each block
    for group_name in ["group_1", "group_2"]:
        for block_num in range(1, 7):
            print(f"\nProcessing {group_name}, Block {block_num}...")
            plot_block_trajectories(group_name, block_num, output_dir, trial_index)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
