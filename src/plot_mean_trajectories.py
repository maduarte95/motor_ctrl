"""
Plot mean trajectories for each block.

For each block and group, we:
1. Compute the mean trajectory for each participant to each target
2. Plot all mean trajectories superimposed (all participants, all targets)

This creates 6 x 2 = 12 figures total.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


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
        4: {"start": 160, "end": 259, "hand": "right", "condition": "45° perturbation"},
        5: {"start": 260, "end": 309, "hand": "right", "condition": "aftereffects"},
        6: {"start": 310, "end": 359, "hand": "left", "condition": "aftereffects"},
    },
    "group_2": {
        1: {"start": 0, "end": 29, "hand": "right", "condition": "baseline"},
        2: {"start": 30, "end": 59, "hand": "left", "condition": "baseline"},
        3: {"start": 60, "end": 159, "hand": "left", "condition": "45° perturbation"},
        4: {"start": 160, "end": 259, "hand": "right", "condition": "45° perturbation"},
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


def normalize_trajectory(df: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    """
    Normalize trajectory to have a fixed number of points.
    Uses linear interpolation to resample trajectory.
    """
    if df is None or len(df) < 2:
        return None

    # Create normalized "time" parameter from 0 to 1
    t_original = np.linspace(0, 1, len(df))
    t_normalized = np.linspace(0, 1, n_points)

    # Interpolate x and y coordinates
    try:
        interp_x = interp1d(t_original, df["x"].values, kind="linear")
        interp_y = interp1d(t_original, df["y"].values, kind="linear")

        x_normalized = interp_x(t_normalized)
        y_normalized = interp_y(t_normalized)

        return pd.DataFrame({"x": x_normalized, "y": y_normalized})
    except:
        return None


def compute_mean_trajectory_per_target(
    participant_dir: Path, block_start: int, block_end: int, config_df: pd.DataFrame
) -> dict:
    """
    Compute mean trajectory for each target in a block for one participant.

    Returns:
        dict: {(target_x, target_y): mean_trajectory_df}
    """
    mean_trajectories = {}

    # Group trials by target
    target_trials = {}
    for trial_num in range(block_start, block_end + 1):
        # Get target for this trial (config trials are 1-indexed)
        trial_config = config_df[config_df["Trial"] == trial_num + 1]
        if len(trial_config) == 0:
            continue

        target_x = trial_config.iloc[0]["TargetX"]
        target_y = trial_config.iloc[0]["TargetY"]
        target_key = (round(target_x, 6), round(target_y, 6))

        if target_key not in target_trials:
            target_trials[target_key] = []

        target_trials[target_key].append(trial_num)

    # Compute mean trajectory for each target
    for target_key, trial_nums in target_trials.items():
        normalized_trajectories = []

        for trial_num in trial_nums:
            df = load_trial_data(participant_dir, trial_num)
            norm_df = normalize_trajectory(df)

            if norm_df is not None:
                normalized_trajectories.append(norm_df)

        if len(normalized_trajectories) > 0:
            # Average across all trials to this target
            mean_x = np.mean([df["x"].values for df in normalized_trajectories], axis=0)
            mean_y = np.mean([df["y"].values for df in normalized_trajectories], axis=0)

            mean_trajectories[target_key] = pd.DataFrame({"x": mean_x, "y": mean_y})

    return mean_trajectories


def plot_mean_trajectories(group_name: str, block_num: int, output_dir: Path):
    """
    Plot mean trajectories for all participants and targets in a block.

    Args:
        group_name: "group_1" or "group_2"
        block_num: Block number (1-6)
        output_dir: Directory to save the figure
    """
    data_dir = Path("data") / group_name
    participants = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    block_info = BLOCK_STRUCTURE[group_name][block_num]

    # Load trial config
    config_file = (
        Path("experiments")
        / f"experiment_{'adapt' if group_name == 'group_2' else 'no_adapt'}_full.csv"
    )
    config_df = pd.read_csv(config_file)

    # Create figure with black background
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Plot all target positions (white circles)
    for target_x, target_y in TARGET_POSITIONS:
        ax.plot(
            target_x,
            target_y,
            "o",
            color="white",
            markersize=15,
            markeredgecolor="white",
            markerfacecolor="white",
            zorder=5,
        )

    # Plot mean trajectories for each participant
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))

    for i, participant_dir in enumerate(participants):
        print(f"  Processing {participant_dir.name}...")

        mean_trajectories = compute_mean_trajectory_per_target(
            participant_dir, block_info["start"], block_info["end"], config_df
        )

        # Plot each mean trajectory (one per target)
        for target_key, mean_df in mean_trajectories.items():
            ax.plot(
                mean_df["x"],
                mean_df["y"],
                color=colors[i],
                alpha=0.7,
                linewidth=2,
            )

    # Add starting point marker at origin
    ax.plot(0, 0, "o", color="red", markersize=12, zorder=10)

    # Format plot - consistent axes for all plots
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("X Position", fontsize=14, color="white")
    ax.set_ylabel("Y Position", fontsize=14, color="white")
    ax.set_title(
        f"{group_name.replace('_', ' ').title()} - Block {block_num}\n"
        f"{block_info['hand'].capitalize()} hand - {block_info['condition'].capitalize()}\n"
        f"Mean trajectories (all participants, all targets)",
        fontsize=16,
        fontweight="bold",
        color="white",
    )
    ax.grid(True, alpha=0.2, color="gray")
    ax.set_aspect("equal")
    ax.tick_params(colors="white", which="both")

    # Make spines white
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{group_name}_block{block_num}_mean_trajectories.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()

    print(f"Saved: {output_file}")


def main():
    """Generate all mean trajectory plots."""
    output_dir = Path("data") / "mean_trajectory_plots"

    # Plot for each group and each block
    for group_name in ["group_1", "group_2"]:
        for block_num in range(1, 7):
            print(f"\nProcessing {group_name}, Block {block_num}...")
            plot_mean_trajectories(group_name, block_num, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
