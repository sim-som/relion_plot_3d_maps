#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
from pathlib import Path
import starfile
import seaborn as sns
import argparse

# Functions:
def plot_map_slices(maps, num_classes, map_shape):
    """Plot slices of the filtered maps."""
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    if num_classes == 1:
        axes = axes.reshape(1,3)
    for i, map_file in enumerate(maps):
        with mrcfile.open(map_file) as mrc:
            map_arr = mrc.data
        # Plot central slices in x, y, z directions
        axes[i, 0].imshow(map_arr[map_shape[0] // 2, :, :], cmap='gray')
        axes[i, 1].imshow(map_arr[:, map_shape[1] // 2, :], cmap='gray')
        axes[i, 2].imshow(map_arr[:, :, map_shape[2] // 2], cmap='gray')
        axes[i, 0].set_xlabel(f"Class {i+1} - Z slice")
        axes[i, 1].set_xlabel(f"Class {i+1} - Y slice")
        axes[i, 2].set_xlabel(f"Class {i+1} - X slice")
    
    return fig, axes

from matplotlib.colors import LogNorm

def plot_orientation_histogram(azimuthal_angles, polar_angles, bins=36, cmap='viridis', 
                               log_scale=False, fig_size=(10, 8), title=None, degrees=False):
    """
    Plot a 2D histogram of 3D orientations given azimuthal and polar angles.
    
    Parameters:
    -----------
    azimuthal_angles : list or array
        List of azimuthal angles (phi). Range [-π, π] or [-180°, 180°].
    polar_angles : list or array
        List of polar angles (theta). Range [0, π] or [0, 180°].
    bins : int or tuple, optional
        Number of bins for the histogram (default=36, which gives 10-degree bins for a full circle).
        Can also be a tuple (n_phi, n_theta) to specify different bin counts for each dimension.
    cmap : str, optional
        Colormap for the histogram (default='viridis').
    log_scale : bool, optional
        Whether to use logarithmic color scaling (default=False).
    fig_size : tuple, optional
        Figure size in inches (default=(10, 8)).
    title : str, optional
        Title for the plot. If None, a default title is used.
    degrees : bool, optional
        If True, input angles are treated as degrees and converted to radians.
        Also affects axis labels and tick marks (default=False).
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
        The created figure and axis objects, allowing for further customization.
    """
    # Convert inputs to numpy arrays if they aren't already
    azimuthal_angles = np.array(azimuthal_angles)
    polar_angles = np.array(polar_angles)
    
    # Convert from degrees to radians if necessary
    if degrees:
        azimuthal_angles = np.radians(azimuthal_angles)
        polar_angles = np.radians(polar_angles)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create the 2D histogram
    norm = LogNorm() if log_scale else None
    h = ax.hist2d(azimuthal_angles, polar_angles, bins=bins, cmap=cmap, norm=norm)
    
    # Add a colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count')
    
    # Set labels and title
    unit_label = "°" if degrees else " [rad]"
    ax.set_xlabel('Azimuthal Angle ("Rot") (φ)' + unit_label)
    ax.set_ylabel('Polar Angle ("Tilt") (θ)' + unit_label)
    if title is None:
        title = '2D Histogram of 3D Orientations'
    ax.set_title(title)
    
    # Set axes limits for azimuthal [-π, π] and polar [0, π]
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, np.pi)
    
    # Add tick marks in terms of π for better readability
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    
    if degrees:
        # If input was in degrees, show tick labels in degrees
        ax.set_xticklabels(['-180°', '-90°', '0°', '90°', '180°'])
        ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°'])
    else:
        # Otherwise show tick labels in terms of π
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        ax.set_yticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
    
    # Add grid for better readability
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Relion 3D refinement results and generate visualizations')
    
    parser.add_argument('--job_dir', type=str, required=True,
                        help='Path to the Relion job directory')
    parser.add_argument("--iteration", "-i", default="-1", type=str)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (defaults to job_dir)')
    parser.add_argument('--bins', type=int, default=36,
                        help='Number of bins for orientation histogram (default: 36)')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use logarithmic color scale for histogram')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to output directory')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively')
    
    return parser.parse_args()

def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up job directory and validate it exists
    job_dir = Path(args.job_dir)
    assert job_dir.exists(), f"Job directory {job_dir} does not exist."
    
    # Set up output directory
    output_dir = Path(args.output_dir) if args.output_dir else job_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get job info
    job_type = job_dir.parent.stem
    job_name = job_dir.stem
    
    # Find filtered maps
    map_glob = f"run_it{args.iteration}_class00?.mrc"
    filtered_maps = list(job_dir.glob(map_glob))
    assert len(filtered_maps) > 0, f"No filtered maps found in {job_dir} with glob pattern {map_glob}."
    NUM_CLASSES = len(filtered_maps)
    
    # Read the first filtered map to get the dimensions and pixel size
    with mrcfile.open(filtered_maps[0]) as mrc:
        MAP_SHAPE = mrc.data.shape
        ANGPIX = mrc.voxel_size.x  # pixel size in Angstroms
    
    # Load particle data
    particles_star_fpath = job_dir / Path(f"run_it{args.iteration}_data.star")
    assert particles_star_fpath.exists(), f"No particle data! Particle star file {particles_star_fpath} does not exist."
    particle_df = starfile.read(particles_star_fpath)["particles"]
    
    # Load refinement/model data
    model_star_fpath = job_dir / Path(f"run_it{args.iteration}_model.star")
    assert model_star_fpath.exists(), f"Model star file {model_star_fpath} does not exist."
    model_dict = starfile.read(model_star_fpath)
    fsc_resolution = model_dict["model_general"]["rlnCurrentResolution"]
    print(f"{job_name}: FSC resolution = {fsc_resolution} Å")
    
    # Plot map slices
    fig_slices, _ = plot_map_slices(filtered_maps, NUM_CLASSES, MAP_SHAPE)
    title = f"Map(s) of {job_type}/{job_name} ({fsc_resolution:.2f} Å)"
    fig_slices.suptitle(title)
    plt.tight_layout()
    
    if args.save_plots:
        slices_output = output_dir / f"{job_type}_{job_name}_map_slices.png"
        fig_slices.savefig(slices_output, dpi=300, bbox_inches='tight')
        print(f"Saved map slices to {slices_output}")
    
    # Plot orientation histogram
    
    fig_orient, _ = plot_orientation_histogram(
        azimuthal_angles=particle_df["rlnAngleRot"],
        polar_angles=particle_df["rlnAngleTilt"],
        degrees=True,
        log_scale=args.log_scale,
        bins=args.bins,
        title=f"Orientation Distribution - {job_type}/{job_name}"
    )

    
    if args.save_plots:
        orient_output = output_dir / f"{job_type}_{job_name}_orientation_histogram.png"
        fig_orient.savefig(orient_output, dpi=300, bbox_inches='tight')
        print(f"Saved orientation histogram to {orient_output}")
    
    # Show plots if requested
    if args.show_plots:
        plt.show()
    elif not args.save_plots:
        # If neither save nor show is specified, show by default
        plt.show()

if __name__ == "__main__":
    main()
