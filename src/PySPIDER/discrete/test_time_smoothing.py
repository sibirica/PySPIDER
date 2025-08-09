#!/usr/bin/env python3
"""
Test script to visualize the effect of temporal smoothing on coarse-grained fields
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Import the SPIDER modules
from .process_library_terms import SRDataset
from ..commons.library import Observable, DerivativeOrder, LibraryPrime
from ..commons.process_library_terms import IntegrationDomain
from ..commons.z3base import LiteralIndex
from .library import CoarseGrainedProduct


def load_lj_data(data_path="discrete/LJ/positions_rho_0.30_savefreq_3.npz"):
    """Load the Lennard-Jones simulation data"""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    positions_orig = data["positions"]
    vs_orig = data["vs"]
    deltat = data["deltat"]
    dims = data["dims"]

    print(f"Data shape: positions {positions_orig.shape}, velocities {vs_orig.shape}")
    print(f"Time step: {deltat}, dimensions: {dims}")

    return positions_orig, vs_orig, deltat, dims


def create_test_datasets(positions, vs, deltat, dims, time_sigma_values=[0, 2.0]):
    """Create SRDataset instances with and without time smoothing"""

    # Basic parameters
    Np, nt = positions.shape[0], positions.shape[2]
    world_size = np.array([dims[0], dims[1], nt])

    # Create observables
    observables = [
        Observable(string="rho", rank=0, indices=()),  # density
        Observable(
            string="v", rank=1, indices=(LiteralIndex(0),)
        ),  # x-velocity component
        Observable(
            string="v", rank=1, indices=(LiteralIndex(1),)
        ),  # y-velocity component
    ]

    # Data dictionary
    data_dict = {
        "v": vs,  # velocity components
    }

    datasets = {}

    for time_sigma in time_sigma_values:
        print(f"\nCreating dataset with time_sigma = {time_sigma}")

        dataset = SRDataset(
            particle_pos=positions,
            data_dict=data_dict,
            observables=observables,
            world_size=world_size,
            kernel_sigma=4,  # spatial smoothing
            cg_res=1,  # coarse-graining resolution
            deltat=deltat,
            time_sigma=time_sigma,  # temporal smoothing
            cutoff=6,
            irreps=[0],  # scalar fields only
        )

        datasets[time_sigma] = dataset

    return datasets


def create_primes():
    """Create the primes for rho and rho[v_x]"""

    # Create the observable v_x (x-component of velocity)
    v_x = Observable(string="v", rank=1, indices=(LiteralIndex(0),))

    # Create the coarse-grained primitives
    cgp_rho = CoarseGrainedProduct(observables=())  # density rho
    cgp_rho_vx = CoarseGrainedProduct(observables=(v_x,))  # rho[v_x]

    # Create the library primes with no derivatives
    prime_rho = LibraryPrime(
        derivative=DerivativeOrder.blank_derivative(0, 0), derivand=cgp_rho
    )
    prime_rho_vx = LibraryPrime(
        derivative=DerivativeOrder.blank_derivative(0, 0), derivand=cgp_rho_vx
    )

    return prime_rho, prime_rho_vx


def compute_coarse_grained_fields(
    datasets, domain_size=(20, 20, 50), domain_offset=(10, 10, 25)
):
    """Compute coarse-grained fields for both datasets"""

    print("\nComputing coarse-grained fields...")

    # Create the primes for rho and rho[v_x]
    prime_rho, prime_rho_vx = create_primes()

    results = {}

    for time_sigma, dataset in datasets.items():
        print(f"Computing for time_sigma = {time_sigma}")

        # Create a domain to evaluate
        min_corner = [domain_offset[0], domain_offset[1], domain_offset[2]]
        max_corner = [
            min_corner[0] + domain_size[0] - 1,
            min_corner[1] + domain_size[1] - 1,
            min_corner[2] + domain_size[2] - 1,
        ]

        domain = IntegrationDomain(min_corner, max_corner)

        # Evaluate both fields
        field_rho = dataset.eval_prime(prime_rho, domain, experimental=True, order=4)
        field_rho_vx = dataset.eval_prime(
            prime_rho_vx, domain, experimental=True, order=4
        )

        print(f"Field shapes: rho {field_rho.shape}, rho[v_x] {field_rho_vx.shape}")
        print(f"rho range: [{np.min(field_rho):.3f}, {np.max(field_rho):.3f}]")
        print(
            f"rho[v_x] range: [{np.min(field_rho_vx):.3f}, {np.max(field_rho_vx):.3f}]"
        )

        results[time_sigma] = {
            "field_rho": field_rho,
            "field_rho_vx": field_rho_vx,
            "domain": domain,
            "prime_rho": prime_rho,
            "prime_rho_vx": prime_rho_vx,
        }

    return results


def create_comparison_movie(
    results, output_path="discrete/time_smoothing_comparison.mp4", fps=10
):
    """Create a 2x2 movie comparing rho and rho[v_x] fields with and without smoothing"""

    print(f"\nCreating comparison movie: {output_path}")

    # Get the fields
    rho_unsmoothed = results[0]["field_rho"]
    rho_vx_unsmoothed = results[0]["field_rho_vx"]
    rho_smoothed = results[list(results.keys())[1]]["field_rho"]
    rho_vx_smoothed = results[list(results.keys())[1]]["field_rho_vx"]
    time_sigma = list(results.keys())[1]

    n_times = rho_unsmoothed.shape[-1]

    # Set up the figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Find global min/max for consistent color scaling for each field type
    rho_vmin = min(np.min(rho_unsmoothed), np.min(rho_smoothed))
    rho_vmax = max(np.max(rho_unsmoothed), np.max(rho_smoothed))
    rho_vx_vmin = min(np.min(rho_vx_unsmoothed), np.min(rho_vx_smoothed))
    rho_vx_vmax = max(np.max(rho_vx_unsmoothed), np.max(rho_vx_smoothed))

    # Initialize plots
    im1 = axes[0, 0].imshow(
        rho_unsmoothed[:, :, 0].T,
        origin="lower",
        vmin=rho_vmin,
        vmax=rho_vmax,
        cmap="viridis",
    )
    im2 = axes[0, 1].imshow(
        rho_smoothed[:, :, 0].T,
        origin="lower",
        vmin=rho_vmin,
        vmax=rho_vmax,
        cmap="viridis",
    )
    im3 = axes[1, 0].imshow(
        rho_vx_unsmoothed[:, :, 0].T,
        origin="lower",
        vmin=rho_vx_vmin,
        vmax=rho_vx_vmax,
        cmap="RdBu_r",
    )
    im4 = axes[1, 1].imshow(
        rho_vx_smoothed[:, :, 0].T,
        origin="lower",
        vmin=rho_vx_vmin,
        vmax=rho_vx_vmax,
        cmap="RdBu_r",
    )

    # Set titles
    axes[0, 0].set_title("ρ (No Smoothing)")
    axes[0, 1].set_title(f"ρ (Smoothed, σ = {time_sigma})")
    axes[1, 0].set_title("ρ[v_x] (No Smoothing)")
    axes[1, 1].set_title(f"ρ[v_x] (Smoothed, σ = {time_sigma})")

    # Add labels
    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0, :], shrink=0.8)
    cbar1.set_label("ρ (density)")
    cbar2 = plt.colorbar(im3, ax=axes[1, :], shrink=0.8)
    cbar2.set_label("ρ[v_x]")

    # Time text
    time_text = fig.suptitle(f"Time step: 0", fontsize=16)

    def animate(frame):
        im1.set_array(rho_unsmoothed[:, :, frame].T)
        im2.set_array(rho_smoothed[:, :, frame].T)
        im3.set_array(rho_vx_unsmoothed[:, :, frame].T)
        im4.set_array(rho_vx_smoothed[:, :, frame].T)
        time_text.set_text(f"Time step: {frame}")
        return [im1, im2, im3, im4, time_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=n_times, interval=1000 // fps, blit=True
    )

    # Save the movie
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, metadata=dict(artist="SPIDER"), bitrate=1800)
    anim.save(output_path, writer=writer)

    plt.close()
    print(f"Movie saved to {output_path}")


def analyze_smoothing_effect(results):
    """Analyze the quantitative effect of smoothing"""

    print("\nAnalyzing smoothing effect:")

    # Get both field types
    rho_unsmoothed = results[0]["field_rho"]
    rho_vx_unsmoothed = results[0]["field_rho_vx"]
    rho_smoothed = results[list(results.keys())[1]]["field_rho"]
    rho_vx_smoothed = results[list(results.keys())[1]]["field_rho_vx"]
    time_sigma = list(results.keys())[1]

    print(f"=== ρ (density) field ===")
    # Compute temporal derivatives for rho
    dt_rho_unsmoothed = np.diff(rho_unsmoothed, axis=-1)
    dt_rho_smoothed = np.diff(rho_smoothed, axis=-1)

    # RMS of temporal derivatives for rho
    rms_dt_rho_unsmoothed = np.sqrt(np.mean(dt_rho_unsmoothed**2))
    rms_dt_rho_smoothed = np.sqrt(np.mean(dt_rho_smoothed**2))

    print(f"RMS temporal derivative (unsmoothed): {rms_dt_rho_unsmoothed:.6f}")
    print(f"RMS temporal derivative (smoothed): {rms_dt_rho_smoothed:.6f}")
    print(
        f"Smoothing factor: {rms_dt_rho_unsmoothed / rms_dt_rho_smoothed:.2f}x reduction"
    )

    # Mean and std of rho fields
    print(f"Field statistics:")
    print(
        f"Unsmoothed - mean: {np.mean(rho_unsmoothed):.6f}, std: {np.std(rho_unsmoothed):.6f}"
    )
    print(
        f"Smoothed - mean: {np.mean(rho_smoothed):.6f}, std: {np.std(rho_smoothed):.6f}"
    )

    # Correlation between rho fields
    correlation_rho = np.corrcoef(rho_unsmoothed.flatten(), rho_smoothed.flatten())[
        0, 1
    ]
    print(f"Correlation between smoothed and unsmoothed: {correlation_rho:.4f}")

    print(f"\n=== ρ[v_x] field ===")
    # Compute temporal derivatives for rho[v_x]
    dt_rho_vx_unsmoothed = np.diff(rho_vx_unsmoothed, axis=-1)
    dt_rho_vx_smoothed = np.diff(rho_vx_smoothed, axis=-1)

    # RMS of temporal derivatives for rho[v_x]
    rms_dt_rho_vx_unsmoothed = np.sqrt(np.mean(dt_rho_vx_unsmoothed**2))
    rms_dt_rho_vx_smoothed = np.sqrt(np.mean(dt_rho_vx_smoothed**2))

    print(f"RMS temporal derivative (unsmoothed): {rms_dt_rho_vx_unsmoothed:.6f}")
    print(f"RMS temporal derivative (smoothed): {rms_dt_rho_vx_smoothed:.6f}")
    print(
        f"Smoothing factor: {rms_dt_rho_vx_unsmoothed / rms_dt_rho_vx_smoothed:.2f}x reduction"
    )

    # Mean and std of rho[v_x] fields
    print(f"Field statistics:")
    print(
        f"Unsmoothed - mean: {np.mean(rho_vx_unsmoothed):.6f}, std: {np.std(rho_vx_unsmoothed):.6f}"
    )
    print(
        f"Smoothed - mean: {np.mean(rho_vx_smoothed):.6f}, std: {np.std(rho_vx_smoothed):.6f}"
    )

    # Correlation between rho[v_x] fields
    correlation_rho_vx = np.corrcoef(
        rho_vx_unsmoothed.flatten(), rho_vx_smoothed.flatten()
    )[0, 1]
    print(f"Correlation between smoothed and unsmoothed: {correlation_rho_vx:.4f}")


def main():
    """Main test function"""

    print("Testing Time Smoothing in Coarse-Grained Fields")
    print("=" * 50)

    # Load data
    positions, vs, deltat, dims = load_lj_data()

    # Subsample data for faster testing (optional)
    subsample_factor = 10  # Use every 10th time step
    positions = positions[:, :, ::subsample_factor]
    vs = vs[:, :, ::subsample_factor]
    print(f"Subsampled to {positions.shape[2]} time steps")

    # Create datasets with different time smoothing
    time_sigma_values = [0, 10]  # No smoothing vs smoothing with σ=10
    datasets = create_test_datasets(
        positions, vs, deltat * subsample_factor, dims, time_sigma_values
    )

    # Compute coarse-grained fields with larger domain
    results = compute_coarse_grained_fields(datasets, domain_size=(75, 75, 150))

    # Analyze the effect
    analyze_smoothing_effect(results)

    # Create movie
    create_comparison_movie(results, fps=5)

    print("\nTest completed successfully!")
    print("Check the generated movie to see the smoothing effect.")


if __name__ == "__main__":
    main()
