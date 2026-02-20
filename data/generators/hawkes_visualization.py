"""
Visualization tools for Hawkes process analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from data.generators.hawkes import HawkesProcess, PoissonProcess, HawkesParameters


def plot_intensity_trajectory(hawkes: HawkesProcess, T: float, ax=None):
    """Plot the conditional intensity λ(t) over time.

    Args:
        hawkes: HawkesProcess instance with simulated events
        T: Time horizon
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    # Compute intensity at fine grid
    t_grid = np.linspace(0, T, 1000)
    intensities = [hawkes.intensity(t, hawkes.events) for t in t_grid]

    # Plot intensity
    ax.plot(t_grid, intensities, 'b-', linewidth=1, label='Conditional intensity λ(t)')
    ax.axhline(y=hawkes.params.mu, color='r', linestyle='--',
               label=f'Baseline μ={hawkes.params.mu}')
    ax.axhline(y=hawkes.params.long_term_intensity, color='g', linestyle=':',
               label=f'Stationary λ*={hawkes.params.long_term_intensity:.2f}')

    # Mark event times
    for t in hawkes.events:
        if t <= T:
            ax.axvline(x=t, color='gray', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity λ(t)')
    ax.set_title('Hawkes Process: Conditional Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def compare_processes(T: float = 3600, window_size: float = 60, seed: int = 42):
    """Compare Hawkes process with Poisson process side by side.

    Args:
        T: Time horizon in seconds (default: 1 hour)
        window_size: Window size for counting events (default: 1 minute)
        seed: Random seed
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Parameters for Hawkes process (moderate clustering)
    hawkes_params = HawkesParameters(mu=0.05, alpha=0.4, beta=0.8)

    # Poisson process with same long-term intensity
    poisson_intensity = hawkes_params.long_term_intensity

    # Simulate both processes
    hawkes = HawkesProcess(hawkes_params)
    hawkes_events = hawkes.simulate(T, seed=seed)

    poisson = PoissonProcess(intensity=poisson_intensity)
    poisson_events = poisson.simulate(T, seed=seed+1)

    # 1. Event timeline comparison
    ax = axes[0, 0]
    ax.eventplot([hawkes_events], orientation='horizontal', colors=['blue'],
                 alpha=0.7, linelengths=0.8, label='Hawkes')
    ax.eventplot([poisson_events], orientation='horizontal', colors=['red'],
                 alpha=0.7, linelengths=0.4, label='Poisson')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Event Timeline Comparison\n(blue=Hawkes, red=Poisson)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Event counts per window
    ax = axes[0, 1]
    hawkes_counts = hawkes.get_event_counts(window_size, T)
    poisson_counts = poisson.get_event_counts(window_size, T)

    time_windows = np.arange(len(hawkes_counts)) * window_size / 60  # Convert to minutes

    ax.plot(time_windows, hawkes_counts, 'b-o', label='Hawkes', markersize=4)
    ax.plot(time_windows, poisson_counts, 'r-s', label='Poisson', markersize=4, alpha=0.7)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(f'Event Count (per {window_size}s window)')
    ax.set_title('Event Counts Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Inter-event time distribution
    ax = axes[1, 0]
    hawkes_iet = hawkes.get_inter_event_times()
    poisson_iet = poisson.get_inter_event_times()

    bins = np.linspace(0, min(100, np.percentile(hawkes_iet, 95)), 30)
    ax.hist(hawkes_iet, bins=bins, alpha=0.5, label='Hawkes', color='blue', density=True)
    ax.hist(poisson_iet, bins=bins, alpha=0.5, label='Poisson', color='red', density=True)
    ax.set_xlabel('Inter-Event Time (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('Inter-Event Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Intensity trajectory for Hawkes
    ax = axes[1, 1]
    plot_intensity_trajectory(hawkes, T, ax=ax)

    plt.tight_layout()
    return fig, {'hawkes': hawkes, 'poisson': poisson}


def demonstrate_clustering():
    """Demonstrate the clustering effect with different alpha values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    T = 600  # 10 minutes
    mu = 0.05
    beta = 1.0
    alphas = [0.0, 0.3, 0.6]  # Low, medium, high clustering

    for idx, alpha in enumerate(alphas):
        # Event timelines
        ax = axes[0, idx]

        if alpha == 0:
            # Pure Poisson
            process = PoissonProcess(intensity=mu)
            title = f'Poisson (no clustering)\nμ={mu}'
        else:
            params = HawkesParameters(mu=mu, alpha=alpha, beta=beta)
            process = HawkesProcess(params)
            title = f'Hawkes α={alpha}, β={beta}\nBranching ratio={alpha/beta:.2f}'

        events = process.simulate(T, seed=42+idx)

        # Plot events as dots
        ax.scatter(events, np.ones(len(events)), marker='|', s=100, alpha=0.7)
        ax.set_xlim(0, T)
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_title(title)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')

        # Event counts in windows
        ax = axes[1, idx]
        window_size = 10  # 10-second windows
        counts = process.get_event_counts(window_size, T)

        time_windows = np.arange(len(counts)) * window_size
        ax.bar(time_windows, counts, width=window_size*0.8, alpha=0.7)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Event Count')
        ax.set_title(f'Events per {window_size}s window\nTotal: {len(events)} events')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Effect of Clustering Parameter α on Trade Arrivals', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def analyze_real_trades(trade_times: List[float], window_seconds: float = 60):
    """Analyze real trade data to check for Hawkes-like clustering.

    Args:
        trade_times: List of trade timestamps (in seconds)
        window_seconds: Window size for analysis

    Returns:
        Dictionary with analysis results
    """
    if len(trade_times) < 10:
        raise ValueError("Need at least 10 trades for analysis")

    trade_times = np.sort(trade_times)
    T = trade_times[-1] - trade_times[0]

    # Compute statistics
    iet = np.diff(trade_times)

    stats = {
        'n_trades': len(trade_times),
        'duration_seconds': T,
        'mean_inter_event': np.mean(iet),
        'std_inter_event': np.std(iet),
        'cv': np.std(iet) / np.mean(iet),  # Coefficient of variation
        'min_inter_event': np.min(iet),
        'max_inter_event': np.max(iet),
    }

    # Poisson has CV = 1, Hawkes has CV > 1 (clustering)
    if stats['cv'] > 1.2:
        stats['clustering_assessment'] = 'Strong clustering (Hawkes-like)'
    elif stats['cv'] > 1.05:
        stats['clustering_assessment'] = 'Moderate clustering'
    else:
        stats['clustering_assessment'] = 'Little clustering (Poisson-like)'

    return stats


if __name__ == '__main__':
    # Run demonstration
    print("Generating Hawkes process visualization...")

    # Main comparison
    fig1, _ = compare_processes(T=3600, window_size=60)
    plt.savefig('hawkes_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: hawkes_comparison.png")

    # Clustering demonstration
    fig2 = demonstrate_clustering()
    plt.savefig('hawkes_clustering.png', dpi=150, bbox_inches='tight')
    print("Saved: hawkes_clustering.png")

    plt.show()

    print("\nHawkes Process Summary:")
    print("- Blue: Hawkes process (with clustering)")
    print("- Red: Poisson process (no clustering)")
    print("\nKey observation: Hawkes shows 'bursts' of activity")
