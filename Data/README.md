This file provides a brief explanation of how the simulation runs.
There is also important explanations of how the data is saved and stored, and what to be careful of when analyzing the data.

LATTICE CONSTRUCTION:

The simulations generate random-walk trajectories on a diluted lattice. The lattice can be one of:
SC (simple cubic)
BCC (body-centered cubic)
FCC (face-centered cubic)
Diamond
A lattice size is first generated, and each site is independently occupied with probability prob. For example, prob = 0.5 means that 50% of lattice sites are removed (site dilution).

HOPPING MATRIX:

Once the lattice is constructed, a hopping matrix W is computed which encodes the hopping rates between sites.
If nearest_neighbor = True, the walker can hop only to nearest neighbors, where the nearest-neighbor list is defined from the fully occupied lattice (even if the lattice is diluted).
If nearest_neighbor = False, long-range hopping is allowed with rates proportional to:
W_ij = 1 / (r_ij)**alpha, where r_ij is the distance between sites i and j and alpha is a free parameter. 
The diagonal elements satisfy:
W_ii = -sum_{j not equal i} W_ij
for probability conservation (basically the on-diagonal elements have to be negative to take into account leaving a site). 
Periodic boundary conditions are always used throughout.

GILLESPIE ALGORITHM RANDOM WALK:

A single Gillespie trajectory proceeds as follows:
The walker starts on a random occupied site.
The total escape rate from that site is computed as
R = -W_ii.
A waiting time tau is drawn from an exponential distribution
P(tau) = R * exp(-R * tau).
Time is advanced by tau. 
The walker hops to a new site, selected with probability proportional to its hopping rate.
Statistics are collected for this trajectory until the cumulative time exceeds t_max (a free parameter).
For diluted lattices, this process is repeated over many lattice realizations, because each diluted configuration has its own structure and hopping matrix. 
For fully occupied lattices (prob = 1), only one configuration is needed.

SAVED DATA FORMAT:

Each simulation produces an .npz file via:
np.savez(
    savefile,
    results=results,
    size=size,
    a=a,
    lattice_type=lattice_type,
    diluted=diluted,
    prob=prob,
    nearest_neighbor=nearest_neighbor,
    A=A,
    alpha=alpha,
    t_max=t_max,
    n_trajectories=n_trajectories,
    n_configs=n_configs,
)
The variables size, prob, alpha, etc. are simply parameters used in the run.
The variable results contains all trajectory data.

STRUCTURE OF RESULTS:

Results is a flat list of dictionaries.
Each entry corresponds to one trajectory (one walker on one lattice configuration):
traj = {
    'wait_times': [...],       # waiting times between hops
    'jump_lengths': [...],     # distance of each hop (with periodic boundary conditions)
    'times': [...],            # cumulative times of each hop
    'sq_disp': [...],          # squared displacement vs. step
    'distinct_sites': [...],   # number of unique sites visited vs. step
    'sites': [...],            # site indices of the walk
}
All trajectories are stored sequentially:
For fully occupied (prob = 1) datasets:
n_configs = 1 and all trajectories belong to the same lattice.
For diluted datasets (prob < 1): 
If there are 10 configs × 1000 trajectories,
trajectories 0–999 → configuration 1
trajectories 1000–1999 → configuration 2
… and so on.
Most data files should have around 10,000 total trajectories. 
If configurational averaging was done, most data files should have 10 configurations with 1,000 trajectories each.

IMPORTANT CAVEATS WHEN AVERAGING:

1. Single trajectories are noisy
Just looking at one trajectory may not be enough to see a meaningful trend.
Choosing how to batch the data in terms of how many to average at a time, etc. might be important.

2. Trajectories have different numbers of steps
The Gillespie algorithm produces irregular time grids, and the total number of hops/time points will vary between trajectories.
Therefore, you cannot take a naive average of sq_disp or times.

3. Long-time averages can be dominated by a few trajectories
If only a small number of walkers survive past a given time, then averaging everything can produce extremely noisy or misleading curves for long-time points.

To address these points, the following helper functions are provided in the notebook:
pad() — pads trajectories with np.nan
contribution_threshold() — determines the last time/step where ≥ X% of trajectories contribute
average_trajectories() — performs the full averaging in a safe and consistent way
These should ensure that the mean squared displacement (MSD) vs. time/step is reliable and comparable across datasets.

SUGGESTED QUANTITIES FOR ML:

mean squared displacement:
vs. step number
vs. time 
Both can be computed using the provided averaging utilities.

Other possible features:
distribution of jump lengths
distribution of waiting times
distinct sites visited vs step (distinct_sites_vs_steps)
