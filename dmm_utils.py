import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
import numpy as np
from scipy.stats import linregress
import torch
import torch.multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange
from cluster_finding import find_cluster_graph


matplotlib.use('Agg')  # for running on server without GUI
torch.set_default_tensor_type(torch.cuda.FloatTensor
                              if torch.cuda.is_available()
                              else torch.FloatTensor)
mp.set_start_method('spawn', force=True)


#Runs a particular DMM after it has been initialized
@torch.no_grad()
def run_dmm(dmm, max_steps, simple, save_steps=0, transient=10, break_threshold=0.5):
    is_solved = torch.zeros(dmm.batch, dtype=torch.bool)
    is_solved_last = torch.zeros_like(is_solved)
    solved_step = torch.zeros(dmm.batch, dtype=torch.int64)
    unsat_moments = torch.zeros(dmm.batch, 4, dtype=torch.get_default_dtype())
    # is_transient = torch.ones(dmm.batch, dtype=torch.bool)
    # is_transient_last = torch.zeros_like(is_transient)
    # transient_end = torch.zeros(dmm.batch, dtype=torch.int64)
    # unsat_traj = []
    v_last = None
    # spin_flips = []
    if not simple: #collects additional trajectories when simple = False
        v_traj = []
        xl_traj = []
        xs_traj = []
        C_traj = []
        G_traj = []
        R_traj = []
        dt_traj = []
    spin_traj = []
    time_traj = []
    integration_time = torch.zeros(dmm.batch, dtype=torch.get_default_dtype())

    # transient_threshold = 1.5 * dmm.n_var ** 0.67
    for step in range(max_steps): #simulates the DMM for an additional time step of duration dt
        if simple:
            unsat_clauses, dt = dmm.compiled_step()
        else:
            unsat_clauses, v, xl, xs, dt, C, G, R = dmm.compiled_step()
        integration_time += dt
        if not simple:
            dt_traj.append(dt)
        # is_transient[unsat_clauses < transient_threshold] = False
        # transient_end[is_transient ^ is_transient_last] = step
        # is_transient_last = is_transient.clone()
        is_solved_i = unsat_clauses == 0
        is_solved = is_solved | is_solved_i
        solved_step[is_solved ^ is_solved_last] = step
        is_solved_last = is_solved.clone()
        n_solved = torch.sum(is_solved).detach().cpu().numpy()
        # n_solved_minibatch = torch.sum(is_solved.view(dmm.param_batch, dmm.minibatch), dim=1)
        if step % 10 == 0:
            print(f'N = {dmm.n_var} M = {dmm.n_clause} Step {step} Solved {n_solved} / {dmm.batch}')
        if step >= transient:
            unsat_moments += ((unsat_clauses * ~is_solved).to(torch.torch.get_default_dtype()).unsqueeze(1))\
                             ** torch.arange(1, 5, dtype=torch.torch.get_default_dtype())
            # unsat_traj.append(unsat_clauses)
            if step < save_steps + transient:
                spin_traj.append(dmm.v.data > 0)
                time_traj.append(integration_time.clone())
                if not simple:
                    v_traj.append(dmm.v.data.clone())
                    xl_traj.append(xl)
                    xs_traj.append(xs)
                    C_traj.append(C)
                    G_traj.append(G)
                    R_traj.append(R)
        if n_solved >= break_threshold * dmm.batch:
        # if (n_solved_minibatch > break_threshold * dmm.minibatch).all():
            break

    #Extracts and writes the CO problem solutions for all instances they are found in a batch
    '''#Can be verified by comparison with the .cnf files produced by dataset.py
    os.makedirs(f'results/{dmm.prob_type}/Solutions', exist_ok=True)
    solutions = ((dmm.v.data > 0).int()).tolist()
    solutions = ["".join(map(str, solutions[i])) for i in range(len(solutions))]
    for i in range(len(solutions)):
        if is_solved[i]:
            print(f'Instance {i:04d} solution: {solutions[i]}')
            solution_file = open(f'results/{dmm.prob_type}/Solutions/n{dmm.n_var}_solution_{i:04d}.txt', 'w')
            solution_file.write(f'{solutions[i]}')
            solution_file.close()'''

    # unsat_traj = torch.stack(unsat_traj, dim=-1)
    if len(spin_traj) > 0:
        spin_traj = torch.stack(spin_traj, dim=-1)  # (batch, n, length)
        time_traj = torch.stack(time_traj, dim=-1)  # (batch, length)
    else:
        spin_traj = torch.zeros(dmm.batch, dmm.n_var, 1)
        time_traj = torch.zeros(dmm.batch, 1)
    time_traj *= dmm.lr
    if simple:
        return is_solved, solved_step, unsat_moments, spin_traj, time_traj, step
    else:
        return is_solved, solved_step, unsat_moments, spin_traj, time_traj, v_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj, dt_traj, step


#Runs the cluster finding function to find avalanches among a given set of logical DOF trajectories
@torch.no_grad()
def avalanche_analysis(spin_traj, time_traj, edges, time_window):
    batch = spin_traj.shape[0]
    cluster_sizes = []
    memory_flag = False
    time_stamps = time_traj // time_window
    mask = torch.diff(time_stamps, dim=1) > 0
    mask = torch.cat([torch.ones(batch, 1, dtype=torch.bool), mask], dim=1)

    try:
        for i in trange(batch):
            spins_i = spin_traj[i].t()[mask[i]].t()
            spin_flips_i = torch.diff(spins_i, dim=1)
            label, cluster_size_i = find_cluster_graph(spin_flips_i, edges[i])
            cluster_sizes.append(cluster_size_i)
    except RecursionError:
        warnings.warn('Maximum recursion depth exceeded.')
        return np.zeros(1, dtype=np.float32), torch.zeros(1), memory_flag
    except torch.cuda.OutOfMemoryError:
        warnings.warn('Out of memory.')
        memory_flag = True
        return np.zeros(1, dtype=np.float32), torch.zeros(1), memory_flag
    if len(cluster_sizes) == 0:
        return np.zeros(1, dtype=np.float32), torch.zeros(1), memory_flag
    else:
        cluster_sizes = torch.cat(cluster_sizes)
        cluster_sizes = cluster_sizes[cluster_sizes > 0].cpu().numpy()
        #For avalanche analysis with anti-instanton extraction (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
        return cluster_sizes, label, memory_flag
        #For standard avalanche analysis (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
        '''return cluster_sizes, memory_flag'''


#Extracts avalanches from DMM data trajectories
#Can additionally extract the number of anti-instantons during a DMM's simulation (must be toggled off/on manually)
def avalanche_analysis_mp(spin_traj, time_traj, edges, pool, minibatch, subprocesses, time_window):
    spin_traj_minibatch = [spin_traj[i * minibatch:(i + 1) * minibatch] for i in range(subprocesses)]
    time_traj_minibatch = [time_traj[i * minibatch:(i + 1) * minibatch] for i in range(subprocesses)]
    edges_minibatch = [edges[i * minibatch:(i + 1) * minibatch] for i in range(subprocesses)]
    p_avalanches = [pool.apply_async(avalanche_analysis, args=(spin_traj_minibatch[process_id],
                                                               time_traj_minibatch[process_id],
                                                               edges_minibatch[process_id], time_window))
                    for process_id in range(subprocesses)]

    cluster_sizes = []
    out_of_memory_flag = False
    #For avalanche analysis with anti-instanton extraction (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
    anti_instantons = 0 #keeps track of number of (temporally) adjascent, identical avalanches (i.e. consisting of the same set of flipping voltages)
    for i, p_avalanche_i in enumerate(p_avalanches):
        cluster_size, label, memory_flag = p_avalanche_i.get()
        cluster_sizes.append(cluster_size)
        set_of_flips_last = []
        for j in range(int(torch.max(label))): #j specifies a particular avalanche
            set_of_flips_j = []
            for n in range(len(label)): #n \in [0, ..., N] specifies a voltage
                if j in label[n]:
                    set_of_flips_j.append(n)
            if set_of_flips_j == set_of_flips_last:
                anti_instantons += 1
            set_of_flips_last = set_of_flips_j
        if memory_flag:
            out_of_memory_flag = True
    cluster_sizes = np.concatenate(cluster_sizes)
    return cluster_sizes, anti_instantons, out_of_memory_flag
    #For standard avalanche analysis (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTION IN benchmark.py)
    '''for p_avalanche_i in p_avalanches:
        cluster_size, memory_flag = p_avalanche_i.get()
        cluster_sizes.append(cluster_size)
        if memory_flag:
            out_of_memory_flag = True
    cluster_sizes = np.concatenate(cluster_sizes)
    return cluster_sizes, out_of_memory_flag'''


#Plots avalanche size distributions for a particular collection of instances
def avalanche_size_distribution(cluster_sizes, name):
    # avalanche_data = []
    # cluster_sizes = np.concatenate(cluster_sizes, axis=0)
    cluster_sizes = cluster_sizes[cluster_sizes > 0]
    if len(cluster_sizes) < 2:
        # avalanche_data.append([0, 0, 0, 0, 0, 0, 0])
        avalanche_data = np.zeros(4)
    else:
        log_cluster_size = np.log10(cluster_sizes)
        log_cluster_size = log_cluster_size[log_cluster_size >= 0]
        # quartiles = np.percentile(log_cluster_size, [25, 50, 75, 100])
        # IQR = quartiles[2] - quartiles[0]
        # bin_width = 2 * IQR * (len(log_cluster_size) ** (-1 / 3))
        mean = np.mean(log_cluster_size)
        std = np.std(log_cluster_size)
        max_size = np.max(log_cluster_size)
        bin_width = 3.5 * std / (len(log_cluster_size) ** (1 / 3))
        bin_width = max(bin_width, 0.02)
        n_bins = int(6 / bin_width)  # assuming all avalanches smaller than 10^6
        n_bins = max(n_bins, 1)
        bins = bin_width * np.arange(n_bins + 1)
        hist, bin_edges = np.histogram(log_cluster_size, bins=bins)
        bin_edges_linear = 10 ** bin_edges
        bin_sizes_linear = np.diff(bin_edges_linear)
        hist = hist / bin_sizes_linear
        hist = hist / hist.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = bin_centers[hist > 0]
        hist = hist[hist > 0]
        try:
            slope, intercept, r, p, se = linregress(bin_centers[1:-8], np.log10(hist)[1:-8])
        except:
            slope, intercept, r, p, se = 0.0, 0.0, 0.0, None, None
        # avalanche_data.append([slope, intercept, r] + quartiles.tolist())
        avalanche_data = np.array([slope, intercept, r, max_size])

        fig, ax = plt.subplots(figsize=(3.0, 1.75))
        ax.scatter(10**(bin_centers), hist, s=20, color='blue')
        # try:
        #     ax.plot(10**(bin_centers), 10**(slope * bin_centers + intercept), 'r--', label=f'{slope:.2f}x+{intercept:.2f} r={r:.2f}', color='red', linestyle='--')
        # except:
        #     pass
        #plt.legend(fontsize=16, loc= 'upper right')
        ax.set_xscale('log')
        ax.set_xlim(0.9, 110)
        ax.set_yscale('log')
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.set_xlabel(r'Cluster Size $s$')
        ax.set_ylabel(r'$P(s)$')
        plt.savefig(f'{name}_{slope:.2f}_{intercept:.2f}_{r:.2f}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    return avalanche_data
