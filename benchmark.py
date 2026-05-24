import os
import numpy as np
import json
import torch
import torch.multiprocessing as mp
from scipy.optimize import curve_fit
from model import DMM
from dmm_utils import run_dmm, avalanche_analysis_mp, avalanche_size_distribution
mp.set_start_method('spawn', force=True)
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go


#############################################################################################################################################################################################
#FREE PARAMETERS DURING SIMULATION
eqn_choice = 'sean_choice' #specifies DMM equations; can ONLY take on the values 'sean_choice', 'diventra_choice', 'yuanhang_choice', and 'zeta_zero' (and 'R_zero', 'rudy_choice', or 'rudy_simple' for XORSAT)
prob_type = '3SAT' #specficies type of CO problem to solve; prob_type can ONLY take on the values '3SAT', '3R3X', OR '5R5X'
simple = False #specifies whether or not data trajectories are plotted (simple = True -> no trajectory plotting)
# batch = 100
batch = 5 #number of instances in a batch
# num_iterations = 2
num_iterations = 1 #number of batches to run
# max_step = 1e6
max_step = 200 #maximum simulation step
tag = '_testing' #tag to add to file names
#big_ns = np.array([[10, 20, 30], [40, 50, 60]]) #array of system sizes N to simulate; include all sizes in which parameters were tuned simultaneously in the same sublist
big_ns = np.array([[10]])
total_param_list = [[1.0, 1.0, 0.02]] #[[0.08, 0.08, 0.35], [0.08, 0.5, 0.35], [0.08, 1.0, 0.35], [0.08, 3.0, 0.6], [0.08, 20.0, 1.0],
                    #[0.5, 0.08, 0.035], [0.5, 0.5, 0.035], [0.5, 1.0, 0.035], [0.5, 3.0, 0.06], [0.5, 20.0, 0.35],
                    #[1.0, 0.08, 0.002], [1.0, 0.5, 0.0035], [1.0, 1.0, 0.02], [1.0, 3.0, 0.035], [1.0, 20.0, 0.2],
                    #[3.0, 0.08, 0.001], [3.0, 0.5, 0.006], [3.0, 1.0, 0.006], [3.0, 3.0, 0.01], [3.0, 20.0, 0.1],
                    #[20.0, 0.08, 0.0006], [20.0, 0.5, 0.001], [20.0, 1.0, 0.002], [20.0, 3.0, 0.006], [20.0, 20.0, 0.035]] #list of parameters to simulate at each system size ([a, b, c], where \beta = a*\beta_{opt}, \zeta = b*\zeta_{opt}, time_window = c)
plotting_transient = 0 #number of initial integration steps which are not plotted
n_plot = 5 #number of instances (out of batch) for which detailed trajectory plots and animations are generated
interactive = True #specifies whether or not interactive (plotly) trajectories are plotted
animation = True #specifies whether or not animations of 3SAT circuit representations are generated
#############################################################################################################################################################################################

from dataset import import_data
from animation_utils import create_3sat_circuit_animation


#A collection of useful functions for fitting to data
def power_law_decay(x, a, b, c):
    return a * x**(-b) + c

def inverse_gaussian(x, mu, lamb):
    return np.sqrt(lamb/(2*math.pi*x**3)) * np.exp(-(lamb*(x-mu)**2)/(2*mu**2*x))

def exponential_decay(x, b):
    return b*np.exp(-b*x)

def log_normal(x, mu, sigma):
    return (1/(x*sigma*np.sqrt(2*math.pi))) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))


#Plots time-to-solution distributions given TTS data on a large set of SAT instances
def tts_distribution(solved_step, prob_type, flattened_big_ns, n, break_t, name):

    plt.figure(figsize=(3.0, 1.75))
    
    #Extracts total number of instances solved
    with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n_solved_{n}_{name}.txt', 'r') as f:
        n_solved = f.readlines()
    total_solved = sum([int(element.strip()) for element in n_solved]) #the total number of solved instances
    solved_step = np.sort(solved_step)

    #Plots all data with logarithmic axes
    '''log_bins = np.logspace(np.log10(min(solved_step)), np.log10(max(solved_step)), 100)
    prob, bins, patches = plt.hist(solved_step, bins=log_bins, density=True, color='blue')'''
    #Fits with logarithmic axes
    '''bin_centers = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    popt, pcov = curve_fit(choose_a_distribution, prob, bin_centers)
    plt.plot(bin_centers, choose_a_distribution(bin_centers, popt[0], popt[1], popt[2]), color='red', linestyle='dashed')'''

    #Plots all data with linear axes
    prob, bins, patches = plt.hist(solved_step, bins=100, density=True, color='blue')
    #Fits with linear axes
    bin_centers = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    #popt, pcov = curve_fit(CHOOSE_A_DISTRIBUTION, prob, bin_centers)
    #plt.plot(bin_centers, CHOOSE_A_DISTRIBUTION(bin_centers, popt[0], popt[1], popt[2]), color='red', linestyle='dashed')
    #Plots TTS IG fits, as in Fig. 4 of manuscript
    '''#plt.plot(bin_centers, inverse_gaussian(bin_centers, 10000, 900), color='red', linestyle='dashed') #[0.08, 20.0]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 1000, 500), color='red', linestyle='dashed') #[3.0, 1.0]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 1150, 4400), color='red', linestyle='dashed') #[20.0, 0.08]
    #plt.plot(bin_centers, inverse_gaussian(bin_centers, 20000, 1000), color='red', linestyle='dashed') #[20.0, 20.0]'''

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'Solution Step $T$')
    plt.ylabel(r'$P(T)$')
    plt.legend(fontsize='10')
    plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()


#Plots trajectories of a given array of data during a DMM's evolution
def trajectory_plotter(traj, time_traj_to_plot, i, j, num_of_trajs, plotting_transient, y_label, file_label, zoom_factor, GR=False):
    for k in range(num_of_trajs):
        if GR == True:
            for l in range(3):
                traj_to_plot = [element[j][k][l] for element in traj[i]]
                plt.plot(time_traj_to_plot, traj_to_plot)
        else:
            traj_to_plot = [element[j][k] for element in traj[i]]
            plt.plot(time_traj_to_plot, traj_to_plot)
    plt.xlabel('Time')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f'{file_label}.png')
    plt.xlim(left = time_traj_to_plot[plotting_transient], right=time_traj_to_plot[plotting_transient+int(len(time_traj_to_plot)/int(zoom_factor))])
    plt.savefig(f'{file_label}_zoomed.png')
    plt.clf()


#Plots interactive trajectories of a given array of data using Plotly
def plotly_trajectory_plotter(traj, time_traj_to_plot, i, j, num_of_trajs, plotting_transient, y_label, file_label, zoom_factor, GR=False):
    fig = go.Figure()
    time_traj_to_plot = np.array(time_traj_to_plot)
    
    for k in range(num_of_trajs):
        if GR == True:
            for l in range(3):
                traj_to_plot = [element[j][k][l] for element in traj[i]]
                fig.add_trace(go.Scatter(x=time_traj_to_plot, y=traj_to_plot, mode='lines', 
                                         name=f'Traj {k} Component {l}'))
        else:
            traj_to_plot = [element[j][k] for element in traj[i]]
            fig.add_trace(go.Scatter(x=time_traj_to_plot, y=traj_to_plot, mode='lines', 
                                     name=f'Traj {k}'))
            
    fig.update_layout(
        title=f"{y_label} Trajectories",
        xaxis_title="Time",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Apply initial transient crop 
    fig.update_xaxes(range=[time_traj_to_plot[plotting_transient], time_traj_to_plot[-1]])
    
    fig.write_html(f'{file_label}.html')


#Main function that initializes/simulates a DMM
#Also extracts supplementary data if desired (e.g., avalanche size distributions, number of anti-instantons, number of solved instances per batch, and TTS distributions)
def param_scaling(param, name, eqn_choice, prob_type, batch, ns, simple, flattened_big_ns, last_iteration, time_window, max_step):
    avalanche_subprocesses = 5
    avalanche_minibatch = int(np.ceil(batch / avalanche_subprocesses))
    pool = mp.Pool(avalanche_subprocesses)

    #Saves parameter value for each batch
    with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/params_{ns}_{name}.json', 'w') as f:
        json.dump(param, f)

    #Initializes lists to extract data trajectories (if simple=False)
    spin_traj = []
    time_traj = []
    v_traj = []
    xl_traj = []
    xs_traj = []
    C_traj = []
    G_traj = []
    R_traj = []
    dt_traj = []
    for n in ns:
        files = []
        for instance_num in range(batch):
            if prob_type == '3SAT':
                file = f'data/p0_080/ratio_4_30/var_{n}/instances/transformed_barthel_n_{n}_r_4.300_p0_0.080_instance_{instance_num+1:03d}.cnf'
            elif prob_type == '3R3X':
                file = f'data/XORSAT/3R3X/{n}/problem_{instance_num:04d}.cnf' #f'../DMM_param_tuning-main/data/XORSAT/3R3X/{n}/problem_{instance_num:04d}_XORgates.cnf'
            elif prob_type == '5R5X':
                file = f'data/XORSAT/5R5X/{n}/problem_{instance_num:04d}.cnf' #f'../DMM_param_tuning-main/data/XORSAT/5R5X/{n}/problem_{instance_num:04d}_XORgates.cnf'
            files.append(file)
        dmm = DMM(files, simple, batch=batch, param=param, eqn_choice=eqn_choice)
        save_steps = 6000 #total number of steps where 
        #save_steps = max_step #total number of steps where 
        transient = 0 #number of steps
        #break_t = 0.5 #proportion of instances which must be solved before a batch is solved
        break_t = 1.0 #proportion of instances which must be solved before a batch is solved
        if simple:
            is_solved, solved_step, unsat_moments, spin_traj_n, time_traj_n, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=break_t)
        else:
            is_solved, solved_step, unsat_moments, spin_traj_n, time_traj_n, v_traj_n, xl_traj_n, xs_traj_n, C_traj_n, G_traj_n, R_traj_n, dt_traj_n, step = \
                run_dmm(dmm, max_step, simple, save_steps, transient, break_threshold=break_t)
            #Extracts data trajectories
            spin_traj.append(spin_traj_n)
            time_traj.append(time_traj_n)
            v_traj.append(v_traj_n)
            xl_traj.append(xl_traj_n)
            xs_traj.append(xs_traj_n)
            C_traj.append(C_traj_n)
            G_traj.append(G_traj_n)
            R_traj.append(R_traj_n)
            dt_traj.append(dt_traj_n)
        solved_step[~is_solved] = max_step
        n_solved = is_solved.sum() #number of solved instances in a batch
        median_step = np.median(solved_step.cpu().numpy()) if n_solved > dmm.batch // 2 \
            else step * (dmm.batch + 1) / (2 * n_solved + 1)
        #Extracts avalanche size distributions
        #For avalanche analysis with anti-instanton extraction (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTIONS IN avalanche_analysis() AND avalanche_analysis_mp() IN dmm_utils.py)
        cluster_size, anti_instantons, out_of_memory_flag = avalanche_analysis_mp(spin_traj_n, time_traj_n, dmm.edges_var, pool,
                                                                 avalanche_minibatch, avalanche_subprocesses, time_window)
        #For standard avalanche extraction (CHOOSE ONLY ONE, SELECT CORRESPONDING OPTIONS IN avalanche_analysis() AND avalanche_analysis_mp() IN dmm_utils.py)
        #cluster_size, out_of_memory_flag = avalanche_analysis_mp(spin_traj_n, time_traj_n, dmm.edges_var, pool,
        #                                                         avalanche_minibatch, avalanche_subprocesses, time_window)
        with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/cluster_sizes_{n}_{name}_{time_window}.txt', 'a') as f:
            for cluster in cluster_size:
                f.write(f'{cluster}\n')
        #Extract number of anti_instantons
        with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/anti_instantons_{n}_{name}.txt', 'a') as f:
            f.write(f'{anti_instantons}\n')
        #Extracts TTS distributions
        with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}.txt', 'a') as f:
            for tts in solved_step:
                if tts < max_step:
                    f.write(f'{tts}\n')
        #Keeps track of n_solved
        with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n_solved_{n}_{name}.txt', 'a') as f:
            f.write(f'{n_solved}\n')
        if last_iteration == True:
            print('Last Iteration!')
            #Plots avalanche size distributions
            with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/cluster_sizes_{n}_{name}_{time_window}.txt', 'r') as f:
                cluster_size = np.array([float(element.strip()) for element in f.readlines()])
            avalanche_stats = avalanche_size_distribution(cluster_size, f'results/{prob_type}/Benchmark/{flattened_big_ns}/{name}_{n}_{time_window}')
            #Plots TTS distributions
            with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/tts_{n}_{name}.txt', 'r') as f:
                solved_step = np.array([float(element.strip()) for element in f.readlines()])
            tts_distribution(solved_step, prob_type, flattened_big_ns, n, break_t, name + '')
        
        #Writes TTS (to solve a median of instances in a batch)
        with open(f'results/{prob_type}/Benchmark/{flattened_big_ns}/steps_{name}.txt', 'a') as f:
            f.write(f'{n} {median_step}\n')
        print(f'N = {n} Done')

    if not simple:
        return spin_traj, time_traj, v_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj, dt_traj


if __name__ == '__main__':
    __spec__ = None
    mp.set_start_method('spawn', force=True)

    if animation:
        if simple:
            print("WARNING: 'animation = True' requires 'simple = False' to collect trajectories. Overriding simple = False.")
            simple = False

    flattened_big_ns = str(big_ns.flatten().tolist()) + tag
    result_dir = f'results/{prob_type}/Benchmark/{flattened_big_ns}'
    os.makedirs(result_dir, exist_ok=True)

    for param_index in range(len(total_param_list)): #iterates over set of parameters
        last_iteration = False
        for iter in range(num_iterations): #performs simulation num_iterations times
            if iter == num_iterations-1:
                last_iteration = True
            for group, ns in enumerate(big_ns):
                if group != 0:
                    with open(f'{result_dir}/steps_{param_index}.txt', 'r') as f:
                        last_tts = f.readlines()[-1].split()[-1]
                    if float(last_tts) >= max_step:
                        break
                try:
                    #Attempts to extract optimal parameters using previous optimization
                    with open(f'parameters/{prob_type}/{flattened_big_ns}/optimal_param_{ns}.json', 'r') as f:
                        all_params = json.load(f)
                    param_i = {'alpha_by_beta': all_params['alpha_by_beta'],
                            'beta': total_param_list[param_index][0]*all_params['beta'],
                            'gamma': all_params['gamma'],
                            'delta_by_gamma': all_params['delta_by_gamma'],
                            'zeta': total_param_list[param_index][1]*all_params['zeta'],
                            'dt_0': all_params['dt_0'], #may need to be adapted iteratively to control numerical errors
                            'time_window': total_param_list[param_index][2], #needs to be varied depending on parameter choices to yield correct avalanche distributions
                            'lr': 1.0,
                            'alpha_inc': 0}
                except:
                    #Initializes default (previously found to be "optimal") parameters if extraction fails (i.e, if no tuning was performing)
                    print(f'No new parameters for n = {ns}')
                    param_i = {'alpha_by_beta': 0.45313481433413916,
                        'beta': 78.83050800202636,
                        'gamma': 0.3635604327568345,
                        'delta_by_gamma': 0.21883211263830715,
                        'zeta': 0.06294441488786634,
                        'dt_0': 1.0,
                        'time_window': 0.5,
                        'lr': 1.0,
                        'alpha_inc': 0}

                if simple:
                    param_scaling(param_i, str(param_index), eqn_choice, prob_type, batch, ns, simple, flattened_big_ns, last_iteration, param_i['time_window'], max_step)
                else:
                    spin_traj, time_traj, v_traj, xl_traj, xs_traj, C_traj, G_traj, R_traj, dt_traj = param_scaling(param_i, str(param_index), eqn_choice, prob_type, batch, ns, simple, flattened_big_ns, last_iteration, param_i['time_window'], max_step)
                   
                    #Calculates number of satisfied clauses
                    num_sat_clauses_traj = []
                    for C_n in C_traj:
                        num_sat_n = [ (C_step < 0.5).sum(dim=-1, keepdim=True).cpu().numpy() for C_step in C_n ]
                        num_sat_clauses_traj.append(num_sat_n)

                    if last_iteration == True:
                        for i in range(len(ns)): #iterates over variable number, could be up to i in range(len(ns))

                            for j in range(min(n_plot, time_traj[i].shape[0])): #iterates over batch, could be up to j in range(batch)
                                time_traj_to_plot = time_traj[i][j]
                                file_label = f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}'

                                #Plots data trajectories
                                trajectory_plotter(v_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Voltages', f'{file_label}_v_{param_index}', 10)
                                trajectory_plotter(xs_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Short Term Memories', f'{file_label}_xs_{param_index}', 10)
                                trajectory_plotter(xl_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Long Term Memories', f'{file_label}_xl_{param_index}', 10)
                                trajectory_plotter(C_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Clause Functions', f'{file_label}_C_{param_index}', 10)
                                trajectory_plotter(G_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Gradient Terms', f'{file_label}_G_{param_index}', 10, GR=True)
                                trajectory_plotter(R_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Rigidity Terms', f'{file_label}_R_{param_index}', 10, GR=True)
                                trajectory_plotter(num_sat_clauses_traj, time_traj_to_plot, i, j, 1, plotting_transient, 'Number of Satisfied Clauses', f'{file_label}_num_sat_clauses_{param_index}', 10)
                                if interactive:
                                    plotly_trajectory_plotter(v_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Voltages', f'{file_label}_v_{param_index}', 10)
                                    plotly_trajectory_plotter(xs_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Short Term Memories', f'{file_label}_xs_{param_index}', 10)
                                    plotly_trajectory_plotter(xl_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Long Term Memories', f'{file_label}_xl_{param_index}', 10)
                                    plotly_trajectory_plotter(C_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Clause Functions', f'{file_label}_C_{param_index}', 10)
                                    plotly_trajectory_plotter(G_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Gradient Terms', f'{file_label}_G_{param_index}', 10, GR=True)
                                    plotly_trajectory_plotter(R_traj, time_traj_to_plot, i, j, 10, plotting_transient, 'Rigidity Terms', f'{file_label}_R_{param_index}', 10, GR=True)
                                    plotly_trajectory_plotter(num_sat_clauses_traj, time_traj_to_plot, i, j, 1, plotting_transient, 'Number of Satisfied Clauses', f'{file_label}_num_sat_clauses_{param_index}', 10)
                                
                                #Plots timestep dt
                                dt_traj_to_plot = [element[j] for element in dt_traj[i]]
                                plt.plot(time_traj_to_plot, dt_traj_to_plot)
                                plt.axhline(1e-1, color='red', linestyle='dashed')
                                #plt.ylim(3e-6, min(3e-1, max(dt_traj_to_plot)))
                                plt.yscale('log')
                                plt.xlabel('Time')
                                plt.ylabel('dt')
                                plt.tight_layout()
                                plt.savefig(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_dt_{param_index}.png')
                                
                                if interactive:
                                    dt_fig = go.Figure()
                                    dt_fig.add_trace(go.Scatter(x=time_traj_to_plot, y=dt_traj_to_plot, mode='lines', name='dt'))
                                    dt_fig.add_hline(y=1e-1, line_dash="dash", line_color="red")
                                    dt_fig.update_layout(title="Timestep (dt) Trajectory", xaxis_title="Time", yaxis_title="dt", yaxis_type="log", template="plotly_white")
                                    dt_fig.write_html(f'results/{prob_type}/Benchmark/{flattened_big_ns}/n{ns[i]}_batch{j}_dt_{param_index}.html')
                                
                                if animation:
                                    if prob_type == '3SAT':
                                        cnf_file = f'data/p0_080/ratio_4_30/var_{ns[i]}/instances/transformed_barthel_n_{ns[i]}_r_4.300_p0_0.080_instance_{j+1:03d}.cnf'
                                    elif prob_type == '3R3X':
                                        cnf_file = f'data/XORSAT/3R3X/{ns[i]}/problem_{j:04d}.cnf'
                                    elif prob_type == '5R5X':
                                        cnf_file = f'data/XORSAT/5R5X/{ns[i]}/problem_{j:04d}.cnf'
                                    else:
                                        cnf_file = None
                                    
                                    if cnf_file and os.path.exists(cnf_file):
                                        clause_idx_list, clause_sign_list, _, _, _, _ = import_data(cnf_file)
                                        create_3sat_circuit_animation(
                                            clause_idx=clause_idx_list[0],
                                            clause_sign=clause_sign_list[0],
                                            v_traj=v_traj[i],
                                            time_traj=time_traj[i],
                                            C_traj=C_traj[i],
                                            batch_idx=j,
                                            output_path=f'{file_label}_circuit_{param_index}'
                                        )

                                plt.clf()
