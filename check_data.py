import pickle
import networkx as nx
import numpy as np

#
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import ndlib.models.dynamic as dyn
import dynetx as dn

from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from tqdm import tqdm
import copy

def create_graph(data, window_size=600):
    # create a list of graphs
    graphs = []
    start_time = data[0][0]
    tmp_graph = nx.Graph()
    node_idx_mapping = {}
    for idx, values in enumerate(data):
        t, i, j = values[:3]

        if i not in node_idx_mapping:
            node_idx_mapping[i] = len(node_idx_mapping)
        if j not in node_idx_mapping:
            node_idx_mapping[j] = len(node_idx_mapping)
        i, j = node_idx_mapping[i], node_idx_mapping[j]
        
        if t - start_time > window_size:
            # create a new graph
            graphs.append(tmp_graph)
            tmp_graph = nx.Graph()
            start_time = t

        if tmp_graph.has_edge(i, j):
            tmp_graph[i][j]['weight'] += 1
        else:
            tmp_graph.add_edge(i, j, weight=1)
        
        if idx == len(data) - 1:
            graphs.append(tmp_graph)
        
            
    # check the graphs
    
    total_nodes = set()
    for graph in graphs:
        total_nodes.update(graph.nodes())
    print('total number of unique nodes: {}'.format(len(total_nodes)))
    # total number of timestamps
    print('total number of timestamps: {}'.format(len(graphs)))
    # total number of temporal edges
    total_edges = 0
    for graph in graphs:
        total_edges += len(graph.edges())
    print('total number of temporal edges: {}'.format(total_edges))
    # average weight of temporal edges
    total_weight = 0
    for graph in graphs:
        for i, j in graph.edges():
            total_weight += graph[i][j]['weight']
    print('average weight of temporal edges: {}'.format(total_weight / total_edges))
    return node_idx_mapping, graphs

def load_sfhh():
    data = []
    with open('src/co-presence/co-presence/tij_pres_SFHH.dat', 'r') as f:
        for line in f:
            t, i, j = line.strip().split(' ')
            data.append((int(t), i, j))
    return data 

def load_hospital():
    data = []
    with open('src/co-presence/co-presence/tij_pres_LH10.dat', 'r') as f:
        for idx, line in enumerate(f):
            print("line: ", idx, end='\r')
            t, i, j = line.strip().split(' ')
            data.append((int(t), i, j))
    print()
    return data




def run_SIR_simulation(dynGraph, n_sims=500, beta = 0.25, gamma = 0.055, seed=2023):
    # run n_sims simulations of SIR model
    # for each set of parameter (β, µ) = {(0.25, 0.055), (0.13, 0.1), (0.13, 0.055), (0.13, 0.01), (0.01, 0.055)}.
    # the simulation is accepted when there is still at least one infectious node when more than half of the total data set time span has elapsed (i.e., |I_{|T|/2}| ≥ 1).
    # If this condition is not met in any of the n_sims simulations, discard the corresponding case
    # return the node states at all time steps of the accepted simulations
    #  For each selected simulation, we assign as ground truth label to each active node (i, t_{i,a}) the state of node i at time t_{i,a}.
    # we consider as initial state a single randomly selected node as seed, setting its state as infectious, with all others susceptible. 

    found_it = True
    for simth in range(n_sims):
        model = dyn.DynSIRModel(dynGraph, seed=seed)
        # Model Configuration
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', beta)
        cfg.add_model_parameter('gamma', gamma)
        # start_pi = 1/dynGraph.time_slice(t_from=0).number_of_nodes()
        # num_infected = int(start_pi * dynGraph.time_slice(t_from=0).number_of_nodes())
        # print('num initial infected: ', num_infected)
        # cfg.add_model_parameter("percentage_infected", start_pi)
        infected_nodes = [np.random.choice(list(dynGraph.time_slice(t_from=0).nodes()))]
        print('infected node: ', infected_nodes)
        cfg.add_model_initial_configuration("Infected", infected_nodes)
        model.set_initial_status(cfg)

        # Simulation execution
        
        # system_status = model.execute_snapshots()
        system_status = []
        snapshots_ids = dynGraph.temporal_snapshots_ids()
        for t in snapshots_ids:
            model.graph = model.dyngraph.time_slice(t_from=t)
            its = model.iteration(True)
            system_status.append(its)
            # proportion of each state
            cs, ci, cr = its['node_count'][0], its['node_count'][1], its['node_count'][2]

            if t == len(snapshots_ids) - 1:
                cs, ci, cr = cs / (cs + ci + cr), ci / (cs + ci + cr), cr / (cs + ci + cr)
                print('final t: {}, ps: {}, pi: {}, pr: {}'.format(t, cs, ci, cr))
            
            if t == len(snapshots_ids) / 2 and ci < 1:
                found_it = False
                print('not valid simulation!')
                print('half t: {}, ps: {}, pi: {}, pr: {}'.format(t, cs, ci, cr))
                break
        
        if not found_it:
            continue
        else:
            break
    # visualize the simulation
    # trends = model.build_trends(system_status)
    # viz = DiffusionTrend(model, trends)
    # viz.plot("diffusion")
    return system_status

def convert_file_to_dynetx_format(path, newpath):
    with open(newpath, 'w') as f:
        with open(path, 'r') as f2:
            for line in f2:
                t, i, j = line.strip().split(' ')
                f.write('{} {} {} {}\n'.format(i, j, '+', t))

def extract_info(data_name):
    if data_name == 'SFHH':
        data = load_sfhh()
    elif data_name == 'LH10':
        data = load_hospital()
    
    node_idx_mapping, graphs = create_graph(data)
    
    # convert graph to directed graph
    # directed_graphs = []    
    # for graph in graphs:
    #     directed_graphs.append(graph.to_directed())

    dynGraph = dn.DynGraph(edge_removal=False)
    for t, graph in enumerate(graphs):
        dynGraph.add_interactions_from(graph.edges(data=True), t=t)

    beta, gamma = (0.25, 0.055)
    print('beta: {}, gamma: {}'.format(beta, gamma))
    system_status = run_SIR_simulation(dynGraph, beta=beta, gamma=gamma, seed=2025)

    label_matrix = np.zeros((len(system_status), len(node_idx_mapping))) - 1 # -1 means not active nodes
    for t, graph in enumerate(graphs):
        status = copy.deepcopy(system_status[t-1]['status'])
        status.update(system_status[t]['status'])
        system_status[t]['status'] = status
        for node in graph.nodes():
            label_matrix[t, int(node)] = status[node]

    return node_idx_mapping, graphs, label_matrix

if __name__ == '__main__':
    node_idx_mapping, graphs, label_matrix = extract_info('SFHH')
    import pdb;pdb.set_trace()