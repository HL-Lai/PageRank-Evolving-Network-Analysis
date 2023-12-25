# Initialization
import os
import time
import pickle
import random
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from joblib import Parallel, delayed

# Dataset
def download_dataset():
    import urllib.request
    print("Downloading data...")

    url = "http://snap.stanford.edu/data/as-733.tar.gz"
    filename = "as-733.tar.gz"

    urllib.request.urlretrieve(url, filename)

    print("Data downloaded.  Now decompress data...")

def decompress_data(filename = "as-733.tar.gz"):
    import tarfile

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall("Data")

    print("Data decompressed.")

folder_name = "Data"
filename = "as-733.tar.gz"

if not os.path.isfile(filename): # download data if it doesn't exist
    download_dataset()
if not os.path.isdir(folder_name):
    decompress_data(filename)
else:
    print("Data exists.")

graph_list = [file for file in os.listdir('Data')]
graph_list.sort()

def cal_time(total_time):
    hour = int(total_time//3600)
    min = int((total_time%3600)//60)
    sec = (total_time%60)
    total_time_str = f"{sec:.2f} second{'s' if sec > 1 else ''}" if total_time < 60 else f"{min} minute{'s' if min > 1 else ''} and {sec} second{'s' if sec > 1 else ''}" if total_time < 3600 else f"{hour} hour{'s' if hour > 1 else ''} {min} minute{'s' if min > 1 else ''} {sec} second{'s' if sec > 1 else ''}"
    return total_time_str

# PageRank vector
if os.path.exists(f'pi_values.pkl'):
    print(f'pi_values.pkl exists.')
    with open(f'pi_values.pkl', 'rb') as f:
        pi_values = pickle.load(f)
else:
    pi_values = []
    for g in tqdm(graph_list):
        graph = nx.read_adjlist('Data/'+g, nodetype = int, create_using=nx.DiGraph)
        pi_t = nx.pagerank(graph)
        pi_values.append(pi_t)
    with open(f'pi_values.pkl', 'wb') as f:
        pickle.dump(pi_values, f)

# Probing

def random_probing(graph, M: int, p=[], alpha=0.15):
    node_list = list(graph.nodes)
    visits = dict.fromkeys(node_list, 0)
    if M == 0: return {k: v for k, v in visits.items()}
    for step in range(M):
        node = random.choice(node_list)
        visits[node] += 1
        if (step+1)%(M//50)==0:
            p.append({k: v / M for k, v in visits.items()})
    return p

def round_robin_probing(graph, M: int, p=[], alpha=0.85):
    node_list = list(graph.nodes)
    num_nodes = len(node_list)
    visits = dict.fromkeys(node_list, 0)
    node_idx = 0
    if M == 0: return {k: v for k, v in visits.items()}
    for step in range(M):
        node = node_list[node_idx]
        visits[node] += 1
        node_idx = (node_idx + 1) % num_nodes
        if (step+1)%(M//50)==0:
            p.append({k: v / M for k, v in visits.items()})
    return p


def proportional_probing(graph, M: int, p=[], alpha=0.85):
    rank_dict = nx.pagerank(graph, alpha=alpha)
    node_list = list(rank_dict.keys())
    weights = list(rank_dict.values())
    visits = dict.fromkeys(node_list, 0)
    if M == 0: return {k: v for k, v in visits.items()}
    for step in range(M):
        if random.random() < alpha:
            node = random.choices(node_list, weights=weights)[0]
        else:
            node = random.choice(node_list)
        visits[node] += 1
        if (step+1)%(M//50)==0:
            p.append({k: v / M for k, v in visits.items()})
    return p


def priority_probing(graph, M: int, p=[], alpha=0.85):
    rank_dict = nx.pagerank(graph, alpha=alpha)
    node_list = list(graph.nodes)
    priority = {node: 0 for node in node_list}
    visits = {node: 0 for node in node_list}
    if M == 0: return {k: v for k, v in visits.items()}
    for step in range(M):
        node = max(priority, key=priority.get)
        visits[node] += 1
        priority[node] = 0
        for other_node in node_list:
            if other_node != node:
                priority[other_node] += rank_dict[other_node]
        if (step+1)%(M//50)==0:
            p.append({k: v / M for k, v in visits.items()})
    return p


M = 500000
Mrange = range(0, M, M//50)
p_list = []

def process_graph(graph_list=graph_list, probe="random", p_list=p_list, M=M):
    for g in tqdm(graph_list):
        graph = nx.read_adjlist('Data/'+g, nodetype=int, create_using=nx.DiGraph)
        match probe:
            case "random": func = random_probing
            case "round_robin": func = round_robin_probing
            case "proportional": func = proportional_probing
            case "priority": func = priority_probing
        p = func(graph, M, p=[], alpha=0.15)
        p_list.append(p)
    return p_list


probes = ["random", "round_robin", "proportional", "priority"]
to_probe = []
gp_dict = {}
gp_list = []


def process_probe(probe, M=M):
    print("Processing graph with {} probing ({} steps)...".format(probe, M))
    start_time = time.time()
    p_list = []

    p_list = process_graph(graph_list, probe, p_list, M=M)
    gp_dict[probe] = p_list
    
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = cal_time(total_time)
    print("Time used for {} probing: {}".format(probe, total_time_str))
    print(f"gp_dict[{probe}]: {len(gp_dict[probe])} x {len(gp_dict[probe][0])} x {len(gp_dict[probe][0][0])}")
    
    with open(f'gp_dict[{probe}].pkl', 'wb') as f:
        pickle.dump(p_list, f)

    return p_list

for probe in probes:
    if os.path.exists(f'gp_dict[{probe}].pkl'):
        print(f'gp_dict[{probe}].pkl exists.')
        with open(f'gp_dict[{probe}].pkl', 'rb') as f:
            gp_dict[probe] = pickle.load(f)
        print(f"{probe} probing data retrieved.\n")
    else:
        to_probe.append(probe)

if len(to_probe) > 0:
    gp_dict.update({probe: result for probe, result in zip(to_probe, Parallel(n_jobs=-1)(delayed(process_probe)(probe, M=M) for probe in to_probe))})

if not os.path.exists(f'gp_dict.pkl'):
    with open(f'gp_dict.pkl', 'wb') as f:
        pickle.dump(gp_dict, f)


# Calculating norm values

def calculate_norm(gp_list, n=len(graph_list)):
    gL1_norms = []
    gLinfty_norms = []
    for i in tqdm(range(len(graph_list))):
        L1_norms = []
        Linfty_norms = []
        for j in range(len(gp_list[i])):
            p = np.fromiter(gp_list[i][j].values(), dtype=float)
            try:
                pa = np.fromiter(pi_values[i].values(), dtype=float)
            except:
                print(i, j)
            L1_norms.append(np.linalg.norm(pa - p, ord=1)) # L1_norms: a list containing L1-norm of all M steps of a graph
            Linfty_norms.append(np.linalg.norm(pa - p, ord=np.inf))
        gL1_norms.append(L1_norms) # gL1_norms: a list containing L1-norms of all graphs
        gLinfty_norms.append(Linfty_norms)
    aL1_norms = (np.sum(gL1_norms, axis=0))/n
    aLinfty_norms = (np.sum(gLinfty_norms, axis=0))/n
    return aL1_norms, aLinfty_norms

L1_dict = {}
Linfty_dict = {}
for probe, gp_list in gp_dict.items():
    with Parallel(n_jobs=-1) as parallel:
        print("Calculating norm of {}...".format(probe))
        start_time = time.time()

        L1_dict[probe], Linfty_dict[probe] = calculate_norm(gp_list)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = cal_time(total_time)
        print("Time used for calculating norm of {} probing: {}".format(probe, total_time_str))
    
with open('L1_dict.pkl', 'wb') as f:
    pickle.dump(L1_dict, f)

with open('Linfty_dict.pkl', 'wb') as f:
    pickle.dump(Linfty_dict, f)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(Mrange, L1_dict["random"], label='Random', linestyle='--')
axs[0].plot(Mrange, L1_dict["round_robin"], label='Round-Robin', linestyle=':')
axs[0].plot(Mrange, L1_dict["proportional"], label='Proportional', linestyle='-.')
axs[0].plot(Mrange, L1_dict["priority"], label='Priority', linestyle='-')
axs[0].set_xlabel('# changes (M)')
axs[0].set_ylabel(r'Average $L_1$ norm')
axs[0].set_title(r'Average $L_1$ norm vs steps')
axs[0].legend()

axs[1].plot(Mrange, Linfty_dict["random"], label='Random', linestyle='--')
axs[1].plot(Mrange, Linfty_dict["round_robin"], label='Round-Robin', linestyle=':')
axs[1].plot(Mrange, Linfty_dict["proportional"], label='Proportional', linestyle='-.')
axs[1].plot(Mrange, Linfty_dict["priority"], label='Priority', linestyle='-')
axs[1].set_xlabel('# changes (M)')
axs[1].set_ylabel(r'Average $L_{\infty}$ norm')
axs[1].set_title(r'Average $L_{\infty}$ norm vs steps')
axs[1].legend()
plt.savefig('Lnorm.png')
plt.show()

# plt.plot(Mrange, L1_dict["random"], label='Random', linestyle='--')
# plt.plot(Mrange, L1_dict["round_robin"], label='Round-Robin', linestyle=':')
# plt.plot(Mrange, L1_dict["proportional"], label='Proportional', linestyle='-.')
# plt.plot(Mrange, L1_dict["priority"], label='Priority', linestyle='-')

# plt.xlabel('# changes (M)')
# plt.ylabel(r'Average $L_1$ norm')
# plt.title(r'Average $L_1$ norm vs steps')
# plt.legend()
# plt.savefig('L1_rnorm.png')
# plt.show()


# plt.plot(Mrange, Linfty_dict["random"], label='Random', linestyle='--')
# plt.plot(Mrange, Linfty_dict["round_robin"], label='Round-Robin', linestyle=':')
# plt.plot(Mrange, Linfty_dict["proportional"], label='Proportional', linestyle='-.')
# plt.plot(Mrange, Linfty_dict["priority"], label='Priority', linestyle='-')

# plt.xlabel('# changes (M)')
# plt.ylabel(r'Average $L_{\infty}$ norm')
# plt.title(r'Average $L_{\infty}$ norm vs steps')
# plt.legend()
# plt.savefig('Linfty_rnorm.png')
# plt.show()