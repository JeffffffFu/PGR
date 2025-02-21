import matplotlib.pyplot as plt 
import networkx as nx
import numpy as np

from pdb import set_trace as st


def plot_attack(adj_train, adj_train_first, victim_edge):
    """
    Note: currently we only support the case for adding edges.
    """
    plt.figure(figsize=(12,7))
    graph = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.DiGraph())
    graph.add_edge(victim_edge[0], victim_edge[1], color='blue', weight=1)
    edges = graph.edges()
    pos = nx.spring_layout(graph)

    colors = []
    for i,j in edges:
        if i == victim_edge[0] and j == victim_edge[1]:
            colors.append('blue')
        elif adj_train_first[i][j] == 0:
            colors.append('g')
        else:
            colors.append('orange')

    weights = [graph[u][v]['weight'] for u,v in edges]
    nx.draw(graph, pos, edges=edges, edge_color=colors, node_size=20, width=1.5)
    plt.show()

def plot_expr(score_dict):
    """
    Make the plot where x_axis represents different experiments.
    """
    fig, ax1 = plt.subplots()

    ax1.set_ylabel("Score for the specific edge")
    num_log = len(score_dict['attack'])

    x_axis = np.linspace(1, num_log, num_log)

    ax1.plot(x_axis, score_dict['attack'], color='r',label='attack')
    ax1.plot(x_axis, score_dict['baseline'], color='g', label='groudtruth')
    ax1.plot(x_axis, score_dict['random'], color='b', label='random')
    ax1.legend(loc = 'upper left')
    ax1.set_xlim(0, num_log+2)

    plt.grid(True)
    plt.title("Attack on 0-1 adjacency matrix")
    plt.show()
    plt.savefig('log.png',dpi=800)