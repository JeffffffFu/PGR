from TIA.topology_privacy_based_inf import topology_audit_based_inf, \
    topology_audit_based_inf_worst_case
from TIA.topology_privacy_based_sim import topology_audit_based_sim, topology_audit_based_sim_classifier, \
     topology_audit_based_sim_PGR_worst_case, \
    topology_audit_based_sim_classifier_PGR_worst_case
from TIA.util import construct_multiple_subgraphs
import networkx as nx
from torch_geometric.utils import to_networkx


def TIA(data, model, dense_matrix, features, regen_adj, labels, device, hops,seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0

    print("TIA_attack")
    G = to_networkx(data, to_undirected=True)
    selected_subgraphs = construct_multiple_subgraphs(G)

    TPL_M=topology_audit_based_sim(selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)
    TPL_C=topology_audit_based_sim_classifier(selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device, seed)
    TPL_I=topology_audit_based_inf( selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device,hops, seed)
    return TPL_M, TPL_C, TPL_I


def TIA_PGR(data, model, dense_matrix, features, regen_adj, labels, device, hops,seed):
    TPL_M=0
    TPL_C=0
    TPL_I=0
    print("TIA_PGR_attack")

    G = to_networkx(data, to_undirected=True)
    selected_subgraphs = construct_multiple_subgraphs(G)
    TPL_M = topology_audit_based_sim_PGR_worst_case( selected_subgraphs, model, dense_matrix, features,
                                                  regen_adj, labels, device, seed)
    TPL_C = topology_audit_based_sim_classifier_PGR_worst_case(selected_subgraphs, model, dense_matrix, features,
                                                  regen_adj, labels, device, seed)

    TPL_I=topology_audit_based_inf_worst_case( selected_subgraphs, model, dense_matrix, features, regen_adj, labels, device,hops, seed)

    return TPL_M, TPL_C, TPL_I

