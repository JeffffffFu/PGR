from baseline.GAP_master.core.methods.node.gap.gap_edp import EdgePrivGAP
from baseline.GAP_master.core.methods.node.gap.gap_inf import GAP
from baseline.GAP_master.core.methods.node.gap.gap_ndp import NodePrivGAP
from baseline.GAP_master.core.methods.node.mlp.mlp import MLP
from baseline.GAP_master.core.methods.node.mlp.mlp_dp import PrivMLP
from baseline.GAP_master.core.methods.node.sage.sage_edp import EdgePrivSAGE
from baseline.GAP_master.core.methods.node.sage.sage_inf import SAGE
from baseline.GAP_master.core.methods.node.sage.sage_ndp import NodePrivSAGE

supported_methods = {
    'gap-inf':  GAP,
    'gap-edp':  EdgePrivGAP,
    'gap-ndp':  NodePrivGAP,
    'sage-inf': SAGE,
    'sage-edp': EdgePrivSAGE,
    'sage-ndp': NodePrivSAGE,
    'mlp':      MLP,
    'mlp-dp':   PrivMLP
}
