from baseline.GAP_master.core.attacks.lsa import LinkStealingAttack
from baseline.GAP_master.core.attacks.nmi import NodeMembershipInference
from baseline.GAP_master.core.attacks.lira import LikelihoodRatioAttack
from baseline.GAP_master.core.attacks.gra import GraphReconstructionAttack


supported_attacks = {
    'lsa': LinkStealingAttack,
    'nmi': NodeMembershipInference,
    'lira': LikelihoodRatioAttack,
    'gra': GraphReconstructionAttack,
}
