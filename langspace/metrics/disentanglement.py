from enum import Enum


class DisentanglementMetric(Enum):
    Z_DIFF = "z-diff"
    Z_MIN_VAR = "z-min-var"
    MIG = "MIG"
    # MODULARITY = "Modularity"
    # EXPLICITNESS = "Explicitness"
    DISENTANGLEMENT = "Disentanglement"
    COMPLETENESS = "Completeness"
    INFORMATIVENESS = "Informativeness"
