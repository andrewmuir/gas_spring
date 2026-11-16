"""Small conservative transport/restriction transitions around zero flow.

These suppress derivative changes at rates far below operating flows. They
do not add compliance or permit mass/energy leakage between control volumes.
"""

VOLUME_FLOW_TRANSITION_M3_S = 1e-11
MASS_FLOW_TRANSITION_KG_S = 1e-8
