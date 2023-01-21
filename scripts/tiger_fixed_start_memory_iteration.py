
from grl.environment import load_spec
from grl.analytical_agent import AnalyticalAgent

if __name__ == "__main__":

    # Get POMDP definition
    spec = load_spec('tiger-alt-start', memory_id=0, n_mem_states=2)

