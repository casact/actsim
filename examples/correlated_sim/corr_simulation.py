import sys
import pandas as pd
# Replace with your actual path to 'src'
sys.path.insert(0, r"c:\Users\jzhng\Modelling\actrisk\src")
from actrisk.core.actsimulator import StochasticSimulator

##### Generate correlated mutivariate distribution
corr_matrix_file = 'examples/correlated_sim/corr_matrix.csv'
dist_list_file = 'examples/correlated_sim/dist_list.json'
simulator = StochasticSimulator("normal", [1,0], "normal",[1,0], 100000, True, 1234) # placeholder parameters for the simulator
simulator.gen_multivariate_corr_simulations(corr_matrix_file, dist_list_file, True)
simulator._all_simulations_data
data = pd.DataFrame(simulator._all_simulations_data)
data_t = data.transpose()
# Compute correlation matrix
correlation_matrix = data_t.corr()
print(correlation_matrix)