from actrisk import StochasticSimulator
from actstats import actuarial as act

freq_dist = 'poisson'
freq_params = (10,)
sev_dist = 'lognormal'
sev_params = (10, 0.5)
act.poisson.ppf(0.8,10)
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 1000, True, 1234, 0.6, 'frank', 0.6)
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234, 0.6)
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234)

simulations = simulator.gen_agg_simulations()
simulator.all_simulations
simulator.calc_agg_percentile(99.2)
simulator.plot_distribution()
simulator.results.mean()
simulator.plot_correlated_variables()
simulator.all_simulations
simulator.analyze_results()
gross_loss = simulator.apply_deductible_and_limit(1000, 10000, 100000, 300000)
gross_loss['amount'] = gross_loss['gross_loss']
simulator.analyze_results(all_simulations=gross_loss)
simulator.all_simulations.to_csv('outputs/all_simulations.csv', index=False)
