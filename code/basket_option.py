import numpy as np
import time
from math import exp
import scipy
from math import sqrt

scipy.random.seed([100])
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time
import pandas_datareader as web


from qiskit import Aer, QuantumRegister, QuantumCircuit, execute, AncillaRegister, transpile
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import WeightedAdder, LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution


rf=0.05
iterations=100000
T= 40/252

def read_data(ticker,strick_price_factor = 10):
    data= web.DataReader('AAPL',data_source='yahoo',start='1-1-2021',end='1-1-2022')["Adj Close"]
    log_return= np.log(1+data.pct_change())
    voltality=round(log_return.std(),2)
    S=data[0]
    sk = S+ strick_price_factor

    mu = (rf - 0.5 * voltality ** 2) * T + np.log(S)
    sigma = voltality * np.sqrt(T)
    mean = np.exp(mu + sigma ** 2 / 2)
    variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
    stddev = np.sqrt(variance)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low = np.maximum(0, mean - 3 * stddev)
    high = mean + 3 * stddev

    return {"data":data,"voltality":voltality,"S":S,"sk":sk,"mu":mu,"sigma":sigma,"low":low,"high":high}


aapl_data = read_data("AAPL")
tsla_data = read_data("TSLA")


def basket_option():
    dt = T
    drift1 = exp((rf-0.5 * aapl_data["voltality"] * aapl_data["voltality"])*dt)
    drift2 = exp((rf-0.5*tsla_data["voltality"]*tsla_data["voltality"])*dt)
    M=1000000
    S1next = 0.0
    S2next = 0.0
    arithPayOff = np.empty(M, dtype=float)
    start=time.time()
    scipy.random.seed([100])
    for i in range(0,M,1):
        Rand1 = scipy.random.randn(1)
        Rand2 = scipy.random.randn(1)
        growthFactor1 = drift1 * exp(aapl_data["voltality"] * sqrt(dt) * Rand1)
        S1next = aapl_data["S"] * growthFactor1
        growthFactor2 = drift2 * exp(tsla_data["voltality"] * sqrt(dt) * (0.5 * Rand1 + sqrt(0.75) * Rand2))
        S2next = tsla_data["S"] * growthFactor2
        # Arithmetic mean
        arithMean = (S1next+S2next)
        arithPayOff[i] = exp(-rf * T) * max((arithMean - (aapl_data["sk"] + tsla_data["sk"])), 0)
    # Standard monte carlo
    Pmean = np.mean(arithPayOff)
    Pstd = np.std(arithPayOff)
    confmc = [Pmean - 1.96*Pstd/sqrt(M), Pmean + 1.96*Pstd/sqrt(M)]
    time_taken=time.time()-start
    return {'Execution_time':time_taken,'option':np.mean(confmc)}


def BQAE():
    num_uncertainty_qubits = 3
    dimension = 2
    num_qubits = [num_uncertainty_qubits] * dimension
    low = np.array([aapl_data["low"],tsla_data["low"]])
    high = np.array([aapl_data["high"],tsla_data["high"]])
    mu =  np.array([aapl_data["mu"],tsla_data["mu"]])#mu * np.ones(dimension)
    cov = np.transpose([aapl_data["sigma"],tsla_data["sigma"]])*np.eye(dimension)

    u = LogNormalDistribution(num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high)))

    # determine number of qubits required to represent total loss
    weights = []
    for n in num_qubits:
        for i in range(n):
            weights += [2 ** i]

    # create aggregation circuit
    agg = WeightedAdder(sum(num_qubits), weights)
    n_s = agg.num_sum_qubits
    n_aux = agg.num_qubits - n_s - agg.num_state_qubits  # number of additional qubits
    strike_price = aapl_data["sk"] + tsla_data["sk"]
    max_value = 2 ** n_s - 1
    low_ = min(low)
    high_ = max(high)
    # mapped_strike_price = strike_price
    # setup piecewise linear objective fcuntion
   
    mapped_strike_price = (
        (strike_price - dimension * low_) / (high_ - low_) * (2**num_uncertainty_qubits - 1)
    )

    c_approx = 0.25

    # setup piecewise linear objective fcuntion
    breakpoints = [0, high_]
    slopes = [0, 1]
    offsets = [0, 0]
    f_min = 0
    f_max = 2 * (2 ** num_uncertainty_qubits - 1) - mapped_strike_price#high_#
    basket_objective = LinearAmplitudeFunction(
        n_s,
        slopes,
        offsets,
        domain=(0, high_),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )

    # define overall multivariate problem
    qr_state = QuantumRegister(u.num_qubits, "state")  # to load the probability distribution
    qr_obj = QuantumRegister(1, "obj")  # to encode the function values
    ar_sum = AncillaRegister(n_s, "sum")  # number of qubits used to encode the sum
    ar = AncillaRegister(max(n_aux, basket_objective.num_ancillas), "work")  # additional qubits

    objective_index = u.num_qubits

    basket_option = QuantumCircuit(qr_state, qr_obj, ar_sum, ar)
    basket_option.append(u, qr_state)
    basket_option.append(agg, qr_state[:] + ar_sum[:] + ar[:n_aux])
    basket_option.append(basket_objective, ar_sum[:] + qr_obj[:] + ar[: basket_objective.num_ancillas])

    # set target precision and confidence level
    epsilon = 0.01
    alpha = 0.05
    start=time.time()
    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
    problem = EstimationProblem(
        state_preparation=basket_option,
        objective_qubits=[objective_index],
        post_processing=basket_objective.post_processing,
    )
    # construct amplitude estimation
    ae = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)

    result = ae.estimate(problem)

    conf_int = (
        np.array(result.confidence_interval_processed)
        / (2 ** num_uncertainty_qubits - 1)
        * (high_ - low_)
    )

    # print(
    #     "Estimated value:    \t%.4f"
    #     % (result.estimation_processed / (2 ** num_uncertainty_qubits - 1) * (high_ - low_))
    # )
    print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))
    print(result.estimation_processed)
    return {'Execution_time':time.time()-start, 'option_QAE':result.estimation_processed / (2 ** num_uncertainty_qubits - 1) * (high_ - low_)}

if __name__=="__main__":
    basket_option_montecarlo = basket_option()
    print(f"basket option using montecarlo:{basket_option_montecarlo}")
    basket_option_qae = BQAE()
    print(f"basket option using qae:{basket_option_qae}")