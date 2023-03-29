# GPs for Bayesian Optimization
Bayesian optimization is a sequential design strategy for global optimization of black-box functions that does not assume any functional forms. In this project you will use Gaussian Process (GP) models to perform Bayesian Optimization of the parameters of Model Predictive Path Integral (MPPI) controller for a pushing task. 

<ins>Due data</ins>: **April 21** for code and demo | **April 24** for presentation

## TODOs

- [x]  **Setup Task Environment**

    Construct a non-trivial set of obstacles in bullet pushing environment from homework 3. Use any pushing dynamics model (e.g. those learned in Homework 3 or Homework 4) in your MPPI. 

- [ ] **Implement Bayesian Optimization Algorithm**
    - [x] Implement 1d Bayesian opt algo using Thompson sampling under `<models>`, run `python tests/test_tsgp.py` for testing.
    
    - [ ] Apply Bayesian Optimization to determine a good set of parameters for MPPI for performing a variety of pushing tasks. 

        Assume that the initial state is known and that you can reset the environment after each run when optimizing the parameters. 
        Be careful to account for the different scales and constraints on the parameters (e.g. the noise variance must be positive). 
    
- [ ] **Prepare Two Baseline Parameter Optmization Methods**

    - [ ] Performance comparisons of MPPI using your optimized parameters and at least two baselines on several pushing tasks. 

        One of the baselines should be to use the manually-defined parameters from the homework. 
        Another baseline should be a different optimization method, e.g. a black-box method like [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES). You may use existing libraries for the baselines.

- [ ] **Writeup a Report**

## Dependencies

- Python >= 3.8
- PyTorch >= 1.11
- GPytorch >= 1.9.1

## Reference
**BOA Algorithm**

<img src="figures/boa_pseudo.png" width="500">

> f: objective function <br>
> X: support of variables <br>
> S: acquisition function <br>
> M: GP model <br>
> D: dataset <br>
> x: variable <br>
> y: observation <br>


**Github Repository**
- [Bayesian machine learning notebooks](https://github.com/krasserm/bayesian-machine-learning) 1.6k
- [Python implementation of bayesian global optimization](https://github.com/fmfn/BayesianOptimization) 6.6k
- [BoTorch](https://github.com/pytorch/botorch) 2.6k
- [GPyTorch](https://github.com/cornellius-gp/gpytorch) 3k

**Others**
- [Bayesian optimization wiki](https://en.wikipedia.org/wiki/Bayesian_optimization)
- [Example blog](https://gdmarmerola.github.io/ts-for-bayesian-optim/)
