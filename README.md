# GPs for Bayesian Optimization
GPs can be used for many other applications, too. One application is called Bayesian Optimization, where you use a GP in the process of optimizing some parameters of a function. In this project you will use GPs for doing Bayesian Optimization of the parameters of MPPI for a pushing task. 

## TODOs

1.  **Setup Task Environment**
    First, construct a non-trivial set of obstacles in bullet pushing environment from homework 3. You can use any pushing dynamics model (e.g. those learned in Homework 3 or Homework 4) in your MPPI. 

2. **Implement Bayesian Optimization Algorithm**
    Then apply  erforming a variety of pushing tasks. 

    Assume that the initial state is known and that you can reset the environment after each run when optimizing the parameters. 

    Be careful to account for the different scales and constraints on the parameters (e.g. the noise variance must be positive). 
3. **Prepare Two Baseline Parameter Optmization Methods**
    Performance comparisons of MPPI using your optimized parameters and at least two baselines on several pushing tasks. 

    One of the baselines should be to use the manually-defined parameters from the homework. 

    Another baseline should be a different optimization method, e.g. a black-box method like [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES). You may use existing libraries for the baselines.

4. **Writeup a Report**

## Dependencies

- Python >= 3.8
- PyTorch >= 1.11
- GPytorch >= 1.9.1

## Reference
**Github Repository**
- [Bayesian machine learning notebooks](https://github.com/krasserm/bayesian-machine-learning) 1.6k
- [Python implementation of bayesian global optimization](https://github.com/fmfn/BayesianOptimization) 6.6k
- [BoTorch](https://github.com/pytorch/botorch) 2.6k
- [GPyTorch](https://github.com/cornellius-gp/gpytorch) 3k

**Others**
- [Bayesian optimization wiki](https://en.wikipedia.org/wiki/Bayesian_optimization)
- [Example blog](https://gdmarmerola.github.io/ts-for-bayesian-optim/)