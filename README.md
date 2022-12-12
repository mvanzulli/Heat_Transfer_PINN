# HeatTransferPINN

## Repo usage

<details>
<summary>(Optional) Python environment setup </summary>

### Create a python environment
```bash
python3 -m venv .venv_InvPINNs
```
### Activate the environment
```bash
. .venv_InvPINNs/bin/activate/
```
### Install dependencies with pip pkg manager
```bash
pip install -r requirements.txt
```
</details>

## Problem statement
We are interested in estimating the heat generation number in a rectangular fin with temperature dependent thermal conductivity and heat generation. The dimensionless governing equation is shown below:

$$\frac{\partial }{\partial x*} \left(  ( 1 + \epsilon_c \theta )  \frac{\partial \theta }{\partial x*} \right)   - N^2 \theta + N^2 G ( 1 + \epsilon_G \theta ) = 0$$

such that:

- $\theta(x) = \frac{T(x) - T_{inf}}{T_{b} - T_{inf}}$


- $N = \sqrt{\dfrac{h P L^2}{ k_0 A }} $


- $G=\dfrac{q_0A}{hP(T_b-T_{inf})}$


- $\epsilon_G=\epsilon(T_b-T_{inf})$ is the non-dimensional heat generation parameter.

- $\epsilon_C=\beta(T_b-T_{inf})$ is the non-dimensional heat generation parameter.



where

- $k$ is the thermal conductivity and varies linearly form $k_0$ value at $T_{inf}$ and reads:

    - $k(T(x))=k_0(1 + \beta(T(x)-T_{inf}))$.

- $q$ is the internal heat source which varies linearly with temperature as follows:

    - $q(T(x))=q_0(1 + \epsilon(T(x)-T_{inf}))$.

- $h$ is the convective coefficient.
- $x*$ is the non-dimensional axial coordinate of the fin.
- $T_b$ is the base temperature.
- $T_{inf}$ the non perturbed flow medium temperatere.

The boundary conditions considered are:

- at $x* = 0 \rightarrow \theta = 1$
- at $x* = 1 \rightarrow \frac{\partial\theta}{\partial x*} = 0$

## Dataset

The base temperature $T_b = 127$ °C and ${T_{inf}} = 27$ °C. The synthetic data is created in COMSOL Multi-Physics Software for different values of G. The data set can be found [here](https://github.com/mvanzulli/inversePINNs/blob/main/src/data/raw/Dataset.csv). $[T_1, T_2,... T_9 ]$ represent the temperature vuales at equi-spaced 9 locations between $x*= 0$ and $x*=1$. The dataset consits of 500 samples of temperature vectors for different $G$ values as follows:


<center><img src="https://user-images.githubusercontent.com/50339940/206868963-e975b3b5-7cff-404b-8f04-487a53ab0791.png" alt="drawing" width="300"/> </center>

## Challenges solved:


1. **Train a surrogate fully connected neural network model** on the provided dataset to learn the
mapping from the temperature field to G. The dataset may be split to train and test data in 80:20
ratio. Investigate the number training samples required for this task by training the model with
10, 20, 30, ..., 400 samples in the training dataset. Plot the Test Mean Squared Error against the
number of training samples. Report the CPU/GPU time taken.

1. Next, take one of the samples from training dataset and **use PINN [3] for estimating the value of
G**. Report the CPU/GPU time taken. You may use equation 4 for computing the residual loss.

1. What do you think are the dominant advantages/disadvantages of PINN compared to the vanilla
neural network model you created in Task 1?

1. Perform an experiment to investigate the **sensitivity of PINN to noise**. Add Gaussian white noise
with zero mean to the temperature vector. Vary the standard deviation of the noise and see how
the PINN predictions change. Provide your observations.

1. **Solve the same problem using PINNs with adaptive activation functions** [1]. Explain your implementation and report your observations. (Hint: related to model convergence while training)

## References

[1] A. D. Jagtap, K. Kawaguchi, and G. E. Karniadakis. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks. Journal of Computational Physics, 404:109136,
2020.

[2] V. Oommen and B. Srinivasan. Solving inverse heat transfer problems without surrogate models-a
fast, data-sparse, physics informed neural network approach. Journal of Computing and Information
Science in Engineering, pages 1–12, 2022.

[3] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning
framework for solving forward and inverse problems involving nonlinear partial differential equations.
Journal of Computational Physics, 378:686–707, 2019.
