# Counterfactual Distance MI Attack

## Attack Overview

Our proposed membership inference (MI) attacks are (Pawelczyk et al (2023)):
- Counterfactual distance attack
- Counterfactual distance LRT attack.

In particular, our attacks take the following form:
```math
M_{\text{Distance}}(\mathbf{x})= \begin{cases} \texttt{MEMBER} & \text{ if } c(\mathbf{x}, \mathbf{x}') \geq \tau_D(\mathbf{x}) \\ \texttt{NON-MEMBER} & \text{ if } c(\mathbf{x}, \mathbf{x}') < \tau_D(\mathbf{x}) \end{cases},
```
where $c(\mathbf{x}, \mathbf{x}')$ denotes the counterfactual distance between $\mathbf{x}$ and $\mathbf{x}' = \mathbf{x} + \delta $

This repo also contains re-implementations of two popular loss-based MI attacks:
- Simple Loss attack (Yeom et al (2018))
- LRT loss attack (Carlini et al (2021)).

The (LRT) loss based attacks have the following form:
```math
M_{\text{Loss}}(\mathbf{x})= \begin{cases} \texttt{MEMBER} & \text{ if } \ell(\theta, \mathbf{x}) \leq \tau_l(\mathbf{x}) \\ \texttt{NON-MEMBER} & \text{ if } \ell(\theta, \mathbf{x}) > \tau_l(\mathbf{x}) \end{cases},
```
where the threshold $\tau$ depends on $\mathbf{x}$ for the LRT attack (Carlini et al (2021)) and is constant for the standard loss based attack.

## Data generating process to determine factors of attack success

Denote by $\gamma$ the class threshold. Denote by $q_{\mathbf{x}_{\alpha}}$ the $100 \times \alpha$-th quantile of the vector $\mathbf{x}$.

**Measurement error**:

$$\varepsilon \sim \mathcal{N}(0, \sigma^2_{\varepsilon})$$

**Design matrix**: 

$$\mathbf{X} \sim \mathcal{N}(\mu_d, \Sigma_d)$$

**True coefficient vector**:

$$\beta_0 \sim U[-1,1]^d$$

$$\beta = \beta_0 \odot \mathbb{I}(|\beta_0| > q_{|\beta_{\alpha}|})$$

$$\beta = \frac{\beta}{||\beta||_2}$$

**Labels**:

$$score =  X \beta + \varepsilon$$

$$p = \frac{1}{1+\exp(-score)}$$

$$y =  \mathbb{I}\big( p  > \gamma \big)$$

**Signal-to-noise ratio**:
$$\frac{||\beta||}{\sigma^2_{\varepsilon}} = \frac{1}{\sigma^2_{\varepsilon}}$$

In the here implmented version, we fix the true weight vector to unit length to make sure that we keep a constant signal-to-noise ratio despite an increase in the feature dimension.


## Credits
If you find this code useful, please consider citing the corresponding work:
```
 @inproceedings{pawelczyk2022privacy,
 title={{On the Privacy Risks of Algorithmic Recourse}},
 author={Pawelczyk, Martin and Lakkaraju, Himabindu and Neel, Seth},
 booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
 year={2023}
}
```
