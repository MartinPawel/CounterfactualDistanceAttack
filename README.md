# CounterfactualDistanceAttack

## Attack Overview

This repo plots results on a variety of different attacks:
- Simple Loss attack (Yeom et al (2018)) -> Threshold on the loss
- LRT loss attack (Carlini et al (2021)) -> Train shadow models & adjust the loss statistic

Our proposed attacks are (Pawelczyk et al (2023)):
- Counterfactual distance attack
- Counterfactual distance LRT attack

## Data generating process to determine factors of attack success

Denote by $\gamma$ the class threshold. Denote by $q_{\mathbf{x}_{\alpha}}$ the $100 \times \alpha$-th quantile of the vector $\mathbf{x}$.

**Measurement error**:

$\varepsilon \sim \mathcal{N}(0, \sigma^2_{\varepsilon})$

**Design matrix**: 

$\mathbf{X} \sim \mathcal{N}(\mu_d, \Sigma_d)$

**True coefficient vector**:

$\beta_0 \sim U[-1,1]^d$

$\beta = \beta_0 \odot \mathbb{I}(|\beta_0| > q_{|\beta_{\alpha}|})$

$\beta = \frac{\beta}{||\beta||_2}$

**Labels**:

$score =  X \beta + \varepsilon$

$p = \frac{1}{1+\exp(-score)}$

$y =  \mathbb{I}\big( p  > \gamma \big)$

**Signal-to-noise ratio**:
$\frac{||\beta||}{\sigma^2_{\varepsilon}} = \frac{1}{\sigma^2_{\varepsilon}}$.

We fix the true weight vector to unit length to make sure that we keep a constant signal-to-noise ratio despite an increase in the feature dimension.


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
