# On the Privavcy Risks of Algorithmic Recourse 
###
As predictive models are increasingly being employed to make consequential decisions, there is
a growing emphasis on developing techniques that can provide algorithmic recourse to affected
individuals. While such recourses can be immensely beneficial to affected individuals, potential
adversaries could also exploit these recourses to compromise privacy. In this code base, we make an 
attempt at investigating if and how an adversary can leverage recourses to infer private information
about the underlying model’s training data. 

## Paper @AISTATS 2023
For a more detailed introduction to these issues presented here please have a look at our paper available on arXiv:

>["On the Privacy Risks of Algorithmic Recourse"](https://arxiv.org/abs/2211.05427). Martin Pawelczyk, Himabindu Lakkaraju* and Seth Neel*.
*In International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR, 2023.

## Attack Overview

Our proposed membership inference (MI) attacks are (Pawelczyk et al (2023)):
- Counterfactual distance attack ($\texttt{CFD}$)
- Counterfactual distance LRT attack ($\texttt{CFD LRT}$).

In particular, our attacks take the following form:
```math
M_{\text{Distance}}(\mathbf{x})= \begin{cases} \texttt{MEMBER} & \text{ if } c(\mathbf{x}, \mathbf{x}') \geq \tau_D(\mathbf{x}) \\ \texttt{NON-MEMBER} & \text{ if } c(\mathbf{x}, \mathbf{x}') < \tau_D(\mathbf{x}) \end{cases},
```
where $c(\mathbf{x}, \mathbf{x}')$ denotes the counterfactual distance between $\mathbf{x}$ and $\mathbf{x}' = \mathbf{x} + \delta $

This repo also contains re-implementations of two popular loss-based MI attacks:
- Simple Loss attack ($\texttt{Loss}$) (Yeom et al (2018))
- LRT loss attack ($\texttt{Loss LRT}$) (Carlini et al (2021)).

The (LRT) loss based attacks have the following form:
```math
M_{\text{Loss}}(\mathbf{x})= \begin{cases} \texttt{MEMBER} & \text{ if } \ell(\theta, \mathbf{z}) \leq \tau_l(\mathbf{z}) \\ \texttt{NON-MEMBER} & \text{ if } \ell(\theta, \mathbf{z}) > \tau_l(\mathbf{z}) \end{cases},
```
where the $\ell(\theta, \mathbf{z})$ denotes the loss (e.g., MSE-Loss or BCE-Loss) on the point $\mathbf{z} = (\mathbf{x}, y)$, and the threshold $\tau$ depends on $\mathbf{x}$ for the LRT attack (Carlini et al (2021)) and is constant for the standard loss based attack. The table below briefly summarizes the assumptions that underlie the various attack algorithms.

<img align="center" width="600" height="200" src="/information_sets.PNG">

## Getting started
### Conda environment
We recommend setting up an extra conda environment for this code to ensure matching versions of the dependencies are installed. To setup the environment and run the notebooks, we assume you have a working installation of Anaconda and Jupyter. If everything is setup, you can run the distance_experiment.ipynb with the default parameters to get an understanding of how the attack works.

### Data generating process to determine factors of attack success
To better understand attack success, we additionally provide the following simple generating process to understand the factors that make membership inference attacks successful. 
Denote by $\gamma$ the class threshold. Denote by $q_{\mathbf{a}_{\alpha}}$ the $100 \times \alpha$-th quantile of an array $\mathbf{a}$.

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

### Data sets
To run experimetns on some real-world data sets, make sure to unzip the data folder, and to download the default data set from openml. The link to this data set can be found in the data/dataset_decscriptions/link.txt.

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
