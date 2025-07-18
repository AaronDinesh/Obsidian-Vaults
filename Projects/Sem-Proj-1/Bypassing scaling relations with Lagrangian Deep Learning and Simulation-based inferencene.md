## High Level Overview
- Provides a proof-of-concept model to to model the cluster number counts without the need for the scaling relations. The scaling relations are over parametrised and also does not take into account non-gravitational phenomena.
- Paper develops a baryonic field-emulator using  Lagrangian DL. Trained on the [CAMELS](https://camels.readthedocs.io/en/latest/) and [IllustrisTNG](https://www.tng-project.org/about/) dataset.
- The model correctly produces the cluster number count from the aforementioned datasets. Their model is also less degenerate than the previous approach (less useless parameters).
- Downsides: The model can only be as good as the simulations its trained on.

## Deep Dive
### Background + Motivation
- Best framework right now is the $\Lambda$CDM model, Dark Energy (DE) is the $\Lambda$ constant. 
- Galaxy Clusters (GC) are good cosmological probes since they are affected by the growth of structures and the geometry of the universe. 
- GC are on the high end of the Halo Mass Function (HMF). HMF is sensitive to cosmological parameters. 
- HMF predicts the number of clusters formed per unit mass as a function of cosmic time. Scaling relations are used to relate X-Ray luminosity  and temperature to the mass of the clusters. Alternatively the mass can also be measured via gravitational lensing. 
- These scaling relations don't predict the observed clusters well because it doesn't account for non gravitational processes (supernova feedback, turbulence, magnetic fields). Parameters and power-law fits were used to describe this. 
- To many parameters used $\rightarrow$ degeneracy in parameters. 
- Scaling relations also don't give insights into the non-gravitational processes in galaxy clusters. 
- [Previous Work]([[Cluster_cosmology_without_scaling_relations.pdf#page=2&selection=42,0,65,9|Cluster_cosmology_without_scaling_relations, page 2]]) that makes used of the scaling relations.
- Kosiba et al. 2024 used simulation based inference techniques to model the parameter count. Only used scaling relations for modelling the detected cluster population. No relations needed during inference.

### This Work + Dataset
- This work tries to model the physical cluster properties without the intermediate step of the scaling relations. 
- Hydrodynamical simulations can model the gravitational and non-gravitational effects based on a set of cosmological parameters. But this is too expensive to perform
- This work makes use of a GPU-Accelerated Dark Matter-only (DMO) simulation with a fast baryonification technique training on previous Hydrodynamical simulations. This method can only be as realistic as the simulations in the training set. 
- Method makes use of the [CAMELS](https://camels.readthedocs.io/en/latest/) dataset for the training set. Makes use of the "Cosmic Variance" subset and the "Latin Hypercube" subset.
- This work models the intra-galaxy cluster gas (ICM) electron number density ($n_e$) and temperature ($T$). They can then calculate the X-Ray emission of the gas.
- Uses post processing on the input simulations to get these temperatures on a regular grid. 
- Makes use of [Cloud-In-Cell (CIC)]() algorithm to spread the simulated particles in co-moving voxels.
- Working resolution (voxel size) in this paper is $0.39 h^{-1} \text{Mpc}$.

### Model Workings
- Work simulates the DM field from scratch using a fast Particle-Mesh (PM) algorithm. Initial condition is $z=6$. Makes use of [JaxPM](https://github.com/DifferentiableUniverseInitiative/JaxPM).
- They then train their ML model to emulate the baryonic properties from the PM-simulated DM instead of the DM field in CAMELS. The PM method to simulate the DM field introduces some smoothing at the smallest scales. The paper does not attempt to correct this, but mention that some [techniques do exist]([[Cluster_cosmology_without_scaling_relations.pdf#page=3&selection=175,5,176,26|Cluster_cosmology_without_scaling_relations, page 3]]).
- Makes use of Lagrangian Deep Learning (LDL) Framework to predict $n_e$ and $T$ from the DM fields. 
- LDL makes use of a particle displacement layer (PDL) and a non-linear activation layer (both of these are learnable layers).
- PDL acts on the simulated DM particles moving them along a modified potential (they use 2 consecutive PDLs). The non-linear activation allows them to introduce the non-linearities of baryonic processes. 
- LDL is a lightweight approach that is designed to follow physical principles (rotation + translation invariance and particles moving along a potential). 

### Math Formalism of Model
- Consider an input overdensity field $\delta(x)$ (which is the DM overdensity if this is the first later). The source term can be modelled as:
$$
f(x) = (1 + \delta(x))^{\gamma}
$$
where $\gamma$ is  a learnable parameter.
- Then they use a radial Fourier filter based on a B-Spline, $\mathcal{B}(\Xi, k)$ where $\Xi$ is the B-Spline parameters:
$$
\hat{O}_{\mathcal{B}}(k) = 1 + \mathcal{B}(\Xi, k)
$$ 
- Then the displacement field applied to the input particles is given as:
$$
dx = \alpha\nabla\mathcal{F}^{-1}\left(\hat{O}_{\mathcal{B}}(k)\hat{f}(k)\right)
$$
*What is $\hat{f}(k)$ here? The paper does not mention this. Could it be the input from a previous layer? Also I think it is implied that $\hat{O}_{\mathcal{B}}(k)$ is applied to $\hat{f}(k)$*
$\alpha$ is an additional learnable parameter and $\mathcal{F}^{-1}$ is the inverse Fourier Transform. 
- This displacement step is applied twice before a modified ReLU is applied:
$$
F(x^{\prime}) = \operatorname{ReLU}(b_1(1+\delta(x^{\prime}))^{\mu} - b_0)
$$
*Note here $x^{\prime}= x + dx$*
where $b_1, b_0, \mu$ are all learnable parameters. The total list of learnable parameters is denoted as $\Theta$.
- The output of the LDL is not a set of particles but a field on a co-moving grid.
- The goal is to reproduce the X-Ray emissivity of the ICM which is $\propto n_e^2T^{1/2}$. So the following loss functions are used
$$
\begin{align*}
\mathcal{L}_{n_e} &= \sum_{i}\left\lVert O_s * \left[T_{true}^{1/2}(x_i)\left(n_{e_LDL}^2(x_i) - n_{e_{true}}^2(x_i)\right)\right]\right\rVert_2\rho_{DM}(x_i) \\
\mathcal{L}_{n_e} &= \sum_{i}\left\lVert O_s * \left[n_{e_{true}}^2\left(T_{LDL}^{1/2}(x_i) -  T_{true}^{1/2}(x_i)\right)\right]\right\rVert_2\rho_{DM}(x_i) \\
\end{align*}
$$
*The lack of a $\lVert \cdot \rVert^2$ is interesting... I would have thought it would make the optimization easier?*
This uses the smoothing operator $\hat{O}_{s}(k) = 1+k^{-n}$. The paper chooses the value for $n$ from this [paper]([[Cluster_cosmology_without_scaling_relations.pdf#page=3&selection=562,30,562,51|Cluster_cosmology_without_scaling_relations, page 3]]). 

### Extending the LDL
- Currently the LDL approach is trained to produce a very specific model (specific redshift and specific parameters). But we want to infer based on any redshift and any parameters.
- We can do this by conditioning the weights on the following parameters: $\theta_{sim} = (\Omega_m, \sigma_8, A_{AGN1}, A_{AGN2}, A_{SN1}, A_{SN2})$. Then they also enforce that the baryonification depends on the redshift. 
- They retrain the base LDL on the Cosmic Variation dataset and output a set of weights $\Theta_{fid}$. Then they use an MLP trained on the LH set to output a weight variation $\delta\Theta$. The LDL model weights are now:
$$
\Theta = \Theta_{fid} + \delta\Theta(\theta_{sim})
$$