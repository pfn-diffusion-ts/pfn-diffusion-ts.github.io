+++
title = "Thompson Sampling using Prior-fitted Diffusion Transformers"
[extra]
authors = [
    {name = "Sihwa Park*", url = "https://sihwa.ai"},
    {name = "Jingsen Zhu*", url = "https://jingsenzhu.github.io/"},
    {name = "Vinamr Jain", url = "https://vinamr.dev"},
    {name = "Sheng-Yen Chou", url = "https://frankccccc.github.io/"},
    {name = "Alexander Terenin", url = "https://avt.im/"},
]
venue = {name = "NeurIPS", date = 2026-12-06, url = "https://neurips.cc/Conferences/2026"}
buttons = [
    {name = "Paper", url = "https://arxiv.org/pdf/2611.14656"},
    {name = "PDF", url = "https://arxiv.org/pdf/2611.14656"},
    {name = "Code", url = "https://github.com/sihwa-park/PFN-Thompson-Sampling"},
]
katex = true
large_card = false
favicon = false
+++

**TLDR**: We show that a diffusion transformer can do in-context Thompson sampling at inference time after pre-training, as in prior-fitted networks.
We built a set of non-Gaussian priors for general black-box optimization, and our results show that pre-trained prior-fitted diffusion transformers with non-Gaussian priors can outperform traditional Gaussian-process pipelines.

---

We are interested in a global optimization problem where we want to find the global optimum of an unknown function `$f : X\to\R$` in a sample-efficient setting.
We assume every evaluation is expensive, so we must balance exploring uncertain regions with exploiting promising observations -- the exploration-exploitation tradeoff.
Bayesian optimization with Gaussian processes (GPs) is a classic approach to this type of problem, but the surrogate is restricted to Gaussian priors [^bayesopt-book] .
More recently, prior-fitted networks (PFNs) have shown that transformers can approximate the posterior predictive distribution `$p(y \mid D, x)$`, suggesting a meta-learning scheme for optimization [^pfn] [^pfns4bo] [^tabpfn].
PFNs support flexible priors while shifting the training burden to the pre-training stage, but need output discretization to represent predictive distributions and still require a decision rule after quantifying uncertainty.
Meanwhile, with the huge success of diffusion models in computer vision, there has been more work using expressive generative models for black-box optimization.
However, much of this work comes with bespoke training loss functions or regularizers to handle the exploration-exploitation tradeoff [^ddom] [^dibo] [^diff-bbo].


# Prior-fitted Diffusion Transformers

In our work, we suggest a framework doing in-context Thompson sampling using pre-trained diffusion transformers [^dit] while keeping the pipeline simple and rooted in decision-theoretic principles.

{% figure(alt=["PFN-Thompson-sampling"], src=["overview.svg"], dark_invert=[true]) %}
**Figure 1.** A visual illustration of our pipeline.
{% end %}

While training, we first sample objective functions `$f$` from our prior and build datasets of input-output pairs to feed into transformers.
Through cross-attention, this information is fused with a sampled diffusion time step `$t$` and intermediate state `$z_t$`.
Unlike the original PFN, our target distribution is the posterior distribution over the maximizer, and we adopt diffusion models with a flow-matching loss [^flow-matching] to learn this distribution.
Hence, a diffusion transformer learns to approximate the distribution over maximizers during training with some priors, and at inference (when solving the actual optimization problem) does Thompson sampling in context.

The suggested framework doesn't need any specialized losses, regularizers, or multiple models interacting in a non-trivial way, but simply learns the target distribution through a diffusion model.
Also, it gives a clean way to do Thompson sampling in a Bayesian setting and still allows great flexibility in the choice of priors.


# Non-Gaussian Priors

This framework is not limited to Gaussian priors.
In other words, we can design our own priors with characteristics that Gaussian processes cannot capture such as priors over unimodal functions.

We propose a non-Gaussian prior family for general optimization problem, called the _envelope-plus-wiggles prior_, where we have a smooth envelope shape around the maximizer and wiggles as you move away from the maximizer.
This construction mirrors many global optimization benchmarks, which have a single global optimum surrounded by many local optima.
Concretely, we propose the family
```
$$
f(x) = E(x; x^*) + g(x)\, W(x),
$$
```
where `$x^*$` is sampled uniformly over the domain, `$E$` and `$W$` are the _envelope_ and _wiggles_, and `$g$` is a gate function, such as a sigmoid, that controls how much of the wiggles are added to the envelope.

We propose a unimodal prior for the envelope,
```
$$
E(x; x^*) = \alpha - \Vert n(x) - n(x^*) \Vert^2,
$$
```
where `$\alpha$` is a randomly-sampled optimum height and `$n$` is a warping function.
If `$n$` is the identity, this is simply a quadratic.
To obtain a more general prior with distortions, we take `$n$` to be an untrained normalizing flow [^real-nvp].
Its randomly initialized weights create a random warping, while we can still control its strength with the standard deviation of the normal distribution where we initialize the weights.
Since this is a bijective transformation, the unimodality is still preserved.

{% figure(alt=["Prior-examples"], src=["prior_3d.svg"], dark_invert=[true]) %}
**Figure 2.** A few sampled functions from the unimodal and EWP priors in 2D.
{% end %}

We also propose two wiggle priors based on GPs using random Fourier features (RFFs) [^rffs] in the paper.
One uses a fixed lengthscale with RFFs, and the other uses different lengthscales for different frequencies in RFFs.

# Experiments

We tested our method on both synthetic and empirical benchmark against diffusion-based and GP-based baselines, as shown in the below.
Our approach is denoted as PFN with different priors in parentheses where EWP-GP and EWP-U represent the two different wiggle priors.

{% figure(alt=["Synthetic"], src=["synthetic.svg"], dark_invert=[true]) %}
**Figure 3.** Comparison of regret in a potentially-model-mismatched setting, where the objectives are given by synthetic benchmark functions under various priors.
{% end %}

{% figure(alt=["Empirical"], src=["empirical.svg"], dark_invert=[true]) %}
**Figure 4.** Comparison of regret on empirical benchmarks, in order to evaluate potential behavior in practice.
{% end %}

The results show that at least one of our priors shows matching or better performance in almost all cases, while baselines perform inconsistently across problem sets.
We also find that the best-performing prior varies across input dimensions and benchmarks.
This is indeed intuitive, since the right exploration-exploitation trade-off should depend on the problem landscape and observations, and training with different priors can give the model more options to handle these better.

# Conclusion
In this work, we suggest a framework that performs Thompson sampling in-context with diffusion transformers.
Our results show that our method performs strong on various benchmarks when trained on the carefully-designed priors.
This work is particularly interesting because it suggests a new way to build and pre-train foundation models for optimization problems.
More broadly, in black-boax optimization it is still unclear how we can define a dataset to build this type of foundation model and design its architecture.
We believe that our work suggests one principled way to build such a dataset, an architecture, and a framework to train a model based on a Bayesian approach.


# Citation

```
@inproceedings{park2026thompson,
  title={Thompson Sampling using Prior-fitted Diffusion Transformers},
  author={Park, Sihwa and Zhu, Jingsen and Jain, Vinamr and Chou, Sheng-Yen and Terenin, Alexander},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```


# References

[^bayesopt-book]: R. Garnett. Bayesian Optimization. Cambridge University Press, 2023.
[^ddom]: S. Krishnamoorthy, S. M. Mashkaria, and A. Grover. Diffusion Models for Black-Box Optimization. International Conference on Machine Learning, 2023.
[^dibo]: T. Yun, K. Om, J. Lee, S. Yun, and J. Park. Posterior Inference with Diffusion Models for High-Dimensional Black-Box Optimization. International Conference on Machine Learning, 2025
[^diff-bbo]: D. Wu, N. L. Kuang, R. Niu, Y.-A. Wang, and R. Yu. Diff-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization
[^ts]: W. R. Thompson. On the Likelihood That One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika, 1933
[^pfn]: S. Müller, N. Hollmann, S. Pineda Arango, J. Grabocka, and F. Hutter. Transformers Can Do Bayesian Inference. International Conference on Learning Representations, 2022
[^pfns4bo]: S. Müller, M. Feurer, N. Hollmann, and F. Hutter. PFNs4BO: In-Context Learning for Bayesian Optimization. International Conference on Machine Learning, 2023
[^tabpfn]: N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. International Conference on Learning Representations, 2023
[^dit]: W. Peebles and S. Xie. Scalable Diffusion Models with Transformers. IEEE/CVF International Conference on Computer Vision, 2023
[^flow-matching]: Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow Matching for Generative Modeling. International Conference on Learning Representations, 2023
[^rffs]: A. Rahimi and B. Recht. Random Features for Large-Scale Kernel Machines. Advances in Neural Information Processing Systems, 2007
[^real-nvp]: L. Dinh, J. Sohl-Dickstein, and S. Bengio. Density Estimation Using Real NVP. International Conference on Learning Representations, 2017

