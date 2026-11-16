+++
title = "Thompson Sampling using Prior-fitted Diffusion Transformers"
[extra]
authors = [
    {name = "Sihwa Park*", url = "https://sihwa.ai"},
    {name = "Jingsen Zhu*", url = "https://jingsenzhu.github.io/"},
    {name = "Vinamr Jain", url = ""},
    {name = "Sheng-Yen Chou", url = ""},
    {name = "Alexander Terenin", url = "https://avt.im/"},
]
venue = {name = "NeurIPS", date = 2026-12-06, url = "https://neurips.cc/Conferences/2026"}
buttons = [
    %%{name = "Paper", url = ""},%%
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

# Background

We are interested in a global optimization problem where we want to find the global optimum of an unknown function `$f : X\to\R$` in a sample-efficient setting.
We assume every evaluation is expensive, so we must balance exploring uncertain regions with exploiting promising observations -- the exploration-exploitation tradeoff.
Bayesian optimization with Gaussian processes (GPs) is a classic approach to this type of problem, but the surrogate is restricted to Gaussian priors [^bayesopt-book] .
More recently, prior-fitted networks (PFNs) have shown that transformers can approximate the posterior predictive distribution `$p(y \mid D, x)`, suggesting a meta-learning scheme for optimization [^pfn] [^pfns4bo] [^tabpfn].
PFNs support flexible priors while shifting the training burden to the pre-training stage, but need output discretization to represent predictive distributions and still require a decision rule after quantifying uncertainty.
Meanwhile, with the huge success of diffusion models in computer vision, there has been more work using expressive generative models for black-box optimization.
However, much of this work comes with bespoke training loss functions or regularizers to handle the exploration-exploitation tradeoff [^ddom] [^dibo] [^diff-bbo].


# Prior-fitted Diffusion Transformers

In our work, we suggest a framework doing in-context Thompson sampling using pre-trained diffusion transformers [^dit] while keeping the pipeline simple and rooted in decision-theoretic principles.

{% <figure alt={["PFN-Thompson-sampling"]} src={["overview.pdf"]} dark_invert={[true]}> %}
**Figure 1.** A visual illustration of our pipeline.
{% </figure> %}

In this framework, we sample objective functions `$f$` from our prior and build datasets of input-output pairs to feed into transformers.
Through cross-attention, this information is fused with a sampled diffusion time step `$t$` and intermediate state `$z_t$`.
Unlike the original PFN, our target distribution is the posterior distribution over the maximizer, and we adopt diffusion models with a flow-matching loss [^flow-matching] to learn this distribution.

# Priors

We propose a non-Gaussian prior family for general optimization problem, called the _envelope-plus-wiggles prior_, where we have a smooth envelope shape around the maximizer and wiggles as you move away from the maximizer.
Concretely, we propose the family
```
$$
f(x) = E(x; x^*) + g(x)\, W(x),
$$
```
where `$x^*$` is sampled uniformly over the domain, `$E$` and `$W$` are the _envelope_ and _wiggles_.
We propose unimodal priors for the envelope and two wiggle priors in the paper.

# Experiments

We tested our method on both synthetic and empirical benchmark against diffusion-based and GP-based baselines, as shown in the below.

{% <figure alt={["Synthetic"]} src={["synthetic.pdf"]} dark_invert={[true]}> %}
**Figure 2.** Comparison of regret in a potentially-model-mismatched setting, where the objectives are given by synthetic benchmark functions under various priors.
{% </figure> %}

{% <figure alt={["Empirical"]} src={["empirical.pdf"]} dark_invert={[true]}> %}
**Figure 3.** Comparison of regret on empirical benchmarks, in order to evaluate potential behavior in practice.
{% </figure> %}


# Conclusion
In this work, we suggest a framework that performs Thompson sampling in-context with diffusion transformers.
Our results show that our method performs strong on various benchmarks when trained on the carefully designed priors.
This work is particularly interesting because it suggests a new way to build and pre-train foundation models for optimization problems.


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

[^simple-regret]: See Chapter 10 of the *Bayesian Optimization* book.
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
[^scaling-law]: J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, O. Vinyals, J. W. Rae, and L. Sifre. An Empirical Analysis of Compute-Optimal Large Language Model Training. Advances in Neural Information Processing Systems, 2022

