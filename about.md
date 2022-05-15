---
layout: page
title: About
permalink: /about/
---

I am currently at [Waymo Research](https://waymo.com/) working on research problems related to autonomous driving.

I earned my master's in computer science from the University of Cambridge supervised by [Prof. Pietro Liò](https://www.cl.cam.ac.uk/~pl219/) and [Prof. Jure Leskovec](https://cs.stanford.edu/people/jure/). During my master's, I worked on problems related to representation learning. I did my bachelor's in computer science at Imperial College London where I worked on unsupervised learning for robotics under the supervision of [Dr. Ronald Clark](https://www.ronnieclark.co.uk/).

I also had the pleasure to spend two summers as an intern at [Google Research](https://research.google/) and one at [Facebook AI Research](https://ai.facebook.com/). At Google Research, I devised methods for quickly estimating how easily models will adapt to novel domains with [Dr. Thomas Mensink](https://www.mensink.nu/) and [Dr. Vittorio Ferrari](https://sites.google.com/view/vittoferrari), and I tried to break video games using reinforcement learning with [Dr. Olivier Bachem](http://olivierbachem.ch/). At Facebok AI Research, I helped build sparse convolutional networks with [Dr. Benjamin Graham](https://scholar.google.com/citations?user=jQkkhlkAAAAJ&fbclid=IwAR10QeH73jEELHjwmHK3rKV3vNnH3El0AZ0V3sHLF8wY8_BosconkfRiPIo).

### Projects & Papers

Here are a few projects I worked on that I think are pretty cool.

[How stable are Transferability Metrics evaluations?](https://arxiv.org/abs/2204.01403) (pre-print): We show that common approaches for evaluating transferability metrics are unstable. We then give some practical tips on how to evaluate them well and provide the first reliable comparison across several vision tasks.

[Transferability Estimation using Bhattacharyya Class Separability](https://arxiv.org/abs/2111.12780) (CVPR, 2022): This paper presents GBC, a fast heuristic approach for estimating how well will models transfer to new domains. GBC eliminates the need for fine-tuning all possible source models in order to pick the best one.

[Neural Distance Embeddings for Biological Sequences](https://arxiv.org/abs/2109.09740) (NeurIPS, 2021): NeuroSEED embeds biological sequences in hyperbolic embedding spaces and solves many useful downstream bioinformatics tasks such as hierarchical clustering or edit distance approximation.

[Learning Graph Search Heuristics](https://physical-reasoning.github.io/assets/pdf/papers/06.pdf) (NeurIPS workshop, 2021): We present PHIL, an imitation learning-based approach for learning graph search heuristics. PHIL also includes a specialized GNN architecture for learning such heuristics well.

<p align="center">
  <img src="/assets/images/phil_vid.gif" width="189"/>
</p>

[Unsupervised Path Regression Networks](https://arxiv.org/abs/2011.14787) (IROS, 2021): We develop a method for training learning-based planners supervised by scene geometries. Our approach outperforms methods supervised by optimal trajectories and is a useful pre-training step in high dimensional problems.

<p align="center">
  <img src="/assets/images/manip_uprn.gif" width="250"/>
</p>

[Emulating and Analysing the Sensitivity of Molecular Diffusion](https://drive.google.com/file/d/1czTYOfRwB7LA4V2tzJ6IQE86rD1cVv9X/view) (coursework): We used multi-fidelity deep gaussian processes to emulate molecular diffusion. Our emulator is faster than high-fidelity simulations and we demonstrate that it follows expected physical properties using Sobol sensitivity indices.