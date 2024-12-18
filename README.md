# MLCommons™ AlgoPerf: Training Algorithms Leaderboard

<br />
<p align="center">
<a href="#"><img width="600" img src=".assets/mlc_logo.png" alt="MLCommons Logo"/></a>
</p>

This repository hosts the official rolling leaderboard for the [**AlgoPerf: Training Algorithms benchmark**](https://github.com/mlcommons/algorithmic-efficiency) by [**MLCommons**](https://mlcommons.org/).
The benchmark measures neural network training speedups due to algorithmic improvements in training algorithms.
The leaderboard tracks the aggregate performance of different algorithms on a variety of [workloads](https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#workloads) and under two different [tuning rulesets](https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#tuning).

> **Leaderboard Version:** 0.5  
> **Last Updated:** 2023-12-18 09:54 UTC

## External Tuning Ruleset Leaderboard

*In the external tuning ruleset, submission must provide workload-agnostic hyperparameter search spaces and they will get* $5$ *tuning trials per workload sampled from this search space.*

<!-- TODO: Add links from the submission names to their code. -->
<!-- BEGIN EXTERNAL TUNING LEADERBOARD -->
| **Rank** | **Submission**            | **Authors**                                                                                                          | **Affiliation**                  | **Framework** | **Score**  |
|----------|---------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------|---------------|------------|
| 1.       | <details><summary>**Distributed Shampoo**</summary>Based on the Distributed Shampoo algorithm of [Anil et al. (2020)](https://arxiv.org/abs/2002.09018) with an implementation tailored to leverage PyTorch performance optimizations. See [Shi et al. (2023)](https://arxiv.org/abs/2309.06497) for details. The submission uses a list of five hyperparameter settings.</details>   | Hao-Jun Shi, Tsung-Hsien Lee, Anna Cai, Shintaro Iwasaki, Wenyin Fu, Yuchen Hao, Mike Rabbat                         | Meta Platforms                   | PyTorch       | **0.7784** |
| 2.       | <details><summary>**Schedule Free AdamW**</summary>An externally tuned version of Schedule Free AdamW ([Defazio et al., 2024](https://openreview.net/forum?id=0XeNkkENuI)) with a list of five hyperparameter configurations.</details>   | Alice Yang, Aaron Defazio, Konstantin Mishchenko                                                                     | Meta AI, Samsung AI              | PyTorch       | **0.7077** |
| 3.       | <details><summary>**Generalized Adam**</summary>Submission with an Adam-style update rule, tuning over the use of Nesterov acceleration and preconditioning. Essentially tuning over AdamW ([Kingma & Ba, 2015](https://openreview.net/forum?id=Bkg6RiCqY7)), NadamW, and SGD ([Robbins & Monro, 1951](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full)) with or without momentum.</details>      | George Dahl, Sourabh Medapati, Zack Nado, Rohan Anil, Shankar Krishnan, Naman Agarwal, Priya Kasimbeg, Vlad Feinberg | Google                           | JAX           | **0.6383** |
| 4.       | <details><summary>**Cyclic LR**</summary>Revisits the work of [Loshchilov & Hutter (2017)](https://openreview.net/forum?id=Skq89Scxx) and [Smith (2017)](https://arxiv.org/abs/1506.01186), coupling NadamW ([Dozat, 2016](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ); [Loshchilov & Hutter, 2019](https://openreview.net/forum?id=Bkg6RiCqY7)) with a cyclic learning rate scheduler. Each cycle involves a linear warmup phase for the LR, followed by cosine annealing.</details>             | Niccolò Ajroldi, Antonio Orvieto, Jonas Geiping                                                                      | MPI-IS, ELLIS Institute Tübingen | PyTorch       | **0.6301** |
| 5.       | <details><summary>**NadamP**</summary>Uses NadamW with an extra tunable hyperparameter $p$ enabling $p$ th root of denominator inside NadamW update rule instead of the default of $2$.</details>                | George Dahl, Sourabh Medapati, Zack Nado, Rohan Anil, Shankar Krishnan, Naman Agarwal, Priya Kasimbeg, Vlad Feinberg | Google                           | JAX           | **0.5909** |
| 6.       | <details><summary>[***Baseline***](/submissions/external_tuning/baseline/)</summary>Baseline using NadamW ([Dozat, 2016](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ); [Loshchilov & Hutter, 2019](https://openreview.net/forum?id=Bkg6RiCqY7)) and a linear learning rate warmup followed by a cosine decay ([Dahl et al., 2023](https://arxiv.org/abs/2306.07179)).</details>            |                                                                                                                      |                                  | JAX           | **0.5707** |
| 7.       | <details><summary>**Amos**</summary>Submission based on the Amos optimizer ([Tian & Parikh, 2022](https://arxiv.org/abs/2210.11693)) with a list of five hyperparameter settings.</details>                  | Ran Tian                                                                                                             | Google                           | JAX           | **0.4918** |
| 8.       | <details><summary>**CASPR Adaptive**</summary>A submission based on ([Duvvuri et al., 2024](https://openreview.net/forum?id=8j9hz8DVi8)) with a list of five hyperparameter configurations.</details>        | Sai Surya Duvvuri, Inderjit S. Dhillon, Cho-Jui Hsieh                                                                | UT Austin, UCLA, Google          | JAX           | **0.4722** |
| 9.       | <details><summary>**LAWA Queue**</summary>Employs Latest Weight Averaging ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407); [Kaddour, 2022](https://openreview.net/forum?id=0OrABUHZuz)) on top of NAdamW ([Dozat, 2016](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ); [Loshchilov & Hutter, 2019](https://openreview.net/forum?id=Bkg6RiCqY7)), maintaining a queue of previous model weights. The queue is periodically updated during training and passed to the competition API for evaluation.</details>            | Niccolò Ajroldi, Antonio Orvieto, Jonas Geiping                                                                      | MPI-IS, ELLIS Institute Tübingen | PyTorch       | **0.3699** |
| 10.      | <details><summary>**LAWA EMA**</summary>Employs Latest Weight Averaging ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407); [Kaddour, 2022](https://openreview.net/forum?id=0OrABUHZuz)) on top of NAdamW ([Dozat, 2016](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ); [Loshchilov & Hutter, 2019](https://openreview.net/forum?id=Bkg6RiCqY7)), maintaining an exponential moving average of the model weights, which is updated periodically during training and returned to the competition API for evaluation.</details>              | Niccolò Ajroldi, Antonio Orvieto, Jonas Geiping                                                                      | MPI-IS, ELLIS Institute Tübingen | PyTorch       | **0.3384** |
| 11.      | <details><summary>**Schedule Free Prodigy**</summary>Combining Schedule-free ([Defazio et al., 2024](https://openreview.net/forum?id=0XeNkkENuI)) with the Prodigy optimizer ([Mishchenko & Defazio, 2024](https://openreview.net/forum?id=WpQbM1kBuy)).</details> | Alice Yang, Aaron Defazio, Konstantin Mishchenko                                                                     | Meta AI, Samsung AI              | PyTorch       | **0.0000** |
<!-- END EXTERNAL TUNING LEADERBOARD -->

## Self-Tuning Ruleset Leaderboard

*In the self-tuning ruleset, submissions must be completely hyperparameter-free.*

<!-- BEGIN SELF-TUNING LEADERBOARD -->
| **Rank** | **Submission**          | **Authors**                                                                                                          | **Affiliation**            | **Framework** | **Score**  |
|----------|-------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------|---------------|------------|
| 1.       | <details><summary>**Schedule Free AdamW**</summary>A self-tuning version of Schedule Free AdamW ([Defazio et al., 2024](https://openreview.net/forum?id=0XeNkkENuI)) using a single hyperparameter configuration.</details> | Alice Yang, Aaron Defazio, Konstantin Mishchenko                                                                     | Meta AI, Samsung AI        | PyTorch       | **0.8542** |
| 2.       | <details><summary>[***Baseline***](/submissions/self_tuning/baseline/)</summary>Baseline using NadamW, a linear learning rate warmup followed by a cosine decay, and a single hyperparameter point ([Dahl et al., 2023](https://arxiv.org/abs/2306.07179)).</details>          |                                                                                                                      |                            | JAX           | **0.8194** |
| 3.       | <details><summary>**NadamW Sequential**</summary>Uses NadamW update rule and runs 3 fixed hyperparameter points sequentially. The intention was for these to be the top 3 hyperparameter points found at one third the self-tuning ruleset step budgets.</details>   | George Dahl, Sourabh Medapati, Zack Nado, Rohan Anil, Shankar Krishnan, Naman Agarwal, Priya Kasimbeg, Vlad Feinberg | Google                     | JAX           | **0.3308** |
| 4.       | <details><summary>**Sinv6 75**</summary>A submission for a task-invariant learned optimizer meta-trained on small tasks. Uses $75$% of the number of steps as target in learned optimizer initialization.</details>            | Abhinav Moudgil                                                                                                      | Mila, Concordia University | JAX           | **0.1420** |
| 5.       | <details><summary>**Sinv6**</summary>A submission for a task-invariant learned optimizer meta-trained on small tasks.</details>               | Abhinav Moudgil                                                                                                      | Mila, Concordia University | JAX           | **0.0903** |
| 6.       | <details><summary>**AdamG**</summary>A submission based on the AdamG optimizer ([Pang et al., 2024](https://arxiv.org/abs/2405.04376)).</details>               | Yijiang Pang                                                                                                         | Michigan State University  | PyTorch       | **0.0000** |
<!-- END SELF-TUNING LEADERBOARD -->

## How to Submit

To submit your algorithm for evaluation on the AlgoPerf leaderboard, please follow these steps:

1. **Implement your algorithm in the AlgoPerf API:** Have a look at our [Getting Started Guide](https://github.com/mlcommons/algorithmic-efficiency/blob/main/GETTING_STARTED.md) and the [Technical Documentation](https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md).
2. **Create a Pull Request:** Fork this repository, create a new branch and add your submission code to a new folder within either `submissions/external_tuning/` or `submissions/self_tuning`. Open a pull request (PR) to the `evaluation` branch of this repository. Make sure to fill out the PR template asking for information such as submission name, authors, affiliations, etc.
3. **PR Review and Evaluation:** The AlgoPerf working group will review your PR. Based on our available resources and the perceived potential of the method, it will be selected for a free evaluation and merged into the `evaluation` branch. The working group will run your submission on all workloads and push the results, as well as the updated leaderboard, to the `main`branch.

## Citation

<!-- TODO: Replace/add the results paper once it is published. -->
If you use the *AlgoPerf benchmark* results, logs, or code in your research, please consider citing our paper as well as the relevant submissions:

> [Dahl, Schneider, Nado, et al.<br/>
> **Benchmarking Neural Network Training Algorithms**<br/>
> *arXiv 2306.07179*](http://arxiv.org/abs/2306.07179)

```bibtex
@Misc{Dahl2023AlgoPerf,
  title         = {{Benchmarking Neural Network Training Algorithms}},
  author        = {Dahl, George E. and Schneider, Frank and Nado, Zachary and Agarwal, Naman and Sastry, Chandramouli Shama and Hennig, Philipp and Medapati, Sourabh and Eschenhagen, Runa and Kasimbeg, Priya and Suo, Daniel and Bae, Juhan and Gilmer, Justin and Peirson, Abel L. and Khan, Bilal and Anil, Rohan and Rabbat, Mike and Krishnan, Shankar and Snider, Daniel and Amid, Ehsan and Chen, Kongtao and Maddison, Chris J. and Vasudev, Rakshith and Badura, Michal and Garg, Ankush and Mattson, Peter},
  year          = {2023},
  archiveprefix = {arXiv},
  eprint        = {2306.07179},
}
```
