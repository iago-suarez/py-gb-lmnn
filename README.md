# py-gb-lmnn
## Python implementation of "Non-linear Metric Learning"

![Demo of figure 2](doc/demo_figure2.png) 

Non-linear Metric Learning shows a powerful way to solve metric-learning and other related problems like dimensionality reduction using gradient-boosting. This approach is robust, parallelizable, much faster than deep learning and require less training data.
I provide an updated implementation using the SciPy ecosystem.

The code is based on the [original matlab code](https://github.com/gabeos/lmnn). I am very grateful to Killian Weinberger for his help. Please refer to the original paper for details:

> [Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012). Non-linear metric learning. In Advances in neural information processing systems (pp. 2573-2581).](https://papers.nips.cc/paper/4840-non-linear-metric-learning)


## Installation

Please use Python 3.7 and install the pip dependencies as:

```
pip install -r requirements.txt
```

## Usage

```
python main.py
```

