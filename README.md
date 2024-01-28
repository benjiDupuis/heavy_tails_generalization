# Code for our experimental results

We provide the code base that we used to produce our main experiments and figures. All our implementation has been done in python, tested with python `3.11.7``.

## Installation

You can create a virtual environment and install the required packages through the provided `requirements.txt` file.
````
python -m pip install -r requirements.txt
````

| :memo:        | This will install a cpu-only version of pytorch, feel free to install torch and torchvision according to your CUDA version.        |
|---------------|:------------------------|


## Tests

We provide some pytests to assess that the installation went well>
````
pytest -x -vv
````
All tests should pass.


## Simulation
All data related scripts may be found in `data`. Models related scripts are in `last_point/model.py`.

The main simulation is implemented in `last_point/simulation.py`. It simulates the Euler Maruyama discretization of the equation studied in our paper. All the parameters are explained in the script itself. 

$$
\widehat{W_{k+1}^S} = \widehat{W_{k+1}^S} - \gamma \nabla V_S(\widehat{W_{k+1}^S}) + \sigma_1 \gamma^{\frac{1}{\alpha}} L_1^\alpha,
$$
with (see the paper for all the notations):
$$
V_S(w) = \widehat{F_S}(w) + \frac{\eta}{2} \Vert w \Vert^2.
$$

We parallelized our experiments in order to use both a $10 \times 10$ grid of hyperparameters and $10$ random seeds. For the sake of this submission, we also implented, in `last_point/__main__.py`, a loop over the seed and hyperparameters. The syntax is:
````
python -m last_point --result_dir [RESULT_DIR] --horizon [NUMBER OF ITERATIONS] --data_type [TRAINING DATASET] 
````
The provided code also allow to act on the range of variation of $(\sigma_!, \alpha)$ and the width of the network, and also on the number of random seeds. Note that, in our implementation, `depth=1` corresponds to a 2 layers fully connected network, and `depth=0` corresponds to a linear model. You can also act on vaious hyperparameters (learning rate,...) and subsample the training dataset by a certain proportion using the `subset` parameter. 


To test that the installation is working, you can run the following command:
````
python -m last_point --subset 0.001 --horizon 3 --n_ergodic 4 
````
It will run a toy example by iterating on only $7$ iterations on $0.1\%$ of the MNIST dataset. If the installation was correct, a folder `results` should appear, containing the `JSON` files produced by the simulation.


## Results examples
The folder `analysis` contains code to produce figures similar as the ones that can be found in the paper. In order to illustrate those scripts, we provide examples result files from our experiments. They correspond to the training of a $2$ layers neural network on $10\%$ of the MNIST dataset (see the paper for hyperparameters details), with varying tail-index and width, on $10$ random seeds.

To compute the estimated bound and plots the averaged accuracy with respect to $\alpha$, you can run:
```
python analysis/plot_ome_seed.py result_example/average_results.json .
```
To compute the correlation (Kendall's $\tau$) between the accuracy gap and $\alpha$, run:
```
python analysis/multiple_kendalls.py result_example/all_results.json .
```
You can also use the argument `--av_path` to obtain the black curve shown in the paper.
To compute the regression of $\alpha$ from the accuracy gap, run:
```
python analysis/regressions.py result_example/average_results.json .
```