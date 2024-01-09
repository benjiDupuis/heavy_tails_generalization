Correct experiments, with noise added in a right way, started the 2023/12/03
experiments with sigma as a scaling factor started the 2023/12/26
experiments with robust mean estimation started the 2023/12/29
Best linear experiment so far: 12_29_14_33
experiments with several seeds estimation started the 2023/12/29


best exp so far: 12-12-16-09-31

todo:
 make the data hyperparameters callable
 vary the width, try to see if the double descent changes
 perturbed regression experiment
 try bigger sigmas


exps to do:
 - vary d ????
 - hinge loss
 - several seeds

First step: Having the experiment working with a linear model, on both types of datasets, in both unparam and overparam regimes

Important idea: divide the gen by the gradients before the alpha_regression
