Correct experiments, with noise added in a right way, started the 2023/12/03
experiments with sigma as a scaling factor started the 2023/12/26
experiments with robust mean estimation started the 2023/12/29
Best linear experiment so far: 12_29_14_33
experiments with several seeds estimation started the 2023/12/29
started correct gradient computation on the (on the whole iterations) on the 2024/01/14


best exp so far: 12-12-16-09-31

Experiments to remember for the paper:
 - linear on MNIST: 2023-12-31_18_14_37
    - sigma_regression: using gradients helps for the correlation, the raw stuff has the good values but not that much correlation
 - n_params evolution on MNIST: 2024-01-04_10_22_56
 - gaussians sigma regression: 2024-01-09_15_07_42
 - gaussians dim regression: 2024-01-09_16_10_06
 - 3 layers MNIST: 2024-01-04_15_06_05
 - gaussian: 2024-01-09_19_12_56 can be used to argue that estimating gradients on the whole stuff does not change much things



experiments MNIST used in final results:
 - sigma varies: 2024-01-14_14_57_14
 - d varies: 2024-01-14_16_36
 - sigma_plot (depth 0): 2023-12-31_18_14_37

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
