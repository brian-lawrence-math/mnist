# Training an MNIST classifier with a custom optimizer

This is a simple experiment to see whether I can do a simple machine learning task
(training an MNIST classifier)
with a custom optimizer that only using the sign of each gradient.

See my blog post about Adam (link to be added)
for an in-depth discussion of why I wanted to try out this optimizer.

## Running the notebook

The MNIST dataset is not included in this repo.
You will need to download it from Kaggle.
Go to [Kaggle's MNIST page](https://www.kaggle.com/datasets/hojjatk/mnist-dataset),
click "Download" in the upper right, and click "Download dataset as zip".
Save the dataset in the repo folder and unzip it.

## About MNIST and the model

MNIST is a computer vision dataset consisting of bitmaps of handwritten digits;
the challenge is to train a classifier to recognize the digit (0-9) from the bitmap.
See [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) for more information on the dataset.

The model is a bog-standard convolutional neural network with ~60K parameters. I use:

- hierarchical downsampling with max pooling
- batch normalization and dropout for regularization
- a total of 8 convolutional layers.

## The experiment

I train an MNIST classifier using three optimizers:
my custom optimizer GradSign, stochastic gradient descent, and Adam.

All three training runs use the same model architecture.
I can't use the same learning rate, since the scale of the learning rate is different for each optimizer.
Instead, for each optimizer I sweep over some reasonable values for the learning rate, and choose the best one.

In each iteration, I train on 1000 batches with a batch size of 32.
(I'm deliberately testing the optimizers on an unusually short training run:
I want to test whether the optimizer can achieve good performance quickly.)

| Optimizer | Best LR | Accuracy |
| --------- | ------- | -------- |
| GradSign  | 0.003   | 98.29 %  |
| Adam      | 0.003   | 98.2 %   |
| SGD       | 0.03    | 98.6 %   |

With this architecture and hyperparameters, all three optimizers get good results (over 98% accuracy on the validation set).
The relative performance among the three seems to be dependent on randomness (see the section on "reproducibility" below).

## Odds and ends

Learning rate scale: The best learning rates turn out to be the same for GradSign and Adam, and very different for SGD.
See my blog post (link to be added) for some speculation as to why:
in short, the learning rate has the same units for GradSign as for Adam.

Why doesn't Adam work better? Adam is supposed to outperform SGD for large, complex models.
I suspect this simple 60K-parameter convolutional network just isn't big enough for Adam to shine:

- Adam does well when the scale of the gradient varies across layers. But this is a small network, and activations are more or less held to the same scale by batch normalization.
- Adam handles sparse gradients well; I expect sparse gradients to be a bigger problem in larger networks, and in more complex datasets.

## Reproducibility

The notebook you see on branch 'public' was run on an older x86 Macbook Pro,
from a Conda environment running Python 3.12.7 and
Pytorch 2.3.1.
Just for fun, I repeated the experiment on a newer Linux machine
(Python 3.14, Pytorch 2.11), and the results were substantially different.
(See branch 'linux'.)
For example, GradSign now achieves 98.4% accuracy on the test data, up from 98.29%.

I did some sleuthing, and surprisingly, the discrepancy seems to be entirely due to numerical instability.
Both systems iterate through the training data in the same order,
and they initialize the models to the same weights.
I hunted down the first discrepancy in the GradSign training run:
A single gradient was computed as +8.0e-8 on one system, and -2.5e-8 on the other.
The two versions run two different CPU kernels,
with algorithms that are mathematically equivalent but numerically different.
Over a long training run, those differences propagate,
resulting in a discrepancy of 0.1% or more in the final evaluation.
