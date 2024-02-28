# OpenAIsaac

**OpenAIsaac** (get it?) is a repo with projects I'm working on to reinforce my understanding of machine learning.

My current area of focus is on **Large Language Models**.

## MLP

This directory is (by default) a random name generator (can be used with any decent raw input text) that generates similar yet unique names as the input data. The MLP neural architecture used is based on the research paper - [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Pytorch is used to make the neural architecture.

MLP is a relatively simple architecture. In this project, it consists of an input layer, a single hidden layer, and an output layer.

#### Here's an example of what the model's output looks like comparing the trained and untrained weights:

![image](/mlp/output.png)

#### Here you can see that without training the goofball just spits out a bunch of random letters, but after training, it spits out words that are much more name-like.

### To run the model locally:

1. Pull down the repo, and run `pip install -r requirements.txt` in the mlp directory. You'll need to be using a version of python that is compatible with Pytorch (like version 3.7).
2. Run `python ./run.py` to train the model and see it generate some predictions. The weights are cached, so subsequent runs won't take as long. You can also run `python ./run.py --compare` to see the untrained predictions compared to the trained predictions, which is kinda neat.

### A general overview of the training process is as follows:

- **Data Preparation**: The dataset of raw words is transformed into simple, sequential input tokens, and split into a training, validation, and test set (80% training, 10% val, 10% test) with inputs and labels. Each input is a three-token vector, while the corresponding label is the next token that follows the input vector. The '.' token is used to signify the start and end of a word.

- **Weight Initialization**: The next step is initializing the weights for each layer. There is an Input layer, a single Hidden layer, and the final Output layer. The input layer is a 10-dimensional embedding table (each unique character/token in the dataset is assigned to a 10 dimension embedding). The hidden layer consists of 200 neurons, each receiving 30 features (since there are 3 input tokens each with 10-dimensional embedding). Biases are randomly initialized. The output layer works the same way and outputs logits that have the original shape of the vocab size. Each logit is a raw non-converted probability of what the next token should be.

- **Training Process**: Finally, in the actual training process - After the forward pass as described above, the final logits are passed through an activation function (softmax) to obtain the probability distribution of each token. The loss is calculated by comparing these probabilities to the expected output labels. During the backward pass, gradients are calculated to determine how changes in each weight and bias would affect the total loss. This is done for each layer in the network, allowing the model to understand how to adjust its parameters to reduce the loss. The learning rate is initially set to 0.1 and is reduced to 0.01 for the second half of the training iterations to adjust the weights of each layer.
