# ISYE Summer Scholars Project with Dr. Huo Ming and Ziyan Wang
This repo is a combination of multiple approaches to get an understanding of meta-reinforcement learning. Most of the work is compiled from these sources mentioned below.
##

## Pytorch Tutorials
The following files included in this folder were adaptions of [Pytorch Tutorials](https://github.com/yunjey/pytorch-tutorial/tree/master) and [Aladdin Persson's Pytorch Introduction]{https://github.com/aladdinpersson/Machine-Learning-Collection}. These files serve as a foundational building block to understand deep learning, and pytorch is one of the commonly used tools for this. The module contains its own tensors, a datatype similar to numpy data arrays, and automatic differentiation engine. Each of the files was picked through the link mentioned above, adapted, annotated with comments, and implemented to cover a wide range of concepts used in future sections. At the most base level, Pytorch Basics is the most generic introduction of the the pytorch module that covers the creation of tensors, one step gradient descents, and pre-trained models. Then, the next files cover their namesakes: Bidirectional LSTMs (Long Short-Term Memory), CNN (Convolution Neural Networks), and a basic neural network example using the MNIST dataset.
##

## Torchopt
While PyTorch is a general-purpose deep learning framework that provides tools for building, training, and deploying neural networks, TorchOpt, on the other hand, is a specialized library built on top of PyTorch, focusing on advanced optimization techniques for meta-learning and reinforcement learning, offering state-of-the-art optimizers and tools for optimization research.

### MAML 
MAML (Model-Agnostic Meta-Learning) is a meta-learning framework designed to enable models to quickly adapt to new tasks with only a few training examples based on this [paper](https://github.com/metaopt/torchopt/tree/main/examples/MAML-RL). It optimizes model parameters such that a small number of gradient updates on new tasks yield good performance. The meta-objective involves training these parameters to ensure that, after a few gradient steps on new task-specific losses, the updated parameters perform well. This meta-optimization process is carried out using stochastic gradient descent (SGD), where gradients of the loss function are computed with respect to the model parameters after gradient descent on task-specific losses.

[testprevspost](/pictures/Test Pre vs Post Reward.png)

[trainprevspost](/pictures/Train Pre vs Post Reward.png)

The key advantage of MAML lies in its versatility and efficiency across different domains, including regression, classification, and reinforcement learning. In regression, MAML demonstrated its ability to adapt to new tasks quickly by fitting sine waves with minimal data. In classification tasks, using datasets like Omniglot and MiniImagenet, MAML showed significant improvements in few-shot learning scenarios, rapidly adapting to new classes with limited data. For reinforcement learning, MAML was tested on standard benchmark environments and proved capable of efficiently adapting policies for new tasks, performing comparably or better than baseline methods with minimal fine-tuning. The MAML model was found through [this github](https://github.com/metaopt/torchopt/tree/main/examples/MAML-RL).

### iMAML
Implicit Model-Agnostic Meta-Learning ([iMAML](https://arxiv.org/abs/1909.04630)) enhances the MAML framework by addressing issues related to computational efficiency and stability in second-order optimization. The goal is to address the limitations of Model-Agnostic Meta-Learning (MAML), which struggles with high computational and memory demands due to high-order derivatives and issues like vanishing gradients, especially with many inner loop optimization steps or larger datasets. Implicit MAML (iMAML) improves upon MAML by adding a regularization term to the inner loop optimization. This adjustment helps mitigate vanishing gradients and reduces computational burdens, making the training process more scalable and efficient. iMAML improves upon MAML by offering a more accurate approximation of the meta-gradient, requiring fewer computational steps and less memory. It achieves better performance and efficiency, as demonstrated by its superior results on benchmarks like Omniglot. Overall, iMAML computes more precise meta-gradients with reduced computational costs and performs better in few-shot learning tasks.

[loss](/pictures/Train Loss vs Time.png)

[ACC](/pictures/Accuracy Over Epochs.png)

This script sets up and runs a meta-learning experiment using the iMAML algorithm with the Omniglot dataset, parsing command-line arguments to configure hyperparameters, initializing a CNN, and employing the Adam optimizer for meta-training and evaluation. It loops through training and testing phases to optimize the model's performance on few-shot learning tasks, aiming to demonstrate whether iMAML enhances the model's accuracy over epochs. High accuracy on the test set would indicate successful generalization to new tasks with limited samples, while evaluating hyperparameters would offer insights for further optimization and improvements. This model is found through [this github](https://github.com/metaopt/torchopt/tree/main/examples/iMAML).

### MGML
Meta-gradient Reinforcement Learning (MGRL) enhances traditional reinforcement learning by optimizing the learning algorithm itself. MGRL, [as explained here](https://github.com/metaopt/torchopt/tree/main/examples/MGRL), is a sophisticated reinforcement learning technique that optimizes learning across multiple tasks by refining the return function. In MGML, the return function is treated as a parametric function with tunable meta-parameters. Key among these parameters are the discount factor (γ), which adjusts the time scale of the return with a value of 1 representing a long-term perspective, and the bootstrapping parameter (θ), which determines the geometric combination of returns over various time scales. By fine-tuning these meta-parameters, MGML enhances the agent's ability to generalize and adapt to different tasks, leading to improved performance across a range of environments. The model code is found through [here](https://arxiv.org/abs/1805.09801).

[discount](/pictures/Learned Discount Factors Over Time.png)

Within the paper, the evaluation of MGML involves testing agents on 57 different Atari games using two distinct protocols. The **Human Starts** protocol initializes episodes to states sampled from human gameplay, providing a realistic starting point for evaluation. In contrast, the **No-ops Starts** protocol begins each episode with a random sequence of no-op (no operation) actions, a method also used during training to ensure consistency. All hyper-parameters, such as batch size, unroll length, learning rate, and entropy cost, are kept consistent with previous studies to ensure a fair comparison. This rigorous evaluation demonstrates MGML's ability to enhance policy performance and generalization in reinforcement learning tasks.

##

## Deep Learning
The deep learning file contains an adaption and implementation of the hugging face space invader simulation to mimic the meta gradient reinforced learning before. The ipynb file is uploaded and viewable [here](https://huggingface.co/alz1319/dqn-SpaceInvadersNoFrameskip-v4) on the HuggingFace website. 
