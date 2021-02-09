# Machine Learning Concept
## 1. Activation Function
Activation function is used in the neuron of the hidden layers for mapping the summed weighted input value to the output value.

The activation function can be divided into two types:    
* Linear Activation Function
* Non-linear Activation Function
The linear activation function will turn the neural network into one layer which is a simple linear regression model.

The non-linear activation function allows the model to create complex mappings between the network’s inputs and outputs. The examples are:

## 2. What is sigmoid? 
- Sigmoid Function
- The output value of the Sigmoid function value ranges from (0,1). The function is differentiable and monotonic 

![equation](https://miro.medium.com/max/224/1*DHN75JRJ_EQgGc0spfqLtQ.png)

![figure](https://miro.medium.com/max/600/0*5euYS7InCmDP08ir.)


## 3. What is the difference between sigmoid and Relu?
The sigmoid function suffers from the vanishing gradient problem since 
error is backpropagated through the layers and decreases dramatically 
with each hidden layer.

ReLU takes an input and directly outputs the maximum value of 0 and input. 
ReLU combines the benefits of a linear activation function (no vanishing gradient) 
while allowing for complex relationships to be modeled in the function. Also, it is
easier to compute than the Sigmoid function.


## 4. Why sigmoid is in output function?

The sigmoid function used as the output function decides which value to be passed as an output and which to be passed.
    

## 5. What is loss? 
 - measure of how far a model's predictions are from its label
 - e.g.
   - Mean Squared Error Loss - for linear regression models
   - Log Loss - for logistic regression models

## 6. What is backpropagation? 
 - BP is an algorithm for supervised learning of artificial neural networks using gradient descent.
 - It updates the weights in such a way that minimizes the loss by giving the nodes with higher error rates lower weights.
 

## 7. What are hyper-parameters? 
 - Its values control the learning process. 
 - Other parameters(e.g. nodes weight) are derived from training.
 - Learning rate is an important hyper-parameter
 
## 8. What is learning rate?
- Learning rate is an important hyperparameter. The gradient descent algorithm multiplies the learning rate by the gradient.
- Often in the range between 0.0 and 1.0.
- The learning rate controls how quickly the model is adapted to the problem. 
    - Small learning rates require more training epochs given the smaller changes made to the weights each update
    - large learning rates result in rapid changes and require fewer training epochs.
## 9. What is L2? 
 - it's regularization term
 - regularization: adding the additional term in the loss function to prevent overfitting

## 10. What is purpose of L2?
 - prevents over-fitting
  - we aim for minimization of loss:
     minimize(Loss(Data|Model))
     - now we want to minimize loss+complexity (or so called Structural Risk Minimization)
     minimize(Loss(Data|Model) + complexity(Model))

     loss term - measures how well the model fits the data

     regularization term - measures model complexity

     Model complexity:
      - function of weights of all the features in the model


## 11. What methods help to prevent over-fitting?
 - cross-validation, regularization L2
