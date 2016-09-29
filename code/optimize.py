import algorithms
import theano
from theano import tensor as T
from abstract import BaseOptimizer
import numpy as np

class Optimizer():

    __loss__ = None
    __update__ = None
    
    def __init__(self, stack):
        self.stack = stack

    def set_loss_function(self, loss_function):
        self.__loss__ = self.stack.process_loss_function(loss_function)
        self.__update__ = self.stack.process_update_function(loss_function)

    def loss(self, validation_data, validation_labels):
        return self.__loss__(validation_data, validation_labels)
    
    def fit(self, training_data, training_labels, validation_data=None, validation_labels=None):
        self.stack.set_training_data(training_data, training_labels)
        
        next_batch = self.stack.next_batch()
        while(next_batch is not None):
            self.__update__(*next_batch)
            next_batch = self.stack.next_batch()

def __from_component(component_name):
    if component_name == "GradientDescent":
        return algorithms.GradientDescent
    
    if component_name == "minibatches":
        return algorithms.Minibatches

    if component_name == "IterationCounter":
        return algorithms.IterationCounter

def __construct_optimizer(settings):
    optimizer = BaseOptimizer()
    for component, parameters in settings:
        optimizer = __from_component(component)(optimizer, parameters)

    #TODO: Better error handling
    if not optimizer.verify():
        print("Construction failed.")
        
    return Optimizer(optimizer)

def build(loss_function, settings):
    optimizer = __construct_optimizer(settings)
    optimizer.set_loss_function(loss_function)

    return optimizer


if __name__ == '__main__':
    X = T.vector('X')
    Y = T.vector('Y')

    a = theano.shared(np.ones(5))
    
    loss = (a*X - Y).mean()

    parameters = [('IterationCounter', {'max_iterations':5}), ('GradientDescent', {'learning_rate':0.1})]
    opt = build(theano.function(inputs=[X,Y], outputs=loss), parameters)
    
    print(opt.loss([1,1,1,1,1], [5,1,4,5,1]))
    opt.fit([1,1,1,1,1], [5,1,4,5,1])
    print(opt.loss([1,1,1,1,1], [5,1,4,5,1]))
    
