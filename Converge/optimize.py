import algorithms
import theano
from theano import tensor as T
from abstract import BaseOptimizer
import numpy as np

class Optimizer():

    __loss__ = None
    __update__ = None
    __parameters__ = None
    
    def __init__(self, stack):
        self.stack = stack

    def set_loss_function(self, loss_function):
        self.__loss__ = self.stack.process_loss_function(loss_function)
        
    def set_parameters_to_optimize(self, parameters_to_optimize):
        self.__parameters__ = parameters_to_optimize
        
    def compute_update_function(self, input_params):
        raw_update = self.stack.process_update_function(self.__parameters__, self.__loss__)
        self.__update__ = theano.function(inputs=input_params, outputs=self.__loss__, updates=raw_update)
        self.__loss__ = theano.function(inputs=input_params, outputs=self.__loss__)
    
    def loss(self, validation_data, validation_labels):
        processed_data, processed_labels = self.stack.process_data(validation_data, validation_labels)
        return self.__loss__(processed_data, processed_labels) 
    
    def fit(self, training_data, training_labels, validation_data=None, validation_labels=None):
        self.stack.set_training_data(training_data, training_labels)

        i = 0
        train_loss = 0
        next_batch = self.stack.next_batch()
        while(next_batch is not None):
            i+=1
            
            message = self.stack.get_message()
            if message is not None:
                print(message)

            processed_batch = self.stack.process_data(*next_batch)
            train_loss += self.__update__(*processed_batch)
            
            if i % 100 == 0:
                print(i)
                print(train_loss/100)
                train_loss = 0

            message = self.stack.get_message()
            if message is not None:
                print(message)
            
            next_batch = self.stack.next_batch()

def __from_component(component_name):
    if component_name == "GradientDescent":
        return algorithms.GradientDescent
    
    if component_name == "Minibatches":
        return algorithms.Minibatches

    if component_name == "IterationCounter":
        return algorithms.IterationCounter

    if component_name == "SampleTransformer":
        return algorithms.SampleTransformer

    if component_name == "GradientClipping":
        return algorithms.GradientClipping
    
def __construct_optimizer(settings):
    optimizer = BaseOptimizer()
    for component, parameters in settings:
        optimizer = __from_component(component)(optimizer, parameters)

    #TODO: Better error handling
    if not optimizer.verify():
        print("Construction failed.")
        
    return Optimizer(optimizer)

def build(loss_function, parameters_to_optimize, input_params, settings):
    optimizer = __construct_optimizer(settings)
    
    optimizer.set_loss_function(loss_function)
    optimizer.set_parameters_to_optimize(parameters_to_optimize)
    optimizer.compute_update_function(input_params)
    
    return optimizer


if __name__ == '__main__':
    X = T.matrix('X')
    Y = T.vector('Y')

    W1 = theano.shared(np.random.randn(2,10))
    W2 = theano.shared(np.random.randn(10,1))
    
    hidden = T.tanh(X.dot(W1))
    output = T.nnet.sigmoid(hidden.dot(W2))   
    
    loss = T.nnet.binary_crossentropy(output.transpose(), Y).mean()

    parameters = [('Minibatches', {'batch_size':2, 'contiguous_sampling':False}),
                  ('IterationCounter', {'max_iterations':10}),
                  ('GradientDescent', {'learning_rate':0.5})]
    
    opt = build(loss,[W1,W2],[X,Y], parameters)

    xor_toy_problem = [[1,0],[1,1],[1,1],[0,0],[0,1],[0,1],[1,0],
                       [1,0],[0,0],[0,1],[1,1],[1,0],[1,1],[0,1]]

    xor_toy_labels = [1,0,0,0,1,1,1,
                      1,0,1,0,1,0,1]
    
    print(opt.loss(xor_toy_problem, xor_toy_labels))
    opt.fit(xor_toy_problem, xor_toy_labels)
    print(opt.loss(xor_toy_problem, xor_toy_labels))
    
