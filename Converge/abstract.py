import theano
from theano import tensor as T

'''
Optimizer interface:
'''
class IOptimizer():

    next_component = None
    
    def __init__(self, next_component, parameters):
        self.next_component = next_component

        for k,v in parameters.items():
            setattr(self,k,v)

    def verify(self):
        return self.valid() and self.next_component.verify()
            
    def process_loss_function(self, loss_function):
        return self.next_component.process_loss_function(loss_function)

    def process_update_function(self, parameters, loss_function):
        return self.next_component.process_update_function(parameters, loss_function)

    def process_data(self, training_data, training_labels):
        return self.next_component.process_data(training_data, training_labels)
    
    def compute_gradient_function(self, parameters, loss_function):
        return self.next_component.compute_gradient_function(parameters, loss_function)
    
    def next_batch(self):
        return self.next_component.next_batch()

    def get_message(self):
        return self.next_component.get_message()
    
    def set_training_data(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

        if self.next_component is not None:
            self.next_component.set_training_data(training_data, training_labels)
        
'''
Base optimizer:
'''
class BaseOptimizer(IOptimizer):

    def __init__(self):
        pass

    def verify(self):
        return True

    def process_loss_function(self, loss_function):
        return loss_function

    def process_update_function(self, parameters, loss_function):
        return []

    def compute_gradient_function(self, parameters, loss_function):
        return T.grad(loss_function, wrt=parameters)
    
    def next_batch(self):
        return self.training_data, self.training_labels

    def get_message(self):
        return None

    def process_data(self, training_data, training_labels):
        return training_data, training_labels
    
