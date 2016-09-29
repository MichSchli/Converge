

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

    def process_update_function(self, loss_function):
        return self.next_component.process_update_function(loss_function)
    
    def compute_gradient_function(self, weights, loss_function):
        return self.next_component.compute_gradient(weights, loss_function)
    
    def next_batch(self):
        return self.next_component.next_batch()

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

    def process_update_function(self, loss_function):
        return loss_function

    def compute_gradient_function(self, weights, loss_function):
        return T.grad(loss_function, wrt=weights)
    
    def next_batch(self):
        return self.training_data, self.training_labels