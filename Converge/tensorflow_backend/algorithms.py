from abstract import IOptimizer
import numpy as np
import tensorflow as tf

class GradientDescent(IOptimizer):

    learning_rate = None

    def valid(self):
        return self.learning_rate is not None

    def process_update_function(self, gradient_function, parameters_to_optimize):
        opt_func = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = opt_func.apply_gradients(zip(gradient_function, parameters_to_optimize))

        return optimizer

class Adam(IOptimizer):

    learning_rate = None
    historical_gradient_weight = 0.9
    historical_moment_weight = 0.999

    def valid(self):
        return self.learning_rate is not None

    def process_update_function(self, gradient_function, parameters_to_optimize):
        opt_func = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                          beta1=self.historical_moment_weight,
                                          beta2=self.historical_gradient_weight)
        optimizer = opt_func.apply_gradients(zip(gradient_function, parameters_to_optimize))

        return optimizer

class AdaGrad(IOptimizer):

    learning_rate = None
    
    def valid(self):
        return self.learning_rate is not None

    def process_update_function(self, gradient_function, parameters_to_optimize):
        opt_func = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        optimizer = opt_func.apply_gradients(zip(gradient_function, parameters_to_optimize))

        return optimizer

    
class GradientClipping(IOptimizer):

    max_norm = None

    def valid(self):
        return self.max_norm is not None

    def process_gradient_function(self, loss_function, parameters_to_optimize):
        gradient = self.next_component.process_gradient_function(loss_function, parameters_to_optimize)
        clipped,_ = tf.clip_by_global_norm(gradient, self.max_norm)
        return clipped


class ModelSaver(IOptimizer):

    model_path = None
    save_function = None

    def valid(self):
        return self.model_path is not None and self.save_function is not None

    def postprocess(self, variables):
        self.next_component.postprocess(variables)
        self.save_function(self.model_path, self.session.run(variables))

#TODO
class EarlyStopper(IOptimizer):

    criteria = None
    evaluate_every_n = 1
    use_development_data = True

    iteration = 0
    
    def valid(self):
        return self.criteria is not None

    def postprocess(self, variables):
        self.next_component.postprocess(variables)
        pass
