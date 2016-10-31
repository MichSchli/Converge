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
    save_every_n = 1

    def valid(self):
        return self.model_path is not None and self.save_function is not None

    def postprocess(self, loss):
        value_of_next = self.next_component.postprocess(loss)

        if value_of_next == 'stop':
            return 'stop'
        
        if self.iteration % self.save_every_n == 0:
            self.save_function(self.model_path)

        return value_of_next

class TrainLossReporter(IOptimizer):
    evaluate_every_n = 1

    cummulative_loss = 0

    def valid(self):
        return True

    def postprocess(self, loss):
        value_of_next = self.next_component.postprocess(loss)

        if value_of_next == 'stop':
            return 'stop'
        
        self.cummulative_loss += loss

        if self.iteration == 1:
            self.cummulative_loss = 0
            print("Initial loss: "+str(loss))
            return value_of_next
                  
        if self.iteration % self.evaluate_every_n == 1:
            average_loss = self.cummulative_loss / float(self.evaluate_every_n)
            self.cummulative_loss = 0

            begin_iteration = self.iteration - self.evaluate_every_n
            end_iteration = self.iteration - 1
            print("Average train loss for iteration "
                  + str(begin_iteration)
                  + "-"
                  + str(end_iteration)
                  + ": "
                  + str(average_loss))

        return value_of_next

            
class EarlyStopper(IOptimizer):

    criteria = None
    evaluate_every_n = 1
    
    previous_validation_score = None
    burnin = 0
    
    def valid(self):
        if self.criteria is None:
            return False

        if self.criteria == 'score_validation_data' and self.scoring_function is None:
            return False

        if self.criteria == 'score_validation_data' and self.comparator is None:
            return False
        
        return self.evaluate_every_n is not None

    def postprocess(self, loss):
        value_of_next = self.next_component.postprocess(loss)

        if value_of_next == 'stop':
            return 'stop'
        
        if self.iteration % self.evaluate_every_n == 0:
            if self.criteria == 'score_validation_data':
                validation_score = self.scoring_function(self.validation_data)

                print("Tested validation score at iteration "+str(self.iteration)+". Result: "+str(validation_score))
                if self.previous_validation_score is not None:
                    if not self.comparator(validation_score, self.previous_validation_score):
                        if self.iteration > self.burnin:
                            print("Stopping criterion reached.")
                        
                            return 'stop'
                        else:
                            print("Ignoring criterion while in burn-in phase.")

                self.previous_validation_score = validation_score

        return value_of_next
                
