from abstract import IOptimizer
import random


class GradientDescent(IOptimizer):

    learning_rate = None

    def valid(self):
        return self.learning_rate is not None

    def process_update_function(self, parameters, loss_function):
        gradient = self.compute_gradient_function(parameters, loss_function)

        update_list = self.next_component.process_update_function(parameters, loss_function)
        lower_count = len(update_list)
        
        update_list += [None]*len(gradient)
        for i in range(len(gradient)):
            update_list[lower_count + i] = (parameters[i], parameters[i] - self.learning_rate * gradient[i])

        return update_list

class IterationCounter(IOptimizer):

    max_iterations = None
    iterations = 0

    def valid(self):
        return self.max_iterations is not None

    def next_batch(self):
        if self.iterations < self.max_iterations:
            self.iterations += 1
            return self.next_component.next_batch()
        else:
            return None

class Minibatches(IOptimizer):

    batch_size = None
    contiguous_sampling = None

    current_batch = None

    def valid(self):
        return self.batch_size is not None and self.contiguous_sampling is not None

    def next_batch(self):
        if self.contiguous_sampling:
            return self.__contiguous_sample()
        else:
            return self.__random_sample()
    
    def __contiguous_sample(self):
        if current_batch is None:
            current_batch = self.next_component.next_batch()
        pass

    def __random_sample(self):
        combined = list(zip(self.training_data, self.training_labels))
        sample = random.sample(combined, self.batch_size)
        return zip(*sample)
