from abstract import IOptimizer



class GradientDescent(IOptimizer):

    learning_rate = None

    def valid(self):
        return self.learning_rate is not None
    
    def compute_update(self, weights, gradient):
        update_list = [None]*len(gradient)
        for i in range(len(gradient)):
            update_list[i] = (weights[i], weights[i] - self.learning_rate * gradient[i])

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
