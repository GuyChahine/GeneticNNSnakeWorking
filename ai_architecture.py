import numpy as np

def get_input(snake_food_info, nb_square):
    
    snake_x, snake_y = snake_food_info['snake_pos']
    food_x, food_y = snake_food_info['food_pos']

    return np.array([
        # Distance bewtween the snake and the walls in 4 direction
        #snake_y,
        #nb_square - snake_y,
        #nb_square - snake_x,
        #snake_x,
        # Calculate the distance between the food and the snake in 4 direction  
        (food_y - snake_y) if (food_y - snake_y) > 0 else 0,
        (snake_y - food_y) if (snake_y - food_y) > 0 else 0,
        (food_x - snake_x) if (food_x - snake_x) > 0 else 0,
        (snake_x - food_x) if (snake_x - food_x) > 0 else 0,
    ]) / nb_square

class NeuralNetwork():
    
    def __relu(self, x):
        return np.where(x > 0, x, 0)
    
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def create_weights(self, l: list[tuple[int, str]]):
        layers = {}
        for i in range(1,len(l)):
            layers["layer" + str(i)] = {
                "weight": np.random.random((
                        l[i-1][0],
                        l[i][0]
                    ))*2-1,
                "bias": np.zeros(l[i][0]),
                "activation": l[i][1]
            }
        return layers
    
    def set_weights(self, weights: dict[dict[np.array, np.array, str]]):
        self.layers = weights
    
    def predict(self, x):
        for key, value in self.layers.items():
            x = ((x @ value['weight']) + value['bias'])
            x = self.activation_refs[value['activation']](x)
        return x
    
    def __init__(self):
        
        self.activation_refs = {
            "relu": self.__relu,
            "sigmoid": self.__sigmoid,
        }
        
def fitness(score: int, step: int):
    return score