import numpy as np

def get_input(
    speed: list[int, int],
    snake_head: list[int, int],
    snake_tail: list[list[int,int]],
    food: list[int,int],
    nb_square: int,
):
    #display(speed, snake_head, snake_tail, food, nb_square)

    head_food = [
        snake_head[0] - food[0] if snake_head[0] - food[0] > 0 else 0, #LEFT
        snake_head[1] - food[1] if snake_head[1] - food[1] > 0 else 0, #UP
        food[0] - snake_head[0] if food[0] - snake_head[0] > 0 else 0, #RIGHT
        food[1] - snake_head[1] if food[1] - snake_head[1] > 0 else 0, #DOWN
    ]
    
    head_wall = [
        snake_head[0], #LEFT
        snake_head[1], #UP
        nb_square - snake_head[0] -1, #RIGHT
        nb_square - snake_head[1] -1, #DOWN
    ]
    
    head_tail = [0]*4
    for encounter in range(snake_head[0]-1, -1, -1):
        if [encounter, snake_head[1]] in snake_tail:
            head_tail[0] =  snake_head[0] - encounter #LEFT
            break
    for encounter in range(snake_head[1]-1, -1, -1):
        if [snake_head[0], encounter] in snake_tail:
            head_tail[1] = snake_head[1] - encounter #UP
            break
    for encounter in range(snake_head[0]+1, nb_square):
        if [encounter, snake_head[1]] in snake_tail:
            head_tail[2] =  encounter - snake_head[0] #RIGHT
            break
    for encounter in range(snake_head[1]+1, nb_square):
        if [snake_head[0], encounter] in snake_tail:
            head_tail[3] =  encounter - snake_head[1] #DOWN
            break
    
    inputs = [head_food, head_wall, head_tail]
    # (nb_permutation to face the direction, index to pop to remove opposite direction) 
    # : [direction]
    permutation = {
        (0,3): [0,1],
        (2,1): [0,-1],
        (0,0): [1,0],
        (1,0): [-1,0],
    }
    
    for key, value in permutation.items():
        for i in range(3):
            if speed == value:
                inputs[i].pop(key[1])
                for _ in range(key[0]):
                    inputs[i].insert(0, inputs[i].pop(-1))
    
    return np.array(inputs).flatten() / nb_square

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