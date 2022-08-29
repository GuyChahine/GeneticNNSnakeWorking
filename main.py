import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from snake import Snake
from ai_architecture import NeuralNetwork, get_input, fitness
from read_write import read_last_generation, write_information, write_last_generation, read_information
from genetic_algorithm import mutate_weights

from os.path import exists
from multiprocessing import Pool
import numpy as np

from utilities import timer, LoadingBar
from time import perf_counter

def NN_playthegame(weights):
    MAX_STEP = 50
    
    snake = Snake(display_mod="training")
    model = NeuralNetwork()
    model.set_weights(weights)
    
    results = snake.get_results()
    while results['alive'] and results['last_eat'] <= MAX_STEP and results['step'] <= 500:
        y = model.predict(get_input(snake.get_info(), snake.nb_square))
        snake.output_handler(y.argmax())
        snake.updater()
        results = snake.get_results()
        
    return (weights, fitness(results['score'], results['step']))

def main(p):
    if exists("models/last_generation.pkl"):
        old_weights = read_last_generation()
        weights = mutate_weights(old_weights)
    else:
        weights = np.array([NeuralNetwork().create_weights(LAYERS) for _ in range(NB_AGENT)])
    
    results = np.array(p.map(NN_playthegame, weights))
    results = results[(results.T[1]).argsort()[::-1]]
    
    write_last_generation(results.T[0])
    write_information(results.T[1])
    
    
    
if __name__ == "__main__":
    NB_AGENT = 128
    LAYERS = [
        (4,""),
        #(8,"relu"),
        #(8,"relu"),
        (4,"sigmoid"),
    ]

    with LoadingBar(50000) as bar:
        with Pool(processes=16) as p:
            for i in range(50000):
                main(p)
                infos = read_information()
                bar(Generation_Number = infos.index[-1], Average_Fitness = np.round(infos.fitness.to_numpy()[-1], 2))
