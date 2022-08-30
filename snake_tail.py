from turtle import pos
import pygame
import sys
from random import randint
import numpy as np

from ai_architecture import NeuralNetwork, get_input
from read_write import read_last_generation

class Snake():
    
    def __event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == ord('z') and self.speed != [0,-1]:
                    self.key = [0,1]
                elif event.key == pygame.K_DOWN or event.key == ord('s') and self.speed != [0,1]:
                    self.key = [0,-1]
                elif event.key == pygame.K_RIGHT or event.key == ord('d') and self.speed != [-1,0]:
                    self.key = [1,0]
                elif event.key == pygame.K_LEFT or event.key == ord('q') and self.speed != [1,0]:
                    self.key = [-1,0]
                    
                elif event.key == ord("i"):
                    self.output_handler(1)
                elif event.key == ord("j"):
                    self.output_handler(0)
                elif event.key == ord("l"):
                    self.output_handler(2)
    
    def output_handler(self, output):
        for key, value in self.output_converter.items():
            if output == key:
                for i in range(4):
                    if self.speed == value[i][0]:
                        self.key = value[i][1]
                        break
    
    def __speed_management(self):
        self.speed = self.key[:]
        
        if self.speed[0] == 1:
            self.snake_pos.insert(0, [
                self.snake_pos[0][0] + self.square_size,
                self.snake_pos[0][1],
            ])
        elif self.speed[0] == -1:
            self.snake_pos.insert(0, [
                self.snake_pos[0][0] - self.square_size,
                self.snake_pos[0][1],
            ])
        elif self.speed[1] == 1:
            self.snake_pos.insert(0, [
                self.snake_pos[0][0],
                self.snake_pos[0][1] - self.square_size,
            ])
        elif self.speed[1] == -1:
            self.snake_pos.insert(0, [
                self.snake_pos[0][0],
                self.snake_pos[0][1] + self.square_size,
            ])
        self.snake_pos.pop(-1)
    
    def __colision_checker(self):
        if self.snake_pos[0][0] == self.food_pos[0][0] and self.snake_pos[0][1] == self.food_pos[0][1]:
            self.score += 1
            self.last_eat = 0
            self.__generate_pos_food()

        for tail in self.snake_pos[1:]:
            if self.snake_pos[0] == tail:
                self.alive = False
                
        if self.snake_pos[-1][0] == self.food_pos[-1][0] and self.snake_pos[-1][1] == self.food_pos[-1][1]:
            self.snake_pos.append(self.snake_pos[-1])
            self.food_pos.pop(-1)
        
        if self.snake_pos[0][0] < 0 or self.snake_pos[0][0] > (self.nb_square*self.square_size)-self.square_size:
            self.alive = False
        elif self.snake_pos[0][1] < 0 or self.snake_pos[0][1] > (self.nb_square*self.square_size)-self.square_size:
            self.alive = False
    
    def updater(self):
        self.step += 1
        self.last_eat += 1
        self.__speed_management()
        self.__colision_checker()
    
    def __gfx_updater(self):
        
        self.game_window.fill(pygame.Color(0,0,0))
        
        for body in self.snake_pos:
            pygame.draw.rect(
            self.game_window,
            pygame.Color(255,255,255),
            pygame.Rect(
                body[0], body[1],
                self.square_size, self.square_size,
            ),
        )
            
        pygame.draw.rect(
            self.game_window,
            pygame.Color(255,0,0),
            pygame.Rect(
                self.food_pos[0][0], self.food_pos[0][1],
                self.square_size, self.square_size,
            ),
        )
    
    def __generate_pos_food(self):
        possibilities = np.array(np.meshgrid(
                np.arange(self.nb_square), np.arange(self.nb_square)
            )).T.reshape(-1,2)
        scaled_snakepos = [[
                value[0] // self.square_size,
                value[1] // self.square_size,
            ] for value in self.snake_pos]
        possibilities = np.delete(possibilities, np.where([((value == scaled_snakepos).all(axis=1)).any() for value in possibilities]), axis=0)
        self.food_pos.insert(0, list(
            possibilities[randint(0, possibilities.shape[0]-1)]*self.square_size
        ))
    
    def get_results(self):
        return {'alive':self.alive, 'score':self.score, 'step':self.step, 'last_eat':self.last_eat}
    
    def get_info(self):
        #print(self.food_pos)
        return [
            self.speed,
            [
                self.snake_pos[0][0] // self.square_size,
                self.snake_pos[0][1] // self.square_size,
            ],
            [
                [
                    tailpos[0] // self.square_size,
                    tailpos[1] // self.square_size, 
                ] for tailpos in self.snake_pos[1:]
            ],
            [
                self.food_pos[0][0] // self.square_size,
                self.food_pos[0][1] // self.square_size,
            ],
            self.nb_square,
        ]
    
    def __init__(
        self,
        display_mod: str = False,
        nb_square: int = 20,
        square_size: int = 30,
        refresh_time: int = 10,
    ):
        
        self.nb_square = nb_square
        self.square_size = square_size
        self.refresh_time = refresh_time
        
        self.output_converter = {
            0: [
                [[0,1], [-1,0]],
                [[0,-1], [1,0]],
                [[1,0], [0,1]],
                [[-1,0], [0,-1]],
            ],
            1: [
                [[0,1], [0,1]],
                [[0,-1], [0,-1]],
                [[1,0], [1,0]],
                [[-1,0], [-1,0]],
            ],
            2: [
                [[0,1], [1,0]],
                [[0,-1], [-1,0]],
                [[1,0], [0,-1]],
                [[-1,0], [0,1]],
            ],
        }
        
        self.frame_size = (self.nb_square * self.square_size, self.nb_square * self.square_size)
        self.snake_pos = [[self.nb_square // 2 * self.square_size]*2]
        self.food_pos = []
        self.key, self.speed = [0,1], [0,1]
        self.alive = True
        self.score, self.step, self.last_eat = [0]*3
        self.__generate_pos_food()
        
        if display_mod == "training":
            pass
        
        elif display_mod == "playing":
            
            errors = pygame.init()
            self.game_window = pygame.display.set_mode(self.frame_size)
            while self.alive:
                self.__event_handler()
                self.updater()
                self.__gfx_updater()
                pygame.display.update()
                pygame.time.Clock().tick(self.refresh_time)
        
        elif display_mod == "testing":
            weights = read_last_generation()[0]
            model = NeuralNetwork()
            model.set_weights(weights)
            
            errors = pygame.init()
            self.game_window = pygame.display.set_mode(self.frame_size)
            while self.alive and self.last_eat < 400:
                self.__event_handler()
                
                x = get_input(*self.get_info())
                y = model.predict(x)
                self.output_handler(y.argmax())
                
                self.updater()
                self.__gfx_updater()
                pygame.display.update()
                pygame.time.Clock().tick(self.refresh_time)
        
        else:
            print("No Display Mode Set")

if __name__ == "__main__":
    snake = Snake(display_mod="testing", refresh_time=100)