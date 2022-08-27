import pygame, sys, random
from read_write import read_last_generation
from ai_architecture import NeuralNetwork, get_input

class Snake():
    
    # Handle all the type of event
    def event_handler(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # If a key is pressed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == ord('z'):
                    self.speed = [0,0]
                    self.speed[1] = 1
                elif event.key == pygame.K_DOWN or event.key == ord('s'):
                    self.speed = [0,0]
                    self.speed[1] = -1
                elif event.key == pygame.K_RIGHT or event.key == ord('d'):
                    self.speed = [0,0]
                    self.speed[0] = 1
                elif event.key == pygame.K_LEFT or event.key == ord('q'):
                    self.speed = [0,0]
                    self.speed[0] = -1
    
    # Handle output of NN
    def output_handler(self,output):
        # NN want to go UP
        if output == 0:
            self.speed = [0,0]
            self.speed[1] = 1
        # NN want to go DOWN
        elif output == 1:
            self.speed = [0,0]
            self.speed[1] = -1
        # NN want to go RIGHT
        elif output == 2:
            self.speed = [0,0]
            self.speed[0] = 1
        # NN want to go LEFT
        elif output == 3:
            self.speed = [0,0]
            self.speed[0] = -1
    
    # Change the position of the snake based on the speed
    def speed_management(self):
        
        # For x axis
        if self.speed[0] == 1:
            self.snake_pos[0] += self.square_size
        elif self.speed[0] == -1:
            self.snake_pos[0] -= self.square_size
            
        # For y axis
        if self.speed[1] == 1:
            self.snake_pos[1] -= self.square_size
        elif self.speed[1] == -1:
            self.snake_pos[1] += self.square_size
    
    # Update position of the snake and the food
    def gfx_updater(self):
        
        # Refill the frame with black
        self.game_window.fill(pygame.Color(0,0,0))
        # Draw or redraw the square of the food
        pygame.draw.rect(self.game_window, pygame.Color(255,0,0), pygame.Rect(self.food_pos[0],self.food_pos[1],self.square_size,self.square_size))
        # Draw or redraw the square of the snake
        pygame.draw.rect(self.game_window, pygame.Color(255,255,255), pygame.Rect(self.snake_pos[0],self.snake_pos[1],self.square_size,self.square_size))
    
    # Generate the position of the food
    def generate_pos_food(self):
        self.food_pos = [random.randint(0,self.nb_square-1)*self.square_size,random.randint(0,self.nb_square-1)*self.square_size]
    
    # Check if the snake collide with a food or a wall
    def colision_checker(self):
        
        # Check if the snake enter in collision with the food
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            # Set the last eat to 0 step
            self.last_eat = 0
            self.generate_pos_food()
            
        # Check if the snake enter in collision with a border
        if self.snake_pos[0] < 0 or self.snake_pos[0] > (self.nb_square*self.square_size)-self.square_size:
            self.alive = False
        elif self.snake_pos[1] < 0 or self.snake_pos[1] > (self.nb_square*self.square_size)-self.square_size:
            self.alive = False
    
    # Main logic
    def updater(self):
        
        # Increment number of step
        self.step += 1
        # Increment last eat
        self.last_eat += 1
        # Change the position of the snake based on the speed
        self.speed_management()
        # Check if the snake collide with a food or a wall
        self.colision_checker()
    
    # Fucntion to get result of this snake
    def get_results(self):
        return {'alive':self.alive, 'score':self.score, 'step':self.step, 'last_eat':self.last_eat}
    
    # Get information for on the game    
    def get_info(self):
        return {
            'snake_pos':[self.snake_pos[0]//self.square_size,self.snake_pos[1]//self.square_size],
            'food_pos':[self.food_pos[0]//self.square_size,self.food_pos[1]//self.square_size]
        }
    
    def __init__(
        self,
        display_mod=False,
        nb_square=20,
        square_size=30,
        refresh_time=10
    ):
        
        # Number of square on the grid
        self.nb_square = nb_square
        # Size of one square on the grid
        self.square_size = square_size
        # Refresh rate of the game
        self.refresh_time = refresh_time
        # Size of the window
        self.frame_size = (self.nb_square*self.square_size,self.nb_square*self.square_size)
        # Starting pos of the snake
        self.snake_pos = [(self.nb_square//2)*self.square_size,(self.nb_square//2)*self.square_size]
        # Initialise the speed of the snake
        self.speed = [1,0]
        # Initialise alive var
        self.alive = True
        # Initialise score of the game
        self.score = 0
        # Initialise step spend on the game
        self.step = 0
        # Initialise last eat of the snake
        self.last_eat = 0
        # Generate the position of the first food
        self.generate_pos_food()
        
        
        # Display mod not set
        
        if display_mod == 'training':
            pass

        elif display_mod == 'playing':
            # Give to var errors the errors
            self.errors = pygame.init()
            # Initialise the window of the game
            self.game_window = pygame.display.set_mode(self.frame_size)
            
            # While the snake is alive
            while self.alive:
                
                # Handle all the type of event
                self.event_handler()
                # Main logic
                self.updater()
                # Update position of the snake and the food
                self.gfx_updater()
                # Update the display
                pygame.display.update()
                # Set the refresh rate of the game
                pygame.time.Clock().tick(self.refresh_time)
        
        elif display_mod == 'testing':
            
            weights = read_last_generation()[-1]
            model = NeuralNetwork()
            model.set_weights(weights)
            
            # Give to var errors the errors
            self.errors = pygame.init()
            
            # Initialise the window of the game
            self.game_window = pygame.display.set_mode(self.frame_size)

            # While the snake is alive
            while self.alive:
                
                # Handle all the type of event
                self.event_handler()
                
                x = get_input(self.get_info(), self.nb_square)
                y = model.predict(x)
                
                self.output_handler(y.argmax())
                
                # Main logic
                self.updater()
                
                # Update position of the snake and the food
                self.gfx_updater()
                
                # Update the display
                pygame.display.update()

                # Set the refresh rate of the game
                pygame.time.Clock().tick(self.refresh_time)

        else:
            print("Display mod not set")

if __name__ == "__main__":
    snake1 = Snake(
        display_mod='testing',
    )