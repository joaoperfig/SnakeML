import random
import math
import time
import msvcrt
import datetime
import glob
import tflearn
import numpy as np
import matplotlib.pyplot as plt

class TFAgent:
    
    def __init__(self, model):
        self.snakeGame = None
        self.model = model
        self.lastDirection = False
        self.move_threshold = 0.25
        
    def getDirection(self):
        observations = self.snakeGame.get_observations()
        predict = self.model.predict((np.array(observations)).reshape(-1, 9, 1))
        predict = predict[0]
        if abs(predict[0]) < self.move_threshold and abs(predict[1]) < self.move_threshold:
            direct = False
        elif abs(predict[0]) > abs(predict[1]):
            # predict[0] is furthest from 0, so predict[1] will be 0
            if predict[0] > 0:
                direct = (1, 0)
            else:
                direct = (-1, 0)
        else:
            # predict[1] is furthest from 0, so predict[0] will be 0
            if predict[1] > 0:
                direct = (0, 1)
            else:
                direct = (0, -1)
        self.lastDirection = direct
        print("AI says: "+str(direct)+" (from "+str(predict)+")")
        return direct
    
    def getLastDirection(self):
        return self.lastDirection
            

class KeyboardAgent:
    
    def __init__(self):
        self.snakeGame = None #the game assigns itself here
        self.lastDirection = False
    
    def getDirection(self):
        if not msvcrt.kbhit():
            direc = False
        else:
            char = msvcrt.getch()
            if char == b'a':
                direc = Direction.left
            elif char == b'w':
                direc = Direction.up
            elif char == b'd':
                direc = Direction.right
            elif char == b's':
                direc = Direction.down
            else:
                direc = False
        self.lastDirection = direc
        return direc

    def getLastDirection(self):
        return self.lastDirection

class TrainingAgent:

    def __init__(self):
        self.snakeGame = None
        self.lastDirection = False

    def getDirection(self):
        dir_number = random.randint(0, 4)
        if dir_number == 1 and self.lastDirection != Direction.right:
            direc = Direction.left
        elif dir_number == 2 and self.lastDirection != Direction.down:
            direc = Direction.up
        elif dir_number == 3 and self.lastDirection != Direction.left:
            direc = Direction.right
        elif dir_number == 4 and self.lastDirection != Direction.up:
            direc = Direction.down
        else:
            direc = False
        self.lastDirection = direc
        return direc

    def getLastDirection(self):
        return self.lastDirection


class Direction:
    left = (-1, 0)
    up = (0, -1)
    right = (1, 0)
    down = (0, 1)

class SnakeGame:
    
    def __init__(self, width, height, agent, render=False, simpleRender=False, record=False):
        agent.snakeGame = self
        self.matrix = []
        self.width = width
        self.height = height
        for i in range(height):
            self.matrix += [[0]*width] # 0=empty 1=snake 2=fruit
        self.snake = []
        self.snake_direction = Direction.up
        self.tail_flag = True #Hold tail for one step
        self.score = 0
        self.game_over = False
        self.shown = False
        self.render = render
        self.simpleRender = simpleRender
        self.agent=agent
        self.record = record
        self.data = []

        
    def set(self, position, value): #Change value at position, position is tuple (x, y)
        try:
            self.matrix[position[1]][position[0]] = value
        except:
            print("Warning: trying to write out of bounds")

    def remove(self, position):
        self.set(position, 0)
        
    def get(self, position): #See value on position, position is tuple (x, y)
        try:
            return self.matrix[position[1]][position[0]]
        except:
            print("Warning: trying to read out of bounds")        
            return 4
        
    def init_snake(self):  #start self.snake (list of positions) add 1s to matrix
        position = (math.floor(self.width/2), math.floor(self.height/2))
        self.snake = [position]
        self.set(position, 1)
        self.add_fruit()

    def display(self):
        if not self.shown:
            self.shown = True
            self.fig, self.ax = plt.subplots()
            self.mat = self.ax.matshow(self.matrix)    
            plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False, labeltop=False)
            #plt.summer()
            plt.title('SnakeGame')
            self.t1 = self.ax.text(-1,-1,'Score = ' + str(self.score))
            return
        else:
            self.t1.set_text('Score = ' + str(self.score))       
            #plt.matshow(self.matrix, 0)
            self.mat.set_data(self.matrix)
            return
    
    def run(self, steptime, wait_start=2): #call step ever x seconds
        if(self.render):
            self.display()
            plt.pause(0.0001)
        if(self.simpleRender):
            self.simpl_display()
        time.sleep(wait_start) #Get ready!
        
        fractionstep = steptime
        last = time.process_time()
        
        while not self.game_over:
            if ( time.process_time() >= last + fractionstep):
                last = time.process_time()
                #time.sleep(steptime)
                self.old_direction = self.snake_direction
                self.move_snake()
                self.step()
                if(self.render):
                    self.display()
                    plt.pause(0.0001)
                if(self.simpleRender):
                    self.simpl_display()
        if (self.record):
            self.save_record()
        return

    def save_record(self):
        self.filename = "data_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        print("Will save data to resources/" + self.filename)

        file = open("../resources/" + self.filename, "w")
        stri = ""
        for dataline in self.data:
            stri += dataline + "\n"
        file.write(stri)
        file.close()
        
    def step(self): #move snake forward, check tail flag, add tiny score
        head_position = (self.snake[0][0] + self.snake_direction[0], self.snake[0][1] + self.snake_direction[1])
        self.snake = [head_position] + self.snake
        self.check_ate_fruit()
        # print(self.get_observations())
        if self.tail_flag:
            self.tail_flag = False
        else:
            self.remove(self.snake[len(self.snake) - 1])
            self.snake = self.snake[:-1]
        self.set(head_position, 1)
        if (self.check_collision()):
            self.game_over = True
        else:
            self.add_data()
        self.score += 1
        return
    
    def add_data(self):
        if (self.agent.getLastDirection()) == False:
            self.data += [str(self.get_observations()) + str(" -> ") + str((0, 0))]
        else:
            self.data += [str(self.get_observations()) + str(" -> ") + str(self.agent.getLastDirection())]

    def move_snake(self): #change snake direction (ask agent?)
        direc = self.agent.getDirection()
        if(direc):
            self.snake_direction = direc
    
    def check_collision(self): #out of bounds or ran over itself (repeated position in snake positions)
        if self.snake[0] in self.snake[1:]: #Head hits body
            return True
        return self.out_of_bounds(self.snake[0])

    def out_of_bounds(self, position): #is position out of board bounds
        if position[0] < 0:
            return True
        if position[1] < 0:
            return True        
        if position[0] >= self.width:
            return True
        if position[1] >= self.height:
            return True
        return False

    def check_ate_fruit(self): #check head on fruit, use tail_flag, add score
        if(self.fruit in self.snake):
            self.tail_flag = True
            self.score += 100
            self.add_fruit()
    
    def add_fruit(self): #set something to 2 (do not put over snake)
        pos = self.snake[0]
        while(pos in self.snake):
            pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
        self.fruit = pos
        self.set(pos, 2)
        return

    def simpl_display(self):
        for i in range(self.width):
            stra = ""
            for j in range(self.height):
                stra += ":#O"[self.matrix[i][j]] + " "
            print(stra)
        print("")

    def distance_fruit(self):
        return (self.snake[0][0] - self.fruit[0],self.snake[0][1] - self.fruit[1])

    def get_observations(self):
        nextpos = [(self.snake[0][0]+Direction.left[0], self.snake[0][1]+Direction.left[1]),
                   (self.snake[0][0]+Direction.up[0],self.snake[0][1]+Direction.up[1]),
                   (self.snake[0][0]+Direction.right[0],self.snake[0][1]+Direction.right[1]),
                   (self.snake[0][0]+Direction.down[0],self.snake[0][1]+Direction.down[1])]
        obs = []
        for el in nextpos:
            if self.out_of_bounds(el) or (el in self.snake):
                obs += [1,]
            else:
                obs += [0,]
        
        if self.distance_fruit()[0] > 0: 
            obs += [1,]
        elif self.distance_fruit()[0] == 0:
            obs += [0,]
        else:
            obs += [-1,]        
    
        if self.distance_fruit()[1] > 0: 
            obs += [1,]
        elif self.distance_fruit()[1] == 0:
            obs += [0,]
        else:
            obs += [-1,]
        obs += [(abs(self.distance_fruit()[0]) + abs(self.distance_fruit()[1]))/max(self.width, self.height), self.old_direction[0], self.old_direction[1],]
        return obs

    def model(self):
        return

    def train_model(self):
        return

    def train(self):
        return

class TrainSnake():
    def __init__(self, width, height, initial_games = 10, render=False):
        self.width = width
        self.height = height
        self.initial_games = initial_games
        self.render = render
        self.max_score = 100
        return

    def play_game(self):
        for _ in range(self.initial_games):
            game = SnakeGame(self.width, self.height, TrainingAgent(),render=self.render)
            game.init_snake()
            game.run(0.000000001)
            if(game.score>self.max_score):
                game.save_record()
                self.max_score = game.score
            print(game.score)

def get_all_data():
    files = glob.glob("../resources/*.txt")
    data = []
    for filename in files:
        f = open(filename,"r")
        for line in f.readlines():
            parts = line.split("->")
            observ = eval(parts[0])
            move = eval(parts[1])
            data += [[observ, move]]
    return data
    
def make_network_and_train(train_data, save_filename=False):
    network = tflearn.layers.core.input_data(shape=[None, 9, 1], name='input')
    network = tflearn.layers.core.fully_connected(network, 30, activation='relu')
    network = tflearn.layers.core.fully_connected(network, 2, activation='linear')
    network = tflearn.layers.estimator.regression(network, optimizer='adam', learning_rate=0.01, loss='mean_square', name='target')
    model = tflearn.DNN(network)    
    X = np.array([i[0] for i in train_data]).reshape(-1, 9, 1)
    y = np.array([i[1] for i in train_data]).reshape(-1, 2)
    if (save_filename):
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = save_filename)
        model.save(save_filename)    
    else:
        model.fit(X,y, n_epoch = 3, shuffle = True)    
    return model    

def auto_game():
    data = get_all_data()
    model = make_network_and_train(data)
    agent = TFAgent(model)
    game = SnakeGame(15,15,agent, render=True, record=False)
    game.init_snake()
    game.run(0.2,6)
    
def normal_game():
    game = SnakeGame(15,15,KeyboardAgent(), render=True, record=True)
    game.init_snake()
    game.run(0.2)    
    
normal_game()
#auto_game()