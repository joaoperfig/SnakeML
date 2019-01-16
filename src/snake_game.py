import random
import math
import time
import msvcrt
import datetime
import glob

import numpy as np
import matplotlib.pyplot as plt

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
        if record:
            self.filename = "data_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +".txt"
            print("Will save data to resources/"+self.filename)
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
    
    def run(self, steptime): #call step ever x seconds
        if(self.render):
            game.display()
            plt.pause(0.0001)
        if(self.simpleRender):
            game.simpl_display()
        time.sleep(1.5) #Get ready!
        
        fractionstep = steptime
        last = time.process_time()
        
        while not self.game_over:
            if ( time.process_time() >= last + fractionstep):
                last = time.process_time()
                #time.sleep(steptime)
                self.move_snake()
                self.step()
                if(self.render):
                    game.display()
                    plt.pause(0.0001)
                if(self.simpleRender):
                    game.simpl_display()
        if (self.record):
            file = open("../resources/"+self.filename, "w")
            stri = ""
            for dataline in self.data:
                stri += dataline + "\n"
            file.write(stri)
            file.close()
        return
        
    def step(self): #move snake forward, check tail flag, add tiny score
        head_position = (self.snake[0][0] + self.snake_direction[0], self.snake[0][1] + self.snake_direction[1])
        self.snake = [head_position] + self.snake
        self.check_ate_fruit()
        if self.tail_flag:
            self.tail_flag = False
        else:
            self.remove(self.snake[len(self.snake) - 1])
            self.snake = self.snake[:-1]
        self.set(head_position, 1)
        if (self.check_collision()):
            self.game_over = True
        else:
            if self.record:
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

    def press(self, event):
        print('pres',event.key)

    def simpl_display(self):
        for i in range(self.width):
            stra = ""
            for j in range(self.height):
                stra += ":#O"[self.matrix[i][j]] + " "
            print(stra)
        print("")
        
    # [left, front, right, angle to apple]
    def get_observations(self): 
        directions = [Direction.left, Direction.up, Direction.right, Direction.down]
        for i in range(len(directions)):
            if directions[i] == self.snake_direction:
                front = directions[i]
                right = directions[(i+1)%3]
                back = directions[(i+2)%3]
                left = directions[(i+3)%3]
        return None

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
    print(data)
        


#game = SnakeGame(15,15,KeyboardAgent(), simpleRender=True, record=True)
game = SnakeGame(15,15,KeyboardAgent(), render=True, record = True)
game.init_snake()
game.run(0.2)
