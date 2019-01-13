import random
import math

class Direction:
    def left():
        return 0
    def up():
        return 1
    def right():
        return 2
    def down():
        return 3

class SnakeGame:
    
    def __init__(self, width, height, agent, render=False):
        self.matrix = []
        self.width = width
        self.height = height
        for i in range(height):
            self.matrix += [[0]*width] # 0=empty 1=snake 2=fruit
        self.snake = []
        self.snake_directrion = Direction.up
        self.tail_flag = False #Hold tail for one step
        self.score = 0
        
    def set(self, position, value): #Change value at position, position is tuple (x, y)
        self.matrix[position[1]][position[0]] = value
        
    def get(self, position): #See value on position, position is tuple (x, y)
        return self.matrix[position[1]][position[0]]
        
    def init_snake(self):  #start self.snake (list of positions) add 1s to matrix
        position = (math.floor(self.width/2), math.floor(self.height/2))
        self.snake = [position]
        self.set(position, 1)
    
    def display(self):
        return
    
    def run(self, steptime): #call step ever x seconds
        return
        
    def step(self): #move snake forward, check tail flag, add tiny score
        return
    
    def move_snake(self,  direction): #change snake direction (ask agent?)
        return
    
    def check_collision(self): #out of bounds or ran over itself (repeated position in snake positions)
        return
    
    def check_ate_fruit(self): #check head on fruit, use tail_flag, add score
        return
    
    def add_fruit(self): #set something to 2 (do not put over snake)
        return
        