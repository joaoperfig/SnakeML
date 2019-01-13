import random
import math
import time

import numpy as np
import matplotlib.pyplot as plt


class Direction:
    left = (-1, 0)
    up = (0, -1)
    right = (1, 0)
    down = (0, 1)

class SnakeGame:
    
    def __init__(self, width, height, agent, render=False):
        self.matrix = []
        self.width = width
        self.height = height
        for i in range(height):
            self.matrix += [[0]*width] # 0=empty 1=snake 2=fruit
        self.snake = []
        self.snake_directrion = Direction.left
        self.tail_flag = False #Hold tail for one step
        self.score = 0
        self.game_over = False;
        
    def set(self, position, value): #Change value at position, position is tuple (x, y)
        self.matrix[position[1]][position[0]] = value

    def remove(self, position):
        self.matrix[position[1]][position[0]] = 0
        
    def get(self, position): #See value on position, position is tuple (x, y)
        return self.matrix[position[1]][position[0]]
        
    def init_snake(self):  #start self.snake (list of positions) add 1s to matrix
        position = (math.floor(self.width/2), math.floor(self.height/2))
        self.snake = [position]
        self.set(position, 1)

    def display(self):
        plt.matshow(self.matrix)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False, labeltop=False)
        plt.summer()
        plt.title('SnakeGame')
        plt.text(-1,-1,'Score = ' + str(self.score))
        plt.show()
        return
    
    def run(self, steptime): #call step ever x seconds
        while not self.game_over:
            game.display()
            self.step()
            if (self.score%1000 == 100): self.add_fruit();
            time.sleep(steptime)
        return
        
    def step(self): #move snake forward, check tail flag, add tiny score
        head_position = (self.snake[0][0] + self.snake_directrion[0], self.snake[0][1] + self.snake_directrion[1])
        if (self.check_collision(head_position)):
            self.game_over = True
        else:
            self.set(head_position, 1)
            self.remove(self.snake[len(self.snake) - 1])
            for i in range(len(self.snake)-1,0,-1):
                position = (self.snake[i-1][0], self.snake[i-1][1])
                self.snake[i] = position
                self.set(position, 1)
            self.snake[0] = head_position
        self.score += 1
        return

    def move_snake(self,  direction): #change snake direction (ask agent?)
        return
    
    def check_collision(self, position): #out of bounds or ran over itself (repeated position in snake positions)
        if self.get(position) == 1:
            return True
        else:
            return False
    
    def check_ate_fruit(self): #check head on fruit, use tail_flag, add score
        if(self.fruit in self.snake):
            tail_flag = True
            self.score += 100
        else:
            tail_flag = False
        return
    
    def add_fruit(self): #set something to 2 (do not put over snake)
        pos = self.snake[0]
        while(pos in self.snake):
            pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
        self.fruit = pos
        self.set(pos, 2)
        return

    def press(self, event):
        print('pres',event.key)


game = SnakeGame(15,15,0)
game.init_snake()
game.run(1)
