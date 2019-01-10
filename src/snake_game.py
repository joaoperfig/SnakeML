import random

class SnakeGame:
    
    def __init__(self, width, height, agent, render=False):
        self.matrix = []
        self.width = width
        self.height = height
        for i in range(width):
            self.matrix += [[0]*height]
        snake = [()]