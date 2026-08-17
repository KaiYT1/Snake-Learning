import pygame
import random
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 30  # Speed rate to see every step play out

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Training')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = "RIGHT"
        self.head = [self.w // 2, self.h // 2]
        self.snake = [
            list(self.head),
            [self.head[0] - BLOCK_SIZE, self.head[1]],
            [self.head[0] - (2 * BLOCK_SIZE), self.head[1]]
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, list(self.head))

        reward = -0.1  # Time penalty to prevent endless circles
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # self._update_ui()
        # self.clock.tick(SPEED)
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Check boundary walls
        if pt[0] > self.w - BLOCK_SIZE or pt[0] < 0 or pt[1] > self.h - BLOCK_SIZE or pt[1] < 0:
            return True
        # Check body self-intersection
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt[0] + 4, pt[1] + 4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right
        else: 
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left

        self.direction = new_dir

        if self.direction == "RIGHT":
            self.head[0] += BLOCK_SIZE
        elif self.direction == "LEFT":
            self.head[0] -= BLOCK_SIZE
        elif self.direction == "DOWN":
            self.head[1] += BLOCK_SIZE
        elif self.direction == "UP":
            self.head[1] -= BLOCK_SIZE
