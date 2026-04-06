import pygame
import math
import neat
import os
import random

# --- 1. SETUP ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Load the track image (Make sure track.png is in the same folder!)
track_image = pygame.image.load("track.png") 

# --- 2. CAR CLASS ---
class Car:
    def __init__(self):
        self.surface = pygame.Surface((20, 40))
        self.surface.fill("red")
        self.surface.set_colorkey((0,0,0))
        
        # Start Position
        self.pos = pygame.Vector2(100, 300) 
        
        # SPREAD THE SWARM: Give them slightly random starting angles
        self.angle = random.randint(-15, 15) 
        
        self.velocity = 0
        self.speed = 500 
        self.radars = []
        self.alive = True 
        self.distance = 0 

    def drive(self, action, dt):
        # Base idle speed so they don't just sit still
        self.velocity += (self.speed * 0.2) * dt 
        
        # Action from AI: [Forward, Backward(Disabled), Left, Right]
        if action[0] > 0.5: self.velocity += self.speed * dt
        # action[1] is reverse, which we completely removed to stop them from driving backwards
        if action[2] > 0.5: self.angle += 100 * dt
        if action[3] > 0.5: self.angle -= 100 * dt

        self.velocity *= 0.9 # Friction
        
        # Movement Math
        rad = math.radians(self.angle)
        self.pos.x += math.sin(rad) * self.velocity * dt
        self.pos.y += math.cos(rad) * self.velocity * dt
        
        self.distance += self.velocity * dt 
        
        # Wall Collision Check
        self.check_collision()
        self.update_sensors()

    def check_collision(self):
        # Screen Bounds
        if self.pos.x <= 0 or self.pos.x >= 800 or self.pos.y <= 0 or self.pos.y >= 600:
            self.alive = False
            return

        # Grass Check (UPDATED WITH YOUR EXACT RGB)
        try:
            pixel = screen.get_at((int(self.pos.x), int(self.pos.y)))
            if pixel[0] == 34 and pixel[1] == 177 and pixel[2] == 76: 
                self.alive = False
        except:
            self.alive = False

    def update_sensors(self):
        self.radars.clear()
        
        # The 5 radar sensors
        for angle_offset in [-90, -45, 0, 45, 90]:
            ray_angle = self.angle + angle_offset
            ray_rad = math.radians(ray_angle)
            ray_len = 0
            
            # March the ray
            while ray_len < 200:
                ray_x = int(self.pos.x + math.sin(ray_rad) * ray_len)
                ray_y = int(self.pos.y + math.cos(ray_rad) * ray_len)
                
                if ray_x <= 0 or ray_x >= 800 or ray_y <= 0 or ray_y >= 600:
                    break
                try:
                    pixel = screen.get_at((ray_x, ray_y))
                    # Radar stops when it sees the exact grass color
                    if pixel[0] == 34 and pixel[1] == 177 and pixel[2] == 76:
                        break
                except:
                    break
                ray_len += 15
            
            # Store data and Draw 
            if self.alive:
                pygame.draw.line(screen, "green", (self.pos.x, self.pos.y), (ray_x, ray_y), 1)
                pygame.draw.circle(screen, "red", (ray_x, ray_y), 3)
                self.radars.append(ray_len)

    def draw(self):
        if self.alive:
            rotated_car = pygame.transform.rotate(self.surface, self.angle)
            rect = rotated_car.get_rect(center=(self.pos.x, self.pos.y))
            screen.blit(rotated_car, rect.topleft)


# --- 3. NEAT EVOLUTION LOOP ---
def eval_genomes(genomes, config):
    cars = []
    nets = []
    ge = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car())
        genome.fitness = 0 
        ge.append(genome)

    running = True
    dt = 0
    timer = 0 # Generation timeout clock

    while running and len(cars) > 0:
        timer += 1
        # Kill generation after roughly 10 seconds to force a mutation
        if timer > 600: 
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            # COLOR DEBUGGER: Click the grass to print its exact color!
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                color = screen.get_at(pos)
                print(f"Color at clicked pixel: R={color[0]}, G={color[1]}, B={color[2]}")

        screen.blit(track_image, (0, 0))

        # Process Each Car backwards
        for i in range(len(cars) - 1, -1, -1):
            car = cars[i]
            
            if car.alive:
                inputs = []
                for radar in car.radars:
                    inputs.append(radar)
                
                while len(inputs) < 5: 
                    inputs.append(200)

                output = nets[i].activate(inputs)
                car.drive(output, dt)
                car.draw()

                # Reward for speed
                ge[i].fitness += car.velocity * dt * 0.1 

            else:
                # Penalty for crashing
                ge[i].fitness -= 1
                cars.pop(i)
                nets.pop(i)
                ge.pop(i)

        pygame.display.flip()
        dt = clock.tick(60) / 1000

# --- 4. SETUP NEAT AND RUN ---
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(eval_genomes, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)