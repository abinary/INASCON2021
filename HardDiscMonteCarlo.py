import math
import numpy as np
import random
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from tqdm import tqdm
import cv2 as cv

# Change the seed (or comment out) to generate a different simulation result
random.seed(1)

# The disc radius is implicitly 1

# Size of the world (L by L)
L = 100

# Num of particles
N = 100

# How long to run the simulation, and how often to record
record_every = 1
num_time_steps = 1000 * record_every
prefix = 'HD_N{0}_L{1}_every{2}'.format(N, L, record_every)
trajectory_filename = prefix + '.txt'
movie_filename = prefix + '.mp4'
#fourcc = cv.VideoWriter_fourcc(*'H264')
fourcc = cv.VideoWriter_fourcc(*'mp4v')

# Step size each time step
StepSize = 0.1

CollisionGridGranularity = min(int(L/2), 100)

collision_grid = []
particles = []

class ScaledCanvas:
    def __init__(self, wh, box):
        #self.image = Image.new("L", wh) # grayscale
        self.image = Image.new("RGB", wh)  # RGB
        self.draw = ImageDraw.Draw(self.image)
        self.box = box

    def circle(self, xy, r, fill = (150, 150, 255)):
        xy = ((xy[0] - self.box[0]) * (self.image.width / self.box[2]),
              (xy[1] - self.box[1]) * (self.image.height / self.box[3]))

        r = r * (self.image.width / self.box[2])

        self.draw.ellipse([(xy[0] - r, xy[1] - r), (xy[0] + r, xy[1] + r)], fill = fill)#, outline="blue")

def InitializeCollisionGrid():
    global collision_grid
    granularity = CollisionGridGranularity + 2
    collision_grid = [[[] for j in range(granularity)] for i in range(granularity)]

def CalculateCollisionGridPosition(xy):
    i = int(xy[0] / L * CollisionGridGranularity) + 1
    j = int(xy[1] / L * CollisionGridGranularity) + 1
    return i, j

def RegisterParticle(id, xy):
    (i, j) = CalculateCollisionGridPosition(xy)
    # Put the particle on the grid
    collision_grid[i][j].append(id)

def RegisterParticleMove(id, xy, updated_xy):
    (i, j) = CalculateCollisionGridPosition(xy)
    collision_grid[i][j].remove(id)
    RegisterParticle(id, updated_xy)

def Norm(xy):
    return math.sqrt(xy[0] * xy[0] + xy[1] * xy[1])

def IsThereACollision(id, xy):
    (i0, j0) = CalculateCollisionGridPosition(xy)

    others = []

    # Put the particle on the grid
    for i in range(i0 - 1, i0 + 2):
        for j in range(j0 - 1, j0 + 2): # important! the last number is not included
            for idx in range(len(collision_grid[i][j])):
                others.append(collision_grid[i][j][idx])

    # Check for collisions against all others
    for idx in range(len(others)):
        if others[idx] != id:
            dist = Norm((particles[others[idx]][0] - xy[0], particles[others[idx]][1] - xy[1]))

            # If there's a collision, stop and report (diameter is 2)
            if dist <= 2:
                return True

    return False

def GenerateStep():
    # This is a way to generate a unit vector towards a random direction
    step = (random.gauss(0, 1), random.gauss(0, 1))
    step_norm = Norm(step)

    return step[0] * (StepSize / step_norm), step[1] * (StepSize / step_norm)


InitializeCollisionGrid()

# Initialize particles without overlaps (collisions)
for i in range(N):
    xy = (random.uniform(1, L - 1), random.uniform(1, L - 1))

    # As long as there is a collision, try another position
    while IsThereACollision(i, xy):
        xy = (random.uniform(1, L - 1), random.uniform(1, L - 1))

    # Register particle
    RegisterParticle(i, xy)
    particles.append(xy)


# Run the simulation

image_size = 2048;

if len(movie_filename) > 0:
    videowriter = cv.VideoWriter(movie_filename, fourcc, 25, (image_size, image_size))

if len(trajectory_filename) > 0:
    trajectory_file = open(trajectory_filename, "w")

#for time_step in range(num_time_steps):
for time_step in tqdm(range(num_time_steps), desc="Running simulation ..."):
    # Generate a step for every particle
    for id in range(N):
        xy = particles[id]

        step = GenerateStep()
        updated_xy = (xy[0] + step[0], xy[1] + step[1])

        # If out of bounds or there's a collision, do not move
        if updated_xy[0] < 1 or updated_xy[0] >= L - 1 or updated_xy[1] < 1 \
                or updated_xy[1] >= L - 1 or IsThereACollision(id, updated_xy):
            continue

        RegisterParticleMove(id, xy, updated_xy)
        particles[id] = updated_xy

    # Record?
    if time_step % record_every == 0:

        # record trajectory?
        if len(trajectory_filename) > 0:
            # Write X,Y coordinates of all particles
            trajectory_file.write(' '.join('{} {}'.format(xy[0], xy[1]) for xy in particles))
            trajectory_file.write('\n')


        # record animation?
        if len(movie_filename) > 0:
            scaled_canvas = ScaledCanvas((image_size, image_size), (0, 0, L, L))

            for id in range(N):
                xy = particles[id]
                scaled_canvas.circle(xy, 1)

            frame = cv.cvtColor(np.array(scaled_canvas.image), cv.COLOR_RGB2BGR)
            videowriter.write(frame)

if len(trajectory_filename) > 0:
    trajectory_file.close()

if len(movie_filename) > 0:
    videowriter.release()

