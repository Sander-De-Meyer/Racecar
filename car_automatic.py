import numpy as np
import pygame

from track import Track

ACCELERATION_UNIT = 0.2
STEERING_UNIT = 0.001
MAX_STEPS = 200

class Own_Space:
    def __init__(self, n):
        self.n = n

class Car:
    def __init__(self, x=0, y=0, width = 40, length = 40, theta=0, speed=0, inv_radius=0, manual=True):
        # x and y are positions
        # theta is the angle of rotation of the car, in the interval [0, 2pi]. 0 means pointing rightward
        # speed is speed of the car
        # inv_radius is related to the wheel position and, together with the speed, is related to the change of rotation in function of time
        # It is defined as the inverse of the radius of the circle the car would trace out if R would remain constant
        # It is positive if the wheels are turned rightward
        # Delta t is 1 second

        self.x, self.y, self.width, self.length, self.theta, self.speed, self.inv_radius, self.manual = x, y, width, length, theta, speed, inv_radius, manual
        self.number_of_steps = 0
        self.action_space = Own_Space(4)
        self.track = Track(r"C:\Users\Sande\Documents\Projecten\Racecar\track2.png")

        self.set_centre()

        if not self.manual:
            print("here")
            self.racecar_image = pygame.image.load(r"C:\Users\Sande\Documents\Projecten\Racecar\racecar_bijgesneden.jpg")  # Load the car image
            # self.racecar_image = pygame.transform.scale(self.racecar_image, (35, 70))  # Scale the image to an appropriate size
            self.racecar_image = pygame.transform.scale(self.racecar_image, (width, length))  # Scale the image to an appropriate size

        self.progress = self.get_progess_angle()

    def __repr__(self):
        return f"Position: ({self.x}, {self.y}) \nRotation: {self.theta} \nSpeed: {self.speed}" 
    
    def state(self):
        return (self.x, self.y, self.theta, self.speed, self.inv_radius)

    def reset(self):
        self.x, self.y, self.width, self.length, self.theta, self.speed, self.inv_radius, self.manual = 0, 0, 40, 40, 0, 0, 0, True
        self.number_of_steps = 0
        self.set_centre()

        if not self.manual:
            print("here")
            self.racecar_image = pygame.image.load(r"C:\Users\Sande\Documents\Projecten\Racecar\racecar_bijgesneden.jpg")  # Load the car image
            # self.racecar_image = pygame.transform.scale(self.racecar_image, (35, 70))  # Scale the image to an appropriate size
            self.racecar_image = pygame.transform.scale(self.racecar_image, (40, 40))  # Scale the image to an appropriate size

        self.progress = self.get_progess_angle()

        return self.state(), {}

    def step(self, action):
        # action is 0 for left, 1 for up, 2 for right, 3 for down
        if action == 0:
            controls = [0, -1]
        elif action == 1:
            controls = [1, 0]
        elif action == 2:
            controls = [0, 1]
        elif action == 3:
            controls = [-1, 0]
        else:
            controls = [0, 0]

        new_state, reward = self.update(controls)
        
        truncated = (self.number_of_steps > MAX_STEPS)        
        terminated = self.check_collision_efficient(self.track)

        self.number_of_steps += 1
        return new_state, reward, terminated, truncated, {}

    def set_centre(self):
        self.centre_x = self.x + (np.cos(self.theta)*self.length)//2
        self.centre_y = self.y - (np.sin(self.theta)*self.length)//2

    def move(self):
        if self.inv_radius == 0:
            self.x += self.speed*np.cos(self.theta)
            self.y -= self.speed*np.sin(self.theta)
        else:
            centre_circle_x = self.x + np.sin(self.theta) / self.inv_radius
            centre_circle_y = self.y + np.cos(self.theta) / self.inv_radius
            angle_traced = self.speed*self.inv_radius
            angle_final = np.sign(self.inv_radius)*np.pi/2 + self.theta - angle_traced
            self.x = centre_circle_x + np.cos(angle_final) / abs(self.inv_radius)
            self.y = centre_circle_y - np.sin(angle_final) / abs(self.inv_radius)
            self.theta -= angle_traced

        new_progess_angle = self.get_progess_angle()
        change = (new_progess_angle - self.progress + np.pi) % (2*np.pi) - np.pi
        self.progress += change

        self.set_centre()

        return change

    def steer_manual(self, controls):
        updown = controls[0]
        rightleft = controls[1]
        self.speed += (updown)*ACCELERATION_UNIT
        self.inv_radius += (rightleft)*STEERING_UNIT
    
    def steer(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.speed += ACCELERATION_UNIT
        if keys[pygame.K_DOWN]:
            self.speed -= ACCELERATION_UNIT
        if keys[pygame.K_LEFT]:
            self.inv_radius -= STEERING_UNIT
        if keys[pygame.K_RIGHT]:
            self.inv_radius += STEERING_UNIT

    def update(self, controls):
        if self.manual:
            self.steer_manual(controls)
        else:
            print("here")
            self.steer()
        progress = self.move()
        # print(f"Progress = {self.progress}")
        # print(f"Centre = {self.centre_x}, {self.centre_y}")
        return self.state(), progress

    def check_collision(self, track):
        iscar = np.zeros((track.width, track.height))
        for y in range(track.height):
            for x in range(track.width):
                if np.sqrt((self.centre_x - x)**2 + (self.centre_y - y)**2) < self.width:
                    iscar[x,y] = 1
                # if (np.cos(self.angle)*self.centre_x + np.sin(self.angle)*self.centre_y) < x < ()
        is_collided = iscar * track.track
        return (np.sum(is_collided))

    def check_collision_efficient(self, track):
        front_left = (self.centre_x  + np.cos(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2, self.centre_y - np.sin(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2)
        front_right = (self.centre_x  + np.cos(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2, self.centre_y - np.sin(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2)
        back_left = (self.centre_x  - np.cos(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2, self.centre_y + np.sin(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2)
        back_right = (self.centre_x  - np.cos(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2, self.centre_y + np.sin(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2)
        # front_left = (self.x  + np.cos(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2, self.y - np.sin(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2)
        # front_right = (self.x  + np.cos(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2, self.y - np.sin(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2)
        # back_left = (self.x  - np.cos(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2, self.y + np.sin(self.theta)*self.length/2 - np.sin(self.theta)*self.width/2)
        # back_right = (self.x  - np.cos(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2, self.y + np.sin(self.theta)*self.length/2 + np.sin(self.theta)*self.width/2)
        for boundary in track.boundaries:
            # if self.is_point_in_rotated_rectangle(boundary[0], boundary[1], self.centre_x, self.centre_y, self.width, self.length, self.theta):
            if (np.sqrt((self.centre_x - boundary[0])**2 + (self.centre_y - boundary[1])**2) < 0.4*self.width):
            # if ((np.sqrt((front_left[0] - boundary[0])**2 + (front_left[0] - boundary[1])**2) < 0.01*self.width) or
            #     (np.sqrt((front_right[0] - boundary[0])**2 + (front_right[0] - boundary[1])**2) < 0.01*self.width) or
            #     (np.sqrt((back_left[0] - boundary[0])**2 + (back_left[0] - boundary[1])**2) < 0.01*self.width) or
            #     (np.sqrt((back_right[0] - boundary[0])**2 + (back_right[0] - boundary[1])**2) < 0.01*self.width)):
                print("Collided")
                return True
        return False

    def is_point_in_rotated_rectangle(self, px, py, cx, cy, width, length, angle):
        """
        Check if a point (px, py) is inside a rotated rectangle centered at (cx, cy)
        with the given width, length, and rotation angle in radians.
        """
        # Translate point to the rectangle's coordinate system
        translated_px = px - cx
        translated_py = py - cy
        
        # Precompute trigonometric values
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Rotate the point by the negative of the rectangle's rotation angle
        rotated_px = translated_px * cos_theta + translated_py * sin_theta
        rotated_py = -translated_px * sin_theta + translated_py * cos_theta
        
        # Check if the rotated point is within the bounds of the axis-aligned rectangle
        half_width = width / 2
        half_length = length / 2
        
        return (-half_length <= rotated_px <= half_length) and (-half_width <= rotated_py <= half_width)


    def draw(self, track):
        # racecar_picture = r'racecar.jpg'#https://flyclipart.com/th-red-racing-car-top-view-race-car-png-210070
        # car_rect = pygame.Rect(self.x, self.y, 20, 10)
        # pygame.draw.rect(screen, (255, 0, 0), car_rect)

        rotated_image = pygame.transform.rotate(self.racecar_image, -90 + (180/np.pi)*self.theta)
        new_rect = rotated_image.get_rect(center=self.racecar_image.get_rect(topleft=(self.x, self.y)).center)
        track.screen.blit(rotated_image, new_rect.topleft)

    def get_progess_angle(self):
        centre = (400, 400)
        return -np.arctan2((centre[0] - self.y),(self.x - centre[1])) + np.pi/2