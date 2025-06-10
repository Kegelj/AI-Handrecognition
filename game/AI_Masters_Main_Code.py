import pygame
import sys
import random
import time
import math
from enum import Enum

# Constants
WIDTH, HEIGHT = 1000, 800
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FPS = 70
goats_killed = 0

class GameState(Enum):
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2
    PAUSED = 3

class Goat:
    def __init__(self):
        self.goat_pos = [WIDTH, 717]
        self.goat_health = 50
        self.goat_speed = 5
        self.attack = 25
        self.goat_radius = 40
        self.load_img()

    def load_img(self):
        try:
            self.image = pygame.image.load('goat.png').convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.goat_radius * 2, self.goat_radius * 2))
        except:
            print("Couldn't load goat image, using circle instead")
            self.image = None

class Squirrel:
    def __init__(self):
        self.squir_pos = [WIDTH, 717]
        self.squir_health = 75
        self.squir_speed = 5
        self.squir_radius = 40
        self.load_img()

    def load_img(self):
        try:
            self.image = pygame.image.load('squirrel.png').convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.squir_radius * 2, self.squir_radius * 2))
        except:
            print("Couldn't load squirrel image, using circle instead")
            self.image = None

class Player:
    def __init__(self):
        self.gun_visible = True # False original
        self.pos = [400, 400]
        self.radius = 40
        self.gravity = 0
        self.jump_power = 20
        self.speed = 7
        self.facing_right = True
        self.health = 50 #100 original - 50= 2 hits
        self.max_health = 50 #100 original
        self.ammo = 50 # 5 Original
        self.max_ammo = 50 # 5 Original
        self.shoot_cooldown = 0
        self.load_images()
        asdasdasd
    def load_images(self):
        try:
            self.image = pygame.image.load('zelda.png').convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.radius * 2, self.radius * 2))
            self.image_left = pygame.transform.flip(self.image, True, False)
        except:
            print("Couldn't load player image, using circle instead")
            self.image = None
            
    def update(self, keys):
        # Horizontal movement
        if keys[pygame.K_LEFT]:
            self.pos[0] -= self.speed
            self.facing_right = False
        if keys[pygame.K_RIGHT]:
            self.pos[0] += self.speed
            self.facing_right = True
            
        # Apply gravity
        self.gravity += 0.8
        self.pos[1] += self.gravity
        
        # Keep player in bounds
        self.pos[1] = max(self.radius, min(HEIGHT - 85, self.pos[1]))
        
        # Update shoot cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
            
    def jump(self):
        if self.pos[1] >= HEIGHT - 85 - self.radius:
            self.gravity = -self.jump_power
            
    def shoot(self):
        if not self.gun_visible:
            print("Can't shoot - gun is hidden!")
            return None
        if self.ammo > 0 and self.shoot_cooldown <= 0:
            self.ammo -= 1
            self.shoot_cooldown = 10  # Cooldown frames
            bullet_x = self.pos[0] + (self.radius + 30 if self.facing_right else -self.radius - 30)
            return {
                'x': bullet_x,
                'y': self.pos[1] - 5,
                'rect': pygame.Rect(bullet_x, self.pos[1] - 5, 50, 20),
                'facing_right': self.facing_right
            }
        return None
        
    def reload(self):
        self.ammo = self.max_ammo
        
    def draw(self, screen):
        if self.image:
            img = self.image if self.facing_right else self.image_left
            screen.blit(img, (self.pos[0] - self.radius, self.pos[1] - self.radius))
        else:
            pygame.draw.circle(screen, GREEN, (int(self.pos[0]), int(self.pos[1])), self.radius)

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.state = GameState.MENU
        self.player = Player()
        self.bullets = []
        self.goats = []
        self.last_goat_spawn = 0
        self.goat_spawn_interval = 2000  # 2 seconds in milliseconds
        self.squir = []
        self.last_squir_spawn = 0
        self.squir_spawn_interval = 2000
        self.load_assets()
        self.setup_audio()
        self.goats_killed = 0

    def spawn_goat(self):
        """Create a new goat at the right edge of the screen"""
        goat = Goat()
        goat.goat_pos = [WIDTH + goat.goat_radius, 717]  # Start just off-screen
        self.goats.append(goat)
        self.last_goat_spawn = pygame.time.get_ticks()

    def update_goats(self):
        current_time = pygame.time.get_ticks()
        
        # Spawn new goat if enough time has passed
        if self.goats_killed < 5 and current_time - self.last_goat_spawn > self.goat_spawn_interval:
            self.spawn_goat()
        
        # Update all goats
        for goat in self.goats[:]:  # Use slice copy to safely modify list during iteration
            if goat.goat_health <= 0:
                self.goats.remove(goat)
                self.goats_killed += 1
                continue
                
            # Simple movement towards the player
            goat.goat_pos[0] -= goat.goat_speed
            
            # Remove goats that go off-screen
            if goat.goat_pos[0] < -goat.goat_radius:
                self.goats.remove(goat)
    
    def draw_goats(self):
        for goat in self.goats:
            if goat.goat_health <= 0:
                continue
                
            if goat.image:
                img = goat.image
                self.screen.blit(img, (goat.goat_pos[0] - goat.goat_radius,
                                      goat.goat_pos[1] - goat.goat_radius))
            else:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                (int(goat.goat_pos[0]), int(goat.goat_pos[1])), 
                                goat.goat_radius)

    def spawn_squir(self):
        """Create a new squirrel at the right edge of the screen"""
        squir = Squirrel()
        squir.squir_pos = [WIDTH + squir.squir_radius, 717]  # Start just off-screen
        self.squir.append(squir)
        self.last_squir_spawn = pygame.time.get_ticks()

    def update_squirs(self):
        current_time = pygame.time.get_ticks()
        
        # Spawn new squirrel if enough time has passed
        if current_time - self.last_squir_spawn > self.squir_spawn_interval:
            self.spawn_squir()
        
        # Update all squirrels
        for squir in self.squir[:]:  # Use slice copy to safely modify list during iteration
            if squir.squir_health <= 0:
                self.squir.remove(squir)
                continue
                
            # Simple movement towards the player
            squir.squir_pos[0] -= squir.squir_speed
            
            # Remove squirrels that go off-screen
            if squir.squir_pos[0] < -squir.squir_radius:
                self.squir.remove(squir)

    def draw_squirs(self):
        for squir in self.squir:
            if squir.squir_health <= 0:
                continue
                
            if squir.image:
                img = squir.image
                self.screen.blit(img, (squir.squir_pos[0] - squir.squir_radius,
                                    squir.squir_pos[1] - squir.squir_radius))
            else:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                (int(squir.squir_pos[0]), int(squir.squir_pos[1])), 
                                squir.squir_radius)
                
    def check_collisions_squir(self):
        for squir in self.squir:
            if squir.squir_health <= 0:
                continue
                
            # Calculate distance between player and squirrel
            dx = self.player.pos[0] - squir.squir_pos[0]
            dy = self.player.pos[1] - squir.squir_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If collision occurs
            if distance < self.player.radius + squir.squir_radius:
                self.player.health -= 2  # Or whatever damage value

    def check_collisions(self):
        for goat in self.goats:
            if goat.goat_health <= 0:
                continue
                
            # Calculate distance between player and goat
            dx = self.player.pos[0] - goat.goat_pos[0]
            dy = self.player.pos[1] - goat.goat_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If collision occurs
            if distance < self.player.radius + goat.goat_radius:
                self.player.health -= 1  # Or whatever damage value

    def load_assets(self):
        # Load bullet image
        try:
            self.bullet_img = pygame.image.load('bullet.png').convert_alpha()
            self.bullet_img = pygame.transform.scale(self.bullet_img, (50, 20))
            self.bullet_img_left = pygame.transform.flip(self.bullet_img, True, False)
        except:
            print("Couldn't load bullet image")
            self.bullet_img = None
            
        # Load gun image
        try:
            self.gun_img = pygame.image.load('gun.png').convert_alpha()
            self.gun_img = pygame.transform.scale(self.gun_img, (100, 60))
            self.gun_img_left = pygame.transform.flip(self.gun_img, True, False)
        except:
            print("Couldn't load gun image")
            self.gun_img = None
            
        # Load background
        try:
            self.ground = pygame.transform.scale(pygame.image.load('ground.png').convert_alpha(), (WIDTH, HEIGHT // 2))
            self.sky = pygame.transform.scale(pygame.image.load('sky.jpeg').convert_alpha(), (WIDTH, HEIGHT))
        except:
            print("Couldn't load background images")
            self.ground = None
            self.sky = None
            
    def setup_audio(self):
        try:
            pygame.mixer.music.load("Game.mp3.mp3")
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play(-1)
            self.gun_toggle_sound = pygame.mixer.Sound("reload.mp3")  
            self.gun_toggle_sound.set_volume(1.0)
        except:
            print("Couldn't load audio files")
            self.gun_toggle_sound = None
            
    def draw_menu(self):
        self.screen.fill((20, 20, 20))
        title_font = pygame.font.SysFont("Comic Sans MS", 80)
        text_font = pygame.font.SysFont(None, 36)
        
        title = title_font.render("AI - MASTERS", True, (0, 255, 0))
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        instructions_old = [
            "left arrow / right arrow - Move",
            "up arrow - Jump",
            "X - Take gun out",
            "SPACE - Shoot (uses ammo)",
            "Z - Reload",
            "P - Pause",
            "",
            "Click START to begin"
        ]

        instructions = [
            "",
            "Thumb - Move right",
            "Pinky - Move left",
            "Pointy - Jump",
            "",
            "",
            "Click START to begin"
        ]

        for i, line in enumerate(instructions):
            txt = text_font.render(line, True, WHITE)
            self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 220 + i * 40))

        button_rect = pygame.Rect(WIDTH // 2 - 100, 600, 200, 60)
        pygame.draw.rect(self.screen, (0, 255, 0), button_rect)
        btn_text = text_font.render("START", True, BLACK)
        self.screen.blit(btn_text, (button_rect.centerx - btn_text.get_width() // 2,
                               button_rect.centery - btn_text.get_height() // 2))
        return button_rect
        
    def draw_gun(self):
        if self.player.gun_visible:
            if self.player.facing_right:
                gun_x = self.player.pos[0] + self.player.radius - 38
                gun_y = self.player.pos[1] - 30
                if self.gun_img:
                    self.screen.blit(self.gun_img, (gun_x, gun_y))
            else:
                gun_x = self.player.pos[0] - self.player.radius - 62
                gun_y = self.player.pos[1] - 30
                if self.gun_img:
                    self.screen.blit(self.gun_img_left, (gun_x, gun_y))
                
    def draw_hud(self):
        font = pygame.font.SysFont(None, 28)
        
        # Draw ammo
        for i in range(self.player.ammo):
            pygame.draw.rect(self.screen, (255, 255, 0), (20 + i * 15, 20, 10, 20))
        ammo_label = font.render("Ammo", True, WHITE)
        self.screen.blit(ammo_label, (20, 0))
        gun_status = "READY" if self.player.gun_visible else "HIDDEN"
        status_color = (0, 255, 0) if self.player.gun_visible else (255, 0, 0)
        status_text = font.render(f"Gun: {gun_status}", True, status_color)
        self.screen.blit(status_text, (20, 120))
            
        # Draw health bar
        health_width = (self.player.health / self.player.max_health) * 200
        pygame.draw.rect(self.screen, (255, 0, 0), (20, 50, health_width, 20))
        health_label = font.render("Health", True, WHITE)
        self.screen.blit(health_label, (20, 75))
        
        # Draw goats killed counter
        kills_text = font.render(f"Goats Killed: {self.goats_killed}", True, WHITE)
        self.screen.blit(kills_text, (WIDTH - 200, 20))
        
    def update_bullets(self):
        for bullet in self.bullets[:]:
            if bullet['facing_right']:
                bullet['x'] += 25
            else:
                bullet['x'] -= 25

            bullet['rect'].x = bullet['x'] 
            if bullet['x'] > WIDTH or bullet['x'] < 0:
                self.bullets.remove(bullet)
                continue
            
            # Check collision with squirrels
            for squir in self.squir[:]:
                if squir.squir_health <= 0:
                    continue
                    
                dx = bullet['x'] - squir.squir_pos[0]
                dy = bullet['y'] - squir.squir_pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < squir.squir_radius:
                    squir.squir_health -= 25
                    squir.squir_pos[0] += 155
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    if squir.squir_health <= 0:
                        self.squir.remove(squir)
                    break
                    
            # Check collision with goats
            for goat in self.goats[:]:
                if goat.goat_health <= 0:
                    continue
                    
                dx = bullet['x'] - goat.goat_pos[0]
                dy = bullet['y'] - goat.goat_pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < goat.goat_radius:
                    goat.goat_health -= 25
                    goat.goat_pos[0] += 155
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    if goat.goat_health <= 0:
                        self.goats.remove(goat)
                        self.goats_killed += 1
                    break

    def draw_bullets(self):
        for bullet in self.bullets:
            if self.bullet_img:
                img = self.bullet_img if bullet['facing_right'] else self.bullet_img_left
                self.screen.blit(img, (bullet['x'], bullet['y']))
            else:
                pygame.draw.rect(self.screen, (255, 0, 0), bullet['rect'])
                
    def run(self):
        running = True
        first_goat_spawned = False
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Spawn first goat after 1 second
            if self.state == GameState.PLAYING and not first_goat_spawned and current_time > 1000:
                self.spawn_goat()
                first_goat_spawned = True
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if self.state == GameState.MENU:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        button_rect = self.draw_menu()
                        if button_rect.collidepoint(event.pos):
                            self.state = GameState.PLAYING
                            first_goat_spawned = False
                            self.last_goat_spawn = pygame.time.get_ticks()
                            
                elif self.state == GameState.PLAYING:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_x:  
                            self.player.gun_visible = not self.player.gun_visible
                            if hasattr(self, 'gun_toggle_sound') and self.gun_toggle_sound:
                                self.gun_toggle_sound.play()
                        if event.key == pygame.K_UP:
                            self.player.jump()
                        if event.key == pygame.K_SPACE:
                            bullet = self.player.shoot()
                            if bullet:
                                self.bullets.append(bullet)
                        if event.key == pygame.K_z:
                            self.player.reload()
                        if event.key == pygame.K_p:
                            self.state = GameState.PAUSED
                            
                elif self.state == GameState.PAUSED:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        self.state = GameState.PLAYING
                        
            if self.state == GameState.MENU:
                self.draw_menu()
                
            elif self.state == GameState.PLAYING:
                keys = pygame.key.get_pressed()
                self.player.update(keys)
                self.update_bullets()
                self.update_goats()
                self.check_collisions()
                
                # Spawn squirrels after 5 goats killed
                if self.goats_killed >= 5:
                    self.update_squirs()
                    self.check_collisions_squir()
                
                # Draw everything
                if self.sky:
                    self.screen.blit(self.sky, (0, 0))
                else:
                    self.screen.fill(BLACK)
                    
                self.draw_bullets()
                self.player.draw(self.screen)
                self.draw_gun()
                self.draw_hud()
                if goats_killed < 5:
                    self.draw_goats()
                
                if self.goats_killed >= 5:
                    self.draw_squirs()
                
                if self.ground:
                    self.screen.blit(self.ground, (0, 755))
                    
                if self.player.health <= 0:
                    self.state = GameState.GAME_OVER
                    
            elif self.state == GameState.GAME_OVER:
                font = pygame.font.SysFont(None, 72)
                text = font.render("GAME OVER", True, (255, 0, 0))
                self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
                pygame.display.flip()
                time.sleep(2)
                running = False
                
            elif self.state == GameState.PAUSED:
                font = pygame.font.SysFont(None, 72)
                text = font.render("PAUSED", True, WHITE)
                self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
                
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
