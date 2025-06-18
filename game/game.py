import pygame
import sys
import random
import time
import math
import os
import pandas
import string
from pathlib import Path
from enum import Enum

# Constants
WIDTH, HEIGHT = 1000, 800
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
FPS = 70
ARROW_SIZE = 60
ARROW_GLOW_TIME = 10
AUTO_RELOAD_THRESHOLD = 5  # ammo count to trigger auto-reload

base_path = os.path.join(os.path.dirname(__file__), "assets\\")

class GameState(Enum):
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2
    PAUSED = 3

class EnemyType(Enum):
    GOAT = 1
    SQUIRREL = 2

class ArrowIndicator:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction  # 'left', 'right', 'up', 'space'
        self.glow_timer = 0
        self.is_glowing = False
        
    def trigger(self):
        self.glow_timer = ARROW_GLOW_TIME
        self.is_glowing = True
        
    def update(self):
        if self.glow_timer > 0:
            self.glow_timer -= 1
        else:
            self.is_glowing = False
            
    def draw(self, screen):
        color = GREEN if self.is_glowing else WHITE
        border_color = (0, 200, 0) if self.is_glowing else GRAY
        
        # Draw border
        pygame.draw.rect(screen, border_color, 
                        (self.x - 5, self.y - 5, ARROW_SIZE + 10, ARROW_SIZE + 10), 3)
        
        # Draw arrow based on direction
        center_x = self.x + ARROW_SIZE // 2
        center_y = self.y + ARROW_SIZE // 2
        
        if self.direction == 'left':
            points = [(self.x + 10, center_y), 
                     (self.x + ARROW_SIZE - 10, self.y + 10),
                     (self.x + ARROW_SIZE - 10, self.y + ARROW_SIZE - 10)]
        elif self.direction == 'right':
            points = [(self.x + ARROW_SIZE - 10, center_y),
                     (self.x + 10, self.y + 10),
                     (self.x + 10, self.y + ARROW_SIZE - 10)]
        elif self.direction == 'up':
            points = [(center_x, self.y + 10),
                     (self.x + 10, self.y + ARROW_SIZE - 10),
                     (self.x + ARROW_SIZE - 10, self.y + ARROW_SIZE - 10)]
        elif self.direction == 'space':
            # Draw a circle for space (shoot)
            pygame.draw.circle(screen, color, (center_x, center_y), 20, 4)
            return
            
        pygame.draw.polygon(screen, color, points)

class Stopwatch():
    def __init__(self):
        self.past_time = None
        self.current_time = None

    def start(self):
        self.past_time = time.time()

    def get_timestamp(self):
        self.current_time = time.time()
        return round((self.current_time - self.past_time),2)

class Enemy:
    def __init__(self, enemy_type, spawn_x=None):
        self.type = enemy_type
        self.pos = [spawn_x or WIDTH, 717]
        self.radius = 40
        self.speed = random.randint(3, 7)  # Random enemy speed
        
        if enemy_type == EnemyType.GOAT:
            self.health = 150
            self.max_health = 150
            self.attack = 25
            self.image_name = "goat.png"
        else:  # SQUIRREL
            self.health = 75
            self.max_health = 75
            self.attack = 35
            self.image_name = "squirrel.png"
            
        self.load_img()

    def load_img(self):
        try:
            self.image = pygame.image.load(base_path + self.image_name).convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.radius * 2, self.radius * 2))
        except:
            print(f"Couldn't load {self.image_name}, using circle instead")
            self.image = None

class Player:
    def __init__(self):
        self.gun_visible = True
        self.pos = [400, 400]
        self.radius = 40
        self.gravity = 0
        self.jump_power = 20
        self.speed = 7
        self.facing_right = True
        self.health = 50
        self.max_health = 50
        self.ammo = 10
        self.max_ammo = 10
        self.shoot_cooldown = 0
        self.reload_cooldown = 0
        self.auto_reload = True
        self.load_images()

    def load_images(self):
        try:
            self.image = pygame.image.load(base_path + 'zelda.png').convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.radius * 2, self.radius * 2))
            self.image_left = pygame.transform.flip(self.image, True, False)
        except:
            print("Couldn't load player image, using circle instead")
            self.image = None
            
    def update(self, keys):
        # Horizontal movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.pos[0] -= self.speed
            self.facing_right = False
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.pos[0] += self.speed
            self.facing_right = True
            
        # Apply gravity
        self.gravity += 0.8
        self.pos[1] += self.gravity
        
        # Keep player in bounds (horizontal and vertical)
        self.pos[0] = max(self.radius, min(WIDTH - self.radius, self.pos[0]))
        self.pos[1] = max(self.radius, min(HEIGHT - 85, self.pos[1]))
        
        # Update cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.reload_cooldown > 0:
            self.reload_cooldown -= 1
            
        # Auto-reload when ammo is low
        if self.auto_reload and self.ammo <= AUTO_RELOAD_THRESHOLD and self.reload_cooldown == 0:
            self.reload()
            
    def jump(self):
        if self.pos[1] >= HEIGHT - 85 - self.radius:
            self.gravity = -self.jump_power
            
    def shoot(self):
        if not self.gun_visible:
            print("Can't shoot - gun is hidden!")
            return None
        if self.ammo > 0 and self.shoot_cooldown <= 0:
            self.ammo -= 1
            self.shoot_cooldown = 10
            bullet_x = self.pos[0] + (self.radius + 30 if self.facing_right else -self.radius - 30)
            return {
                'x': bullet_x,
                'y': self.pos[1] - 5,
                'rect': pygame.Rect(bullet_x, self.pos[1] - 5, 50, 20),
                'facing_right': self.facing_right
            }
        return None
        
    def reload(self):
        if self.reload_cooldown == 0:
            self.ammo = self.max_ammo
            self.reload_cooldown = 30  # Reload cooldown
        
    def draw(self, screen):
        if self.image:
            img = self.image if self.facing_right else self.image_left
            screen.blit(img, (self.pos[0] - self.radius, self.pos[1] - self.radius))
        else:
            pygame.draw.circle(screen, GREEN, (int(self.pos[0]), int(self.pos[1])), self.radius)

class Game:
    def __init__(self):
        self.stopwatch = Stopwatch()
        self.stopwatch.start()

        self.player_name = ""
        self.name_imput_active = False
        self.pressed_keys = {}
        self.gestures = []
        self.timestamps = []
        
        # Initialize pygame
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Masters Game")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.state = GameState.MENU
        self.player = Player()
        self.bullets = []
        self.enemies = []
        self.last_enemy_spawn = 0
        self.enemy_spawn_interval = 2000
        
        # Enemy tracking
        self.enemies_killed = {
            EnemyType.GOAT: 0,
            EnemyType.SQUIRREL: 0
        }
        self.total_kills = 0
        
        self.time_Interval = 500 #ms
        self.fullscreen = False
        self.user_screen = pygame.display.Info()
        
        # Arrow indicators for gesture control
        self.arrows = {
            'left': ArrowIndicator(20, HEIGHT // 2, 'left'),
            'right': ArrowIndicator(WIDTH - 80, HEIGHT // 2, 'right'),
            'up': ArrowIndicator((WIDTH // 2) - 30, (HEIGHT // 2) - 300, 'up'),
            'space': ArrowIndicator((WIDTH // 2) - 30, HEIGHT // 2, 'space')
        }
        
        # Load assets
        self.load_assets()
        self.setup_audio()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.user_screen.current_w, self.user_screen.current_h), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    def spawn_enemy(self):
        """Spawn random enemy type"""
        # After 10 total kills, spawn more squirrels
        if self.total_kills > 10:
            enemy_type = random.choice([EnemyType.GOAT, EnemyType.SQUIRREL, EnemyType.SQUIRREL])
        elif self.total_kills > 5:
            enemy_type = random.choice([EnemyType.GOAT, EnemyType.SQUIRREL])
        else:
            enemy_type = EnemyType.GOAT
            
        enemy = Enemy(enemy_type, WIDTH + 40)
        self.enemies.append(enemy)
        self.last_enemy_spawn = pygame.time.get_ticks()

    def update_enemies(self):
        current_time = pygame.time.get_ticks()
        
        # Spawn new enemies
        if current_time - self.last_enemy_spawn > self.enemy_spawn_interval:
            self.spawn_enemy()
        
        # Update existing enemies
        for enemy in self.enemies[:]:
            if enemy.health <= 0:
                self.enemies.remove(enemy)
                self.enemies_killed[enemy.type] += 1
                self.total_kills += 1
                continue
                
            enemy.pos[0] -= enemy.speed
            
            if enemy.pos[0] < -enemy.radius:
                self.enemies.remove(enemy)
    
    def draw_enemies(self):
        for enemy in self.enemies:
            if enemy.health <= 0:
                continue
                
            if enemy.image:
                self.screen.blit(enemy.image, (enemy.pos[0] - enemy.radius,
                                             enemy.pos[1] - enemy.radius))
            else:
                color = (255, 255, 255) if enemy.type == EnemyType.GOAT else (150, 75, 0)
                pygame.draw.circle(self.screen, color, 
                                (int(enemy.pos[0]), int(enemy.pos[1])), 
                                enemy.radius)
                
    def check_collisions(self):
        for enemy in self.enemies:
            if enemy.health <= 0:
                continue
                
            dx = self.player.pos[0] - enemy.pos[0]
            dy = self.player.pos[1] - enemy.pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.player.radius + enemy.radius:
                damage = 2 if enemy.type == EnemyType.SQUIRREL else 1
                self.player.health -= damage

    def load_assets(self):
        try:
            self.bullet_img = pygame.image.load(base_path + 'bullet.png').convert_alpha()
            self.bullet_img = pygame.transform.scale(self.bullet_img, (50, 20))
            self.bullet_img_left = pygame.transform.flip(self.bullet_img, True, False)
        except:
            print("Couldn't load bullet image")
            self.bullet_img = None
            
        try:
            self.gun_img = pygame.image.load(base_path + 'gun.png').convert_alpha()
            self.gun_img = pygame.transform.scale(self.gun_img, (100, 60))
            self.gun_img_left = pygame.transform.flip(self.gun_img, True, False)
        except:
            print("Couldn't load gun image")
            self.gun_img = None
            
        try:
            self.ground = pygame.transform.scale(pygame.image.load(base_path + 'ground.png').convert_alpha(), (WIDTH, HEIGHT // 2))
            self.sky = pygame.transform.scale(pygame.image.load(base_path + 'sky.jpeg').convert_alpha(), (WIDTH, HEIGHT))
        except:
            print("Couldn't load background images")
            self.ground = None
            self.sky = None
            
    def setup_audio(self):
        try:
            pygame.mixer.music.load(base_path + "Game.mp3.mp3")
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play(-1)
            self.gun_toggle_sound = pygame.mixer.Sound(base_path + "reload.mp3")  
            self.gun_toggle_sound.set_volume(1.0)
        except:
            print("Couldn't load audio files")
            self.gun_toggle_sound = None

    def draw_game_over_screen(self):
        font = pygame.font.SysFont(None, 72)
        text = font.render("GAME OVER", True, (255, 0, 0))
        self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
        pygame.display.flip()

        time.sleep(2)
        
        self.screen.fill((0, 0, 0))
        font = pygame.font.SysFont(None, 48)
        title = font.render("Game Over", True, (255, 0, 0))
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        prompt = font.render("Enter your name:", True, (255, 255, 255))
        self.screen.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2, 200))

        name_surface = font.render(self.player_name, True, (0, 255, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (WIDTH // 2 - 150, 260, 300, 50), 2)
        self.screen.blit(name_surface, (WIDTH // 2 - 140, 265))

    def draw_menu(self):
        self.screen.fill((20, 20, 20))
        title_font = pygame.font.SysFont("Comic Sans MS", 80)
        text_font = pygame.font.SysFont(None, 36)
        
        title = title_font.render("AI - MASTERS", True, (0, 255, 0))
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        instructions = [
            "",
            "Left Arrow / A - Move left",
            "Right Arrow / D - Move right", 
            "Up Arrow / W - Jump",
            "SPACE - Shoot (auto-reload enabled)",
            "Z - Manual reload",
            "P - Pause",
            "F - Fullscreen",
            "",
            "Click START to begin"
        ]

        for i, line in enumerate(instructions):
            txt = text_font.render(line, True, WHITE)
            self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 220 + i * 30))

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
        small_font = pygame.font.SysFont(None, 24)
        
        # Ammo display
        for i in range(self.player.ammo):
            pygame.draw.rect(self.screen, (255, 255, 0), (20 + i * 15, 20, 10, 20))
        ammo_label = font.render("Ammo", True, WHITE)
        self.screen.blit(ammo_label, (20, 0))
        
        # Gun status
        gun_status = "READY" if self.player.gun_visible else "HIDDEN"
        status_color = (0, 255, 0) if self.player.gun_visible else (255, 0, 0)
        status_text = font.render(f"Gun: {gun_status}", True, status_color)
        self.screen.blit(status_text, (20, 120))
        
        # Auto-reload indicator
        if self.player.auto_reload:
            auto_text = small_font.render("AUTO-RELOAD ON", True, (0, 255, 255))
            self.screen.blit(auto_text, (20, 145))
            
        # Health bar
        health_width = (self.player.health / self.player.max_health) * 200
        pygame.draw.rect(self.screen, (255, 0, 0), (20, 50, health_width, 20))
        health_label = font.render("Health", True, WHITE)
        self.screen.blit(health_label, (20, 75))
        
        # Enhanced kill tracking
        goat_kills = self.enemies_killed[EnemyType.GOAT]
        squirrel_kills = self.enemies_killed[EnemyType.SQUIRREL]
        
        kills_y = 20
        goat_text = font.render(f"Goats: {goat_kills}", True, WHITE)
        self.screen.blit(goat_text, (WIDTH - 150, kills_y))
        
        squirrel_text = font.render(f"Squirrels: {squirrel_kills}", True, (150, 75, 0))
        self.screen.blit(squirrel_text, (WIDTH - 150, kills_y + 30))
        
        total_text = font.render(f"Total: {self.total_kills}", True, (255, 215, 0))
        self.screen.blit(total_text, (WIDTH - 150, kills_y + 60))
        
    def draw_arrows(self):
        """Draw the arrow indicators for gesture control"""
        for arrow in self.arrows.values():
            arrow.update()
            arrow.draw(self.screen)
            
        # Draw labels
        font = pygame.font.SysFont(None, 20)
        labels = [
            ("LEFT", self.arrows['left'].x + 15, self.arrows['left'].y + 65),
            ("RIGHT", self.arrows['right'].x + 10, self.arrows['right'].y + 65),
            ("JUMP", self.arrows['up'].x + 15, self.arrows['up'].y + 65),
            ("SHOOT", self.arrows['space'].x + 10, self.arrows['space'].y + 65)
        ]
        
        for text, x, y in labels:
            label = font.render(text, True, WHITE)
            self.screen.blit(label, (x, y))
        
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
            
            # Check bullet collisions with enemies
            for enemy in self.enemies[:]:
                if enemy.health <= 0:
                    continue
                    
                dx = bullet['x'] - enemy.pos[0]
                dy = bullet['y'] - enemy.pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < enemy.radius:
                    enemy.health -= 25
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    if enemy.health <= 0:
                        self.enemies.remove(enemy)
                        self.enemies_killed[enemy.type] += 1
                        self.total_kills += 1
                    break

    def draw_bullets(self):
        for bullet in self.bullets:
            if self.bullet_img:
                img = self.bullet_img if bullet['facing_right'] else self.bullet_img_left
                self.screen.blit(img, (bullet['x'], bullet['y']))
            else:
                pygame.draw.rect(self.screen, (255, 0, 0), bullet['rect'])
                
    def handle_otp(self, key):
        if not self.pressed_keys.get(key, False):
            self.pressed_keys[key] = True

            # Trigger arrow indicators and log gestures
            if key in (pygame.K_LEFT, pygame.K_a):
                self.arrows['left'].trigger()
                self.gestures.append(1)
                self.timestamps.append(self.stopwatch.get_timestamp())
            elif key in (pygame.K_RIGHT, pygame.K_d):
                self.arrows['right'].trigger()
                self.gestures.append(2)
                self.timestamps.append(self.stopwatch.get_timestamp())
            elif key in (pygame.K_UP, pygame.K_w):
                self.arrows['up'].trigger()
                self.gestures.append(3)
                self.timestamps.append(self.stopwatch.get_timestamp())
            elif key == pygame.K_SPACE:
                self.arrows['space'].trigger()
                self.gestures.append(4)
                self.timestamps.append(self.stopwatch.get_timestamp())

    def extract_infos(self):
        game_id = "".join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(6))
        User = self.player_name
        df = pandas.DataFrame({"game_id": game_id,
                               "user_name": User,
                               "user_input": self.gestures,
                               "timestamp": self.timestamps,
                               "goat_kills": self.enemies_killed[EnemyType.GOAT],
                               "squirrel_kills": self.enemies_killed[EnemyType.SQUIRREL]})
        df.to_csv(f'game/logs/{User}_{game_id}.csv')

    def run(self):
        running = True
        first_enemy_spawned = False

        while running:
            current_time = pygame.time.get_ticks()
        
            if self.state == GameState.PLAYING and not first_enemy_spawned and current_time > 1000:
                self.spawn_enemy()
                first_enemy_spawned = True
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if self.state == GameState.MENU:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        button_rect = self.draw_menu()
                        if button_rect.collidepoint(event.pos):
                            self.state = GameState.PLAYING
                            first_enemy_spawned = False
                            self.last_enemy_spawn = pygame.time.get_ticks()
                            
                elif self.state == GameState.PLAYING:
                    if event.type == pygame.KEYDOWN:
                        self.handle_otp(event.key)
                        
                        if event.key == pygame.K_x:  
                            self.player.gun_visible = not self.player.gun_visible
                            if hasattr(self, 'gun_toggle_sound') and self.gun_toggle_sound:
                                self.gun_toggle_sound.play()
                        if event.key == pygame.K_UP or event.key == pygame.K_w:
                            self.player.jump()                           
                        if event.key == pygame.K_SPACE:
                            bullet = self.player.shoot()
                            if bullet:
                                self.bullets.append(bullet)
                        if event.key == pygame.K_z:
                            self.player.reload()
                        if event.key == pygame.K_p:
                            self.state = GameState.PAUSED
                        if event.key == pygame.K_f:
                            self.toggle_fullscreen()
                            
                    if event.type == pygame.KEYUP:
                        self.pressed_keys[event.key] = False
                        
                elif self.state == GameState.PAUSED:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        self.state = GameState.PLAYING
                        
                elif self.state == GameState.GAME_OVER and hasattr(self, 'game_over_phase') and self.game_over_phase == 2:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_BACKSPACE:
                            self.player_name = self.player_name[:-1]
                        elif event.key == pygame.K_RETURN:
                            self.extract_infos()
                            self.state = GameState.MENU
                            # Reset game state
                            self.enemies_killed = {EnemyType.GOAT: 0, EnemyType.SQUIRREL: 0}
                            self.total_kills = 0
                            self.player = Player()
                            self.bullets = []
                            self.enemies = []
                        else:
                            if len(self.player_name) < 10 and event.unicode.isprintable():
                                self.player_name += event.unicode

            if self.state == GameState.MENU:
                self.draw_menu()
                
            elif self.state == GameState.PLAYING:
                keys = pygame.key.get_pressed()
                
                self.player.update(keys)
                self.update_bullets()
                self.update_enemies()
                self.check_collisions()
                
                # Draw everything
                self.screen.fill(BLACK)
                if self.sky:
                    self.screen.blit(self.sky, (0, 0))
                else:
                    self.screen.fill(BLACK)
                    
                self.draw_bullets()
                self.player.draw(self.screen)
                self.draw_gun()
                self.draw_hud()
                self.draw_enemies()
                self.draw_arrows()  # Draw gesture indicators
                
                if self.ground:
                    self.screen.blit(self.ground, (0, 755))
                    
                if self.player.health <= 0:
                    self.state = GameState.GAME_OVER
                    self.game_over_phase = 1
                    self.game_over_timer = pygame.time.get_ticks()
                    self.player_name = ""
                    
            elif self.state == GameState.GAME_OVER:
                if not hasattr(self, 'game_over_phase'):
                    self.game_over_phase = 1
                    self.game_over_timer = pygame.time.get_ticks()
                    
                if self.game_over_phase == 1:
                    if current_time - self.game_over_timer < 2000:
                        font = pygame.font.SysFont(None, 72)
                        text = font.render("GAME OVER", True, (255, 0, 0))
                        self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
                    else:
                        self.game_over_phase = 2

                elif self.game_over_phase == 2:
                    self.screen.fill((75, 156, 211))
                    font = pygame.font.SysFont(None, 48)
                    title = font.render("Game Over", True, (210, 10, 46))
                    self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

                    prompt = font.render("Enter your name:", True, (255, 255, 255))
                    self.screen.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2, 200))

                    name_surface = font.render(self.player_name + "|", True, (0, 255, 0))
                    pygame.draw.rect(self.screen, (255, 255, 255), (WIDTH // 2 - 150, 260, 300, 50), 2)
                    self.screen.blit(name_surface, (WIDTH // 2 - 140, 265))

            elif self.state == GameState.PAUSED:
                font = pygame.font.SysFont(None, 72)
                text = font.render("PAUSED", True, WHITE)
                self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
                
            pygame.display.flip()
            self.clock.tick(FPS)
            
        # Cleanup
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()