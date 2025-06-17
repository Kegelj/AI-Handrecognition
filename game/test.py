import pygame

pygame.init()

clock = pygame.time.Clock()
running = True
while running:
    print(clock.get_time())
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            running = False


pygame.quit()