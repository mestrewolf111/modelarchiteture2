import os
import time
LABIRINTO_LINHAS = 9
LABIRINTO_COLUNAS = 9
import random
class LabirintoAmbiente:
    def __init__(self):
        self.labirinto = self.criar_labirinto()
        self.linhas, self.colunas = len(self.labirinto), len(self.labirinto[0])
        self.agente_pos = (1, 1)
        self.final_pos = (1, self.colunas - 2)
        self.actions = ['w', 's', 'a', 'd']  # Lista de ações possíveis

    def step(self, action):
        next_pos = self.mover_agente(self.agente_pos, action)
        reward = self.calcular_recompensa(next_pos)
        self.agente_pos = next_pos
        done = (self.agente_pos == self.final_pos)
        return next_pos, reward, done

    def criar_labirinto(self):
        labirinto = [['#' for _ in range(LABIRINTO_COLUNAS)] for _ in range(LABIRINTO_LINHAS)]

        # Set the initial position of the agent and final position
        start_pos = (1, 1)
        final_pos = (1, LABIRINTO_COLUNAS - 2)
        labirinto[start_pos[0]][start_pos[1]] = 'A'  # 'A' denotes the starting position of the agent
        labirinto[final_pos[0]][final_pos[1]] = 'F'  # 'F' denotes the final position

        # Create paths within the maze using depth-first search algorithm
        stack = [start_pos]
        visited = set()

        def is_valid(pos):
            x, y = pos
            return 1 <= x < LABIRINTO_LINHAS - 1 and 1 <= y < LABIRINTO_COLUNAS - 1

        while stack:
            current = stack[-1]
            visited.add(current)
            x, y = current
            neighbors = [(x + dx, y + dy) for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]]
            unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited and is_valid(neighbor)]

            if unvisited_neighbors:
                next_cell = random.choice(unvisited_neighbors)
                nx, ny = next_cell
                labirinto[nx][ny] = ' '  # ' ' denotes an open space in the maze
                labirinto[(x + nx) // 2][(y + ny) // 2] = ' '  # Carve out walls
                stack.append(next_cell)
            else:
                stack.pop()

        return labirinto

    def calcular_recompensa(self, pos):
        x, y = pos
        if self.labirinto[x][y] == ' ':
            self.labirinto[x][y] = 'P'  # Mark the position as visited
            return 1  # Positive reward for moving to an open space
        return -1  # No penalty for incorrect moves

    def mover_agente(self, agente_pos, movimento):
        direcoes = {
            'w': (-1, 0),  # Cima
            's': (1, 0),   # Baixo
            'a': (0, -1),  # Esquerda
            'd': (0, 1),   # Direita
        }
        nova_pos = tuple(sum(x) for x in zip(agente_pos, direcoes[movimento]))

        if self.labirinto[nova_pos[0]][nova_pos[1]] != '#':
            return nova_pos
        return agente_pos

    def imprimir_labirinto(self):
        os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela do terminal
        for i in range(len(self.labirinto)):
            for j in range(len(self.labirinto[i])):
                if (i, j) == self.agente_pos:
                    print('A', end=' ')
                else:
                    print(self.labirinto[i][j], end=' ')
            print()

    def inicializar_estado(self):
        return (1, 1)  # Posição inicial do agente (linha, coluna)