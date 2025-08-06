#!/usr/bin/env python3

from collections import deque
import heapq

print("\n--- SEARCH: DFS, BFS, A* ---")

grid = [
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1, 0]
]

start = (0, 0)
goal = (3, 3)

def get_neighbors(pos):
    x, y = pos
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 4 and 0 <= ny < 4 and grid[nx][ny] == 0:
            yield (nx, ny)

def bfs(start, goal):
    queue = deque([start])
    visited = set()
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        visited.add(node)
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                queue.append(neighbor)
    return False

print("BFS found goal:", bfs(start, goal))

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal):
    open_list = [(0, start)]
    cost_so_far = {start: 0}
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            return True
        for next in get_neighbors(current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heapq.heappush(open_list, (priority, next))
    return False

print("A* found goal:", a_star(start, goal))

print("\n--- PLANNING: Simple STRIPS-style logic ---")
print("Initial: robot in room1. Goal: robot in room3.")
print("Action: move(X, Y) if connected(X, Y).")