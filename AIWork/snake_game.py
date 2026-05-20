import os
import sys
import time
import random
import msvcrt

# ========== 常量定义 ==========
WIDTH = 40
HEIGHT = 20
TICK = 0.15

WALL = '#'
HEAD = '@'
BODY = 'O'
FOOD = '$'
EMPTY = ' '

DIRECTIONS = {
    'w': (0, -1),
    's': (0, 1),
    'a': (-1, 0),
    'd': (1, 0),
}

OPPOSITE = {
    'w': 's',
    's': 'w',
    'a': 'd',
    'd': 'a',
}


# ========== 函数实现 ==========
def init_game():
    """初始化游戏状态"""
    snake = [(5, 3), (4, 3), (3, 3)]
    direction = 'd'
    next_direction = 'd'
    game_over = False
    score = 0

    food = generate_food(snake, WIDTH, HEIGHT)

    return {
        'snake': snake,
        'direction': direction,
        'next_direction': next_direction,
        'food': food,
        'score': score,
        'game_over': game_over,
        'width': WIDTH,
        'height': HEIGHT,
    }


def generate_food(snake, width, height):
    """在空白位置随机生成食物，无可选位置返回 None"""
    candidates = []
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if (x, y) not in snake:
                candidates.append((x, y))

    if not candidates:
        return None
    return random.choice(candidates)


def get_input(current_dir):
    """非阻塞检测键盘输入，返回合法方向或 None"""
    if not msvcrt.kbhit():
        return None

    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()

    if key not in DIRECTIONS:
        return None

    # 禁止反向
    if OPPOSITE.get(key) == current_dir:
        return None

    return key


def move_snake(snake, direction, food):
    """移动蛇，返回 (new_head, ate_food)"""
    dx, dy = DIRECTIONS[direction]
    head_x, head_y = snake[0]
    new_head = (head_x + dx, head_y + dy)

    snake.insert(0, new_head)

    ate_food = False
    if new_head == food:
        ate_food = True
    else:
        snake.pop()

    return new_head, ate_food


def check_collision(head, snake, width, height):
    """检测碰撞，撞墙或撞自身返回 True"""
    x, y = head

    # 撞墙
    if x < 1 or x >= width - 1 or y < 1 or y >= height - 1:
        return True

    # 撞自身
    if head in snake[1:]:
        return True

    return False


def render(snake, food, score, width, height):
    """渲染游戏画面"""
    os.system('cls')

    # 构建二维网格，默认填充空白
    grid = [[EMPTY for _ in range(width)] for _ in range(height)]

    # 画墙壁（边框）
    for x in range(width):
        grid[0][x] = WALL
        grid[height - 1][x] = WALL
    for y in range(height):
        grid[y][0] = WALL
        grid[y][width - 1] = WALL

    # 放食物
    if food is not None:
        fx, fy = food
        grid[fy][fx] = FOOD

    # 放蛇身（从尾部到蛇头，确保蛇头覆盖）
    for i in range(len(snake) - 1, 0, -1):
        sx, sy = snake[i]
        grid[sy][sx] = BODY
    # 放蛇头
    hx, hy = snake[0]
    grid[hy][hx] = HEAD

    # 逐行打印
    for row in grid:
        print(''.join(row))

    print(f'得分：{score}')


def show_game_over(score):
    """显示游戏结束画面"""
    os.system('cls')
    print('\n' * 5)
    print(' ' * 15 + '游戏结束！')
    print(' ' * 15 + f'得分：{score}')
    print('\n' * 3)
    print(' ' * 12 + '按任意键退出...')
    msvcrt.getch()


def game_loop():
    """游戏主循环"""
    game = init_game()

    snake = game['snake']
    direction = game['direction']
    next_direction = game['next_direction']
    food = game['food']
    score = game['score']
    game_over = game['game_over']
    width = game['width']
    height = game['height']

    while not game_over:
        # 渲染
        render(snake, food, score, width, height)

        # 获取输入（方向缓冲）
        key = get_input(direction)
        if key is not None:
            next_direction = key

        # 移动
        direction = next_direction
        new_head, ate_food = move_snake(snake, direction, food)

        # 碰撞检测
        if check_collision(new_head, snake, width, height):
            game_over = True
            break

        # 吃食物
        if ate_food:
            score += 1
            food = generate_food(snake, width, height)

        # 等待
        time.sleep(TICK)

    show_game_over(score)


def main():
    """主入口"""
    os.system('cls')
    print('=' * 40)
    print(' ' * 10 + '贪吃蛇 Snake Game')
    print('=' * 40)
    print()
    print('  WASD 控制方向：')
    print('    W - 上')
    print('    A - 左')
    print('    S - 下')
    print('    D - 右')
    print()
    print('  吃到 $ 得分，撞墙或撞自己游戏结束')
    print()
    print(' ' * 6 + '按任意键开始游戏...')
    msvcrt.getch()

    game_loop()


if __name__ == '__main__':
    main()
