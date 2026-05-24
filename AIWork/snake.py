"""
贪吃蛇游戏 — Windows 控制台版
使用 msvcrt 非阻塞输入 + ctypes 控制台 API
"""

import msvcrt
import sys
import time
import random
from ctypes import windll, byref, c_short, c_ulong, c_int, Structure, POINTER

# ─── 控制台 API ───────────────────────────────────────────────
STD_OUTPUT_HANDLE = -11
kernel32 = windll.kernel32
h_stdout = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)


class COORD(Structure):
    _fields_ = [("X", c_short), ("Y", c_short)]


class CONSOLE_CURSOR_INFO(Structure):
    _fields_ = [("dwSize", c_ulong), ("bVisible", c_int)]


def hide_cursor():
    """隐藏控制台光标"""
    info = CONSOLE_CURSOR_INFO()
    info.dwSize = 1
    info.bVisible = 0
    kernel32.SetConsoleCursorInfo(h_stdout, byref(info))


def show_cursor():
    """恢复显示控制台光标"""
    info = CONSOLE_CURSOR_INFO()
    info.dwSize = 25
    info.bVisible = 1
    kernel32.SetConsoleCursorInfo(h_stdout, byref(info))


def set_cursor(x, y):
    """将光标移动到控制台指定位置"""
    kernel32.SetConsoleCursorPosition(h_stdout, COORD(x, y))


# ─── 游戏常量 ──────────────────────────────────────────────────
WIDTH = 40   # 含边框
HEIGHT = 20  # 含边框
FPS = 0.15   # 每帧秒数

# 方向常量（dx, dy）
DIRECTIONS = {
    'w': (0, -1),
    's': (0, 1),
    'a': (-1, 0),
    'd': (1, 0),
}
# 反方向映射
OPPOSITE = {
    'w': 's',
    's': 'w',
    'a': 'd',
    'd': 'a',
}


# ─── 游戏核心 ──────────────────────────────────────────────────
class SnakeGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        # 蛇：坐标列表，索引 0 为蛇头
        start_x = WIDTH // 2 - 2
        start_y = HEIGHT // 2
        self.snake = [(start_x, start_y),
                      (start_x - 1, start_y),
                      (start_x - 2, start_y)]
        self.direction = 'd'   # 当前移动方向
        self.next_dir = 'd'    # 缓冲的下一个方向
        self.food = None
        self.score = 0
        self.game_over = False
        self._spawn_food()

    def _spawn_food(self):
        """在空闲位置随机生成食物"""
        occupied = set(self.snake)
        candidates = [(x, y)
                      for x in range(1, self.width - 1)
                      for y in range(1, self.height - 1)
                      if (x, y) not in occupied]
        if candidates:
            self.food = random.choice(candidates)
        else:
            # 没有空闲位置（蛇已占满），视为胜利
            self.game_over = True

    def set_direction(self, key):
        """设置移动方向，含反方向保护"""
        key = key.lower()
        if key in DIRECTIONS and key != OPPOSITE[self.direction]:
            self.next_dir = key

    def tick(self):
        """推进一帧，返回 True 表示游戏继续"""
        if self.game_over:
            return False

        # 应用方向
        self.direction = self.next_dir

        # 计算新蛇头位置
        dx, dy = DIRECTIONS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # 检查是否吃到食物
        ate = (new_head == self.food)

        # 构建新蛇
        if ate:
            self.snake = [new_head] + self.snake
            self.score += 1
        else:
            self.snake = [new_head] + self.snake[:-1]

        # 碰撞检测：撞墙
        nx, ny = new_head
        if nx <= 0 or nx >= self.width - 1 or ny <= 0 or ny >= self.height - 1:
            self.game_over = True
            return False

        # 碰撞检测：撞自身（注意吃食物时蛇变长，需排除蛇头自身）
        if self.snake.count(new_head) > 1:
            self.game_over = True
            return False

        # 食物被吃掉，生成新食物
        if ate:
            self._spawn_food()

        return True

    def build_frame(self):
        """构建当前帧的字符串（帧缓冲区）"""
        # 初始化空网格（全部填空格）
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # 画边框
        for x in range(self.width):
            grid[0][x] = '#'
            grid[self.height - 1][x] = '#'
        for y in range(self.height):
            grid[y][0] = '#'
            grid[y][self.width - 1] = '#'

        # 画蛇身（除蛇头外）
        for segment in self.snake[1:]:
            x, y = segment
            if 0 <= y < self.height and 0 <= x < self.width:
                grid[y][x] = 'O'

        # 画蛇头
        hx, hy = self.snake[0]
        if 0 <= hy < self.height and 0 <= hx < self.width:
            grid[hy][hx] = '@'

        # 画食物
        if self.food:
            fx, fy = self.food
            if 0 <= fy < self.height and 0 <= fx < self.width:
                grid[fy][fx] = '*'

        # 转字符串
        lines = [''.join(row) for row in grid]
        return '\n'.join(lines)


# ─── 主程序 ────────────────────────────────────────────────────
def main():
    # 设置控制台窗口大小与颜色（黑底绿字）
    import os as _os
    _os.system('mode con: cols=50 lines=25')
    _os.system('color 0A')
    sys.stdout.write('\033]0;贪吃蛇\007')
    sys.stdout.flush()

    try:
        # 隐藏光标
        hide_cursor()

        # 创建游戏
        game = SnakeGame()

        # 帧循环
        while not game.game_over:
            frame_start = time.time()

            # 非阻塞读取按键
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'w', b's', b'a', b'd',
                          b'W', b'S', b'A', b'D'):
                    game.set_direction(ch.decode())

            # 推进一帧
            game.tick()

            # 绘制画面
            set_cursor(0, 0)
            sys.stdout.write(game.build_frame())
            sys.stdout.write(f'\n得分: {game.score}')
            sys.stdout.flush()

            # 帧率控制
            elapsed = time.time() - frame_start
            sleep_time = FPS - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ─── 游戏结束画面 ────────────────────────────────────
        set_cursor(0, 0)
        frame = game.build_frame()
        lines = frame.split('\n')
        # 在画面中央叠加 GAME OVER 信息
        center_y = game.height // 2 - 1
        msg1 = '  GAME OVER  '
        msg2 = f'  得分: {game.score}  '
        start_x1 = (game.width - len(msg1)) // 2
        start_x2 = (game.width - len(msg2)) // 2

        overlay = []
        for i, line in enumerate(lines):
            if i == center_y:
                line = line[:start_x1] + msg1 + line[start_x1 + len(msg1):]
            elif i == center_y + 1:
                line = line[:start_x2] + msg2 + line[start_x2 + len(msg2):]
            overlay.append(line)

        sys.stdout.write('\n'.join(overlay))
        sys.stdout.write('\n按任意键退出...')
        sys.stdout.flush()

        # 等待按键
        msvcrt.getch()

    finally:
        # 确保退出时恢复光标显示
        show_cursor()
        set_cursor(0, game.height + 3)


if __name__ == '__main__':
    main()
