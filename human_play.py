# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
@modifier: Junguang Jiang
"""

from __future__ import print_function
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


# 请仔细阅读Human这个类
class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        """设置人类玩家的编号，黑：1，白：2"""
        self.player = p

    def get_action(self, board):
        """根据棋盘返回动作"""
        try:
            location = input("Your move: ") # 从键盘上读入位置，eg. "0,2"代表第0行第2列
            if isinstance(location, str):  # 如果location确实是字符串
                location = [int(n, 10) for n in location.split(",")] # 将location转换为对应的坐标点
            move = board.location_to_move(location) # 坐标点转换为一个一维的值move,介于[0,width*height)
        except Exception as e: #异常情况下
            move = -1
        if move == -1 or move not in board.availables: # 如果move值不合法
            print("invalid move")
            move = self.get_action(board) # 重新等待输入
        return move

    def __str__(self):
        return "Human {}".format(self.player)


# 以下函数可以略读
"""
@param n_in_row,width,height # 几子棋，棋盘宽度，高度
@model_file,ai_first # 载入的模型文件，是否AI先下棋
@n_playout,use_gpu # AI每次进行蒙特卡洛的模拟次数，是否使用GPU
"""
def run(n_in_row, width, height, 
        model_file, ai_first, 
        n_playout, use_gpu): 
    try:
        board = Board(width=width, height=height, n_in_row=n_in_row) # 产生一个棋盘
        game = Game(board) # 加载一个游戏

        # ############### 人类 VS AI ###################
        best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=use_gpu) # 加载最佳策略网络
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=n_playout) # 生成一个AI玩家
        human = Human() # 生成一个人类玩家

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=ai_first, is_shown=1) # 开始游戏
    except KeyboardInterrupt:
        print('\n\rquit')

def usage():
    print("-s 设置棋盘大小，默认为6")
    print("-r 设置是几子棋，默认为4")
    print("-m 设置每步棋执行MCTS模拟的次数，默认为400")
    print("-i ai使用哪个文件中的模型，默认为model/6_6_4_best_policy.model")
    print("--use_gpu 使用GPU进行运算")
    print("--human_first 让人类先下")


if __name__ == '__main__':
    import sys, getopt

    height = 10
    width = 10
    n_in_row = 6
    use_gpu = False
    n_playout = 800
    model_file = "model/10_10_6_best_policy_3.model"
    ai_first=True

    opts, args = getopt.getopt(sys.argv[1:], "hs:r:m:i:", ["use_gpu", "graphics", "human_first"])
    for op, value in opts:
        if op == "-h":
            usage()
            sys.exit()
        elif op == "-s":
            height = width = int(value)
        elif op == "-r":
            n_in_row = int(value)
        elif op == "--use_gpu":
            use_gpu = True
        elif op == "-m":
            n_playout = int(value)
        elif op == "-i":
            model_file = value
        elif op == "--human_first":
            ai_first=False
    run(height=height, width=width, n_in_row=n_in_row, use_gpu=use_gpu, n_playout=n_playout,
        model_file=model_file, ai_first=ai_first)
