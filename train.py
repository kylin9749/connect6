# -*- coding: utf-8 -*-

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch



class TrainPipeline():
    def __init__(self, init_model=None, board_width=6, board_height=6,
                 n_in_row=4, n_playout=400, use_gpu=False, is_shown=False,
                 output_file_name="", game_batch_number=1500):
        # 游戏和棋盘参数
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # 训练参数
        self.learn_rate = 2e-3  #学习率α ：0.002
        self.lr_multiplier = 1.0  # 根据 KL散度 适应性的调整学习率 
        self.temp = 1.0  # 温度参数t
        self.n_playout = n_playout  # 每次move的 模拟playout次数
        self.c_puct = 5 #c_put常量
        self.buffer_size = 10000 #缓冲区大小
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size) 
        self.play_batch_size = 1 
        self.epochs = 5  #  每次 update 的 train_steps
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = game_batch_number #训练局数
        self.best_win_ratio = 0.0
        # 纯蒙特卡索搜索训练参数
        # 目的可以是作为真正训练的模型的对手
        self.pure_mcts_playout_num = 1000 #纯蒙特卡洛搜索模拟次数
        self.use_gpu = use_gpu 
        self.is_shown = is_shown
        self.output_file_name = output_file_name #输出的txt文件名
        #初始化神经网络
        self.policy_value_net = PolicyValueNet(self.board_width,
                                               self.board_height,
                                               model_file=init_model,
                                               use_gpu=self.use_gpu
                                               )
        #
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)


    def get_equi_data(self, play_data):
        """旋转和平移，增大本次play_data(因为同一个局面对应着其他三个等价局面)
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state]) #盘面旋转90度
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """为训练收集self-play的数据"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]#刚刚的zip元组类型转换成list
            self.episode_len = len(play_data) #事件数
            # 通过旋转平移得到四个等价的局面，对本次数据进行增大/增元 （augment）
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)#将增大后的数据存入缓冲队列

    def policy_update(self):
        """使用训练结果取更新策略网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size) #从data_buffer中，随机获取batch_size个元素
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        通过和纯MCTS玩家对战评估当前策略
        Note: 这仅仅是为了监控训练的进程
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=self.is_shown)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """开始训练"""
        #打开两个txt文件
        with open("info/"+str(self.board)+"_loss_"+self.output_file_name+".txt",'w') as loss_file:
            loss_file.write("self-play次数,loss,entropy\n")
        with open("info/"+str(self.board)+"_win_ration"+self.output_file_name+".txt", 'w') as win_ratio_file:
            win_ratio_file.write("self-play次数, pure_MCTS战力， 胜率\n")
        
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)#执行一次重0到分出胜负的模拟，并收集数据
                print("对局 i:{}, 事件数（走了多少步）:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    with open("info/" + str(self.board) + "_loss_" + self.output_file_name + ".txt", 'a') as loss_file:
                        loss_file.write(str(i+1)+','+str(loss)+','+str(entropy)+'\n')
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("当前的 self-play 对局: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    with open("info/" + str(self.board) + "_win_ration" + self.output_file_name + ".txt",
                              'a') as win_ratio_file:
                        win_ratio_file.write(str(i+1)+','+str(self.pure_mcts_playout_num)+','+str(win_ratio)+'\n')
                    self.policy_value_net.save_model('./model/'+str(self.board_height)
                                                     +'_'+str(self.board_width)
                                                     +'_'+str(self.n_in_row)+
                                                     '_current_policy_'+output_file_name+'.model')
                    if win_ratio >= self.best_win_ratio:
                        print("更好的策略产生!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./model/'+str(self.board_height)
                                                     +'_'+str(self.board_width)
                                                     +'_'+str(self.n_in_row)+
                                                     '_best_policy_'+output_file_name+'.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 50000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')
        loss_file.close()
        win_ratio_file.close()

def usage():
    print("-s 设置棋盘大小，默认为6")
    print("-r 设置是几子棋，默认为4")
    print("-m 设置每步棋执行MCTS模拟的次数，默认为400")
    print("-o 训练好的模型存入文件的标识符（注意：程序会根据模型的参数自动生成文件名的前半部分）")
    print("-n 设置训练局数，默认为1500")
    print("--use_gpu 使用GPU进行训练")
    print("--graphics 当进行模型评估时，显示对战界面")


if __name__ == '__main__':
    import sys, getopt

    height = 10
    width = 10
    n_in_row = 6
    use_gpu = False
    n_playout = 800
    is_shown = False
    output_file_name = ""
    game_batch_number = 1500
    init_model_name = None
    battle=False

    opts, args = getopt.getopt(sys.argv[1:], "hs:r:m:go:n:i:", ["use_gpu", "graphics"])
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
        elif op == "-g" or op == "--graphics":
            is_shown = True
        elif op == "-o":
            output_file_name = value
        elif op == "-i":
            init_model_name = value
        elif op == "-n":
            game_batch_number = int(value)

    training_pipeline = TrainPipeline(board_height=height, board_width=width,
                                      n_in_row=n_in_row, use_gpu=use_gpu,
                                      n_playout=n_playout, is_shown=is_shown,
                                      output_file_name=output_file_name,
                                      init_model=init_model_name,
                                      game_batch_number=game_batch_number)
    training_pipeline.run()
