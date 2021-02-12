import optuna
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import time
import pickle

from IPython.display import clear_output

from . import simulation_colab


class assess_framework():
    '''
    このクラスでは、パラメータ推定時の目的関数を評価し、最適な目的関数を求める。
    目的関数の候補はSSD,SAD,KL,ZNCCである。
    目的関数の評価では、まず適当なパラメータを定めて人流シミュレーションした結果をヒートマップとしてパラメータ推定モデルに入力する。
    そのモデルが推定したパラメータと正解のパラメータを比較した結果を評価することで目的関数の評価を行う。
    (参考:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
    5.各処理の組み合わせの評価)

    This class assesses objective functions used to optimize the arguments.
    The objective functions assessed in this class are SSD, SAD, KL, and ZNCC.
    (Reference:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
    5.各処理の組み合わせの評価)
    '''
    def __init__(self, maps, people_num, v_arg, repul_h, repul_m, target, R, min_p, p_arg, wall_x, wall_y, in_target_d,
                 dt,
                 save_params, v_range, repul_h_range, repul_m_range, p_range, n_trials):
        '''
        maps: 3次元numpy.ndarray。パラメータ推定する人流データを表現するヒートマップ(各時刻においてGridの中にいる人数を表す)。
        
        people_num, v_arg, repul_h, repul_m, target, R, min_p, p_arg, wall_x, wall_y, in_target_d,
        dt, save_params: 人流シミュレーションに使う変数。simulation.pyのpeople_flowを参照

        v_range: (2,2)の形であるリスト型の変数。人の速さの平均と標準偏差を推定するとき、それぞれがとりうる値の範囲を表す。

        repul_h_range: (2,2)の形であるリスト型の変数。人の間の反発力に関する2つのパラメータを推定するとき、それぞれがとりうる値の範囲を表す。
        (詳細:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)

        repul_m_range: (2,2)の形であるリスト型の変数。人と壁の間の反発力に関する2つのパラメータを推定するとき、それぞれがとりうる値の範囲を表す。
        (詳細:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)

        p_range: (2,2)の形であるリスト型の変数。次の目的地に移動する確率の平均と標準偏差を推定するとき、それぞれがとりうる値の範囲を表す。
        なお、パラメータ推定では、次の目的地に移動する確率はどの目的地においても等しいものとして計算する。

        n_trials: 推定したパラメータを最適化する回数
        '''
        self.maps = maps
        self.people_num = people_num
        self.v_arg = v_arg
        self.repul_h = repul_h
        self.repul_m = repul_m
        self.target = target
        self.R = R
        self.min_p = min_p
        self.p_arg = p_arg
        self.wall_x = wall_x
        self.wall_y = wall_y
        self.in_target_d = in_target_d
        self.dt = dt
        self.save_format = "heat_map"
        self.save_params = save_params

        self.v_range = v_range
        self.repul_h_range = repul_h_range
        self.repul_m_range = repul_m_range
        self.p_range = p_range
        self.n_trials = n_trials

        self.study = dict()

    def __SSD(self, trial):
        # 目的関数SSDの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], [repul_h_arg1, repul_h_arg2],
                                [repul_m_arg1, repul_m_arg2], self.target, self.R, self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum((ans - result) ** 2)

    def __SAD(self, trial):
        # 目的関数SADの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], [repul_h_arg1, repul_h_arg2],
                                [repul_m_arg1, repul_m_arg2], self.target, self.R, self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum(abs(ans - result))

    def __KL(self, trial):
        # 目的関数KLの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], [repul_h_arg1, repul_h_arg2],
                                [repul_m_arg1, repul_m_arg2], self.target, self.R, self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        result /= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1))
        # 0除算によりnanになっていたらその部分を除く
        if np.isnan(np.sum(result)):
            result = result[~np.isnan(result)].reshape((-1, 30, 30))
        ans /= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1))
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(result * np.log((result + epsilon) / (ans + epsilon)))

    def __ZNCC(self, trial):
        # 目的関数ZNCCの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], [repul_h_arg1, repul_h_arg2],
                                [repul_m_arg1, repul_m_arg2], self.target, self.R, self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        ans = np.asarray(ans)
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        result -= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1)) / (result.shape[1] * result.shape[2])
        ans -= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1)) / (ans.shape[1] * ans.shape[2])
        numerator = np.sum(result * ans, axis=(1, 2))
        denominator1 = np.sqrt(np.sum(result ** 2, axis=(1, 2)))
        denominator2 = np.sqrt(np.sum(ans ** 2, axis=(1, 2)))
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(numerator / (denominator1 * denominator2 + epsilon))

    def __optimize(self, objective, n_trials, name):
        # 推定パラメータを最適化する
        sampler = optuna.samplers.CmaEsSampler()
        self.study[name] = optuna.create_study(sampler=sampler)
        self.study[name].optimize(objective, n_trials=n_trials)

    def paint_opt(self, name):
        # 推定したパラメータの値と目的関数の値の関係、最適化回数とそれまでの最高の目的関数の値の関係をグラフ化
        # 推定したパラメータの値を格納するリスト
        x = dict()
        # 目的関数の値を格納するリスト
        y = list()
        # それまでの最高の目的関数の値を格納するリスト
        y_best = list()
        # 各パラメータの値のリストを作る
        for param_name in self.study[name].trials[0].params.keys():
            x[param_name] = list()
        for trial in self.study[name].trials:
            for param_name, param in trial.params.items():
                x[param_name].append(param)
            y.append(trial.value)
            y_best.append(np.min(y))
        param_num = len(self.study[name].trials[0].params)

        # 各グラフのサイズを(5,5)としてグラフ化
        plt.figure(figsize=((param_num + 1) * 5, 5))
        i = 1
        for param_name in self.study[name].trials[0].params.keys():
            ax = plt.subplot(1, param_num + 1, i)
            ax.set_xlabel(param_name)
            ax.set_ylabel('value')
            ax.scatter(x[param_name], y)
            i += 1
        ax = plt.subplot(1, param_num + 1, param_num + 1)
        ax.set_xlabel('trial')
        ax.set_ylabel('best_value')
        ax.plot(y_best)

    def save_opt(self, name):
        # optunaのオブジェクトstudy(パラメータの最適化に関するデータを持つ)をピクル化して保存
        filename = name + '.txt'
        with open(filename, 'wb') as f:
            pickle.dump(self.study[name], f)

    def whole_opt(self):
        # すべてのパラメータを最適化する
        self.__optimize(self.__SSD, self.n_trials, 'SSD')
        self.paint_opt('SSD')
        self.save_opt('SSD')

        self.__optimize(self.__SAD, self.n_trials, 'SAD')
        self.paint_opt('SAD')
        self.save_opt('SAD')

        self.__optimize(self.__KL, self.n_trials, 'KL')
        self.paint_opt('KL')
        self.save_opt('KL')

        self.__optimize(self.__ZNCC, self.n_trials, 'ZNCC')
        self.paint_opt('ZNCC')
        self.save_opt('ZNCC')

    def assess_v(self, ave, sd):
        # 積分を用いて、速さに関する推定したパラメータを評価
        dv = 0.1
        v = 0
        distance = 0
        while v <= self.v_range[0][1] + self.v_range[1][1] ** 3:
            a = np.exp(-(v - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
            b = (self.v_arg[1] / sd) * np.exp(
                (v - self.v_arg[0]) ** 2 / (2 * self.v_arg[1] ** 2) - (v - ave) ** 2 / (2 * sd ** 2))
            distance += a * np.log(b) * dv
            v += dv
        return distance

    def assess_repul_h(self, param1, param2):
        # 積分を用いて、人の間の反発力に関する推定したパラメータを評価
        dx = 0.1
        x = 0
        distance = 0
        while x <= self.wall_x / 10:
            distance += abs(param1 * np.exp(-x / param2) - self.repul_h[0] * np.exp(-x / self.repul_h[1])) * dx
            x += dx
        return distance

    def assess_repul_m(self, param1, param2):
        # 積分を用いて、人と壁の間の反発力に関する推定したパラメータを評価
        dx = 0.1
        x = 0
        distance = 0
        while x <= self.wall_x / 10:
            distance += abs(param1 * np.exp(-x / param2) - self.repul_m[0] * np.exp(-x / self.repul_m[1])) * dx
            x += dx
        return distance

    def assess_p(self, ave, sd):
        # 積分を用いて、次の目的地に移動する確率に関する推定したパラメータを評価
        dp = 0.1
        p = self.min_p
        distance = 0
        for i in range(len(self.p_arg)):
            while p <= 1:
                a = np.exp(-(p - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
                b = (self.p_arg[i][1] / sd) * np.exp(
                    (p - self.p_arg[i][0]) ** 2 / (2 * self.p_arg[i][1] ** 2) - (p - ave) ** 2 / (2 * sd ** 2))
                distance += a * np.log(b) * dp
                p += dp
            p = self.min_p
        distance /= len(self.p_arg)
        return distance

    def __get_best_params(self, name):
        # 各最適化において最高の目的関数の値を得たときのパラメータを返す
        best_params = list()
        for param in self.study[name].best_params.values():
            best_params.append(param)
        return best_params

    def assess(self):
        # 速さ、人の間の反発力、人と壁の間の反発力、次の目的地に移動する確率に関するパラメータをそれぞれ推定するのに最適な目的関数を決定する
        # そして、最適な目的関数の組み合わせを返す
        best_combi = dict()
        distance_v = dict()
        distance_repul_h = dict()
        distance_repul_m = dict()
        distance_p = dict()

        params = self.__get_best_params('SSD')
        distance_v['SSD'] = abs(self.assess_v(params[0], params[1]))
        distance_repul_h['SSD'] = abs(self.assess_repul_h(params[2], params[3]))
        distance_repul_m['SSD'] = abs(self.assess_repul_m(params[4], params[5]))
        distance_p['SSD'] = abs(self.assess_p(params[6], params[7]))

        params = self.__get_best_params('SAD')
        distance_v['SAD'] = abs(self.assess_v(params[0], params[1]))
        distance_repul_h['SAD'] = abs(self.assess_repul_h(params[2], params[3]))
        distance_repul_m['SAD'] = abs(self.assess_repul_m(params[4], params[5]))
        distance_p['SAD'] = abs(self.assess_p(params[6], params[7]))

        params = self.__get_best_params('KL')
        distance_v['KL'] = abs(self.assess_v(params[0], params[1]))
        distance_repul_h['KL'] = abs(self.assess_repul_h(params[2], params[3]))
        distance_repul_m['KL'] = abs(self.assess_repul_m(params[4], params[5]))
        distance_p['KL'] = abs(self.assess_p(params[6], params[7]))

        params = self.__get_best_params('ZNCC')
        distance_v['ZNCC'] = abs(self.assess_v(params[0], params[1]))
        distance_repul_h['ZNCC'] = abs(self.assess_repul_h(params[2], params[3]))
        distance_repul_m['ZNCC'] = abs(self.assess_repul_m(params[4], params[5]))
        distance_p['ZNCC'] = abs(self.assess_p(params[6], params[7]))

        best_combi[min(distance_v, key=distance_v.get) + '_v'] = self.__get_best_params(
            min(distance_v, key=distance_v.get))
        best_combi[min(distance_repul_h, key=distance_repul_h.get) + '_repul_h'] = self.__get_best_params(
            min(distance_repul_h, key=distance_repul_h.get))
        best_combi[min(distance_repul_m, key=distance_repul_m.get) + '_repul_m'] = self.__get_best_params(
            min(distance_repul_m, key=distance_repul_m.get))
        best_combi[min(distance_p, key=distance_p.get) + '_p'] = self.__get_best_params(
            min(distance_p, key=distance_p.get))

        return best_combi

    def __graph_v(self, ave, sd):
        # 平均と標準偏差から、速さの正規分布を満たすx,yの値を返す。グラフを作るのに用いる。
        dv = 0.1
        v = 0
        x = list()
        y = list()
        while v <= self.v_range[0][1] + self.v_range[1][1] ** 3:
            a = np.exp(-(v - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
            x.append(v)
            y.append(a)
            v += dv
        return x, y

    def __graph_repul_h(self, param1, param2):
        # 2つのパラメータから、人の間の反発力の関数を満たすx,yの値を返す。グラフを作るのに用いる。
        dr = 0.1
        r = 0
        x = list()
        y = list()
        while r <= self.wall_x / 10:
            x.append(r)
            y.append(param1 * np.exp(-r / param2))
            r += dr
        return x, y

    def __graph_repul_m(self, param1, param2):
        # 2つのパラメータから、人と壁の間の反発力の関数を満たすx,yの値を返す。グラフを作るのに用いる。
        dr = 0.1
        r = 0
        x = list()
        y = list()
        while r <= self.wall_x / 10:
            x.append(r)
            y.append(param1 * np.exp(-r / param2))
            r += dr
        return x, y

    def __graph_p(self, ave, sd):
        # 平均と標準偏差から、次の目的地に移動する確率の正規分布を満たすx,yの値を返す。グラフを作るのに用いる。
        dp = 0.1
        p = self.min_p
        x = list()
        y = list()
        while p <= 1:
            a = np.exp(-(p - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
            x.append(p)
            y.append(a)
            p += dp
        return x, y

    def __assess_v_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、速さの正規分布のグラフを描画する。
        names = ['SSD', 'SAD', 'KL', 'ZNCC']
        i = 1

        for name in names:
            params = self.__get_best_params(name)[:2]
            x_guess, y_guess = self.__graph_v(params[0], params[1])
            x_real, y_real = self.__graph_v(self.v_arg[0], self.v_arg[1])
            ax = plt.subplot(whole_row, 4, i + (row - 1) * 4)
            ax.plot(x_guess, y_guess, label='inferred')
            ax.plot(x_real, y_real, label='correct')
            ax.set_xlabel(name + '_v')
            ax.set_ylabel('probability')
            ax.legend()
            i += 1

    def __assess_repul_h_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、人の間の反発力のグラフを描画する。
        names = ['SSD', 'SAD', 'KL', 'ZNCC']
        i = 1

        for name in names:
            params = self.__get_best_params(name)[2:4]
            x_guess, y_guess = self.__graph_repul_h(params[0], params[1])
            x_real, y_real = self.__graph_repul_h(self.repul_h[0], self.repul_h[1])
            ax = plt.subplot(whole_row, 4, i + (row - 1) * 4)
            ax.plot(x_guess, y_guess, label='inferred')
            ax.plot(x_real, y_real, label='correct')
            ax.set_xlabel(name + '_repul_h')
            ax.set_ylabel('force')
            ax.legend()
            i += 1

    def __assess_repul_m_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、人と壁の間の反発力のグラフを描画する。
        names = ['SSD', 'SAD', 'KL', 'ZNCC']
        i = 1

        for name in names:
            params = self.__get_best_params(name)[4:6]
            x_guess, y_guess = self.__graph_repul_m(params[0], params[1])
            x_real, y_real = self.__graph_repul_m(self.repul_m[0], self.repul_m[1])
            ax = plt.subplot(whole_row, 4, i + (row - 1) * 4)
            ax.plot(x_guess, y_guess, label='inferred')
            ax.plot(x_real, y_real, label='correct')
            ax.set_xlabel(name + '_repul_m')
            ax.set_ylabel('force')
            ax.legend()
            i += 1

    def __assess_p_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、次の目的地に移動する確率の正規分布のグラフを描画する。
        names = ['SSD', 'SAD', 'KL', 'ZNCC']
        k = 1

        for name in names:
            for i in range(len(self.p_arg)):
                params = self.__get_best_params(name)[6:]
                x_guess, y_guess = self.__graph_p(params[2 * i], params[2 * i + 1])
                x_real, y_real = self.__graph_p(self.p_arg[i][0], self.p_arg[i][1])
                ax = plt.subplot(whole_row, 4, k + i + (row - 1) * 4)
                ax.plot(x_guess, y_guess, label='inferred')
                ax.plot(x_real, y_real, label='correct')
                ax.set_xlabel(name + '_p')
                ax.set_ylabel('probability')
                ax.legend()
            k += 1

    def assess_paint(self):
        # 速さ、人の間の反発力、人と壁の間の反発力、確率に関するパラメータをそれぞれ4つの目的関数を用いて推定した結果をグラフにする。
        # 16個のグラフが表示される。
        plt.figure(figsize=(5 * 4, 5 * (3 + len(self.p_arg))))
        self.__assess_v_paint(1, 4)
        self.__assess_repul_h_paint(2, 4)
        self.__assess_repul_m_paint(3, 4)
        self.__assess_p_paint(3 + len(self.p_arg), 4)






class assess_framework_detail():
    '''
    このクラスでは、パラメータ推定時の目的関数を評価し、最適な目的関数を求める。
    目的関数の候補はSSD,SAD,KL,ZNCCである。
    ただし、このクラスでは、1種類のパラメータ(速さ、人の間の反発力、人と壁の間の反発力、確率のうち1つ)のみ推定し、他の種類のパラメータは正解
    を与えた上で推定する。
    したがって、最適化は16回行われることになる。(4種類それぞれのパラメータにつき4つの目的関数を試す)
    (参考:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
    5.各処理の組み合わせの評価)

    This class assesses objective functions used to optimize the arguments.
    The objective functions assessed in this class are SSD, SAD, KL, and ZNCC.
    There are four kinds of parameters for optimization, and in this class, only one kind of parameters is optimized and other parameters are given
    correct values in each optimization.
    Therefore, optimization takes place 16 times and the best combination of objective functions for each kind of parameters is decided.
    (Reference:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
    5.各処理の組み合わせの評価)
    '''
    def __init__(self, maps, people_num, v_arg, repul_h, repul_m, target, R, min_p, p_arg, wall_x, wall_y, in_target_d,
                 dt,
                 save_params, v_range, repul_h_range, repul_m_range, p_range, n_trials):
        '''
        maps: 3次元numpy.ndarray。パラメータ推定する人流データを表現するヒートマップ(各時刻においてGridの中にいる人数を表す)。
        
        people_num, v_arg, repul_h, repul_m, target, R, min_p, p_arg, wall_x, wall_y, in_target_d,
        dt, save_params: 人流シミュレーションに使う変数。simulation.pyのpeople_flowを参照

        v_range: (2,2)の形であるリスト型の変数。人の速さの平均と標準偏差を推定するとき、それぞれがとりうる値の範囲を表す。

        repul_h_range: (2,2)の形であるリスト型の変数。人の間の反発力に関する2つのパラメータを推定するとき、それぞれがとりうる値の範囲を表す。
        (詳細:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)

        repul_m_range: (2,2)の形であるリスト型の変数。人と壁の間の反発力に関する2つのパラメータを推定するとき、それぞれがとりうる値の範囲を表す。
        (詳細:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)

        p_range: (2,2)の形であるリスト型の変数。次の目的地に移動する確率の平均と標準偏差を推定するとき、それぞれがとりうる値の範囲を表す。
        なお、パラメータ推定では、次の目的地に移動する確率はどの目的地においても等しいものとして計算する。

        n_trials: 推定したパラメータを最適化する回数
        '''

        self.maps = maps
        self.people_num = people_num
        self.v_arg = v_arg
        self.repul_h = repul_h
        self.repul_m = repul_m
        self.target = target
        self.R = R
        self.min_p = min_p
        self.p_arg = p_arg
        self.wall_x = wall_x
        self.wall_y = wall_y
        self.in_target_d = in_target_d
        self.dt = dt
        self.save_format = "heat_map"
        self.save_params = save_params

        self.v_range = v_range
        self.repul_h_range = repul_h_range
        self.repul_m_range = repul_m_range
        self.p_range = p_range
        self.n_trials = n_trials

        self.study = dict()

    def __SSD_v(self, trial):
        # 速さに関するパラメータを推定するときの目的関数SSDの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum((ans - result) ** 2)

    def __SSD_repul_h(self, trial):
        # 人の間の反発力に関するパラメータを推定するときの目的関数SSDの実装
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum((ans - result) ** 2)

    def __SSD_repul_m(self, trial):
        # 人と壁の間の反発力に関するパラメータを推定するときの目的関数SSDの実装
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum((ans - result) ** 2)

    def __SSD_p(self, trial):
        # 次の目的地に移動する確率に関するパラメータを推定するときの目的関数SSDの実装
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum((ans - result) ** 2)

    def __SAD_v(self, trial):
        # 速さに関するパラメータを推定するときの目的関数SADの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum(abs(ans - result))

    def __SAD_repul_h(self, trial):
        # 人の間の反発力に関するパラメータを推定するときの目的関数SADの実装
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum(abs(ans - result))

    def __SAD_repul_m(self, trial):
        # 人と壁の間の反発力に関するパラメータを推定するときの目的関数SADの実装
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum(abs(ans - result))

    def __SAD_p(self, trial):
        # 次の目的地に移動する確率に関するパラメータを推定するときの目的関数SADの実装
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        return np.sum(abs(ans - result))

    def __KL_v(self, trial):
        # 速さに関するパラメータを推定するときの目的関数KLの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        result /= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1))
        # 0除算によりnanになっていたらその部分を除く
        if np.isnan(np.sum(result)):
            result = result[~np.isnan(result)].reshape((-1, 30, 30))
        ans /= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1))
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(result * np.log((result + epsilon) / (ans + epsilon)))

    def __KL_repul_h(self, trial):
        # 人の間の反発力に関するパラメータを推定するときの目的関数KLの実装
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        result /= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1))
        # 0除算によりnanになっていたらその部分を除く
        if np.isnan(np.sum(result)):
            result = result[~np.isnan(result)].reshape((-1, 30, 30))
        ans /= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1))
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(result * np.log((result + epsilon) / (ans + epsilon)))

    def __KL_repul_m(self, trial):
        # 人と壁の間の反発力に関するパラメータを推定するときの目的関数KLの実装
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        result = result / np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1))
        # 0除算によりnanになっていたらその部分を除く
        if np.isnan(np.sum(result)):
            result = result[~np.isnan(result)].reshape((-1, 30, 30))
        ans /= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1))
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(result * np.log((result + epsilon) / (ans + epsilon)))

    def __KL_p(self, trial):
        # 次の目的地に移動する確率に関するパラメータを推定するときの目的関数KLの実装
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        result = result / np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1))
        # 0除算によりnanになっていたらその部分を除く
        if np.isnan(np.sum(result)):
            result = result[~np.isnan(result)].reshape((-1, 30, 30))
        ans /= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1))
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(result * np.log((result + epsilon) / (ans + epsilon)))

    def __ZNCC_v(self, trial):
        # 速さに関するパラメータを推定するときの目的関数ZNCCの実装
        v_ave = trial.suggest_uniform("average_of_v", self.v_range[0][0], self.v_range[0][1])
        v_sd = trial.suggest_uniform("standard_deviation_of_v", self.v_range[1][0], self.v_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        ans = np.asarray(ans)
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        result -= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1)) / (result.shape[1] * result.shape[2])
        ans -= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1)) / (ans.shape[1] * ans.shape[2])
        numerator = np.sum(result * ans, axis=(1, 2))
        denominator1 = np.sqrt(np.sum(result ** 2, axis=(1, 2)))
        denominator2 = np.sqrt(np.sum(ans ** 2, axis=(1, 2)))
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(numerator / (denominator1 * denominator2 + epsilon))

    def __ZNCC_repul_h(self, trial):
        # 人の間の反発力に関するパラメータを推定するときの目的関数ZNCCの実装
        repul_h_arg1 = trial.suggest_uniform("arg1_of_repul_h", self.repul_h_range[0][0], self.repul_h_range[0][1])
        repul_h_arg2 = trial.suggest_uniform("arg2_of_repul_h", self.repul_h_range[1][0], self.repul_h_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        ans = np.asarray(ans)
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        result -= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1)) / (result.shape[1] * result.shape[2])
        ans -= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1)) / (ans.shape[1] * ans.shape[2])
        numerator = np.sum(result * ans, axis=(1, 2))
        denominator1 = np.sqrt(np.sum(result ** 2, axis=(1, 2)))
        denominator2 = np.sqrt(np.sum(ans ** 2, axis=(1, 2)))
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(numerator / (denominator1 * denominator2 + epsilon))

    def __ZNCC_repul_m(self, trial):
        # 人と壁の間の反発力に関するパラメータを推定するときの目的関数ZNCCの実装
        repul_m_arg1 = trial.suggest_uniform("arg1_of_repul_m", self.repul_m_range[0][0], self.repul_m_range[0][1])
        repul_m_arg2 = trial.suggest_uniform("arg2_of_repul_m", self.repul_m_range[1][0], self.repul_m_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
                                self.R, self.min_p,
                                self.p_arg, self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        ans = np.asarray(ans)
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        result -= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1)) / (result.shape[1] * result.shape[2])
        ans -= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1)) / (ans.shape[1] * ans.shape[2])
        numerator = np.sum(result * ans, axis=(1, 2))
        denominator1 = np.sqrt(np.sum(result ** 2, axis=(1, 2)))
        denominator2 = np.sqrt(np.sum(ans ** 2, axis=(1, 2)))
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(numerator / (denominator1 * denominator2 + epsilon))

    def __ZNCC_p(self, trial):
        # 次の目的地に移動する確率に関するパラメータを推定するときの目的関数ZNCCの実装
        p_ave = trial.suggest_uniform("average_of_p", self.p_range[0][0], self.p_range[0][1])
        p_sd = trial.suggest_uniform("standard_deviation_of_p", self.p_range[1][0], self.p_range[1][1])
        # "disrupt_point"を与えられた人流データの長さ分にする
        model = simulation_colab.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
                                self.min_p,
                                [[p_ave, p_sd]], self.wall_x, self.wall_y, self.in_target_d, self.dt,
                                len(self.maps) * self.save_params[1] + 1,
                                self.save_format, self.save_params)
        result = model.simulate()
        result = np.asarray(result)
        ans = self.maps.copy()
        ans = np.asarray(ans)
        # ヒートマップの時間方向の長さが一致しないとき、後方の長い部分を切り捨てる
        if len(ans) > len(result):
            ans = np.delete(ans, np.s_[len(result):], axis=0)
        elif len(result) > len(ans):
            result = np.delete(result, np.s_[len(ans):], axis=0)
        result -= np.sum(result, axis=(1, 2)).reshape((len(result), 1, 1)) / (result.shape[1] * result.shape[2])
        ans -= np.sum(ans, axis=(1, 2)).reshape((len(ans), 1, 1)) / (ans.shape[1] * ans.shape[2])
        numerator = np.sum(result * ans, axis=(1, 2))
        denominator1 = np.sqrt(np.sum(result ** 2, axis=(1, 2)))
        denominator2 = np.sqrt(np.sum(ans ** 2, axis=(1, 2)))
        # 0除算を防ぐための微小な値
        epsilon = 0.001
        return np.sum(numerator / (denominator1 * denominator2 + epsilon))

    def __optimize(self, objective, n_trials, name):
        # 推定パラメータを最適化する
        sampler = optuna.samplers.CmaEsSampler()
        self.study[name] = optuna.create_study(sampler=sampler)
        self.study[name].optimize(objective, n_trials=n_trials)

    def paint_opt(self, name):
        # 推定したパラメータの値と目的関数の値の関係、最適化回数とそれまでの最高の目的関数の値の関係をグラフ化
        # 推定したパラメータの値を格納するリスト
        x = dict()
        # 目的関数の値を格納するリスト
        y = list()
        # それまでの最高の目的関数の値を格納するリスト
        y_best = list()
        # 各パラメータの値のリストを作る
        for param_name in self.study[name].trials[0].params.keys():
            x[param_name] = list()
        for trial in self.study[name].trials:
            for param_name, param in trial.params.items():
                x[param_name].append(param)
            y.append(trial.value)
            y_best.append(np.min(y))
        param_num = len(self.study[name].trials[0].params)

        # 各グラフのサイズを(5,5)としてグラフ化
        plt.figure(figsize=((param_num + 1) * 5, 5))
        i = 1
        for param_name in self.study[name].trials[0].params.keys():
            ax = plt.subplot(1, param_num + 1, i)
            ax.set_xlabel(param_name)
            ax.set_ylabel('value')
            ax.scatter(x[param_name], y)
            i += 1
        ax = plt.subplot(1, param_num + 1, param_num + 1)
        ax.set_xlabel('trial')
        ax.set_ylabel('best_value')
        ax.plot(y_best)

    def save_opt(self, name):
        # optunaのオブジェクトstudy(パラメータの最適化に関するデータを持つ)をピクル化して保存
        filename = name + '.txt'
        with open(filename, 'wb') as f:
            pickle.dump(self.study[name], f)

    def whole_opt(self):

        # すべてのパラメータを最適化する
        self.__optimize(self.__SSD_v, self.n_trials, 'SSD_v')
        self.paint_opt('SSD_v')
        self.save_opt('SSD_v')

        self.__optimize(self.__SSD_repul_h, self.n_trials, 'SSD_repul_h')
        self.paint_opt('SSD_repul_h')
        self.save_opt('SSD_repul_h')

        self.__optimize(self.__SSD_repul_m, self.n_trials, 'SSD_repul_m')
        self.paint_opt('SSD_repul_m')
        self.save_opt('SSD_repul_m')

        self.__optimize(self.__SSD_p, self.n_trials, 'SSD_p')
        self.paint_opt('SSD_p')
        self.save_opt('SSD_p')

        self.__optimize(self.__SAD_v, self.n_trials, 'SAD_v')
        self.paint_opt('SAD_v')
        self.save_opt('SAD_v')

        self.__optimize(self.__SAD_repul_h, self.n_trials, 'SAD_repul_h')
        self.paint_opt('SAD_repul_h')
        self.save_opt('SAD_repul_h')

        self.__optimize(self.__SAD_repul_m, self.n_trials, 'SAD_repul_m')
        self.paint_opt('SAD_repul_m')
        self.save_opt('SAD_repul_m')

        self.__optimize(self.__SAD_p, self.n_trials, 'SAD_p')
        self.paint_opt('SAD_p')
        self.save_opt('SAD_p')

        self.__optimize(self.__KL_v, self.n_trials, 'KL_v')
        self.paint_opt('KL_v')
        self.save_opt('KL_v')

        self.__optimize(self.__KL_repul_h, self.n_trials, 'KL_repul_h')
        self.paint_opt('KL_repul_h')
        self.save_opt('KL_repul_h')

        self.__optimize(self.__KL_repul_m, self.n_trials, 'KL_repul_m')
        self.paint_opt('KL_repul_m')
        self.save_opt('KL_repul_m')

        self.__optimize(self.__KL_p, self.n_trials, 'KL_p')
        self.paint_opt('KL_p')
        self.save_opt('KL_p')

        self.__optimize(self.__ZNCC_v, self.n_trials, 'ZNCC_v')
        self.paint_opt('ZNCC_v')
        self.save_opt('ZNCC_v')

        self.__optimize(self.__ZNCC_repul_h, self.n_trials, 'ZNCC_repul_h')
        self.paint_opt('ZNCC_repul_h')
        self.save_opt('ZNCC_repul_h')

        self.__optimize(self.__ZNCC_repul_m, self.n_trials, 'ZNCC_repul_m')
        self.paint_opt('ZNCC_repul_m')
        self.save_opt('ZNCC_repul_m')

        self.__optimize(self.__ZNCC_p, self.n_trials, 'ZNCC_p')
        self.paint_opt('ZNCC_p')
        self.save_opt('ZNCC_p')

    def assess_v(self, ave, sd):
        # 積分を用いて、速さに関する推定したパラメータを評価
        dv = 0.1
        v = 0
        distance = 0
        while v <= self.v_range[0][1] + self.v_range[1][1] ** 3:
            a = np.exp(-(v - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
            b = (self.v_arg[1] / sd) * np.exp(
                (v - self.v_arg[0]) ** 2 / (2 * self.v_arg[1] ** 2) - (v - ave) ** 2 / (2 * sd ** 2))
            distance += a * np.log(b) * dv
            v += dv
        return distance

    def assess_repul_h(self, param1, param2):
        # 積分を用いて、人の間の反発力に関する推定したパラメータを評価
        dx = 0.1
        x = 0
        distance = 0
        while x <= self.wall_x / 10:
            distance += abs(param1 * np.exp(-x / param2) - self.repul_h[0] * np.exp(-x / self.repul_h[1])) * dx
            x += dx
        return distance

    def assess_repul_m(self, param1, param2):
        # 積分を用いて、人と壁の間の反発力に関する推定したパラメータを評価
        dx = 0.1
        x = 0
        distance = 0
        while x <= self.wall_x / 10:
            distance += abs(param1 * np.exp(-x / param2) - self.repul_m[0] * np.exp(-x / self.repul_m[1])) * dx
            x += dx
        return distance

    def assess_p(self, ave, sd):
        # 積分を用いて、次の目的地に移動する確率に関する推定したパラメータを評価
        dp = 0.1
        p = self.min_p
        distance = 0
        for i in range(len(self.p_arg)):
            while p <= 1:
                a = np.exp(-(p - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
                b = (self.p_arg[i][1] / sd) * np.exp(
                    (p - self.p_arg[i][0]) ** 2 / (2 * self.p_arg[i][1] ** 2) - (p - ave) ** 2 / (2 * sd ** 2))
                distance += a * np.log(b) * dp
                p += dp
            p = self.min_p
        distance /= len(self.p_arg)
        return distance

    def __get_best_params(self, name):
        # 各最適化において最高の目的関数の値を得たときのパラメータを返す
        best_params = list()
        for param in self.study[name].best_params.values():
            best_params.append(param)
        return best_params

    def assess(self):
        # 速さ、人の間の反発力、人と壁の間の反発力、次の目的地に移動する確率に関するパラメータをそれぞれ推定するのに最適な目的関数を決定する
        # そして、最適な目的関数の組み合わせを返す
        best_combi = dict()
        distance_v = dict()
        distance_repul_h = dict()
        distance_repul_m = dict()
        distance_p = dict()

        params = self.__get_best_params('SSD_v')
        distance_v['SSD_v'] = abs(self.assess_v(params[0], params[1]))
        params = self.__get_best_params('SAD_v')
        distance_v['SAD_v'] = abs(self.assess_v(params[0], params[1]))
        params = self.__get_best_params('KL_v')
        distance_v['KL_v'] = abs(self.assess_v(params[0], params[1]))
        params = self.__get_best_params('ZNCC_v')
        distance_v['ZNCC_v'] = abs(self.assess_v(params[0], params[1]))
        best_combi[min(distance_v, key=distance_v.get)] = self.__get_best_params(min(distance_v))

        params = self.__get_best_params('SSD_repul_h')
        distance_repul_h['SSD_repul_h'] = abs(self.assess_repul_h(params[0], params[1]))
        params = self.__get_best_params('SAD_repul_h')
        distance_repul_h['SAD_repul_h'] = abs(self.assess_repul_h(params[0], params[1]))
        params = self.__get_best_params('KL_repul_h')
        distance_repul_h['KL_repul_h'] = abs(self.assess_repul_h(params[0], params[1]))
        params = self.__get_best_params('ZNCC_repul_h')
        distance_repul_h['ZNCC_repul_h'] = abs(self.assess_repul_h(params[0], params[1]))
        best_combi[min(distance_repul_h, key=distance_repul_h.get)] = self.__get_best_params(min(distance_repul_h))

        params = self.__get_best_params('SSD_repul_m')
        distance_repul_m['SSD_repul_m'] = abs(self.assess_repul_m(params[0], params[1]))
        params = self.__get_best_params('SAD_repul_m')
        distance_repul_m['SAD_repul_m'] = abs(self.assess_repul_m(params[0], params[1]))
        params = self.__get_best_params('KL_repul_m')
        distance_repul_m['KL_repul_m'] = abs(self.assess_repul_m(params[0], params[1]))
        params = self.__get_best_params('ZNCC_repul_m')
        distance_repul_m['ZNCC_repul_m'] = abs(self.assess_repul_m(params[0], params[1]))
        best_combi[min(distance_repul_m, key=distance_repul_m.get)] = self.__get_best_params(min(distance_repul_m))

        params = self.__get_best_params('SSD_p')
        distance_p['SSD_p'] = abs(self.assess_p(params[0], params[1]))
        params = self.__get_best_params('SAD_p')
        distance_p['SAD_p'] = abs(self.assess_p(params[0], params[1]))
        params = self.__get_best_params('KL_p')
        distance_p['KL_p'] = abs(self.assess_p(params[0], params[1]))
        params = self.__get_best_params('ZNCC_p')
        distance_p['ZNCC_p'] = abs(self.assess_p(params[0], params[1]))
        best_combi[min(distance_p, key=distance_p.get)] = self.__get_best_params(min(distance_p))

        return best_combi

    def __graph_v(self, ave, sd):
        # 平均と標準偏差から、速さの正規分布を満たすx,yの値を返す。グラフを作るのに用いる。
        dv = 0.1
        v = 0
        x = list()
        y = list()
        while v <= self.v_range[0][1] + self.v_range[1][1] ** 3:
            a = np.exp(-(v - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
            x.append(v)
            y.append(a)
            v += dv
        return x, y

    def __graph_repul_h(self, param1, param2):
        # 2つのパラメータから、人の間の反発力の関数を満たすx,yの値を返す。グラフを作るのに用いる。
        dr = 0.1
        r = 0
        x = list()
        y = list()
        while r <= self.wall_x / 10:
            x.append(r)
            y.append(param1 * np.exp(-r / param2))
            r += dr
        return x, y

    def __graph_repul_m(self, param1, param2):
        # 2つのパラメータから、人と壁の間の反発力の関数を満たすx,yの値を返す。グラフを作るのに用いる。
        dr = 0.1
        r = 0
        x = list()
        y = list()
        while r <= self.wall_x / 10:
            x.append(r)
            y.append(param1 * np.exp(-r / param2))
            r += dr
        return x, y

    def __graph_p(self, ave, sd):
        # 平均と標準偏差から、次の目的地に移動する確率の正規分布を満たすx,yの値を返す。グラフを作るのに用いる。
        dp = 0.1
        p = self.min_p
        x = list()
        y = list()
        while p <= 1:
            a = np.exp(-(p - ave) ** 2 / (2 * sd ** 2)) / np.sqrt(2 * math.pi * sd ** 2)
            x.append(p)
            y.append(a)
            p += dp
        return x, y

    def __assess_v_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、速さの正規分布のグラフを描画する。
        names = ['SSD_v', 'SAD_v', 'KL_v', 'ZNCC_v']
        i = 1

        for name in names:
            params = self.__get_best_params(name)
            x_guess, y_guess = self.__graph_v(params[0], params[1])
            x_real, y_real = self.__graph_v(self.v_arg[0], self.v_arg[1])
            ax = plt.subplot(whole_row, 4, i + (row - 1) * 4)
            ax.plot(x_guess, y_guess, label='inferred')
            ax.plot(x_real, y_real, label='correct')
            ax.set_xlabel(name)
            ax.set_ylabel('probability')
            ax.legend()
            i += 1

    def __assess_repul_h_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、人の間の反発力のグラフを描画する。
        names = ['SSD_repul_h', 'SAD_repul_h', 'KL_repul_h', 'ZNCC_repul_h']
        i = 1

        for name in names:
            params = self.__get_best_params(name)
            x_guess, y_guess = self.__graph_repul_h(params[0], params[1])
            x_real, y_real = self.__graph_repul_h(self.repul_h[0], self.repul_h[1])
            ax = plt.subplot(whole_row, 4, i + (row - 1) * 4)
            ax.plot(x_guess, y_guess, label='inferred')
            ax.plot(x_real, y_real, label='correct')
            ax.set_xlabel(name)
            ax.set_ylabel('force')
            ax.legend()
            i += 1

    def __assess_repul_m_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、人と壁の間の反発力のグラフを描画する。
        names = ['SSD_repul_m', 'SAD_repul_m', 'KL_repul_m', 'ZNCC_repul_m']
        i = 1

        for name in names:
            params = self.__get_best_params(name)
            x_guess, y_guess = self.__graph_repul_m(params[0], params[1])
            x_real, y_real = self.__graph_repul_m(self.repul_m[0], self.repul_m[1])
            ax = plt.subplot(whole_row, 4, i + (row - 1) * 4)
            ax.plot(x_guess, y_guess, label='inferred')
            ax.plot(x_real, y_real, label='correct')
            ax.set_xlabel(name)
            ax.set_ylabel('force')
            ax.legend()
            i += 1

    def __assess_p_paint(self, row, whole_row):
        # 正解パラメータと推定パラメータについて、次の目的地に移動する確率の正規分布のグラフを描画する。
        names = ['SSD_p', 'SAD_p', 'KL_p', 'ZNCC_p']
        k = 1

        for name in names:
            for i in range(len(self.p_arg)):
                params = self.__get_best_params(name)
                x_guess, y_guess = self.__graph_p(params[2 * i], params[2 * i + 1])
                x_real, y_real = self.__graph_p(self.p_arg[i][0], self.p_arg[i][1])
                ax = plt.subplot(whole_row, 4, k + i + (row - 1) * 4)
                ax.plot(x_guess, y_guess, label='inferred')
                ax.plot(x_real, y_real, label='correct')
                ax.set_xlabel(name)
                ax.set_ylabel('probability')
                ax.legend()
            k += 1

    def assess_paint(self):
        # 速さ、人の間の反発力、人と壁の間の反発力、確率に関するパラメータをそれぞれ4つの目的関数を用いて推定した結果をグラフにする。
        # 16個のグラフが表示される。
        plt.figure(figsize=(5 * 4, 5 * (3 + len(self.p_arg))))
        self.__assess_v_paint(1, 4)
        self.__assess_repul_h_paint(2, 4)
        self.__assess_repul_m_paint(3, 4)
        self.__assess_p_paint(3 + len(self.p_arg), 4)
