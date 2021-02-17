import optuna
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import time
import pickle

from IPython.display import clear_output

from . import simulation_colab

class inference_framework():
    '''
    このクラスでは、assessment_framework.pyにあるクラスassess_frameworkによって求められた最適な目的関数を用いてパラメータ推定をする。

    This class infers parameters using objective functions decided by "assess_framework" in "assessment_framework.py."
    '''
    def __init__(self, maps, people_num, target, R, min_p, wall_x, wall_y, in_target_d, dt,
                 save_params, v_range, repul_h_range, repul_m_range, p_range, n_trials):
        '''
        maps: 3次元numpy.ndarray。パラメータ推定する人流データを表現するヒートマップ(各時刻においてGridの中にいる人数を表す)。
        
        people_num, target, R, min_p, wall_x, wall_y, in_target_d,
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
        self.target = target
        self.R = R
        self.min_p = min_p
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
        model = simulation.people_flow(self.people_num, [v_ave, v_sd], [repul_h_arg1, repul_h_arg2],
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
        model = simulation.people_flow(self.people_num, [v_ave, v_sd], [repul_h_arg1, repul_h_arg2],
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

    def set_study(self, name, path):
        with open(path, 'rb') as f:
            self.study[name] = pickle.load(f)

    def save_opt(self, name):
        # optunaのオブジェクトstudy(パラメータの最適化に関するデータを持つ)をピクル化して保存
        filename = name + '.txt'
        with open(filename, 'wb') as f:
            pickle.dump(self.study[name], f)

    def whole_opt(self, infer_dict=None):
        # すべてのパラメータを最適化する
        if infer_dict is None:
            infer_dict = {"v":"ZNCC","repul_h":"ZNCC","repul_m":"KL","p":"ZNCC"}

        if "SSD" in infer_dict.values():
            self.__optimize(self.__SSD, self.n_trials, 'SSD')
            self.paint_opt('SSD')
            self.save_opt('SSD')
        if "SAD" in infer_dict.values():
            self.__optimize(self.__SAD, self.n_trials, 'SAD')
            self.paint_opt('SAD')
            self.save_opt('SAD')
        if "KL" in infer_dict.values():
            self.__optimize(self.__KL, self.n_trials, 'KL')
            self.paint_opt('KL')
            self.save_opt('KL')
        if "ZNCC" in infer_dict.values():
            self.__optimize(self.__ZNCC, self.n_trials, 'ZNCC')
            self.paint_opt('ZNCC')
            self.save_opt('ZNCC')

        inferred_params = dict()
        if "v" in infer_dict:
            inferred_params['average_of_v'] = self.study[infer_dict["v"]].best_params['averafe_of_v']
            inferred_params['standard_deviation_of_v'] = self.study[infer_dict["v"]].best_params['standard_deviation_of_v']
        if "repul_h" in infer_dict:
            inferred_params['arg1_of_repul_h'] = self.study[infer_dict["repul_h"]].best_params['arg1_of_repul_h']
            inferred_params['arg2_of_repul_h'] = self.study[infer_dict["repul_h"]].best_params['arg2_of_repul_h']
        if "repul_m" in infer_dict:
            inferred_params['arg1_of_repul_m'] = self.study[infer_dict["repul_m"]].best_params['arg1_of_repul_m']
            inferred_params['arg2_of_repul_m'] = self.study[infer_dict["repul_m"]].best_params['arg2_of_repul_m']
        if "p" in infer_dict:
            inferred_params['average_of_p'] = self.study[infer_dict["p"]].best_params['averafe_of_p']
            inferred_params['standard_deviation_of_p'] = self.study[infer_dict["p"]].best_params['standard_deviation_of_p']

        return inferred_params


class inference_framework_detail():
    '''
    このクラスでは、inference_frameworkで推定した結果を、さらに"assessment_framework.py"にあるassessment_framework_detailで求めた
    最適な目的関数を用いて推定する。評価時のように、1回の最適化では1種類のパラメータだけを推定して、他のパラメータは固定する。

    This class infers more exact parameters using objective functions decided by "assess_framework_detail" in "assessment_framework.py."
    As well as the assessment by "assessment_framework_detail," only one kind of parameter is inferred in each optimization.
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
        model = simulation.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
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
        model = simulation.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
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
        model = simulation.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
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
        model = simulation.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
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
        model = simulation.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
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
        model = simulation.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
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
        model = simulation.people_flow(self.people_num, [v_ave, v_sd], self.repul_h, self.repul_m, self.target, self.R,
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
        model = simulation.people_flow(self.people_num, self.v_arg, [repul_h_arg1, repul_h_arg2], self.repul_m, self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, [repul_m_arg1, repul_m_arg2], self.target,
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
        model = simulation.people_flow(self.people_num, self.v_arg, self.repul_h, self.repul_m, self.target, self.R,
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

    def set_study(self, name, path):
        with open(path, 'rb') as f:
            self.study[name] = pickle.load(f)

    def save_opt(self, name):
        # optunaのオブジェクトstudy(パラメータの最適化に関するデータを持つ)をピクル化して保存
        filename = name + '.txt'
        with open(filename, 'wb') as f:
            pickle.dump(self.study[name], f)

    def whole_opt(self, infer_dict=None):
        # すべてのパラメータを最適化する
        if infer_dict is None:
            infer_dict = {"v":"KL","repul_h":"ZNCC","repul_m":"SAD","p":"KL"}

        if "v" in infer_dict:
            if infer_dict["v"] == "SSD":
                self.__optimize(self.__SSD_v, self.n_trials, 'SSD_v')
                self.paint_opt('SSD_v')
                self.save_opt('SSD_v')
            if infer_dict["v"] == "SAD":
                self.__optimize(self.__SAD_v, self.n_trials, 'SAD_v')
                self.paint_opt('SAD_v')
                self.save_opt('SAD_v')
            if infer_dict["v"] == "KL":
                self.__optimize(self.__KL_v, self.n_trials, 'KL_v')
                self.paint_opt('KL_v')
                self.save_opt('KL_v')
            if infer_dict["v"] == "ZNCC":
                self.__optimize(self.__ZNCC_v, self.n_trials, 'ZNCC_v')
                self.paint_opt('ZNCC_v')
                self.save_opt('ZNCC_v')
        if "repul_h" in infer_dict:
            if infer_dict["repul_h"] == "SSD":
                self.__optimize(self.__SSD_repul_h, self.n_trials, 'SSD_repul_h')
                self.paint_opt('SSD_repul_h')
                self.save_opt('SSD_repul_h')
            if infer_dict["repul_h"] == "SAD":
                self.__optimize(self.__SAD_repul_h, self.n_trials, 'SAD_repul_h')
                self.paint_opt('SAD_repul_h')
                self.save_opt('SAD_repul_h')
            if infer_dict["repul_h"] == "KL":
                self.__optimize(self.__KL_repul_h, self.n_trials, 'KL_repul_h')
                self.paint_opt('KL_repul_h')
                self.save_opt('KL_repul_h')
            if infer_dict["repul_h"] == "ZNCC":
                self.__optimize(self.__ZNCC_repul_h, self.n_trials, 'ZNCC_repul_h')
                self.paint_opt('ZNCC_repul_h')
                self.save_opt('ZNCC_repul_h')
        if "repul_m" in infer_dict:
            if infer_dict["repul_m"] == "SSD":
                self.__optimize(self.__SSD_repul_m, self.n_trials, 'SSD_repul_m')
                self.paint_opt('SSD_repul_m')
                self.save_opt('SSD_repul_m')
            if infer_dict["repul_m"] == "SAD":
                self.__optimize(self.__SAD_repul_m, self.n_trials, 'SAD_repul_m')
                self.paint_opt('SAD_repul_m')
                self.save_opt('SAD_repul_m')
            if infer_dict["repul_m"] == "KL":
                self.__optimize(self.__KL_repul_m, self.n_trials, 'KL_repul_m')
                self.paint_opt('KL_repul_m')
                self.save_opt('KL_repul_m')
            if infer_dict["repul_m"] == "ZNCC":
                self.__optimize(self.__ZNCC_repul_m, self.n_trials, 'ZNCC_repul_m')
                self.paint_opt('ZNCC_repul_m')
                self.save_opt('ZNCC_repul_m')
        if "p" in infer_dict:
            if infer_dict["p"] == "SSD":
                self.__optimize(self.__SSD_p, self.n_trials, 'SSD_p')
                self.paint_opt('SSD_p')
                self.save_opt('SSD_p')
            if infer_dict["p"] == "SAD":
                self.__optimize(self.__SAD_p, self.n_trials, 'SAD_p')
                self.paint_opt('SAD_p')
                self.save_opt('SAD_p')
            if infer_dict["p"] == "KL":
                self.__optimize(self.__KL_p, self.n_trials, 'KL_p')
                self.paint_opt('KL_p')
                self.save_opt('KL_p')
            if infer_dict["p"] == "ZNCC":
                self.__optimize(self.__ZNCC_p, self.n_trials, 'ZNCC_p')
                self.paint_opt('ZNCC_p')
                self.save_opt('ZNCC_p')

        inferred_params = dict()
        if "v" in infer_dict:
            inferred_params['average_of_v'] = self.study[infer_dict["v"]].best_params['averafe_of_v']
            inferred_params['standard_deviation_of_v'] = self.study[infer_dict["v"]].best_params['standard_deviation_of_v']
        if "repul_h" in infer_dict:
            inferred_params['arg1_of_repul_h'] = self.study[infer_dict["repul_h"]].best_params['arg1_of_repul_h']
            inferred_params['arg2_of_repul_h'] = self.study[infer_dict["repul_h"]].best_params['arg2_of_repul_h']
        if "repul_m" in infer_dict:
            inferred_params['arg1_of_repul_m'] = self.study[infer_dict["repul_m"]].best_params['arg1_of_repul_m']
            inferred_params['arg2_of_repul_m'] = self.study[infer_dict["repul_m"]].best_params['arg2_of_repul_m']
        if "p" in infer_dict:
            inferred_params['average_of_p'] = self.study[infer_dict["p"]].best_params['averafe_of_p']
            inferred_params['standard_deviation_of_p'] = self.study[infer_dict["p"]].best_params['standard_deviation_of_p']

        return inferred_params
