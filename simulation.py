import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import time


class people_flow():
    '''
    このクラスでは、Social Force Model (SFM)を用いて、指定した人数の人々が入口から入って出口から出るまでをシミュレーションする。
    This class simulates people's flow from the entrance to the exit,
    using Social Force Model.
    '''

    def __init__(self, people_num, v_arg, repul_h, repul_m, target, R,
                 min_p, p_arg, wall_x, wall_y, in_target_d, dt, disrupt_point=None, save_format=None, save_params=None):
        '''
        入力された引数をプロパティに格納する。
        Keep the input parameter as property

        people_num: シミュレーションする人数。
        people_num: The number of people who enter the simulated area

        v_arg: 人の速さに関連する2つの要素を持つリスト型の変数。1つ目の要素は平均の速さ、2つ目の要素は速さの標準偏差を表す。
        v_arg: List type argument that has two elements. This is about people's speed.
        The first element is mean speed of people, and
        the second is standard deviation of people's speed.

        repul_h: 人の間の反発力に関連する2つの要素を持つリスト型の変数。
        (詳細:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)
        repul_h: List type argument that has two elements. This is about repulsion between people.
        The first and second element is the parameters that contribute to the repulsion.
        (Details:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)

        repul_m: 人と壁の間の反発力に関連する2つの要素を持つリスト型の変数。
        (詳細:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)
        repul_m: List type argument that has two elements. This is about repulsion between a person and a wall.
        The first and second element is the parameters that contribute to the repulsion.
        (Details:人流シミュレーションのパラメータ推定手法(https://db-event.jpn.org/deim2017/papers/146.pdf)
        3.人流シミュレーションモデル 3.1 Social Force Model)

        target: 形が(N,2)のリスト型の変数。Nは目的地の数。2次元目の要素は目的地のxy座標を表す。最後の目的地は出口と見なされる。
        target: Multidimensional list type argument, the shape of which is (N, 2).
        N is the number of place people stop by at. The second dimension's elements are XY coordinates
        that represent the location of the place.

        R: 人(粒子として表される)の半径
        R: The radius of a person (they are regarded as a particle in the model).

        min_p: 目的地にいる人が次の目的地に移動する確率の最小値。この確率の逆数が目的地での滞在時間の期待値として使われる。
        min_p: This parameter represents minimum probability of people's going to the next destination.
        The inverse of the probability is used as time to stay at the destination.

    　　 p_arg: 目的地にいる人が次の目的地に移動する確率を決定する2次元のリスト型の変数。1次元目の要素数は目的地の数から1を引いた数の因数。
        2次元目の1つ目の要素は確率の平均、2つ目の要素は標準偏差となる。この確率の逆数が目的地に滞在する時間の期待値として使われる。
        なお、1次元目の要素数が目的地の数から1引いた数より小さいとき、numpyのようにブロードキャストしてしようされる。
        p_arg: Two dimensional list type argument. The first dimension has the elements of a factor of target's first dimension minus 1.
        These arguments designate averages and standard deviations of normal distributions, which represent the probability with which people go to
         the next destination.
        The first element of the second dimension designates an average, and the second designates a standard deviation.
        The inverse of the probability is used as time to stay at the destination.

        wall_x: 壁のx座標(左端は0であり右端がこの変数により決定される)
        wall_x: The X-coordinate of the wall (the left end is 0, and the right end is designated by this parameter.)

        wall_y: 壁のy座標(下端は0であり上端がこの変数により決定される)
        wall_y: The Y-coordinate of the wall (the bottom is 0, the top is designated by this parameter.)

        in_target_d: 人と目的地の間の距離がこの変数より小さければ、その人は目的地に到着したと見なす。)
        in_target_d: If the distance between a person and the destination is smaller than in_target_d,
        it is considered that the person reaches the destination.

        dt: 人の状態(位置、速さなど)を更新するときに利用する微小時間の大きさ
        dt: Infinitesimal difference of time that represents how frequently the model updates
        conditions of people.

        disrupt_point: 経過時間がこの変数を超えたらシミュレーションを停止する。(シミュレーション1/dt回で経過時間の1単位となる)
        disrupt_point: Simulation is disrupted when passed time has surpassed "disrupt_point."
        Passed time is calculated by multiplying the inverse of "dt" by times of simulation.

        save_format: シミュレーション結果を保存する形式を指定する。現在は"heat_map"だけしか使えません。"None"であれば結果を保存しません。
        save_format: A format in which the results of simulation will be saved. Only "heat_map" is allowed now.
        If "None," the results won't be saved.

        save_params: 結果を保存するのに使う変数。"heat_map"であれば2つの要素を持ったリスト型の変数が必要で、
        1つ目の要素はヒートマップの行数と列数を指定するタプル、2つ目の要素は保存する頻度を指定する変数。頻度の単位は"disrupt_point"と同じ。
        save_params: Parameters that are used when the results are saved. If save_format is "heat_map," a list of two elements is needed.
        One is a tuple that designates the number of rows and columns of heat maps,
        and another is a float or int type number that represents an interval at which how frequently the model saves the result.
        The unit of frequency is the same as "disrupt_point."
        '''

        try:
            self.people_num = people_num

            self.v_arg = np.asarray(v_arg)
            if len(v_arg) != 2:
                raise TypeError("The length of v_arg is mistaken.")

            self.repul_h = np.asarray(repul_h)
            if len(repul_h) != 2:
                raise TypeError("The length of repul_h is mistaken.")

            self.repul_m = np.asarray(repul_m)
            if len(repul_m) != 2:
                raise TypeError("The length of repul_m is mistaken.")

            self.target = np.asarray(target)

            self.R = R
            self.min_p = min_p

            self.p_arg = p_arg
            if (len(self.target) - 1) % len(p_arg) != 0:
                raise TypeError("The length of p_arg is mistaken.")

            self.wall_x = wall_x
            self.wall_y = wall_y
            self.in_target_d = in_target_d

            if save_format != None:
                self.save_format = save_format
                if save_format != "heat_map":
                    raise ValueError("The save format you designated is not allowed.")

            if save_params != None:
                self.save_params = save_params
                if save_format == "heat_map":
                    if len(save_params) != 2:
                        raise TypeError("The length of save_params is mistaken.")
                    if wall_x % save_params[0][0] != 0 or wall_y % save_params[0][1] != 0:
                        raise ValueError("The shape of the heat map is mistaken.")
                    if save_params[1] < dt:
                        raise ValueError("The interval of saving results is shorter than that of updating results.")

            self.dt = dt
            self.disrupt_point = disrupt_point

        except TypeError as error:
            print(error)
        except ValueError as error:
            print(error)

    def __sincos(self, x1, x2, on_paint=True):
        '''
        sinθとcosθを計算する。θはx1とx2を結ぶ直線とx軸との角度。
        Calculates sinθ and cosθ. Theta is the angle between x axis and a line between x1 and x2.
        '''

        # x1,x2の組の個数が2以上かそうでないかで場合分けする
        if x1.ndim == 2:
            if x2.ndim == 2:
                # 0除算を防ぐためにシミュレーションされていない人に対してははrに1を足す
                r = np.sqrt((x2[:, 0] - x1[:, 0]) ** 2 + (x2[:, 1] - x1[:, 1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[:, 1] - x1[:, 1]) / r
                cos = on_paint * (x2[:, 0] - x1[:, 0]) / r
                return sin, cos
            else:
                r = np.sqrt((x2[0] - x1[:, 0]) ** 2 + (x2[1] - x1[:, 1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[1] - x1[:, 1]) / r
                cos = on_paint * (x2[0] - x1[:, 0]) / r
                return sin, cos

        else:
            if x2.ndim == 2:
                # 0除算を防ぐためにシミュレーションされていない人に対してははrに1を足す
                r = np.sqrt((x2[:, 0] - x1[0]) ** 2 + (x2[:, 1] - x1[1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[:, 1] - x1[1]) / r
                cos = on_paint * (x2[:, 0] - x1[0]) / r
                return sin, cos
            else:
                r = np.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[1] - x1[1]) / r
                cos = on_paint * (x2[0] - x1[0]) / r
                return sin, cos

    def __force(self, x, v, v_opt, target_num, in_target, on_paint):
        '''
        人に働く力を計算する
        calculates force acting on people
        '''

        sin, cos = self.__sincos(x, self.target[target_num], on_paint)
        fx = on_paint * (v_opt * cos - v[:, 0])
        fy = on_paint * (v_opt * sin - v[:, 1])

        # 人の間に働く力を計算する
        # calculate repulsion between people
        for i in range(len(x)):
            if not on_paint[i]:
                continue

            on_paint[i] = False
            sin, cos = self.__sincos(x[i], x, on_paint)
            on_paint[i] = True
            r = np.sqrt((x[:, 0] - x[i, 0]) ** 2 + (x[:, 1] - x[i, 1]) ** 2)
            f_repul_h = self.repul_h[0] * (math.e ** (-r / self.repul_h[1]))
            fx_repul_h = f_repul_h * cos
            fy_repul_h = f_repul_h * sin

            # 人から遠ざかる方向に反発力が働くのでマイナスにする
            fx[i] -= np.sum(fx_repul_h)
            fy[i] -= np.sum(fy_repul_h)

        # 人と壁の間に働く力を計算する
        # calculate repulsion between a person and walls
        fx += self.repul_m[0] * (math.e ** (-x[:, 0] / self.repul_m[1]))
        fx -= self.repul_m[0] * (math.e ** (-(self.wall_x - x[:, 0]) / self.repul_m[1]))

        return fx, fy

    def __calculate(self, x, v, v_opt, p, target_num,
                    target, in_target, stay_target, on_paint):
        '''
        人の状態を更新する
        updates conditions of people
        '''

        # シミュレーションする場所にいない人は更新しない
        # if a person is not on the area, it doesn't update
        x[:, 0] += on_paint * v[:, 0] * self.dt
        x[:, 1] += on_paint * v[:, 1] * self.dt

        # 誤って壁の位置より飛び出ていたら壁の内側に戻す
        for i in range(len(x)):
            if x[i, 0] > self.wall_x:
                x[i, 0] = self.wall_x
            if x[i, 0] < 0:
                x[i, 0] = 0
            if x[i, 1] > self.wall_y:
                x[i, 1] = self.wall_y
            if x[i, 1] < 0:
                x[i, 1] = 0

        fx, fy = self.__force(x, v, v_opt, target_num, in_target, on_paint)

        v[:, 0] += fx * self.dt
        v[:, 1] += fy * self.dt

        # 目的地と人の間の距離を計算する
        target_d = np.sqrt((self.target[target_num, 0] - x[:, 0]) ** 2
                           + (self.target[target_num, 1] - x[:, 1]) ** 2)
        for i in range(len(x)):
            if not on_paint[i]:
                continue
            # 目的地が出口であればそのまま続ける
            # if people's target is the exit, they continue until they are out of range.
            if target_num[i] == len(self.target) - 1:
                continue

            # 目的地との距離が"in_target_d"より近ければ到着したと見なす
            if target_d[i] < self.in_target_d:
                in_target[i] = True

            if in_target[i]:
                stay_target[i] += self.dt
                # 滞在時間の期待値が過ぎれば次の目的地に進む
                if stay_target[i] > (1 / p[i, target_num[i]]):
                    target_num[i] += 1
                    in_target[i] = False
                    stay_target[i] = 0.0
                # 滞在時間が過ぎる前に遠ざかったら、また近づくようにする
                if target_d[i] > self.in_target_d:
                    in_target[i] = False

        return x, v, target_num, in_target, stay_target

    def __initialize(self):
        '''
        シミュレーションに使う変数を初期化する
        initializes parameters
        '''
        x = list()
        v_opt = list()
        v = list()
        p = list()
        target_num = list()
        in_target = list()
        stay_target = list()
        on_paint = list()
        for i in range(self.people_num):
            # 入口を広めにして、スタート位置はランダムにする
            x.append([(self.wall_x * 0.1) * np.random.rand() + self.wall_x * 0.45, self.wall_y])
            v_opt.append(abs(np.random.normal(loc=self.v_arg[0], scale=self.v_arg[1])))
            v.append([0, -v_opt[i]])

            p_target = list()
            for k in range(len(self.p_arg)):
                for m in range((len(self.target) - 1) // len(self.p_arg)):
                    p_candidate = np.random.normal(loc=self.p_arg[k][0], scale=self.p_arg[k][1])
                    # 次の目的地に移動する確率が"min_p"より小さかったり1より大きかったりしたらその値にする
                    if p_candidate < self.min_p:
                        p_candidate = self.min_p
                    elif p_candidate > 1:
                        p_candidate = 1
                    p_target.append(p_candidate)
            p_target.append(self.min_p)
            p.append(p_target)

            target_num.append(0)
            in_target.append(False)
            stay_target.append(0.0)
            on_paint.append(False)

        # numoy.ndarrayにする
        x = np.asarray(x)
        v_opt = np.asarray(v_opt)
        v = np.asarray(v)
        p = np.asarray(p)
        target_num = np.asarray(target_num, dtype=int)
        in_target = np.asarray(in_target)
        stay_target = np.asarray(stay_target)
        on_paint = np.asarray(on_paint, dtype=bool)
        return x, v_opt, v, p, target_num, in_target, stay_target, on_paint

    def __start_paint(self, x, on_paint):
        '''
        入口が混んでいなければ入場を許可する
        allow people to enter the area if the entrance is not crowded
        '''
        for i in range(len(x)):
            if x[i, 1] == self.wall_y and on_paint[i] == False:
                for k in range(len(x)):
                    if on_paint[k] == True:
                        if np.abs(x[i, 0] - x[k, 0]) < self.R * 1.5 or np.abs(x[i, 1] - x[k, 1]) < self.R * 2:
                            break
                    if k == len(x) - 1:
                        on_paint[i] = True
        return on_paint

    def __judge_end(self, x, target_num, on_paint):
        '''
        出口に着いたら描画や計算をやめる
        全員出口に着いていなくなったら、シミュレーションを終了する
        if people reach the exit, they are painted no longer
        if all people are out of range, the model finishes to paint and the simulation ends
        '''
        target_d = np.sqrt((self.target[-1, 0] - x[:, 0]) ** 2
                           + (self.target[-1, 1] - x[:, 1]) ** 2)
        for i in range(len(x)):
            if target_d[i] < self.in_target_d and target_num[i] == len(self.target) - 1:
                on_paint[i] = False

        end_flag = False
        if np.sum(on_paint) == 0:
            end_flag = True

        return on_paint, end_flag

    def __paint(self, x, target, on_paint):
        '''
        エリアにいる人を全員描画する
        paints all people and targets on the area
        '''
        ax = plt.axes()
        plt.xlim(0, self.wall_x)
        plt.ylim(0, self.wall_y)
        for i in range(len(x)):
            if not on_paint[i]:
                continue
            # 人の描画
            particle = pt.Circle(xy=(x[i, 0], x[i, 1]), radius=self.R, fc='b', ec='b')
            ax.add_patch(particle)
        for i in range(len(target)):
            if i < len(target) - 1:
                # 目的地の描画
                obj = pt.Rectangle(xy=(target[i, 0] - self.R, target[i, 1] - self.R), width=self.R * 2,
                                   height=self.R * 2, fc='y', ec='y', fill=True)
                ax.add_patch(obj)
            else:
                # 出口の描画
                exit = pt.Rectangle(xy=(target[i, 0] - self.R / 2, target[i, 1]), width=self.R / 2,
                                    height=self.R / 2, fc='r', ec='r', fill=True)
                ax.add_patch(exit)
        # 入口の描画
        entrance = pt.Rectangle(xy=(self.wall_x * 0.45, self.wall_y * 0.99), width=self.wall_x * 0.1,
                                height=self.wall_y * 0.01, fc='r', ec='r', fill=True)

        ax.add_patch(entrance)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_color("coral")
        ax.spines["bottom"].set_color("coral")
        ax.spines["left"].set_color("coral")
        ax.spines["right"].set_color("coral")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.plot()
        plt.pause(interval=0.01)
        plt.gca().clear()

    def __heat_map(self, x, on_paint):
        '''
        各グリッドの中にいる人数を示すヒートマップを作る
        make a heat map that shows the number of people in each grid
        '''
        map = np.zeros(shape=(self.save_params[0][0], self.save_params[0][1]))
        # 指定した行、列1つあたりのx,yの範囲を計算する
        rough_x = self.wall_x / self.save_params[0][0]
        rough_y = self.wall_y / self.save_params[0][1]
        # 各人がヒートマップ上のどこにいるか計算する
        location_x = (x[:, 0] // rough_x).astype(int)
        location_y = (x[:, 1] // rough_y).astype(int)
        for i in range(len(x)):
            if on_paint[i]:
                # 計算した場所が保存するヒートマップの範囲外であったら範囲内にする(一番端にいると範囲外になる)
                if location_x[i] >= self.save_params[0][0]:
                    location_x[i] = self.save_params[0][0] - 1
                if location_x[i] < 0:
                    location_x[i] = 0
                if location_y[i] >= self.save_params[0][1]:
                    location_y[i] = self.save_params[0][1] - 1
                if location_y[i] < 0:
                    location_y[i] = 0
                map[location_x[i], location_y[i]] += 1

        return map

    def simulate(self):
        '''
        人流をシミュレーションする
        simulates people's flow
        '''
        # シミュレーションにかかった時間を記録する
        start = time.perf_counter()
        x, v_opt, v, p, target_num, in_target, stay_target, on_paint = self.__initialize()
        end_flag = False
        if self.save_format == "heat_map":
            self.maps = list()
            save_times = 0
            passed_time = 0

        while not end_flag:

            x, v, target_num, in_target, stay_target = \
                self.__calculate(x, v, v_opt, p, target_num,
                                 self.target, in_target, stay_target, on_paint)

            on_paint = self.__start_paint(x, on_paint)
            on_paint, end_flag = self.__judge_end(x, target_num, on_paint)
            if self.save_format == "heat_map":
                if passed_time > save_times * self.save_params[1]:
                    self.maps.append(self.__heat_map(x, on_paint))
                    save_times += 1
                passed_time += self.dt

            self.__paint(x, self.target, on_paint)

            # "disrupt_point"が指定されていたら、それを超えたときにシミュレーションを終了する
            if self.disrupt_point and passed_time > self.disrupt_point:
                print("The simulation was dirupted.")
                break

        # シミュレーションにかかった時間を記録
        end = time.perf_counter()
        # かかった時間を出力
        print("It took " + str(end - start) + " s.")
        return self.maps if self.save_format == "heat_map" else None