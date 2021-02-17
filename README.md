# people_flow
論文「人流シミュレーションのパラメータ推定手法」( https://db-event.jpn.org/deim2017/papers/146.pdf )の実装(非公式、一部改変)  
・追記(2021/2/17) &nbsp; Google Colaboratoryでの実行例を公開しました。リンクは[こちら](https://colab.research.google.com/drive/10w2SaKBmlPgTQHj2IKeE6H5WmHPth04f?usp=sharing)

Unofficial implementation of "a way to infer parameters of people flow's simulation." (partially changed)( https://db-event.jpn.org/deim2017/papers/146.pdf )  
・P.S.(2021/2/17) &nbsp; I made a public Google Colaboratory page. This page demonstrates what I explain below. [Here](https://colab.research.google.com/drive/10w2SaKBmlPgTQHj2IKeE6H5WmHPth04f?usp=sharing) is the link.

# 概要(Outline)
上記のリンク先の論文の実装(非公式)  
要約すると、やったことは
1. Social Force Model (SFM)を用いた人流モデルの実装
2. 人流データのパラメータの最適化に使う、目的関数の評価モデルの実装
3. 人流データのパラメータを求めるモデルを、2.の結果をもとに構築
---
Unofficial implementation of the paper above  
To sum up, what I've done is
1. Constructing people flow's model using Social Force Model (SFM).
2. Constructing an assessing model of objective functions used for parameter optimization of actual people flow data applied to the simulation model.
3. Based on the result of 2., constructing a model inferring parameters of actual people flow data applied to the simulation model.

# 論文からの変更点(Changes from the paper)
1. 人流シミュレーションにおいて遷移パラメータ(論文参照)を目的地ごとに決めるのではなく人ごとに正規分布で決定した。
2. パラメータ推定方法の評価において、ヒートマップの作り方はGridのみを用い、Voronoiは使わなかった。また、ヒートマップのGridの目の粗さに関しても評価せず、固定した。
3. パラメータ推定において、論文通りの推定法に加え、1種類のパラメータのみを推定して他のパラメータを固定する推定方法も実装した。
---
1. Deciding "transition parameters" for each people based on normal distribution, not for each destination as in the paper.
2. For making heat maps, using only Grid style and not using Voronoi style. Furthermore, not assessing roughness of the Grid.
3. In addition to the inferring model in the paper, making a model that infers only one kind of parameter and others are given.

# パッケージとしての利用(Usage as a Package)
## 実行に必要なパッケージ
- numpy (version 1.19.2)
- matplotlib (version 3.3.2)
- optuna (version 2.4.0)
## simulation.py
人流のシミュレーションを行う。  
simulating people flow

1. 引数を指定(詳細はsimulation.pyを参照)  
Deciding parameters (details:simulation.py)
```python
people_num = 30
v_arg = [6,2]
repul_h = [5,5]
repul_m = [2,2]
target = [[60,240],[120,150],[90,60],[240,40],[200,120],[170,70],[150,0]]
R = 3
min_p = 0.1
p_arg = [[0.5,0.1]]
wall_x = 300
wall_y = 300
in_target_d = 3
dt = 0.1
save_format = "heat_map"
save_params = [(30,30),1]
```
2. シミュレーションするインスタンスを生成  
making an instance for simulation
```python
import people_flow
model = people_flow.simulation.people_flow(people_num,v_arg,repul_h,repul_m,target,R,min_p,p_arg,wall_x,wall_y,in_target_d,dt,save_format=save_format,save_params=save_params)
```
3. シミュレーションを実行(結果のヒートマップを得る)  
do simulation (getting heat maps as a result)
```python
maps = model.simulate()
```
mapsは3次元numpy.ndarray  
実行中、シミュレーション状況がmatplotlibにより描画される。  
"maps" is a three dimentional numpy.ndarray.  
While simulating, people flow is drawn by matlotlib.

## assessment_framework.py
パラメータ推定における目的関数の評価を行う。  
assessment_frameworkとassessment_framework_detailがありassessment_framework_detailは1種類ずつパラメータを推定して他を固定する方法を使っているが、パッケージとしての実行の手順は同じ。 

assessing objective functions in parameter inference  
There are two classess, assessment_framework and assessment_framework_detail, and assessment_framework_detail adopts a way to infer a kind of parameter in an optimization, but how to use them as a package is the same.

1. 引数を指定(詳細はassessment_framework.pyを参照)  
Deciding parameters (details:assessment_framework.py)
```python
people_num = 30
v_arg = [6,2]
repul_h = [5,5]
repul_m = [2,2]
target = [[60,240],[120,150],[90,60],[240,40],[200,120],[170,70],[150,0]]
R = 3
min_p = 0.1
p_arg = [[0.5,0.1]]
wall_x = 300
wall_y = 300
in_target_d = 3
dt = 0.1
save_params = [(30,30),1]

v_range = [[3,8],[0.5,3]]
repul_h_range = [[2,8],[2,8]]
repul_m_range = [[2,8],[2,8]]
p_range = [[0.1,1],[0.01,0.5]]
n_trials = 10
```
2. 評価するインスタンスを生成  
making an instance for assessment
```python
assessment = people_flow.assessment_framework.assess_framework(maps, people_num, v_arg, repul_h, repul_m, target, R, min_p, p_arg, wall_x, wall_y, in_target_d, dt, 
              save_params, v_range, repul_h_range, repul_m_range, p_range, n_trials)
```
3. すべての目的関数を用いてパラメータ推定を実行  
opt_list=["SSD","KL"]などとして引数にこれを指定することで、最適化に用いる目的関数を選択できる。分割して実行するときや特定のものを再最適化するときに使える。  
inferring parameters using every objective function
```python
assessment.whole_opt()
```
4. 最高の目的関数の組み合わせを求める  
getting the best combination of objective functions
```python
best_combination = assessment.assess()
```
5. パラメータ推定結果を正解と比較してグラフ化  
making graphs comparing correct parameters and inferred parameters
```python
assessment.assess_paint()
```


## inference_framework.py
パラメータ推定を行う。  
inference_frameworkとinference_framework_detailがありinference_framework_detailは1種類ずつパラメータを推定して他を固定する方法を使っている。パッケージとしての実行の手順は引数の指定以外同じ。  
inference_framework_detailを使う場合、inference_frameworkでパラメータを推定した後、そのパラメータを用いてさらに一つずつパラメータ最適化をinference_framework_detailで行う。

inferring parameters  
There are two classess, inference_framework and inference_framework_detail, and inference_framework_detail adopts a way to infer a kind of parameter in an optimization, but how to use them as a package is the same except for deciding parameters.  
If you use inference_framework_detail, you should first infer parameters on inference_framework. After that, you should optimize each parameter on inference_framework_detail based on inferred parameters.

1. 引数を指定(詳細はinference_framework.pyを参照)  
Deciding parameters (details:inference_framework.py)
```python
# inference_frameworkの場合
# in case of inference_framework
people_num = 30
target = [[60,240],[120,150],[90,60],[240,40],[200,120],[170,70],[150,0]]
R = 3
min_p = 0.1
wall_x = 300
wall_y = 300
in_target_d = 3
dt = 0.1
save_params = [(30,30),1]

v_range = [[3,8],[0.5,3]]
repul_h_range = [[2,8],[2,8]]
repul_m_range = [[2,8],[2,8]]
p_range = [[0.1,1],[0.01,0.5]]
n_trials = 10
```

```python
# inference_framework_detailの場合
# in case of inference_framework_detail
people_num = 30
v_arg = [6,2]
repul_h = [5,5]
repul_m = [2,2]
target = [[60,240],[120,150],[90,60],[240,40],[200,120],[170,70],[150,0]]
R = 3
min_p = 0.1
p_arg = [[0.5,0.1]]
wall_x = 300
wall_y = 300
in_target_d = 3
dt = 0.1
save_params = [(30,30),1]

v_range = [[3,8],[0.5,3]]
repul_h_range = [[2,8],[2,8]]
repul_m_range = [[2,8],[2,8]]
p_range = [[0.1,1],[0.01,0.5]]
n_trials = 10
```
2. 推定するインスタンスを生成  
making an instance for inference
```python
# inference_frameworkの場合
# in case of inference_framework
inference = people_flow.inference_framework.inference_framework(maps, people_num, target, R, min_p, wall_x, wall_y, in_target_d, dt, 
              save_params, v_range, repul_h_range, repul_m_range, p_range, n_trials)
```

```python
# inference_framework_detailの場合
# in case of inference_framework_detail
inference_detail = people_flow.inference_framework.inference_framework_detail(maps, people_num, v_arg, repul_h, repul_m, target, R, min_p, p_arg, wall_x, wall_y, in_target_d, dt, 
              save_params, v_range, repul_h_range, repul_m_range, p_range, n_trials)
```
3. パラメータ推定を実行(推定したパラメータを得る)  
infer_dict={"v":"SSD", "repul_h":"KL", "repul_m":"ZNCC", "p":"ZNCC"}などとして引数にこれを指定することで、最適化に用いる目的関数を選択できる。デフォルト(inference_framework.pyを参照)とは異なる目的関数を用いたいときに使える。  
inferring parameters (getting inferred parameters)
```python
# inference_frameworkの場合
# in case of inference_framework
inferred_params = inference.whole_opt()
```

```python
# inference_framework_detailの場合
# in case of inference_framework_detail
inferred_params_detail = inference_detail.whole_opt()
```
