# dslr — Data Science × Logistic Regression

`src/stats42/` に、統計と機械学習の部品を自作してまとめています。
この README は、各関数の公式を統計の教科書の記法で整理したものです。
(数式は GitHub の LaTeX 表示を前提にしています)

## 記法

| 記号 | 意味 |
|---|---|
| $x_1, \dots, x_n$ | 標本(欠損値 NaN を除いたもの) |
| $n$ | 標本サイズ(要素数) |
| $x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$ | 昇順に並べた標本(順序統計量) |
| $\bar{x}$ | 標本平均 |
| $s^2$, $s$ | 標本分散、標本標準偏差 |
| $\mathrm{ddof}$ | 自由度の補正値($\mathrm{ddof}=1$ が標本、$0$ が母集団) |
| $m_k$ | $k$ 次の中心モーメント |
| $Q(p)$ | $p$ パーセント点($0 \le p \le 100$) |

## モジュール構成

| ファイル | 内容 |
|---|---|
| `_core.py` | 基本部品: `ft_dropna`, `ft_count`, `ft_sum`, `ft_min`, `ft_max`, `ft_sqrt`, `ft_sort` |
| `central.py` | 代表値: `ft_mean`, `ft_median`, `ft_mode` |
| `dispersion.py` | ばらつき: `ft_var`, `ft_std`, `ft_range`, `ft_iqr` |
| `position.py` | 位置: `ft_percentile`, `ft_quartiles` |
| `shape.py` | 形状: `ft_skewness`, `ft_kurtosis` |
| `relation.py` | 2変数の関係: `ft_corr` |
| `preprocessing.py` | 前処理: `ft_standardize`, `ft_normalize` |
| `logistic.py` | ロジスティック回帰: `ft_sigmoid`, `ft_predict`, `ft_cost`, `ft_gradient`, `ft_gradient_descent`, `ft_sgd`, `ft_minibatch_gd` |

---

## 1. 基本部品 (`_core.py`)

**欠損値の除去** `ft_dropna`: NaN は自分自身と等しくない($x_i \ne x_i$)ので、
$x_i = x_i$ を満たす要素だけを残す。

**要素数** `ft_count`:

$$n = \sum_{i=1}^{n} 1$$

**合計** `ft_sum`:

$$S = \sum_{i=1}^{n} x_i$$

**最小値・最大値** `ft_min`, `ft_max`:

$$x_{\min} = x_{(1)}, \qquad x_{\max} = x_{(n)}$$

**平方根** `ft_sqrt`(ニュートン法): $y_0 \ge \sqrt{a}$ から出発し、

$$y_{k+1} = \frac{1}{2}\left(y_k + \frac{a}{y_k}\right)$$

を $y_{k+1} \ge y_k$ になるまで繰り返す
($y_0 \ge \sqrt{a}$ なら $y_1$ 以降は単調に減少して $\sqrt{a}$ に収束する)。

**整列** `ft_sort`: マージソート($O(n \log n)$、安定)。
$x_{(1)} \le \dots \le x_{(n)}$ を返す。

---

## 2. 代表値 (`central.py`)

**算術平均** `ft_mean`:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**中央値** `ft_median`:

$$
\tilde{x} =
\begin{cases}
x_{\left(\frac{n+1}{2}\right)} & (n \text{ が奇数}) \\[4pt]
\dfrac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2} & (n \text{ が偶数})
\end{cases}
$$

**最頻値** `ft_mode`:

$$\hat{x} = \operatorname*{arg\,max}_{v} \; \#\{\, i : x_i = v \,\}$$

同数なら最小の値を返す。

---

## 3. ばらつき (`dispersion.py`)

**分散** `ft_var`:

$$s^2 = \frac{1}{n - \mathrm{ddof}} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

- $\mathrm{ddof}=1$: 不偏分散(標本分散)
- $\mathrm{ddof}=0$: 母分散 $\sigma^2$

**標準偏差** `ft_std`:

$$s = \sqrt{s^2} = \sqrt{\frac{1}{n - \mathrm{ddof}} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

既定は $\mathrm{ddof}=1$(pandas と同じ)。

**範囲** `ft_range`:

$$R = x_{(n)} - x_{(1)}$$

**四分位範囲** `ft_iqr`:

$$\mathrm{IQR} = Q_3 - Q_1 = Q(75) - Q(25)$$

---

## 4. 位置 (`position.py`)

**パーセント点** `ft_percentile`(線形補間。numpy / pandas の既定、Excel の `PERCENTILE.INC` と同じ定義):

$$h = (n - 1)\frac{p}{100} + 1, \qquad k = \lfloor h \rfloor, \qquad f = h - k$$

$$Q(p) = x_{(k)} + f \left( x_{(k+1)} - x_{(k)} \right)$$

($k = n$ のときは $Q(p) = x_{(n)}$)

> パーセント点の定義は複数ある(Hyndman & Fan の 9 種)。
> 例: $\{1,2,3,4\}$ の $Q(25)$ は、この定義では $1.75$、
> 最近傍順位では $1$、$h = (n+1)p/100$ の定義では $1.25$。

**四分位数** `ft_quartiles`:

$$Q_1 = Q(25), \qquad Q_2 = Q(50) = \tilde{x}, \qquad Q_3 = Q(75)$$

---

## 5. 形状 (`shape.py`)

$k$ 次の中心モーメント:

$$m_k = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^k$$

**歪度** `ft_skewness(x, bias=False)`(左右の非対称さ。0 で対称、正で右に裾が長い):

$$g_1 = \frac{m_3}{m_2^{3/2}} \;\; (\texttt{bias=True}), \qquad
G_1 = \frac{\sqrt{n(n-1)}}{n-2} \, g_1 \;\; (\texttt{bias=False},\; n \ge 3)$$

**尖度(超過尖度)** `ft_kurtosis(x, bias=False)`(裾の厚さ。正規分布で 0):

$$g_2 = \frac{m_4}{m_2^{2}} - 3 \;\; (\texttt{bias=True}), \qquad
G_2 = \frac{n-1}{(n-2)(n-3)} \left[ (n+1)\, g_2 + 6 \right] \;\; (\texttt{bias=False},\; n \ge 4)$$

$g_1$, $g_2$ は母集団の式、$G_1$, $G_2$ は pandas の `skew()`, `kurt()` と同じ小標本補正(既定)。

---

## 5.5 相関 (`relation.py`)

**ピアソンの積率相関係数** `ft_corr`:

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \cdot \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

$x_i$, $y_i$ は同じ対象(同じ生徒など)の 2 つの測定値。$x$ と $y$ は同じ長さで、NaN を含まない。

$-1 \le r \le 1$。$r = 1$ で完全な正の直線関係、$r = -1$ で完全な負の直線関係、$0$ で直線的な関係なし。

分子は偏差の積の合計(共分散の $n$ 倍)。$x_i$, $y_i$ がともに平均より上、またはともに下なら
積は正、片方だけ上なら負になる。同じ向きにずれる対象が多いほど $r$ は $1$ に近づく。

分母は $x$ と $y$ のばらつき(偏差平方和)の積の平方根。これで割ると、単位や尺度に依存せず
$[-1, 1]$ に収まる。同じことを、共分散と標準偏差で書くと次のようになる(約分で $n-1$ は消える)。

$$r = \frac{s_{xy}}{s_x \, s_y}, \qquad
s_{xy} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

- $\sum (x_i - \bar{x})^2 = 0$ または $\sum (y_i - \bar{y})^2 = 0$(全部同じ値)のときは分母が $0$ で定義できない。
- 直線的な関係だけを測る。曲線状の関係は強くても $r$ が小さくなることがある。散布図で目視して確かめる。
- 相関は因果関係を意味しない。

---

## 6. 前処理 (`preprocessing.py`)

学習データから求めた定数を、テストデータにもそのまま使う。

**標準化(z スコア)** `ft_standardize`:

$$z_i = \frac{x_i - \bar{x}}{s}$$

変換後は平均 $0$、標準偏差 $1$。

**min-max 正規化** `ft_normalize`:

$$x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$

学習データでは $[0, 1]$ に収まる。

---

## 7. ロジスティック回帰 (`logistic.py`)

特徴量行列 $X \in \mathbb{R}^{m \times (n+1)}$(先頭列は切片用の $1$)、
ラベル $y^{(i)} \in \{0, 1\}$、重み $\theta \in \mathbb{R}^{n+1}$、$m$ は標本数。

**シグモイド関数** `ft_sigmoid`:

$$g(z) = \frac{1}{1 + e^{-z}}$$

**仮説(確率)** `ft_predict`:

$$h_\theta(x) = g(\theta^{\mathsf T} x) = P(y = 1 \mid x;\, \theta)$$

**損失(クロスエントロピー)** `ft_cost`:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m}
\Big[ y^{(i)} \log h_\theta(x^{(i)}) + \big(1 - y^{(i)}\big) \log\big(1 - h_\theta(x^{(i)})\big) \Big]$$

**勾配** `ft_gradient`:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m}
\big( h_\theta(x^{(i)}) - y^{(i)} \big)\, x^{(i)}_j
\qquad\Longleftrightarrow\qquad
\nabla J = \frac{1}{m} X^{\mathsf T} \big( h_\theta(X) - y \big)$$

**バッチ勾配降下法** `ft_gradient_descent`($\alpha$: 学習率):

$$\theta := \theta - \alpha \, \nabla J(\theta)$$

全標本で勾配を 1 回計算して更新する。

**確率的勾配降下法 (SGD)** `ft_sgd`(各エポックで並べ替え、1 標本ごとに更新):

$$\theta := \theta - \alpha \big( h_\theta(x^{(i)}) - y^{(i)} \big)\, x^{(i)}$$

**ミニバッチ勾配降下法** `ft_minibatch_gd`(各エポックで並べ替え、$b$ 個の組 $B$ ごとに更新):

$$\theta := \theta - \alpha \, \frac{1}{b} \sum_{i \in B}
\big( h_\theta(x^{(i)}) - y^{(i)} \big)\, x^{(i)}$$

$b = m$ ならバッチ GD、$b = 1$ なら SGD と一致する。

**多クラス分類 (one-vs-all)**: 各クラス $k$ について「$k$ か、それ以外か」の
2 値分類器 $\theta^{(k)}$ を学習し、予測は確率が最大のクラスとする。

$$\hat{y} = \operatorname*{arg\,max}_{k} \; h_{\theta^{(k)}}(x)$$

---

## 8. 学習と予測の流れ (`logreg_train.py` / `logreg_predict.py`)

7 章は「部品」(シグモイド、損失、勾配、勾配降下法)の説明だった。
この章は、その部品を使って **データから寮を予測するまでの手順** を、
順番に丁寧に説明する。まだ実装していない部分の設計メモでもある。

### 8.1 全体像

```
dataset_train.csv ──> logreg_train.py ──> 重みファイル
                                              │
dataset_test.csv ──────────────────> logreg_predict.py ──> houses.csv
```

- `logreg_train.py` は学習用データ(寮の答え付き)から、寮を見分けるための **重み** を作って保存する。
- `logreg_predict.py` は保存された重みを読み、答えのないテストデータの寮を予測して `houses.csv` に書く。
- 2 つのプログラムは **重みファイルだけでつながる**。学習側で決めたことは、
  すべて重みファイルに入れて予測側に渡す(8.5 参照)。

### 8.2 課題の要求(課題PDF Version 5 より)

| 項目 | 内容 |
|---|---|
| `logreg_train` の入力 | `dataset_train.csv` |
| `logreg_train` の出力 | 重みを入れたファイル(形式は自由) |
| `logreg_predict` の入力 | `dataset_test.csv` と重みファイル |
| `logreg_predict` の出力 | `houses.csv`(下記の形式、**exactly**) |
| 学習方法 | 勾配降下法で誤差を最小にする(必須) |
| 多クラス | one-vs-all(one-vs-rest) |
| 合格ライン | テストデータで、scikit-learn の `accuracy_score` が **98% 以上** |

`houses.csv` の形式:

```
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
...
```

- ヘッダーは `Index,Hogwarts House`。
- 1 行が 1 人の生徒で、`Index` は `dataset_test.csv` の `Index` 列をそのまま使う。
- 行の数は、テストデータの行の数(400 行)と一致させる。**1 人も飛ばしてはいけない**。

### 8.3 データの事実(確認済み)

| | 行数 | `Hogwarts House` 列 |
|---|---|---|
| `dataset_train.csv` | 1600 | 答えが入っている |
| `dataset_test.csv` | 400 | **全部空**(これを予測する) |

- 数値の科目は 13 個(`Index` を除く)。
- どちらのデータにも、欠損値(NaN)がある。
  特徴量の候補にした 9 科目(`Charms`、`Astronomy`、`Herbology`、`Ancient Runes`、
  `Divination`、`Muggle Studies`、`History of Magic`、`Flying`、`Transfiguration`)でも、
  テストデータ 400 行のうち **64 行** は、どれか 1 科目以上が NaN。

### 8.4 学習 (`logreg_train.py`) の手順

#### 手順 1: 特徴量を選ぶ

ここまでの 3 つのスクリプトの結果から、次のように考える。

| 判断 | 科目 | 根拠 |
|---|---|---|
| 外す | `Arithmancy`、`Care of Magical Creatures` | histogram で 4 寮の分布がほぼ同じ。寮の手がかりにならない。 |
| 片方だけ使う | `Astronomy` と `Defense Against the Dark Arts` | 相関係数が $-1.0$ で、片方は他方の定数倍(重複)。 |
| 候補 | `Charms`、`Astronomy`(または Defense...)、`Herbology`、`Ancient Runes`、`Divination`、`Muggle Studies` | pair_plot で寮の色が分かれる。 |
| どれか 1 つ | `History of Magic`、`Transfiguration`、`Flying` | どれも Gryffindor を分ける科目で、互いに強く相関している(同じ情報)。 |

> 最終的に何を使うかは、精度(8.7)を比べて決める。
> 説明できること(なぜその科目を選んだか)が、評価では大事。

選んだ科目の名前と順番は、**重みファイルに保存する**(8.5)。
予測側で同じ順番に並べ直さないと、重みと特徴量の対応がずれる。

#### 手順 2: 欠損値(NaN)を処理する

NaN のままだと、平均や勾配の計算が全部 NaN になって壊れる。対処は 2 通り。

| 方法 | やり方 | 長所 | 短所 |
|---|---|---|---|
| A. 行を落とす | 選んだ科目のどれかが NaN の生徒を、学習から外す | 簡単。偽の値を作らない | データが減る |
| B. 値で埋める | NaN を、その科目の **学習データの平均** で置き換える | データが減らない | 偽の値が入る |

> **重要**: 予測側(テストデータ)では、方法 A は使えない。
> `houses.csv` には **全員分** の予測が必要で、NaN がある生徒も飛ばせないから。
> つまり、**予測側は必ず方法 B(埋める)**。
> 学習側を A にするか B にするかは自由だが、「学習と予測で同じ前処理」にしたいなら、
> 両方で B にするのが素直。

埋める値(平均)は、**学習データから** 求めた値を使う。テストデータの平均は使わない。
(標準化と同じ理由。手順 3 を参照)

標準化の後に平均で埋めると、その値は $0$ になる。つまり
「標準化したあとの NaN を $0$ にする」と書いても、同じ結果になる。

#### 手順 3: 標準化する

$$z_i = \frac{x_i - \bar{x}_{\text{train}}}{s_{\text{train}}}$$

(`ft_standardize(x, mean, std)`)

- 理由 1: 科目ごとに点数の大きさがまったく違う(`Arithmancy` は 5 万点台、`Herbology` は 1 前後)。
  そのまま使うと、勾配降下法が収束しにくい(学習率を小さくしないと発散する)。
- 理由 2: 重みの大きさを、科目どうしで比べられるようになる。
- **$\bar{x}_{\text{train}}$ と $s_{\text{train}}$ は、学習データだけから求める**。
  テストデータの平均や標準偏差で標準化してはいけない。
  (予測時にはテストデータ全体が手に入るとは限らず、学習と別の基準になってしまうため)
  学習で求めた平均と標準偏差は、重みファイルに保存して、予測側で使い回す。

#### 手順 4: 切片の列を足す

$X$ の先頭に、すべて $1$ の列を足す。これが切片 $\theta_0$ に掛かる。
`logistic.py` は「先頭列が 1」の $X$ を前提にしている。

$$X = \begin{pmatrix} 1 & z_{1,1} & \cdots & z_{1,n} \\ \vdots & & & \vdots \\ 1 & z_{m,1} & \cdots & z_{m,n} \end{pmatrix}$$

($m$ は生徒数、$n$ は特徴量の数)

#### 手順 5: 寮ごとに、答えを 0 と 1 にする(one-vs-all)

4 つの寮 $k \in \{\text{Gryffindor}, \text{Hufflepuff}, \text{Ravenclaw}, \text{Slytherin}\}$ について、

$$y^{(k)}_i = \begin{cases} 1 & (\text{生徒 } i \text{ の寮が } k) \\ 0 & (\text{それ以外}) \end{cases}$$

という 0/1 のラベルを作る。例えば Gryffindor 用の分類器は、
「Gryffindor か、それ以外か」だけを見分ける。

#### 手順 6: 寮ごとに、勾配降下法で重みを学習する

4 つの寮それぞれについて、**独立に** 次を行う。

1. $\theta^{(k)} = \mathbf{0}$(すべて 0)から始める。
2. 次の更新を繰り返す(`ft_gradient_descent`)。

$$\theta^{(k)} := \theta^{(k)} - \alpha \cdot \frac{1}{m} X^{\mathsf T}\big(h_{\theta^{(k)}}(X) - y^{(k)}\big)$$

3. 損失 $J(\theta^{(k)})$ の減り方が $\mathrm{tol}$ より小さくなったら止める
   (または、最大エポック数に達したら止める)。

決めるパラメータ:

| パラメータ | 意味 | 目安 |
|---|---|---|
| 学習率 $\alpha$ | 1 回の更新の大きさ | 大きすぎると発散、小さすぎると遅い。標準化済みなら $0.1$ 前後から試す |
| 最大エポック数 | 更新回数の上限 | 損失の履歴を見て、十分に収束する数にする |
| $\mathrm{tol}$ | 止める基準 | 損失の変化がこれより小さければ収束とみなす |

- 損失の履歴(`history`)は、毎回少しずつ減っていくはず。
  増えたら、学習率が大きすぎる。グラフにして確認するとよい。
- ボーナスとして、同じことを SGD(`ft_sgd`)やミニバッチ GD(`ft_minibatch_gd`)でもできる。
  同じ 4 回の学習を、最適化の方法だけ変えて比べる。

### 8.5 重みファイルに入れるもの

予測側が「学習と全く同じ前処理」をするために、次をすべて保存する。

| 保存するもの | 使う場面 |
|---|---|
| 4 寮ぶんの重み $\theta^{(k)}$(切片を含む) | 予測そのもの |
| 寮の名前と、その順番 | 4 つの重みがどの寮のものか |
| 特徴量の名前と、その順番 | 重みと科目の対応を保つ |
| 各特徴量の平均 $\bar{x}_{\text{train}}$ | 標準化(手順 3)と NaN の埋め合わせ(手順 2) |
| 各特徴量の標準偏差 $s_{\text{train}}$ | 標準化(手順 3) |

形式は自由。JSON は人が読めて、名前と値を対応させやすい。
(`json` モジュールで足りる。CSV や `.npy` でもよいが、名前の対応が崩れやすい)

### 8.6 予測 (`logreg_predict.py`) の手順

1. 引数からテストデータ(CSV)と重みファイルを読む。
2. 重みファイルの **特徴量の順番** で、テストデータから科目の列を取り出す。
3. NaN を、保存した平均で埋める。
4. 保存した平均と標準偏差で標準化する(テストデータから計算し直さない)。
5. 先頭に $1$ の列を足す。
6. 4 つの寮それぞれの確率を計算する(`ft_predict`)。

$$p_k = g\big(\theta^{(k)\mathsf T} x\big)$$

7. 確率が最大の寮を選ぶ。

$$\hat{y} = \operatorname*{arg\,max}_{k}\; p_k$$

8. `Index` と予測した寮を、`houses.csv` に書く。

確率の合計は $1$ にならない(4 つの分類器を別々に学習したため)。
でも、選ぶのは「一番大きいもの」なので問題ない。

### 8.7 精度の確認

テストデータには答えがない(`Hogwarts House` が空)ので、
提出前に自分で精度を測るには、**学習データを分ける** 必要がある。

1. `dataset_train.csv` を、学習用(例: 8 割)と検証用(例: 2 割)に分ける。
2. 平均や標準偏差は、学習用だけから求める(検証用はテストデータの代わり)。
3. 検証用で、予測した寮と本当の寮を比べる。

$$\mathrm{accuracy} = \frac{\text{予測が当たった生徒の数}}{\text{生徒の総数}}$$

- 課題の合格ラインは 98%。検証用で 98% 前後なら、テストでも期待できる。
- 特徴量の組み合わせや、学習率を比べるときの基準にもなる。
- 最終的な提出用の重みは、`dataset_train.csv` の全部を使って学習し直してもよい。

### 8.8 決めることと、よくある失敗

**これから決めること**

| 項目 | 選択肢 |
|---|---|
| 使う特徴量 | 8.4 手順 1 の候補から選ぶ(精度で比べる) |
| 学習時の NaN | 行を落とす / 平均で埋める(予測側は必ず「埋める」) |
| 学習率・エポック数・`tol` | 損失の履歴を見て決める |
| 重みファイルの形式 | JSON など |

**よくある失敗**

- テストデータの平均や標準偏差で標準化する(学習と基準がずれる)。
- 予測側で、特徴量の並び順が学習時と違う(重みが別の科目に掛かる)。
- 切片の列($1$)を、学習側だけ、または予測側だけに足す。
- 学習率が大きすぎて、損失が増えて発散する。
- `houses.csv` のヘッダーや列の順番が、指定と違う。
- NaN のある生徒を、予測の出力から飛ばしてしまう(行数が 400 にならない)。

---

## 参考文献

### Wikipedia(日本語)

関数・統計知識ごとの参考リンク。

| 関数 / 内容 | 記事 |
|---|---|
| `ft_mean` 算術平均 | [算術平均](https://ja.wikipedia.org/wiki/%E7%AE%97%E8%A1%93%E5%B9%B3%E5%9D%87) |
| `ft_median` 中央値 | [中央値](https://ja.wikipedia.org/wiki/%E4%B8%AD%E5%A4%AE%E5%80%A4) |
| `ft_mode` 最頻値 | [最頻値](https://ja.wikipedia.org/wiki/%E6%9C%80%E9%A0%BB%E5%80%A4) |
| `ft_var` 分散 | [分散 (確率論)](https://ja.wikipedia.org/wiki/%E5%88%86%E6%95%A3_(%E7%A2%BA%E7%8E%87%E8%AB%96)) |
| `ft_var` / `ft_std` の `ddof=1` | [不偏分散](https://ja.wikipedia.org/wiki/%E4%B8%8D%E5%81%8F%E5%88%86%E6%95%A3) |
| `ft_std` 標準偏差 | [標準偏差](https://ja.wikipedia.org/wiki/%E6%A8%99%E6%BA%96%E5%81%8F%E5%B7%AE) |
| `ft_iqr` 四分位範囲 | [四分位範囲](https://ja.wikipedia.org/wiki/%E5%9B%9B%E5%88%86%E4%BD%8D%E7%AF%84%E5%9B%B2) |
| `ft_percentile` / `ft_quartiles` | [分位数](https://ja.wikipedia.org/wiki/%E5%88%86%E4%BD%8D%E6%95%B0) |
| 昇順に並べた標本 $x_{(i)}$ | [順序統計量](https://ja.wikipedia.org/wiki/%E9%A0%86%E5%BA%8F%E7%B5%B1%E8%A8%88%E9%87%8F) |
| `ft_skewness` 歪度 | [歪度](https://ja.wikipedia.org/wiki/%E6%AD%AA%E5%BA%A6) |
| `ft_kurtosis` 尖度 | [尖度](https://ja.wikipedia.org/wiki/%E5%B0%96%E5%BA%A6) |
| `ft_corr` ピアソンの積率相関係数 | [ピアソンの積率相関係数](https://ja.wikipedia.org/wiki/%E3%83%94%E3%82%A2%E3%82%BD%E3%83%B3%E3%81%AE%E7%A9%8D%E7%8E%87%E7%9B%B8%E9%96%A2%E4%BF%82%E6%95%B0) |
| `ft_sigmoid` | [シグモイド関数](https://ja.wikipedia.org/wiki/%E3%82%B7%E3%82%B0%E3%83%A2%E3%82%A4%E3%83%89%E9%96%A2%E6%95%B0) |
| `ft_sgd` 確率的勾配降下法 | [確率的勾配降下法](https://ja.wikipedia.org/wiki/%E7%A2%BA%E7%8E%87%E7%9A%84%E5%8B%BE%E9%85%8D%E9%99%8D%E4%B8%8B%E6%B3%95) |
| ロジスティック回帰全般 | [ロジスティック回帰](https://ja.wikipedia.org/wiki/%E3%83%AD%E3%82%B8%E3%82%B9%E3%83%86%E3%82%A3%E3%83%83%E3%82%AF%E5%9B%9E%E5%B8%B0) |
| `ft_cost` クロスエントロピー | [交差エントロピー](https://ja.wikipedia.org/wiki/%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC) |
| `ft_gradient_descent` / `ft_minibatch_gd` | [勾配降下法](https://ja.wikipedia.org/wiki/%E5%8B%BE%E9%85%8D%E9%99%8D%E4%B8%8B%E6%B3%95) |
| `ft_sqrt` ニュートン法 | [ニュートン法](https://ja.wikipedia.org/wiki/%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%88%E3%83%B3%E6%B3%95) |
| `ft_sort` マージソート | [マージソート](https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%BC%E3%82%B8%E3%82%BD%E3%83%BC%E3%83%88) |
| `ft_standardize` z スコア | [標準得点](https://ja.wikipedia.org/wiki/%E6%A8%99%E6%BA%96%E5%BE%97%E7%82%B9) |

### 教科書

- 秋山裕『統計学基礎講義』第3版、慶應義塾大学出版会、2022年4月(414ページ)
- Sebastian Raschka, Yuxi (Hayden) Liu, Vahid Mirjalili 著、クイープ 訳、福島真太朗 監訳『Python機械学習プログラミング PyTorch&scikit-learn編』(impress top gear)、インプレス、2022年12月、ISBN 978-4-295-01558-1
