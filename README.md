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
