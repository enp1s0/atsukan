![ATSUKAN LOGO](https://gitlab.momo86.net/mutsuki/atsukan/raw/master/docs/atsukan.svg)

冬ですね．  
熱燗の時期ですね．  
GPUで汎用計算をしている者としては自分の計算の熱で熱燗を作るのは夢です．  
これはそんなプロジェクト．

## 燗サーバー
流石に普段使っている計算サーバーの近くに液体を持ち込むのは気がひけるので，新たに計算サーバーを立てます．

|    品目     |     型番                   |  入手元   | 
|:------------|:---------------------------|:----------|
|  CPU        | AMD Athlon 200GE           | Amazon    |
|  M/B        | A320M PRO-VD/S             | TSUKUMO   |
|  RAM        | DDR4 2666MHz PC4-21300 8GB | Amazon    |
|  SSD        | Kingston SSD SA400S37/120G | Amazon    |
|  Graphics   | ASUSU GTX650-E-1GD5        | Yahoo Auc |

## 燗アプリケーション
### ラフな要求
- 燗アルゴリズム(行列計算等)，計算型(float/double)，使用GPU ID，ログの表示方式(csv/human-friendly)をプログラムの引数として指定
- お酒を温める
- GPUの温度上昇速度や消費電力を評価値として最適な並列数や燗アルゴリズムの問題サイズを調整

### 開発言語
- CUDA/C++

### 使用ライブラリ
- cutf (https://gitlab.momo86.net/mutsuki/cutf)
- cxxopts (https://github.com/jarro2783/cxxopts)

### 開発時間
- お正月にはぜひ熱燗を楽しみたいので納期は今年中
- 12/15にサーバーを実家に設置し試験運用を行いためプロトタイプは12/15までに完成


