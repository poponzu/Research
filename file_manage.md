# Case1 LSTM-SAE_bike
- ver1  trainloss, valloss追加。
- ver5   pollution_ver5をbikeのデータに置き換えた
- ver6   ver5から、batch_size = 73, dropout = 0.1, seq_len = 30に変更した, noiseなしにして実行。元のモデルと比較する
- ver7 ver6にnoiseを追加
- ver8 ver6のhidden_layers = [33,5,47]に変更してみる
- ver9 ver6のhidden_layers = [33,47,5]に変更
- ver10 ver6のepoch数を元論文のパラメータに揃えて実行
- ver11 ver1のepochs_finetuneをepochs_finetune = 355に変更して実行中
# Case2 LSTM-SAE_pollution
- ver2 trainloss, valloss追加。
- ver3 AEの学習曲線追加・コードの構造化
- ver4 swap noise10%を3回入れて実験,(add_noise関数の引数にnoiseの量を指定するもの追加した方がいいかも)
- ver5 すべてのepochを100回にした。これからの実験のため noiseあり
- ver6 ver5からnoiseなしにして、hidden_layers = [45,60,5]にかえてみる
- ver7 論文にパラメータすべて揃えてepochｓも本来と同じ回数。10_29寝る前に実行する
- ver8 add_swapnoiseにどれくらいの量のノイズ入れるかを入力できるようにする 100eopchで実行中
- ver9 ver7にnoise三つ0.10を追加してみる
- ver10 noise0.05,0.10,0.15の順で加える epoch_fine_tuneの回数をふやした。
- ver11　ver9をもとにgaussian noiseを実装 すべてepoch100で実行
- ver12 ver11を論文にパラメータすべて揃えてepochｓも本来と同じ回数でじっこうした
- ver13 masking noise 実装する！！！ 論文にパラメータすべて揃えて実行するmaskingnoiseの実装は("https://stackoverflow.com/questions/54633038/how-to-add-masking-noise-to-numpy-2-d-matrix-in-a-vectorized-manner")をさんこうにした　。epoch全て100
- ver14 ver13のepochを論文に揃える
- ver15 swapnoiseがswapできていなかったので、それを治して、ver7をもう一度実行してます
- ver16 maskingnoiseをswapnoise関数をもとに改良
- ver17 gaussianとmask混ぜる bikeのver10が終わり次第実行する
# 10_29時点
noiseなしという条件下では、
- bikeはver1が一番精度良い
- pollutionはver2が一番精度良い

bikeの方が論文と評価値がかけ離れている。

verの横に精度ものせれるようにしたいね
