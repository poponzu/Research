# LSTM-SAE_pollution
- ver2 trainloss, valloss追加。
- ver3 AEの学習曲線追加・コードの構造化
- ver4 swap noise10%を3回入れて実験,(add_noise関数の引数にnoiseの量を指定するもの追加した方がいいかも)
- ver5 すべてのepochを100回にした。これからの実験のため
# LSTM-SAE_bike
- ver1  trainloss, valloss追加。
- ver5   pollution_ver5をbikeのデータに置き換えた
- ver6   ver5から、batch_size = 73, dropout = 0.1, seq_len = 30に変更した, noiseなしにして実行。元のモデルと比較する
- ver7 ver6にnoiseを追加
- ver8 ver6のhidden_layers = [33,5,47]に変更してみる