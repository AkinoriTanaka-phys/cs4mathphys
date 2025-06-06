# Pythonと機械学習

Pythonはプログラミング言語初学者でも比較的気軽に使え、他の言語への取っ掛かりにもなる良い言語ですが 実行環境を自前で整えるのは（全くの知識ゼロからだと）意外と敷居が高く、以前は入門のためにいくらか勉強が必要でした。ところが 2024年現在では ブラウザ上無料で（！）実行できる環境が数多くあり、何の準備もなしに直ぐに Python を書いて動かせるようになりました。以下ではそのようなサービスの一つであるである Google Colaboratory を用いることを前提としています。

> [!TIP]
> <details class="memo">
> <summary>Google Colaboratory について</summary>
> <blockquote>
> 
> **立ち上げ方**
> 
> 1. https://colab.research.google.com/?hl=ja にアクセス
> 2. Googleにログイン（すでにしている場合は3）
> 3. ノートブックを新規作成 / あるいはすでに作成したノートブックを選択
> 
> **使い方**
> 
> ノートブックは、セルと呼ばれるパーツが縦に並んで構成されています。セルには二種類あります：
> - コードのセル
>     - Python の コードを書いて動かすことのできるセル。
>     - 動かすには、そのセルを選択しながら `shift` + `enter` か、セルの冒頭に示されている再生マークをクリックする。
> - テキストのセル
>     - テキストを書く用のセル。次のコードセルで何をやっているのかの説明を書くと良い。
>     - テキストを装飾したい場合は、\$\$ などで囲むと LaTeXが使える。また、HTML や markdown が有効。
> 
> ウインドウの左上の追加するボタンからこれらのセルを追加できます。また、左側のアイコンは、上から順に
> - 目次
>     - テキストセルの markdown 記法の heading (#を6個まで重ねて見出し扱いにできる) から自動生成。
> - 検索と置換
> - 現在読み込まれている変数の情報
> - 機密情報の管理
>     - 外部に接続する際の鍵情報など：参考 https://qiita.com/suzuki_sh/items/4817e3423f2989bbb9ed
> - プログラム実行している仮想マシンのファイル構造
>     - 例えば python で画像を保存したりできます。
> 
> 他にも色々ありますが、使っていくうちに覚えてられるかと思います。
> </blockquote>
> </details>


1. [基礎編](section1/preface.md)
    - [1-1. Pythonの基本文法](section1/1-1.md)
    - [1-2. クラスとカプセル化](section1/1-2.md)
    - [1-3. その他の役立つ文法](section1/1-3.md)
2. [よく使われるライブラリ](section2/preface.md)
    - [2-1. 配列と図のプロット（numpy & matplotlib）](section2/2-1.md)
    - [2-2. 数学的な処理（sympy & scipy）](section2/2-2.md)
    - [2-3. データサイエンス系（pandas & seaborn）](section2/2-3.md)
3. [機械学習ライブラリ](section3/preface.md)
    - [3-1. 軽量な機械学習（scikit learn）](section3/3-1.md)
    - [3-2. 深層学習ライブラリ1（Keras, Tensorboard）](section3/3-2.md)
    - [3-3. 深層学習ライブラリ2（TensorFlow & PyTorch & JAX）](section3/3-3.md)
4. その他の話題
    - 4-1. 言語モデル/拡散モデルのライブラリ（transformers & diffusers）
    - [4-2. 強化学習の環境ライブラリ（Gymnasium & PettingZoo）](section4/4-2.md)
    - 4-3. マルチGPUの使用
    - 4-3. ハイパラ調整（Optuna?）

## 付録

- 強化学習入門
    1. [rl-1. Statelessな場合](rl_np/1.md)
    2. [rl-2. Statefulな場合1](rl_np/2.md)
    3. [rl-3. Statefulな場合2](rl_np/3.md)

# 参考文献

- Pythonの基本といくつかのライブラリについては "科学技術計算のためのPython入門" 中久喜 健司 (著) https://www.amazon.co.jp/%E7%A7%91%E5%AD%A6%E6%8A%80%E8%A1%93%E8%A8%88%E7%AE%97%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AEPython%E5%85%A5%E9%96%80-%E2%80%95%E2%80%95%E9%96%8B%E7%99%BA%E5%9F%BA%E7%A4%8E%E3%80%81%E5%BF%85%E9%A0%88%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA%E3%80%81%E9%AB%98%E9%80%9F%E5%8C%96-%E4%B8%AD%E4%B9%85%E5%96%9C-%E5%81%A5%E5%8F%B8/dp/4774183881 が良い気がします。
- 以下の講義ノートはとてもためになると思います。セクションごとについているコラムも読んでいて楽しいです：https://github.com/kaityo256/python_zero 