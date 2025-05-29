
# Unix/Linuxシェル操作の入門

コンピュータの用途としてプログラミングがあるのは言うまでもないことだと思います。プログラミングは自動化の技術ですが、コンピュータの操作そのものを「プログラミング」できるというのは自然な考えでしょう。その恩恵を授かるための第一歩が Unix/Linuxシェル操作 を習得することです。これらのツールは便利な反面、「ユーザーが操作の意味をわかっている」ことが前提となっているため、実は使用には責任が伴います。勉強のためには間違ってもダメージの少ない環境が欲しいところですが、こちらも2025年現在では ブラウザ上無料で（！）使い捨てで実行できる環境があります。以下では Google Cloud Shell の使用を前提とします。

> [!TIP]
> <details>
> <summary>Google Cloud Shell の立ち上げ方</summary>
> 
> Google Cloud Shell は Googleアカウントから使えるUnix/Linuxシェルの環境です：
> - 無料でも使えます：https://cloud.google.com/shell/pricing?hl=ja
>
> 使うには基本的にGoogleアカウントにログインした状態でブラウザから適当なURLにアクセスするだけです。最初に動かす際は承認が必要です。使う際にはいくつかの選択肢があります：
> 1. エフェメラルモード：https://shell.cloud.google.com/?hl=fromcloudshell=true&show=terminal&pli=1&ephemeral=true
>       - セッションを切ると作った環境はすべて消去される「一時的（＝エフェメラル）な」モード
>       - 慣れないうちはこちらでやっておくと、何かを失敗しても起動しなおせばやり直せるので安心です
> 2. 通常モード：https://shell.cloud.google.com/?hl=fromcloudshell=true&show=terminal&pli=1
>       - セッションを切っても環境が保存されたままのモード（`$HOME` 領域以下 5GB まで使えるらしい）
>       - 120日間アクセスがない場合は `$HOME` 領域以下 は削除される（メール通知が来る）らしいです。
>       - 一度 エフェメラルモード で動かすと再度承認が求められる？
> </details>

1. 基礎編
    - [1-1. ファイル周辺の操作コマンド](section1/1-1.md)
    - 1-2. コマンドを組み合わせる方法
    - 1-3. シェルでプログラミング
    - 1-4. シェルの設定
2. 実用編
    - pythonの仮想環境
    - sshによる別マシンへのリモート接続
    - gitによるバージョン管理
    - docker/singularityによる仮想環境

## 参考文献

- Linuxの教科書（メールアドレス登録から無料でpdf版が手に入れられます）：https://linuc.org/textbooks/linux/
- python仮想環境： https://zenn.dev/tigrebiz/articles/2822fb4de256d8
- ssh：
- OpenPBS：