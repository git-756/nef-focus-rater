# NEF Focus Rater

ニコンの RAW（NEF）からピンボケ写真を見つけて、レーティング（★）を書き込むデスクトップアプリです。
書き込んだ★は、NX Studio などレーティングに対応したビューアで確認できます。

- フォルダを選ぶだけで、中の NEF をまとめて解析します
- NEF に内蔵された一番大きい JPEG プレビュー（多くの Z 機ではフルサイズ）で判定するので、RAW 現像は不要で高速です
- カメラが記録した **AF エリア** の位置で判定できます（AF 情報がない写真は画像全体で判定）
- 背景ボケの大きい写真をピンボケと誤判定しにくい評価方法です（ブロック分割＋ノイズ補正）
- スコアの分布をヒストグラムで見ながらしきい値を決められ、1枚ずつ等倍で確認できます
- 書き換えるのは `XMP:Rating` だけです。画像データには触れません

> NEF ファイルを直接書き換えるツールです。初めて使うときは、必ずコピーしたファイルで試してください。

## 必要なもの

- [uv](https://docs.astral.sh/uv/)（Python 3.11 以上は uv が自動で用意します）
  - macOS: `brew install uv` または `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ExifTool](https://exiftool.org/)
  - macOS: `brew install exiftool`
  - Windows: exiftool.org から入手し、`exiftool(-k).exe` を `exiftool.exe` に名前を変えて PATH の通った場所に置く
  - 場所を指定したいときは環境変数 `NEF_FOCUS_RATER_EXIFTOOL` にパスを設定

## インストールと起動

```bash
git clone https://github.com/git-756/nef-focus-rater.git
cd nef-focus-rater
uv sync                            # .venv を作って依存をインストール

uv run nef-focus-rater             # GUI を起動（uv run python -m nef_focus_rater でも可）
```

## 使い方

1. 「フォルダを選択…」で NEF の入ったフォルダを選び、「解析開始」を押す
2. スコアの低い順に一覧が出ます。行を選ぶと、右側に全体プレビューと等倍の切り出しが表示されます
3. しきい値を「下位 ◯%」かヒストグラムのクリックで調整します。赤い行が書き込み対象です
   - 個別に対象から外したり加えたりするには「書込」列のチェックを切り替えます（スペースキーでも可）
4. 「★を書き込む」を押すと、確認のあとで対象の写真に★が書き込まれます

### オプション

| 項目 | 説明 |
|---|---|
| 解析解像度 | フル解像度は細かなピントのズレまで検出できます。高感度ノイズが多いときは 1/2 が安定することがあります |
| 判定 | 「AFエリア優先」はカメラが記録した AF エリア内で判定します。「画像全体」は最もシャープな部分で判定します |
| すでに★がある写真は変更しない | 自分で付けた★を上書きしません（初期設定オン） |
| 同名の JPEG にも書き込む | RAW+JPEG で撮った場合に、JPEG 側にも同じ★を書き込みます |
| 元ファイル（_original）を残す | exiftool のバックアップを残します（初期設定オフ） |

### コマンドライン（解析のみ）

```bash
uv run nef-focus-rater-cli scan /path/to/photos             # スコアを一覧表示
uv run nef-focus-rater-cli scan /path/to/photos --csv out.csv
```

## しくみ

1. exiftool で、内蔵プレビュー（JpgFromRaw / PreviewImage）の位置、AF エリア、現在の★をまとめて読み取ります
2. 一番大きいプレビューをファイルから直接読み、グレースケールにしてラプラシアン（エッジの強さ）を計算します
3. 画像を 8×6 のブロックに分け、シャープなブロック上位の分散をスコアにします。AF エリアがある場合は、その範囲のスコアも計算します
4. 平坦な部分の値をノイズ量とみなして差し引き、高感度ノイズでスコアが水増しされるのを防ぎます
5. 書き込みは `exiftool -overwrite_original -P -XMP:Rating=N` です。`-P` でファイルの更新日時を保ちます

スコアは撮影条件（被写体・ISO・ピクチャーコントロールのシャープネスなど）で大きく変わるため、
絶対値ではなく、同じフォルダ内の相対的な順位（下位 ◯%）で判定するのが基本です。

## 注意

- NX Studio で編集やレーティングをした写真は、調整ファイル（`NKSC_PARAM` フォルダ）の情報が優先されることがあります
- 解析は内蔵 JPEG プレビューで行います。プレビューがカメラ内で縮小されている場合は判定精度が落ちます（一覧の「プレビュー」列で確認できます）
- ニコン Z シリーズ以外の NEF でも動作しますが、AF エリア情報は機種によって記録されていないことがあります

## テスト

```bash
uv run pytest                      # dev グループ（pytest）は uv sync で一緒に入ります
```

実際の NEF でテストするには、`tests/data/` に NEF を置いてください（`.gitignore` でリポジトリには含めません）。

## ライセンス

MIT License（[LICENSE](LICENSE)）

このアプリは次のソフトウェアを利用します。各ライセンスに従ってください。

- [PySide6](https://pypi.org/project/PySide6-Essentials/)（Qt for Python）: LGPLv3 / GPLv2 / 商用
- [OpenCV](https://opencv.org/): Apache License 2.0
- [NumPy](https://numpy.org/): BSD License
- [ExifTool](https://exiftool.org/): Perl と同じライセンス（別途インストールして外部コマンドとして実行）

「Nikon」「NEF」「NX Studio」はニコンの商標です。このプロジェクトはニコンとは関係ありません。
