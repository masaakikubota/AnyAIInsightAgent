# AnyAI UI Kit（ビルドレス）

- 目的: エンジニアがこのキットだけで「どのプロダクトでも」UI の実装方針がわかり、素早く着手できるようにする
- 方式: 純粋な HTML/CSS/JS（外部依存最小）
- ブランディング: すべてのページで左上に `assets/AnyAI_logo.png` を表示、ファビコンは `assets/AnyAI_icon.png`

## 構成

```
core/               # トークン・基本コンポーネント・ユーティリティ
docs/               # コンポーネント一覧とパターン
templates/          # 実働想定テンプレート（アプリ/サイト/フォーム/テーブル/認証/状態）
refactors/          # AnyTag/AnyX のリファイン例
assets/             # ロゴとアイコン
index.html          # 概要・導線
```

## 使い方

1. `core/anyai.tokens.css` を読み込む（色/間隔/タイポの統一）
2. `core/anyai.components.css` と `core/anyai.utilities.css` を状況に応じて
3. すべての HTML の `<head>` に: `<link rel="icon" href="assets/AnyAI_icon.png">`
4. すべてのページの左上（ヘッダー）に: `<img src="assets/AnyAI_logo.png" alt="AnyAI">`（高さ 28px）

## ダークモード

`<html data-theme="dark">` または `localStorage.anyai-theme = "dark"` で切替可能。トークンで自動反映。

## 既存 HTML との統合

- 任意のページに対し、`<header class="anyai-header">...</header>` と `core/*.css` を追加
- 独自スタイルは、まずトークン（`--anyai-*`）を参照して上書きを推奨
