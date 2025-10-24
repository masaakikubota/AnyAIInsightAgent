# Tribe-Interview UI & Data Flow 概要（2025-10-24）

## 1. ページ構成

- 単一ページ `tribe_interview.html`
- サイドバー項目: "Tribe Interview"
- 主コンテンツ: 2 カラムレイアウト
  - 左: 入力フォーム（スクロール対応）
  - 右: ステージ進捗カード（Tribe / Combination / Persona / QA / Embedding）

## 2. 入力フォームセクション

1. **プロジェクト情報**
   - プロジェクト名
   - Product Category（必須）
   - Country / Region（必須）
   - モード選択（Product / Communication）
2. **製品/タグライン入力**
   - テキストボックス（製品詳細 or タグライン）
   - 画像アップロード（ドラッグ＆ペースト対応、任意）
3. **生成パラメータ**
   - ペルソナ生成数（組み合わせあたり、min 1 / max 10 推奨）
   - インタビュー生成数（1〜10 想定）
4. **Google Sheets 設定**
   - スプレッドシート URL / ID + Set ボタン
   - シート選択（Tribe_SetUp など 5 種。デフォルトをプリセット）
5. **送信ボタン**
   - `開始` ボタン。投稿後は disabled / spinner 表示

## 3. ステージ進捗カード

| ステージ | 表示要素 |
|----------|-----------|
| Tribe Generation | ステータス、生成行数、アクティブ属性一覧 | 
| Combination | 組み合わせ件数 |
| Persona Prompts | 生成済み/総数、リンク（`Persona_SetUp` 範囲） |
| QA Responses | 進捗バー、出力セルレンジ |
| Embeddings | ベクトル生成件数 |

- 右カラム下部に成果物リンク（ローカル JSON、runs フォルダ）
- エラー時は該当ステージカードを赤表示し、再実行ガイドを提示

## 4. データフロー

1. `POST /tribe_interview/jobs`
   - フォーム送信（FormData）。画像は multi-part。
   - レスポンス: `job_id`
2. `GET /tribe_interview/jobs/{job_id}`
   - 2s 間隔でポーリング。`stage`, `status`, `message`, `metrics`, `artifacts` を取得。
   - ステージ毎に UI を更新。
3. 完了時
   - ステージ全て `completed` になり、成果物リンク（JSON）を有効化。
   - `Restart` / `New Job` ボタンを表示。

## 5. バリデーション

- 必須項目：Product Category, Country/Region, Persona/Interview 数値, Spreadsheet URL
- シート名が未選択の場合は警告
- モードが communication の場合、タグライン入力を必須にする
- 画像ファイルは MIME type チェック（image/*） + サイズ上限 5MB（予定）

## 6. エラー表示

- API エラーはトースト + ステージカード上にメッセージ
- ネットワーク断は再試行ボタンを表示
- LLM ステージで失敗した場合は詳細ログリンク（runs ディレクトリ）を表示

## 7. TODO
- フロント側で JSON 化された成果物への参照ガイド（ダウンロードボタン）
- Accessibility チェック（フォーカス制御・aria 属性）
- レイアウト崩れ防止のレスポンシブ対応

