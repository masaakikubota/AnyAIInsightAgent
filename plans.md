# AnyAI Tribe-Interview 統合計画（2025-10-24）

この文書では Seeds 機能と Interview 機能を統合し、`Tribe_SetUp` → `Tribe_Combination` → `Persona_SetUp` → `QA_LLM` → `QA_Embedding` の順でシートを更新する新パイプラインの実装方針とタスクを整理する。

---

## 1. ゴール
- シングル UI / シングル API でトライブ生成からインタビュー結果のベクトル化まで一気通貫で処理できるようにする。
- LLM 呼び出しと計算ロジックを段階的に実行し、各ステージを Google Sheets に反映・再開できるようにする。
- 参照論文（https://arxiv.org/html/2510.08338v2）に沿った属性セットとプロンプト設計を採用し、再現性を確保する。
- Dashboard 系機能を排除し、Seeds / Interview のコード重複を解消する。

---

## 2. ユースケースと入出力整理

| フロー段階 | 入力 | 出力 / 更新対象 | 備考 |
|-------------|------|----------------|------|
| 0. 初期設定 | ユーザが `ProductCategory`, `Country/Region`, 製品詳細 or タグライン（テキスト/画像任意）、モード（製品/コミュニケーション）、トライブ組み合わせ当たりのペルソナ数、インタビュー数、シート名（任意）を入力 | `/tribe_interview/jobs` への POST | モードに応じて後続プロンプトを切替。画像は任意で保存。 |
| 1. トライブ生成 | 上記入力 | `Tribe_SetUp`（列 B〜V 固定） | `gemini-pro-latest` をバッチ 3 で呼び、各属性を最大 10 行生成。ヘッダは Gender〜Communication Tool を固定。リトライ最大 5 回。 |
| 2. 組み合わせ展開 | `Tribe_SetUp` の行 | `Tribe_Combination` | `Tribe_SetUp` と同一ヘッダを持ち、属性の全組み合わせをアルゴリズムで展開。制限なし。 |
| 3. ペルソナプロンプト生成 | 各組み合わせ, ユーザ入力 | `Persona_SetUp` | `gemini-flash-latest`（fallback `gpt-4.1`）で組み合わせごとに指定件数のプロンプトを生成。リトライ 5 回。 |
| 4. インタビュー応答生成 | `Persona_SetUp` の各プロンプト + 製品詳細/タグライン | `QA_LLM` | 指定インタビュー数だけ `gemini-flash-latest`（fallback `gpt-4.1`）で自然文回答を生成。組み合わせ行の B 列以降に水平配置。リトライ 5 回。 |
| 5. Embedding | `QA_LLM` の各セル | `QA_Embedding` | `gemini-embedding-001`（fallback `text-embedding-3-small`）で対応セルに埋め込みベクトルを出力。 |

---

## 3. アーキテクチャ刷新方針

1. **UI 統合**
   - `persona.html` と `interview.html` を統廃合し、新しい `tribe_interview.html`（仮称）を作成。
   - 入力フォームに必須項目（カテゴリ・地域）、モード選択、生成数指定、画像アップロード、スプレッドシート設定（Set→Select）を集約。
   - 進捗表示をステージ単位（Tribe / Combination / Persona / QA / Embedding）で切り替え。

2. **API / マネージャ統一**
   - `/persona`・`/interview` ルータを統合し、`/tribe_interview` ルータを新設。Dashboard エンドポイントは削除。
   - `MassPersonaJobManager` と `InterviewJobManager` を再構成し、`TribeInterviewJobManager`（仮）として一本化。
   - 共通の `RunConfig` 相当（Tribe/Persona/QA 設定 + シート名）を `TribeInterviewJobConfig` として定義。

3. **ジョブ実行パイプライン**
   - ステージごとにメソッドを分割し、各ステージ完了時にジョブディレクトリへ成果物保存＋シート反映。
   - LLM 呼び出しは `retry=5`（指数バックオフ）を共通実装で扱う。
   - 失敗時はステージ単位で停止し、`status=failed` とメッセージを記録。

4. **シート書き込み**
   - `Tribe_SetUp`：既定ヘッダ（B1〜V1）を事前チェックし、`batch_update_values` で 2 行目以降を埋める。
   - `Tribe_Combination`：ヘッダをトライブと揃え、組み合わせ結果を横並びで書き込む。
   - `Persona_SetUp`：組み合わせ ID、プロンプトテキスト、関連属性を整理し縦方向に追加。
   - `QA_LLM` / `QA_Embedding`：組み合わせ行と列位置を一致させるため、行インデックス管理テーブルを保持。

5. **ファイルと成果物**
   - `runs/tribe_interview/<job_id>/` 配下に以下を保存：`config.json`, `tribes.json`, `combinations.json`, `personas.json`, `qa_llm.json`, `qa_embedding.json`, `summary.json`。
   - 画像をアップロードした場合は同ディレクトリに格納し、パスを config に記録。

6. **ダッシュボード削除**
   - `/persona/dashboard/*` ルート、`PersonaResponseJobManager`、関連モデル・テンプレートを削除。
   - UI の Dashboard タブも削除し、新 UI へリダイレクト。

---

## 4. 実装タスク一覧

### PHASE 0: 設計・準備
1. プロンプトドラフト作成
   - トライブ生成（属性 18 項目、最大 10 行）用プロンプト
   - ペルソナ生成（モード別）プロンプト
   - インタビュー質問テンプレート（製品モード / コミュニケーションモード）
   - 失敗時リトライ時の挙動（温度、シード等）を決定

2. データモデル定義
   - `TribeInterviewJobConfig`（カテゴリ・地域・シート設定・生成数・モード・画像パス）
   - `TribeInterviewJobProgress`（stage, status, counts, artifacts）
   - `TribeDefinition`, `PersonaPrompt`, `InterviewRecord` など中間構造体

3. 旧機能の棚卸し
   - Dashboard 系コードとビュー、利用のない API をチェックして削除対象を明確化

### PHASE 1: バックエンド土台
1. 新ルータ `/tribe_interview` 作成、旧 `/persona` `/interview` との互換レイヤー（暫定リダイレクトまたは 410）を整備
2. `TribeInterviewJobManager` 実装
   - ステージ遷移ロジック（ingest → combine → persona → qa → embed）
   - ステージごとのリトライ（最大5回）とエラーハンドリング
   - 進捗・成果物記録
3. Google Sheets 書き込みユーティリティ拡張
   - ヘッダ整合チェック、既存シートのクリア/追記の判断
   - 同一ジョブ内で行インデックスを管理するマッピングテーブル作成

### PHASE 2: ステージ実装
1. トライブ生成
   - LLM 呼び出し実装、バッチ3並列、最大10行、属性欠損時の再要求ロジック
   - `Tribe_SetUp` への書き込み
2. トライブ組み合わせ
   - 組み合わせロジック（属性ごとの可変要素を抽出、直積生成）
   - `Tribe_Combination` への書き込み
3. ペルソナプロンプト生成
   - モード別プロンプト、指定件数の生成
   - `Persona_SetUp` への書き込み
4. インタビュー生成
   - モード別質問テンプレート、指定回数の LLM 呼び出し
   - `QA_LLM` 行列のマッピング
5. 埋め込み生成
   - `gemini-embedding-001` 呼び出し、ベクトル形式のセル書き込み（JSON or カンマ区切り）

### PHASE 3: フロントエンド統合
1. 新 UI 作成（`tribe_interview.html`）
   - 入力フォーム（カテゴリ等）とシート設定 UI（Set→Select）
   - 画像アップロード（ドラッグ＆ペースト）
   - ステージ進捗カード・成果物ダウンロードリンク
2. サイドバー等で既存 Seeds / Interview エントリを統合
3. API 呼び出し・ポーリング実装（`/tribe_interview/jobs`, `/tribe_interview/jobs/{id}`）

### PHASE 4: クリーニングと移行
1. Dashboard コード・テンプレート削除
2. 旧 UI（persona.html / interview.html）削除とリダイレクト
3. README / ドキュメント更新
4. 既存テストの整理、新パイプラインのユニット/統合テスト追加

---

## 5. スケジュール目安

| フェーズ | 期間目安 | 主な成果物 |
|----------|-----------|------------|
| PHASE 0 | 2 日 | プロンプト草案、モデル定義、削除対象リスト |
| PHASE 1 | 3 日 | 新ジョブマネージャ / ルータ骨格、シートユーティリティ下地 |
| PHASE 2 | 5 日 | 各ステージ処理実装・LLM 呼び出し・シート更新 |
| PHASE 3 | 3 日 | 新 UI・API 連結、ポーリング・成果物表示 |
| PHASE 4 | 2 日 | 旧機能削除、ドキュメント更新、テスト整備 |

※ 実際には LLM プロンプト調整や再試行ロジック検証に追加バッファを見込む。このスケジュールを初期目安に、進捗に応じて調整する。

---

## 6. リスクと対策

| リスク | 内容 | 対策 |
|--------|------|------|
| LLM 出力の揺らぎ | 属性が欠落／フォーマット崩れ | プロンプトで JSON 固定、validator 導入、5 回までリトライ |
| 組み合わせ爆発 | 属性直積により行数が膨大 | 入力時に注意喚起、進捗表示、処理時間見積りを UI に表示 |
| Sheets 書き込み失敗 | レート制限/ネットワーク障害 | 共通リトライ処理 + バックオフ、部分的なローカル保存で再開可能に |
| 既存ジョブとの互換性 | 旧 UI/API が動かなくなる | リリース前に十分周知、旧ルートは HTTP 410 を返し明記 |
| Embedding サイズ | ベクトルをセルに書き込む際の形式揺れ | JSON 文字列で統一、後続処理が必要なら別シートへ転記 |

---

## 7. 次アクション
1. プロンプトドラフト（ステージ1〜5）を作成し、レビューを受ける。
2. 新しい `TribeInterviewJobConfig` / `Progress` のモデルスケッチを `app/models.py` に追加。
3. ダッシュボード関連コードの依存を洗い出し、削除リストを固める。
4. UI モック（ワイヤーフレーム）を描き、入力項目とステージ表示の導線を確認する。

以上を完了後、PHASE 1 の実装に着手する。
