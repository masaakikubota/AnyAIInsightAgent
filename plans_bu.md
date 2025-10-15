# AnyAI Marketing Agent 開発計画（2025-10-14更新）

## 1. 概要
AnyAI Marketing Agent は、LLM を用いた合成消費者調査パイプラインです。Interview／Cleansing／Scoring／Dashboard を FastAPI ベースで提供し、Google Sheets・Gemini・OpenAI と連携します。本ドキュメントは現状の実装状況と引き継ぎタスクを整理したものです。

## 2. 完了済みマイルストーン
- Interview v0.1：方向性YAML → ペルソナ生成 → QA トランスクリプト自動生成。
- SSR Mapper：自由回答を Likert 分布に写像するパイプライン統合。
- Dashboard Generator：AnyAI UI Master に準拠した静的 HTML を LLM 二段構成で生成。
- Cleansing Manager：原データ整形パイプラインとUIを提供。
- Mass Persona Manager：200件規模の大量ペルソナ生成と Sheets 連携。
- Persona Builder：個別ペルソナ構築とUI提供。
- Persona Responder：ペルソナ応答生成・JSON 化。
- Result Dashboard Suite（2025-10-13）：`/dashboard` UI、フィルタリングAPI、CSV/PNG/PDF エクスポート。
- Interview Enhancements（2025-10-13）：発話CSV トライブ学習（任意）、手動刺激画像対応、QA回答/スコアの `QA_Answer`/`QA_Embedded` 反映。
- Interview Tribe/Persona Sheet Integration（実装中）：トライブ/ペルソナ数を変数化し、`Tribe_SetUp`/`Persona_SetUp` へのバッチ出力、ContextWindow 固定、500k tokens 制限を実装（UI/バックエンド実装完了、検証・仕上げタスクが残り）。

## 3. 進捗中テーマ
### 3.1 Interview Tribe/Persona Sheet Integration（検証・仕上げフェーズ）
- 実装済み要素：トライブ変数、CSVトークン制限、Tribe_SetUp/Persona_SetUp への書き込み、共通質問生成。
- 2025-10-14 更新:
  - ✅ Sheets 出力結果 QA（列マッピング、ID、SessionID）を実施し、ローカルユニットテストで検証（`tests/test_interview_sheet_mapping.py`）。
  - ✅ SessionID 割り当て計画を LLM プロンプトに明示化し、10件単位の ContextWindow を保証。
  - ✅ `max_rounds == questions_per_persona` をサーバ側・Pydantic バリデーションで強制。
  - ✅ CSV 無し／境界ケースのスモークテストを追加し、`Tribe_SetUp`/`Persona_SetUp` 出力整合性を確認。

### 3.2 AnyAI Scoring パイプライン高速化（新規）
- 目的：Scoring ジョブを I/O 待機時間に左右されない非同期パイプラインへ再構成し、LLM 呼び出しの実効スループットを向上させる。
- 実施予定タスク（詳細は §5 参照）:
  1. スコアリング問い合わせの非同期パイプライン化。
  2. LLM レスポンス検証を専用ワーカーにオフロード。
  3. パーシャル結果をストリーミングで書き込み。
  4. LLM レスポンスキャッシュの導入。
  5. 入力シートを 500 行単位のチャンクに分割し、それぞれ独立ジョブとして実行。完了チャンクのみ Sheets を更新、失敗チャンクは自動再実行するチャンクスケジューラを実装。

### 3.2 Dashboard キャッシュ戦略
- Parquet キャッシュ基盤とテストスクリプトは整備済み。残りはベースライン計測とドキュメント更新。

### 3.3 今後の大型タスク（Phase 2 / Phase 3）
- アクセスログ／監査証跡整備（middleware + BigQuery 出力）。
- LLM プロバイダ統合強化（フォールバック／レート制御）。
- データフロー最適化（async batching / streaming）。
- エラー処理・リトライ・通知強化。
- パフォーマンス監視・SLO整備。
- フェーズ3：多言語対応、マルチモーダル刺激、リアルタイム協働、レート制限・課金、カスタムダッシュボードテンプレート。

## 4. 作業ロードマップ
| 項目 | 期待成果 | 状態 |
|------|-----------|------|
| SSR 分位点可視化 | `dashboard_data.py` / UI 更新 | ✅ 完了 |
| Dashboard キャッシュ戦略 | 計測メモ、Parquet キャッシュ実装、`scripts/test_dashboard_cache.py` | 🚧 計測待ち |
| Interview トライブ/ペルソナ統合 | Sheets 出力、共通質問、ContextWindow 制御、500k token バリデーション | ✅ 完了（2025-10-14） |
| Scoring パイプライン高速化 | 非同期化・検証ワーカー・ストリーミング書き込み・LLMキャッシュ・チャンク分割 | ⏳ 未着手 |
| ログ/監査証跡整備 | middleware・BigQuery スキーマ | ⏳ 未着手 |
| LLM プロバイダ統合 | フォールバック/レート制御 | ⏳ 未着手 |
| データフロー最適化 | Job Manager 改修、性能計測 | ⏳ 未着手 |
| パフォーマンス監視・SLO | 計測ダッシュボード | ⏳ 未着手 |
| Phase3 拡張 (i18n, マルチモーダル 等) | PoC / 設計 | ⏳ 未着手 |

## 5. 直近の ToDo
1. **Scoring パイプライン高速化 設計**：非同期パイプライン化、検証ワーカー分離、ストリーミング書き込み、LLM キャッシュ、500行チャンク分割と自動再実行スケジューラの設計ドキュメントを作成し、PoC タスクをチケット化。
2. **ドキュメント更新**：README の Interview セクション、`Tribe_Category` 説明を最新仕様に合わせる。
3. **キャッシュ計測**：`docs/perf/dashboard_cache_baseline.md` の TODO を消化し、結果を共有。

## 6. 参照ファイル & 主変更点
- `app/models.py`：トライブ関連パラメータ、トークン制限。
- `app/main.py`：フォーム取り込み、CSVトークン推定、ファイル保存。
- `app/interview_manager.py`：トライブ生成、Sheets 出力、共通質問配布。
- `app/services/interview_llm.py`：トライブ・ペルソナ・インタビュー生成ロジック。
- `app/static/interview.html`：UI刷新、派生値計算、CSV/画像アップロード。
- `Tribe_Category`：Tribe_SetUp 用ヘッダー。
- `tests/test_interview_sheet_mapping.py`：Tribe/Persona 出力整合性スモークテスト。
- ※ Scoring パイプライン高速化はこれから着手予定のため参照ファイルは未確定。

## 7. リスクと対応策
| リスク | 内容 | 対応策 |
|--------|------|--------|
| シート列不整合 | LLM 出力が列定義と合わない可能性 | QA でチェックし、実装側で空文字にフォールバック。 |
| CSV 大量投入 | 500k tokens 超の CSV | 事前推定でエラー返却。境界ケースを検証。 |
| ContextWindow 未指定 | 同一バッチでコンテキスト混線 | SessionID assignment をプロンプトへ明示（2025-10-14 実装済）。 |
| ダッシュボードキャッシュ未検証 | ヒット率・TTL 未評価 | テストスクリプトで計測し、`docs/perf/dashboard_cache_baseline.md` を更新。 |

## 8. ハンドオフメモ
- サービスアカウント：`anyagent-cep@anyai-playground.iam.gserviceaccount.com` の Sheets 権限を全処理で確認する仕組みあり。権限不足時はエラーとなるため共有設定を要確認。
- 実行ログ：`runs/server.log`。各ジョブのサマリは `runs/<module>/<job_id>/summary.json`。
- 参考：`scripts/test_dashboard_cache.py`、`docs/perf/dashboard_cache_baseline.md`、`Tribe_Category`。

以上。引き継ぎは README 更新とダッシュボードキャッシュ計測（§5）から着手するのがおすすめです。EOF
