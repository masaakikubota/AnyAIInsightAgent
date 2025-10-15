# AnyAI Scoring 実装計画（詳細）

このドキュメントは `plans.md` に記載した Scoring 改修テーマを実装段階に落とし込むための具体的なタスクリストです。各ステージは順番に着手し、完了時に `plans.md` と同期してください。

---

## 1. 非同期パイプライン化

### 1.1 設計
- [x] 現在の `JobManager.run_job` フローを分解し、以下のコンポーネントを定義する（`docs/design/scoring_pipeline.md` 参照）。
  - `ScoringProducer` : 行/ブロック単位のタスクを作成し `asyncio.Queue` に投入。
  - `ScoringInvoker` : LLM 呼び出し担当。並列度管理と 429 レート調整を実装。
  - `ValidationWorker` : LLM レスポンスを検証し整形。失敗時のフォールバックもここで処理。
  - `SheetWriter` : 部分結果を書き戻すワーカー。一定件数/時間で flush。
- [ ] `RunConfig` にパイプライン関連設定（`pipeline_queue_size` など）が必要か検討。
- [x] `checkpoint.json` 更新タイミングを ValidationWorker で統一する仕様を記載。

### 1.2 実装
- [x] `app/worker.py` にパイプライン用クラス or 内部 `async` 関数を追加（テキストスコアリング向け PoC を導入）。
- [x] 既存の同期ループを置き換え、Producer → Invoker → Validation → Writer の順にデータを流す（テキスト/Video 共通化済み、Retry パスもパイプライン経由に移行）。
- [x] Video モードのダウンロード/アップロード/一時ファイル削除を `ScoringPipeline` に統合し、`cleanup_queue` で後処理を自動化。
- [ ] 例外ハンドリングを統一し、致命的なエラーでパイプラインを停止する際はキューに `None` を投入して consumer を終了させる（キャンセル時のリカバリと Video 対応が未完）。

### 1.3 テスト
- [ ] 小規模シートで動作確認、`processed_rows` と Sheets 更新が正しく行われるか検証。
- [ ] 429 レート調整が動くシミュレーションテスト（モック）を追加。

---

## 2. LLM レスポンス検証ワーカー

### 2.1 設計
- [ ] Validation ステージで実行する処理：JSON schema 検証、スコア整形、フォールバック判定、キャッシュ書き込み。
- [ ] `ThreadPoolExecutor` か `ProcessPoolExecutor` のどちらを採用するか決定（CPU 使用率を基準に）。
- [ ] スレッドプールサイズ、タイムアウト、失敗時の再実行ロジックを整理。

### 2.2 実装
- [ ] `asyncio.to_thread` もしくは `loop.run_in_executor` で検証処理をオフロード。
- [ ] ValidationWorker からの戻り値を SheetWriter 用フォーマットに整形。

### 2.3 テスト
- [ ] 異常 JSON / 欠損スコアのテストケースを追加し、フォールバックが正しく動くか確認。
- [ ] 高並列で検証ワーカーが詰まらないかベンチマークを実施。

---

## 3. パーシャル結果のストリーミング書き込み

### 3.1 設計
- [ ] SheetWriter にキューを持たせ、`flush_interval`（秒）と `flush_batch_size`（件数）の設定を決定（案: 2 秒 / 50 件、要検証）。
- [ ] Google Sheets API のレートを順守するための backoff ポリシーを定義。
- [ ] `apply_updates_to_sheet` を再利用するか、新しい書き込み関数を用意するか決める。

### 3.2 実装
- [x] ValidationWorker から `(row_index, col_offset, scores, metadata)` をキュー投入（テキストモードで導線構築済み）。
- [x] SheetWriter が一定条件で `batch_update_values` を発行し、成功したエントリを削除（flush_interval / flush_unit_threshold を暫定値で実装）。
- [ ] フラッシュ失敗時はリトライし、上限到達でパイプラインにエラー通知（GoogleSheetsError のリカバリ設計が未実装）。

### 3.3 テスト
- [ ] 書き込みキューが詰まらないか、長時間テスト。
- [ ] Sheets 側で結果が逐次反映されるか手動確認。

---

## 4. LLM キャッシュ導入

### 4.1 設計
- [x] キャッシュキー：`hash(utterance, categories[], system_prompt, provider, model)`。
- [x] 保持形式：JSON (ローカル) or SQLite。PoC は JSON で開始し、必要に応じて拡張。保存先は `runs/scoring/cache.json` を想定。
- [x] TTL（例: 24h）と最大件数（例: 10,000）を設定。メトリクスとしてヒット率を計測。
- [x] キャッシュ無効化オプション（`RunConfig` にフラグ追加）を検討。

### 4.2 実装
- [x] シンプルなキャッシュクラスを作成（読み込み・保存・cleanup）。
- [x] ScoringInvoker で LLM 呼び出し前にキャッシュ確認、ヒットしたら Validation に直接流す。
- [x] フォールバック含む最終スコアが決定した時点でキャッシュへ保存。

### 4.3 テスト
- [x] キャッシュヒット時に LLM を呼ばず処理できるか確認。
- [x] TTL・最大件数が正しく機能するかユニットテスト。
- [ ] キャッシュ統計（ヒット率など）を出力してメトリクス可視化の要否を判断。

---

## 5. 500 行チャンク分割と自動再実行

### 5.1 設計
- [x] `RunConfig` に `chunk_row_limit=500` / `chunk_retry_limit` を追加し、親ジョブと子チャンクジョブのメタ関係を整理。
- [x] `JobManager` にチャンク再実行ロジックを追加。`chunk_retry_limit` を参照して失敗チャンクのみ自動リトライ（同一ジョブ内で再試行）。
- [x] チャンク状態を記録するためのメタファイル（`chunk_meta.json`）を定義。項目: `chunk_id`, `row_range`, `status`, `retry_count`, `last_error`。
- [x] 自動再実行ポリシー：最大リトライ回数（`chunk_retry_limit`）と指数バックオフ（min(60, 2^retry) 秒）を導入。
- [ ] すべてのチャンクが完了したタイミングで親ジョブを completed にし、UI に進捗を集約表示。

### 5.2 実装
- [ ] `create_job` / `run_job` のエントリポイントを改修し、親ジョブからチャンクジョブを生成。
- [ ] Sheets 書き込みはチャンク完了時にのみ実施。パイプライン側ではチャンク内バッファを保持し、チャンク終了時に一括 flush。
- [ ] 失敗チャンクは `retry_count < limit` の場合自動でキューに再投入、上限超過で親ジョブにエラーを伝搬。

### 5.3 テスト
- [ ] 1,500 行以上のシートで 3 チャンク動作を検証。
- [ ] 1 チャンクだけ意図的に失敗させ、再実行と最終エラー表示が期待どおりか確認。
- [ ] 再開（resume）時にチャンク状態を正しく読み込めるかテスト。

---

## 6. 検証とリリース準備
- [ ] ログフォーマット更新：パイプライン各ステージの処理時間、キャッシュヒット率、チャンク進捗を記録。
- [ ] README / UI の説明文更新。
- [ ] 大規模シート（>10,000 行）を用いたベンチマーク結果を `docs/perf/scoring_pipeline.md` にまとめる。
- [ ] リリースノート作成（Breaking change があれば明記）。
- [x] テストシナリオ整理：チャンク再実行（成功/失敗）、429 多発時の並列調整、ストリーミング書き込み障害などを記録。


---

- [x] チャンク再実行フローを実装（`chunk_retry_limit` に基づく自動リトライ、`chunk_meta` 反映）。
- [x] LLM キャッシュ基盤の導入（`RunConfig` オプション追加・スコアリング経路統合）。
- [x] 非同期パイプライン（タスク 1.2 / 3.2）に着手し、Producer/Invoker/Validation/Writer をコード化（テキストモード PoC 完了）。
- [ ] ValidationWorker / SheetWriter の並列実行モデルを決定し、ストリーミング書き込み PoC を Video/Retry フローへ拡張。

- [ ] **非同期パイプライン拡張**  
  - Video モード/リトライパスにもパイプラインを適用し、既存の同期処理を段階的に廃止。  
  - 429 調整・停止処理をキュー経由で統一し、`_terminate` フローとエラーハンドリング基盤を整備。
- [x] **Video パイプライン統合詳細設計**  
  - `download_video_to_path` / `upload_video_to_gemini` を Producer 前処理として実装し、`file_parts` を Invoker に受け渡すフローを追加。  
  - Validation/Writer で Cleanup キューを管理し、Writer 終了時に一時ファイル削除を走らせる実装を完了。  
  - 今後は例外時のクリーンアップ保証と再試行時のリソース管理を検証する。
- [ ] **Validation/Writer ワーカー設計の具体化**  
  - JSON 検証とスコア整形をワーカーに移す際のスレッドプールサイズ、タイムアウト、再実行ポリシーを決定。  
  - SheetWriter のフラッシュ条件（件数・時間）とリトライポリシーを RFC にまとめる。
- [ ] **チャンク実装の検証強化**  
  - ユニットテストで成功／失敗パターンをカバーし、`chunk_meta` と `checkpoint.json` の整合性を確認。  
  - 1 万行規模のシートで手動検証し、自動再実行とシート書き戻しが期待どおりに動くか確認。
- [ ] **親ジョブ集約と UI 連携**  
  - 全チャンク完了時に親ジョブを `completed/failed` へ更新し、Queue API ですべてのチャンク進捗を返す。  
  - UI/README の文言を更新してチャンク処理・再実行・ストリーミング書き込みを説明。
- [ ] **パフォーマンス検証 & ドキュメント整備**  
  - 長時間ロードテスト（429・Sheets 障害含む）でメトリクスを取得し、`docs/perf/scoring_pipeline.md` に記録。  
  - リリースノート草案を作成（Breaking change の有無を確認）。  
  - キャッシュヒット率の収集方法を検討し、必要であればダッシュボードに追加。
---

## 7. 参考リンク
- 設計メモ: `docs/design/scoring_pipeline.md`
- キャッシュベンチマーク（未作成）: `docs/perf/scoring_cache.md`
