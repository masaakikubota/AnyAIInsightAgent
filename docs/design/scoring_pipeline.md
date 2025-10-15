# AnyAI Scoring パイプライン設計メモ

最終更新: 2025-10-14  
対象: `app/worker.py` を中心とした Scoring ジョブ実行処理

---

## 1. コンポーネント構成

```
ChunkScheduler --> ScoringPipeline
                   ├─ Producer (async)
                   ├─ InvokerPool (async workers)
                   ├─ ValidationPool (executor workers)
                   └─ SheetWriter (async)
```

### 1.1 ChunkScheduler
- 入力シートを 500 行単位で分割し、`ChunkJob` を生成。
- `chunk_meta.json` に `chunk_id`, `row_range`, `status`, `retry_count`, `last_error`, `updated_at` を記録。
- チャンク成功時に Sheets へ反映し、親ジョブの進捗を更新。失敗時は自動再キューイング（最大 3 回）。

### 1.2 Producer
- `ChunkJob` が割り当てられた行・カテゴリブロックを読み込み、`TaskPayload` を `asyncio.Queue`（`invoke_queue`）へ投入。
- キューの最大長（例: 2 * concurrency）を超える場合は backpressure が掛かる。
- チェックポイント復元時は完了済みブロックをスキップ。

### 1.3 InvokerPool
- `invoke_queue` から `TaskPayload` を取得し、LLM へリクエストを送信。
- 429 レートリミット発生時は動的に並列度を調整（現行ロジックをコルーチン単位へ移植）。
- 成功結果 or 例外情報を `validation_queue` に送る。キャッシュヒット時は LLM をバイパス。
- `invoke_queue` が空 & Producer から `None` 受信で終了。

### 1.4 ValidationPool
- `validation_queue` を消費し、JSON スキーマ検証・スコア整形・フォールバック判定を実施。
- CPU バウンド処理のため `ThreadPoolExecutor` を利用（ワーカー数は CPU コア数に応じて設定）。
- 成功時は `(row_index, block_index, col_offset, scores_meta)` を `writer_queue` に push。
- 失敗時は `TaskPayload.retry()` を用いて InvokerPool に再投入するか、最大リトライ超過でチャンクを失敗扱いにする。

### 1.5 SheetWriter
- `writer_queue` を消費し、バッファリング後に `batch_update_values` を呼び出す。
- フラッシュ条件:
  - バッファ件数 >= `flush_batch_size`（例: 50）
  - 最終フラッシュから `flush_interval`（例: 2 秒）経過
  - チャンク完了時（`None` 受信時）は即フラッシュ
- 書き込み失敗時は指数バックオフで再試行し、5 回失敗でチャンクを失敗として報告。
- フラッシュ完了後に checkpoint を更新し、`job.processed_rows` をインクリメント。

### 1.6 Chunk Retry Policy
- `RunConfig.chunk_retry_limit`（既定 3 回）までチャンクを自動再実行。
- 失敗時は `chunk_meta.json` の `retry_count` をインクリメントし、`status=failed`、`last_error` にメッセージを保存。
- `retry_count < chunk_retry_limit` の場合、ChunkScheduler がチャンクを再度キューに投入。バックオフは `min(60, 2 ** retry_count)` 秒を推奨。
- 上限超過で親ジョブに失敗を伝搬し、UI へエラーを表示。必要に応じて手動再実行 API（TODO）を提供。

---

## 2. データモデル

### 2.1 TaskPayload
```python
TaskPayload = TypedDict({
    "chunk_id": str,
    "row_index": int,        # 0-based absolute index
    "block_index": int,
    "col_offset": int,
    "utterance": str,
    "categories": List[Category],
    "retry_count": int,
})
```

### 2.2 ValidationResult
```python
ValidationResult = TypedDict({
    "chunk_id": str,
    "row_index": int,
    "block_index": int,
    "col_offset": int,
    "scores": List[Optional[float]],
    "provider": str,
    "model": str,
})
```

### 2.3 ChunkMeta
```json
{
  "chunk_id": "chunk-0001",
  "row_start": 5,
  "row_end": 504,
  "status": "running",      // pending | running | completed | failed
  "retry_count": 1,
  "last_error": null,
  "updated_at": "2025-10-14T09:30:00Z"
}
```

---

## 3. 並列制御

| キュー | 役割 | サイズ/制御 |
|--------|------|--------------|
| `invoke_queue` | Producer → Invoker | `maxsize = concurrency * 2`（自然な backpressure） |
| `validation_queue` | Invoker → Validation | 無制限（Executor がバックプレッシャを掛ける） |
| `writer_queue` | Validation → SheetWriter | `maxsize = 100`（書き込み遅延を検知） |

### 3.1 レートリミット調整
- InvokerPool の各タスクは共有 `ConcurrencyController` を参照して await。
- 429 発生時は `ConcurrencyController.reduce()` で並列数を 0.7 倍にし、`clean_batches` をリセット。
- 再度安定すれば `increase()` で 1 ずつ回復。

### 3.2 チャンク並列
- 初期バージョンではチャンクはシリアル実行（Queue 内で順番に処理）。
- 将来的な並列化に備え、ChunkScheduler はチャンクごとに独立した `ScoringPipeline` インスタンスを生成できる設計にしておく。

---

## 4. 例外ハンドリング

### 4.1 Invoker
- ネットワークエラー: 最大 `cfg.max_retries` まで指数バックオフで再実行。
- 429: ConcurrencyController が並列度を縮小し、タスクはリトライ。
- 400/500 など致命的エラー: `TaskPayload` に `retry_count` を設定し ValidationQueue にエラーとして渡す。

### 4.2 Validation
- JSON decode 失敗・スコア長不足などは `TaskPayload.retry()` を呼び、`retry_count < cfg.max_retries` の場合 InvokerQueue へ戻す。
- 上限超過時は `ChunkMeta.last_error` を更新し、SheetWriter に空の更新を送らない。

### 4.3 SheetWriter
- `batch_update_values` 失敗時は最大 5 回再試行。失敗継続でチャンク失敗扱い。
- フラッシュ時に `update_buffer` をクリア。失敗で再試行中はバッファ保持。

### 4.4 パイプライン終了
- Producer → InvokerQueue へ `None`
- Invoker → ValidationQueue へ `None`
- Validation → WriterQueue へ `None`
- SheetWriter が `None` を受け取ると最終フラッシュ → `pipeline.done()` を発火。

---

## 5. チェックポイント更新タイミング

| タイミング | 更新内容 |
|-----------|----------|
| ValidationResult 受信時 | `checkpoint.json` に `completed_blocks` を追加 |
| チャンク完了時 | `chunk_meta.json` の `status` を `completed` に更新 |
| Sheets 書き込み成功時 | `job.run_meta` の `processed_blocks` を加算 |
| チャンク失敗時 | `chunk_meta` に `last_error` と `retry_count` を記録 |

`checkpoint.json` 例:
```json
{
  "completed_blocks": ["12:0", "12:1", "13:0"],
  "updated_at": "2025-10-14T09:45:00Z"
}
```

---

## 6. キャッシュ仕様（概要）

- キャッシュキー: SHA256(utterance + categories JSON + system prompt hash + provider + model)
- 保存先: `runs/scoring/cache.json`
- エントリ:
```json
{
  "value": {
    "scores": [0.85, 0.1, 0.65],
    "provider": "gemini",
    "model": "gemini-flash-lite-latest",
    "timestamp": "2025-10-14T09:00:00Z"
  },
  "ts": 1697274000.0
}
```
- TTL: 24 時間（設計時点）  
- 最大件数: 10,000（超過時は古い順に削除）
- キャッシュヒット時は Validation ステージを通さず、直接 WriterQueue に結果を投げる。

---

## 7. 未決事項 / 次ステップ
- キャッシュの永続化形式を JSON から SQLite に切り替えるか要検討（並列アクセス時のロック管理を考慮）。
- ChunkScheduler と UI の連携（親ジョブ画面にチャンク進捗を表示）について仕様策定。
- Metrics 収集（処理時間、キャッシュヒット率、Sheets 書き込み遅延など）をどこでログ出力するか決定。

---

## 8. 実装順序の提案
1. ChunkScheduler とチャンクメタ構造を先に導入（同期処理のままでも可）。
2. Producer / Invoker / Validation / Writer を導入し、非同期パイプライン化。
3. キャッシュ処理を追加。
4. 最後に自動再実行と UI 表示を整備。

以上が初期設計。実装中に仕様変更があれば本ドキュメントを更新すること。
