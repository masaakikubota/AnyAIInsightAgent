# Dashboard 機能削除対象リスト（2025-10-24）

Seeds/Interview 統合に伴い、以下の Dashboard 関連資産を削除予定。

## API / マネージャ
- `app/routers/persona.py`
  - `/dashboard/*` ルート一式（generate, build, list runs, get run, query, artifact download）
- `app/persona_response_manager.py`
  - Dashboard ビルド・キャッシュ・クエリロジック全体
- `app/services/dashboard.py`
- `app/services/dashboard_data.py`
- `app/services/clients.py` の Dashboard 用関数（`call_openai_dashboard_plan` など）
- `scripts/test_dashboard_cache.py`

## モデル / スキーマ
- `app/models.py`
  - `DashboardRequest` / `DashboardResponse` など dashboard 系モデル
- `app/models.py` の `DashboardBuildRequest`, `DashboardRunSummary`, `DashboardRunDetail`, `DashboardFilters`, `DashboardQueryRequest` ほか関連型

## ドキュメント / プラン
- `plans_bu.md` の Dashboard 項目
- `docs/perf/dashboard_cache_baseline.md`
- `UI_plan.md` Dashboard セクション

## フロントエンド
- `app/static/dashboard.html`
- `app/static/anyai/` 内 Dashboard 関連スクリプトがあれば（現状は共通利用のため精査）
- サイドバー (`app/static/anyai/sidebar.js`) の Dashboard リンク

## 外部サブプロジェクト
- `external/AnyAI_video_analysis` は別目的のダッシュボードだが、今回の統合対象外。影響を受けないことを明記しておく。

---

### 備考
- Dashboard 連携を削除することで、`PersonaResponseJobManager` に依存している箇所はなくなる予定だが、Seeds/Interview 統合後の UI からリンクがある場合は合わせて削除。
- 既存ランの成果物 (`runs/persona_response/...`) は移行対象外。必要なら別途アーカイブ手順を案内。

