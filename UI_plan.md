# UI_plan.md — AnyAI UI 要件とデザイン方針

---

## 1. 目的と原則
- **目的**: Interview / Persona / Cleansing / Dashboard 各モジュールのUIを、開発・運用双方で使いやすく一貫した体験にする。
- **原則**:
  - 一貫性: AnyAI UI Master（tokens/components/utilities）に準拠。
  - 可観測性: 状態・進捗・失敗を即時に可視化。
  - パフォーマンス: 初期表示<1.5s、操作応答<100ms、表/グラフのインタラクション<200msを目標。
  - アクセシビリティ: キーボード操作・スクリーンリーダ対応、コントラスト比WCAG AA。
  - 国際化: ja/ en。日付・数値・小数点/桁区切りにロケール反映。

---

## 2. 情報アーキテクチャ
- ヘッダー: ブランド/ナビ（Interview, Persona, Cleansing, Dashboard）, 実行環境, 認証/設定。
- 左パネル（必要時）: フィルタ/フォーム/検索。
- メイン: カード型セクション（フォーム、進捗、テーブル、グラフ）。
- 右パネル（オプション）: ヘルプ/ドキュメント/ショートカット。
- フッター: バージョン、モデル情報、実行ID、サポートリンク。

---

## 3. 画面別要件

### 3.1 Interview (/interview)
- フォーム
  - プロジェクト名、カテゴリ、言語、刺激（URL/テキスト/シート）、ペルソナ数、SSR有効、メモ。
  - バリデーション（必須/形式、URL/数値/上限）。
- 実行
  - [作成]→ジョブ起動→ジョブID表示→進捗SSE/ポーリング(2s)。
  - ステージ: direction → personas → interviews → artifacts → done。
- 進捗カード
  - ステータス/ステージ、生成済みカウント、ETA、再試行回数、ログ断片。
- 成果物
  - artifacts テーブル（path/size/createdAt/ダウンロード）。
  - summary.json のプレビュー（JSON/整形）。

### 3.2 Persona (/persona)
- タブ: Seeds（発話取込） | Builder（詳細生成） | Catalog（一覧）
- Seeds
  - シート/CSV取込、プレビュー、クレンジングルール適用。
- Builder
  - 方向性YAML + 属性（age/income/region/attitude）→ プロンプト生成。
  - 重複検知（類似度しきい値）と置換/保持選択。
- Catalog
  - テーブル（persona_id、属性、作成モデル、品質フラグ、更新日）。
  - 検索・フィルタ・一括操作（再生成/削除/エクスポート）。

### 3.3 Cleansing (/cleansing)
- 入力: シート/CSV/テキスト、正規化ルール（trim, lower, NFKC 等）。
- 実行: ルール適用シミュレーション → 確定 → 書き戻し。
- 結果: 変更差分の行ごとハイライト、エラー行抽出、エクスポート。

### 3.4 Dashboard (/dashboard)
- フィルタパネル
  - region, age_band, income_band, occupation, attitude, model_version, run_id, product。
  - フィルタの保存/共有（URLクエリ化）。
- 指標カード
  - 平均PI、KS類似度、サンプル数、Top2Box/Bottom2Box。
- 可視化
  - 分布（1..5 pmf）/CDF、地域ヒートマップ、年齢トレンド、商品間スロープ、ワードクラウド。
- テーブル
  - SKU×セグメントの集計、並び替え/カラム選択/エクスポート。
- 自動サマリー
  - LLM生成。要点3–5件、バイアス注意・サンプル偏りの警告。

---

## 4. コンポーネント設計
- フォーム
  - Text, Number, Select, MultiSelect, Toggle, File/URL, DateTime, Tag。
  - 入力支援: プレースホルダ、例、説明、エラー文、制限表示。
- テーブル
  - 仮想スクロール、列リサイズ、固定ヘッダー、CSV/JSON出力。
- グラフ
  - Plotlyベース。ズーム/凡例/PNG保存、ライト/ダーク対応。
- カード/レイアウト
  - anyai.components.css のカード/グリッド/モーダルを使用。
- トースト/アラート
  - 成功/注意/警告/エラー（色とアイコン）。ユーザー操作に近接表示。

---

## 5. デザイン方針（AnyAI UI Master 準拠）
- トークン: anyai.tokens.css（色/余白/フォント/シャドウ）。
- カラー: 
  - Primary: #5B6CFF、Success: #22C55E、Warning: #F59E0B、Danger: #EF4444。
  - Neutral: パネル/境界は控えめ。背景は高コントラスト。
- タイポ: Noto Sans JP、見出し600、本文400、要素ラベル700。
- 角丸/影: カード半径18px、shadow-sm〜md。フラットすぎない立体感。
- アイコン: AnyAI_icon.png、用途別にシステムアイコン（SVG）。
- ダークモード: prefers-color-schemeで自動、切替トグル。

---

## 6. 体験設計（UX）
- 空状態（Empty）: 初回利用時のガイドカード、サンプルデータ投入ボタン。
- ローディング: スケルトン/プログレスバー、過度なスピナー禁止。
- エラー: 原因・対処・再試行。ジョブID/ログリンクを提示。
- 成功: 次アクション提示（「ダッシュボードで表示」等）。
- Undo/Redo: 破壊的操作に対して短時間のUndoトースト。
- ショートカット: /, g i, g p, g d 等の移動・検索。

---

## 7. 非機能要件
- パフォーマンス
  - 画像・大規模表は遅延ロード、IntersectionObserver活用。
  - グラフはオフスクリーンで計算、Idleで前計算キャッシュ。
- 安定性
  - APIリトライ（指数バックオフ）、部分成功の提示、フォールバックUI。
- セキュリティ
  - サービスアカウント鍵はブラウザに出さない。ダウンロードURLは署名付き。
- 可観測性
  - アクションログ（UI計測ID, run_id）、エラーレポート送出。

---

## 8. 実装スタックと構成
- 静的配信: `app/static/*.html` + AnyAI UI MasterのコアCSS/JS。
- グラフ: Plotly（または軽量代替）。
- ビルド不要のプレーンJS優先、将来Next.jsへ移行可能なDOM構造に。
- i18n: data-i18n属性 + 辞書JSON。
- 状態: URLクエリ/LocalStorage（ユーザー設定/フィルタの保存）。

---

## 9. コンポーネント/ページ単位の受け入れ基準（抜粋）
- フォーム送信
  - 必須項目未入力で送信不可。エラーが上部と各項目に表示される。
- 進捗表示
  - status/stageが2秒間隔で更新。完了時に自動停止・成功トースト。
- テーブル
  - 1万行でもスクロールが滑らか。列ソートは100ms以内。
- グラフ
  - フィルタ変更後200ms以内に再レンダリング。PNG保存ボタンが機能。
- ダークモード
  - OS切替に追随。コントラスト比 AA を満たす。

---

## 10. ワイヤーフレーム（テキスト）
```
Header [Logo] [Interview] [Persona] [Cleansing] [Dashboard] ... [User]

/interview
  [Form Card]  プロジェクト/カテゴリ/刺激/数/SSR/メモ [実行]
  [Progress Card]  Status ●●  Stage personas (32/200)  ETA 05:12  [Logs]
  [Artifacts Table] path | size | createdAt | [Download]

/persona
  [Tabs] Seeds | Builder | Catalog
  Seeds: [Sheet/CSV Import] [Preview Table] [Apply Rules]
  Builder: [YAML + attributes] [Generate] [Dedup]
  Catalog: [Table] [Search] [Bulk Actions]

/cleansing
  [Input Sources] [Rules] [Simulate] [Apply] [Export]

/dashboard
  [Filters Left] age/income/region/... [Save Filter]
  [KPI Cards] MeanPI | KS | N | Top2Box
  [Charts] Dist | Heatmap | Trend | Slope | WordCloud
  [Table] SKU x Segment [Export]
```

---

## 11. 今後の拡張
- モバイル最適化（最小レイアウト幅360px, タッチ最適化）。
- マルチラン比較（複数run_idを並列表示）。
- 共有リンクの短縮URL化、埋め込みカード生成（OGP）。
- コンポーネントのStorybook化、UIスナップショットテスト。

---

## 12. トラッキング指標（UXメトリクス）
- Time to First Interaction (TTFI)
- フィルタ適用〜グラフ更新のレイテンシ中央値
- エラー率（API/レンダリング）
- ダッシュボードの滞在時間・エクスポート利用率
- Job成功率・平均完了時間


