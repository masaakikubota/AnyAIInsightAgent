# AnyAIMarketingAgentV2
- FastAPI ベースのスコアリングエージェント。アップロードされたCSVの発話×カテゴリをGemini→OpenAIの順で評価し、[-1,1]でスコアリングした結果CSVや監査ログを生成。
- `app/main.py` にAPIエントリポイント、`app/worker.py` が非同期ワーカーでCSVのブロック処理・リトライ制御、`app/services` 下にLLMクライアントやCSV入出力が配置。
- フロントエンドは `app/static/index.html` の静的ページでフォーム送信・進捗監視・ログ表示を提供。