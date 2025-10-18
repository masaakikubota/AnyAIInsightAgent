# Credentials Setup

This directory should contain the Google OAuth client configuration and any
runtime API keys required by the AnyAI Video Analysis dashboard. These files
must **not** be committed to source control.

Required files:

1. `client_secrets.json` – OAuth 2.0 credentials downloaded from Google Cloud.
2. `token.json` – Created automatically after the first OAuth consent flow.
3. `Keys.txt` – Optional key-value pairs (e.g., `GEMINI_API_KEY="..."`).

Sample placeholders are provided in this repository. Copy the corresponding
`*.example` files, populate them with real values, and ensure the resulting
files stay ignored via `.gitignore`.
