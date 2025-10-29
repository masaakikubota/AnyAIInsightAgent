# AnyAI Frontend Reference Guide

This document bundles the reusable UI building blocks that appear across the AnyAI tooling pages. Components are grouped by the element they influence (layout shell, forms, queue system, etc.) so you can bring over the entire “combo” — markup, CSS, and JavaScript hooks — without breaking layouts.

Unless otherwise noted, core styles live in `/static/anyai/css/main.css` and the most complete examples can be seen in `app/static/index.html`. When applying a combo, make sure the relevant CSS is already in `main.css` or copied into the page-level `<style>` block, and that any JavaScript helpers are initialised.

---

## 1. Layout Shell

### 1.1 Tool Frame (`anyai-layout` / `anyai-sidebar` / `anyai-content`)
- **Purpose:** Consistent shell with fixed sidebar and scrollable content region.
- **Markup:**
  ```html
  <div class="anyai-layout">
    <aside class="anyai-sidebar" data-active-tool="scoring"></aside>
    <main class="anyai-content">
      <!-- page content -->
    </main>
  </div>
  ```
- **CSS:** root rules in `main.css` under the layout section (sticky sidebar widths, background colours).
- **References:** `app/static/index.html`, `app/static/cleansing.html`, `app/static/comment-enhancer.html`.

### 1.2 Two-Column Application Grid (`anyai-app`, `anyai-app-primary`, `anyai-app-support`)
- **Purpose:** Split primary workflow from supporting cards with responsive sticky sidebar.
- **Markup:**
  ```html
  <div class="anyai-app">
    <section class="anyai-app-primary" aria-labelledby="tool-heading">
      <!-- main form -->
    </section>
    <aside class="anyai-app-support">
      <!-- support cards (history, queue, log, etc.) -->
    </aside>
  </div>
  ```
- **CSS:** Grid definitions in `main.css` (`--anyai-app-primary-span`, sticky support column).
- **References:** `app/static/index.html`, `app/static/comment-enhancer.html`, `app/static/video-analysis.html` (after refactor).

---

## 2. Form + Action Combos

### 2.1 Form Container (`anyai-form`, `anyai-form-section`)
- **Purpose:** Standard card-like container with banded sections.
- **Markup:**
  ```html
  <form id="runner-form" class="anyai-form">
    <section class="anyai-form-section" aria-labelledby="section-heading">
      <header>
        <h2 id="section-heading" class="mt-0">入力設定</h2>
        <p class="field-hint">セクションの説明テキスト。</p>
      </header>
      <!-- section fields -->
    </section>
    <!-- additional sections -->
  </form>
  ```
- **CSS:** spacing, background, border-radius handled in `main.css` (see `.anyai-form`, `.anyai-form-section`).
- **References:** `app/static/index.html`, `app/static/tribe_interview.html`, `app/static/video-analysis.html`.

### 2.2 Field With Inline Action (`field-with-action`, `input-action-btn`)
- **Purpose:** Combine a text/url input with an adjacent button (e.g., “Set”).
- **Markup:**
  ```html
  <div class="field">
    <label for="spreadsheet-url">スプレッドシートURL / ID</label>
    <p class="field-hint">対象シートを指定し、Setで一覧を読み込みます。</p>
    <div class="field-with-action">
      <input class="input" type="url" id="spreadsheet-url" placeholder="https://docs.google.com/spreadsheets/d/..." required>
      <button type="button" class="btn btn-primary input-action-btn" id="btn-spreadsheet-set">Set</button>
    </div>
  </div>
  ```
- **CSS:** inline styles for flex alignment are provided per page (see `app/static/index.html` and copied to other pages as needed).
- **JS:** the button ID (`btn-spreadsheet-set`) is listened to by each tool’s script to fetch sheet metadata.
- **References:** `app/static/index.html:933`, `app/static/cleansing.html:379`, `app/static/video-analysis.html:325`.

### 2.3 Sheet Selector Grid (`sheet-selection-grid`, `field-info-inline`)
- **Purpose:** Layout for paired `<select>` elements with hints and inline tooltips.
- **Markup:**
  ```html
  <div class="sheet-selection-grid">
    <div class="field" id="analysis-sheet-field">
      <label for="sheet_keyword">分析テキスト シート</label>
      <select id="sheet_keyword" class="js-custom-select" disabled data-placeholder="Setで読み込んでください">
        <option value="">Setで読み込んでください</option>
      </select>
      <p class="field-hint">Setで取得したシート一覧から選択します。</p>
      <details class="field-info field-info-inline">
        <summary class="info-toggle"><span class="info-toggle-icon" aria-hidden="true">i</span><span>詳細</span></summary>
        <div class="info-panel">シート選択に関する追加説明。</div>
      </details>
    </div>
    <!-- additional fields -->
  </div>
  ```
- **CSS:** `main.css` includes grid behaviour, tooltip styling, and disabled states.
- **JS:** options are populated via the spreadsheet list handler (see index.js section around `populateSheetOptions`).
- **References:** `app/static/index.html:929`, `app/static/comment-enhancer.html:467`, `app/static/video-analysis.html:334`.

### 2.4 Custom Select Wrapper (`js-custom-select` enhancer)
- **Purpose:** Replace native `<select>` with accessible combobox pattern.
- **Implementation:** call `setupCustomSelect(selectElement)` for each `.js-custom-select`. The helper lives inline in pages like `index.html` (search for `function setupCustomSelect`).
- **Markup:** native `<select>` remains in place; the enhancer injects `.custom-select`, `.custom-select-toggle`, `.custom-select-menu`, and `.custom-select-option` elements.
- **CSS:** large block in `main.css` and mirrored in relevant pages (padding, arrow icon, focus states).
- **References:** initialiser in `app/static/index.html:673`, re-used in `app/static/comment-enhancer.html:680`, `app/static/video-analysis.html:460`.

### 2.5 Validation Banner (`validation-banner`)
- **Purpose:** Display contextual info/error/success messages below the sheet selectors.
- **Markup:** `<div id="validation-banner" class="validation-banner" role="status" aria-live="polite"></div>`
- **CSS:** `[data-state]` variants defined in `main.css` and page-level overrides (index.html, video-analysis).
- **JS:** helper `showValidationBanner(state, message)` toggles `data-state` (`error`, `info`, `success`).
- **References:** `app/static/index.html:602`, `app/static/comment-enhancer.html:501`, `app/static/video-analysis.html:352`.

### 2.6 Form Actions + Floating Proxy (`anyai-form-actions`, `anyai-action-bar`)
- **Purpose:** Keep main action buttons accessible at the bottom of the viewport.
- **Markup:**
  ```html
  <div class="anyai-form-actions" id="scoring-action-area">
    <button type="button" class="btn btn-outline" id="stop-button" disabled>停止</button>
    <button type="submit" class="btn btn-primary" id="run-button">実行</button>
    <button type="button" class="btn btn-soft-accent" id="queue-add-button">キューに追加</button>
  </div>

  <div class="anyai-action-bar" role="region" aria-label="スコアリング操作">
    <button type="button" class="btn btn-outline" data-proxy-for="stop-button">停止</button>
    <button type="button" class="btn btn-primary" data-proxy-for="run-button">実行</button>
    <button type="button" class="btn btn-soft-accent" data-proxy-for="queue-add-button">キューに追加</button>
  </div>
  ```
- **CSS:** `.anyai-form-actions` and `.anyai-action-bar` styles in `main.css`; additional positioning tweaks copied into `app/static/index.html` & `video-analysis.html`.
- **JS:** helper `syncActionProxies` mirrors disabled/busy states and triggers the underlying button’s `click` (see index.html’s `initActionProxies`).
- **References:** `app/static/index.html:1160-1217`, `app/static/video-analysis.html:372-397`.

### 2.7 Miscellaneous Form Utilities
- **Info Toggle:** `.info-toggle` and `.info-panel` styles are in `main.css`; used alongside `<details class="field-info field-info-inline">`.
- **Grid Utilities:** `.grid-2`, `.grid-3` patterns appear in `main.css` and handle responsive column layouts.
- **Checkbox Rows:** `.checkbox-grid.two-col` (see `app/static/dashboard.html:210`).

---

## 3. Job Queue System

### 3.1 Queue Support Metrics Card (`anyai-support-card support-metrics`)
- **Purpose:** Summarise queue state (counts, start button) outside of the standard panel.
- **Markup:**
  ```html
  <div class="anyai-support-card support-metrics">
    <div class="support-header">
      <h2 class="mt-0">ジョブキュー</h2>
      <button id="btn-queue-start" class="btn btn-primary" type="button">キュー開始</button>
    </div>
    <div id="queue-status" class="queue-status"></div>
    <div id="queue-list" class="stack-sm"></div>
  </div>
  ```
- **References:** `app/static/index.html`, `app/static/video-analysis.html`, `app/static/comment-enhancer.html`.
- **Notes:** The list container stays empty in markup; rows are injected by `refreshQueue()` in each page’s script.

### 3.2 Queue Row Pattern (`queue-row`, `queue-actions`)
- **Purpose:** Inline row layout for queue items with reorder / edit / delete controls.
- **Markup (JS-generated example):**
  ```html
  <div class="row queue-row" data-job-id="job_123">
    <div class="queue-row-info">
      <strong>キャンペーンコメント改善</strong>
      <div class="queue-row-meta">
        <span><code>job_123</code></span>
        <span>status: queued</span>
        <span>comment-enhancer</span>
      </div>
    </div>
    <div class="queue-actions">
      <button class="btn btn-ghost" type="button">↑</button>
      <button class="btn btn-ghost" type="button">↓</button>
      <button class="btn btn-soft-accent queue-edit-button" type="button">✎</button>
      <button class="btn btn-primary queue-save-button" type="button">編集を保存</button>
      <button class="btn btn-danger queue-delete-button" type="button">✕</button>
    </div>
  </div>
  ```
- **CSS:** Lightweight helpers (`.queue-row`, `.queue-row-info`, `.queue-row-meta`, `.queue-actions`, `.queue-save-button`, `.queue-delete-button`) live in the page-level `<style>` blocks.
- **JS:** Rows are built inside each page’s `refreshQueue()` (`index.html`, `video-analysis.html`, `comment-enhancer.html`). The scripts call `/queue`, render status chips, wire the move/edit/delete actions, and keep `form.dataset.editing` in sync.

### 3.3 Queue Toast (`queue-toast`)
- **Purpose:** Inline notification when queue operations occur.
- **Markup:** `<div id="queue-toast" class="queue-toast" role="status" aria-live="polite" aria-hidden="true"></div>` (see `index.html`, `video-analysis.html`, `comment-enhancer.html`).
- **JS:** Utility `showQueueToast(message, { state })` toggles visibility and auto-hides via `setTimeout`.

---

## 4. Support Cards

### 4.1 Recent Sheet History
- **Markup:**
  ```html
  <section class="anyai-support-card">
    <div class="support-header">
      <h2 class="mt-0">最近のシート</h2>
    </div>
    <p class="field-hint" id="history-empty">実行履歴はまだありません。</p>
    <ul id="history-list" class="column gap-2"></ul>
  </section>
  ```
- **CSS:** `.support-header` (flex alignment), list styling either in `main.css` or page-level styles (`#history-list li` hover state).
- **JS:** functions `loadHistory`, `renderHistory`, `addToHistory` (see `app/static/index.html` and reused in video-analysis).

### 4.2 Execution Log Card
- **Markup:**
  ```html
  <section class="anyai-support-card">
    <div class="support-header">
      <h2 class="mt-0">実行ログ</h2>
    </div>
    <div id="log-container"><span class="text-subtle">Awaiting configuration...</span></div>
  </section>
  ```
- **CSS:** `#log-container` styles per page (monospace font, scrollable area). For consistent layout reuse `main.css` definitions or copy from index.
- **JS:** append log messages via `appendLog(message, isError)`; used in most tool scripts.

---

## 5. Buttons & States

- **btn btn-primary / btn btn-outline / btn btn-soft-accent / btn btn-secondary btn-strong-accent / btn btn-danger / btn btn-subtle:** colour ramps and box-shadows defined in `main.css` under the button section. Use existing class combinations to avoid discrepancies.
- **Loading State:** `setButtonBusy(button, label)` helper (see index.html) temporarily sets `aria-busy`, disables the button, and locks width.
- **Proxy Buttons:** Any button in the floating `anyai-action-bar` requires `data-proxy-for="target-button-id"`. JS will manage disabled/busy states automatically.

---

## 6. Additional Combos

- **API Key Accordion:** `<details class="section-toggle">` with a nested `.stack` of fields and an outline submit (`app/static/index.html:1037`, `.section-toggle` styles in `main.css`).
- **Toggle Pills (`mode-pill`, `toggle-switch`):** used in scoring mode selector (`index.html`). Ensure the inline CSS block for `.mode-pill`/`.toggle-switch` is copied.
- **Status Badge (`status-badge`, `data-state`):** appears in validation/support cards (see `index.html:1176`).

---

## 7. Usage Checklist

1. **Identify the component** you need (e.g., full job queue card).
2. **Copy both markup and related CSS/JS helpers** from the cited file if they are not already in `main.css` or the shared script.
3. **Wire up IDs and data attributes** (`data-proxy-for`, `data-queue-*`, select IDs). Consistency is essential for the existing scripts to hook correctly.
4. **Call initialisers** after injecting markup:
   - `setupCustomSelect(...)` for each `.js-custom-select`.
   - Queue manager: `window.AnyAIQueueManager.init({ root, log })`.
   - Action proxies: `initActionProxies()` / `syncActionProxies()`.
5. **Test the layout** in both desktop and mobile breakpoints to ensure sticky support cards and action bars behave as expected.

Keeping components packaged with their full markup/CSS/JS context prevents subtle layout regressions and makes new tooling pages feel immediately consistent with the existing AnyAI experience.
