;(function (window, document) {
  if (window.AnyAILoading) {
    return;
  }

  const state = {
    overlay: null,
    container: null,
    logo: null,
    text: null,
    dots: null,
    counter: 0,
    interval: null,
    dotCount: 0,
    completionTimer: null,
  };

  const keyState = {
    ready: null,
    status: null,
    promise: null,
    resolve: null,
    overlayTemp: false,
    modal: null,
    messageEl: null,
    errorEl: null,
    form: null,
    geminiInput: null,
    openaiInput: null,
    persistInput: null,
    submitBtn: null,
    refreshBtn: null,
  };

  function ensureKeyStyles() {
    if (document.getElementById('anyai-key-modal-style')) {
      return;
    }
    const style = document.createElement('style');
    style.id = 'anyai-key-modal-style';
    style.textContent = `
.anyai-loading-overlay.has-key-prompt {
  backdrop-filter: blur(4px);
}
.anyai-key-modal {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: min(90vw, 360px);
  background: rgba(255, 255, 255, 0.98);
  color: #1f2937;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 24px 48px rgba(15, 23, 42, 0.25);
  display: none;
  z-index: 2;
}
.anyai-loading-overlay.has-key-prompt .anyai-key-modal {
  display: block;
}
.anyai-key-modal h2 {
  margin: 0 0 0.75rem;
  font-size: 1.1rem;
  font-weight: 700;
  color: #111827;
}
.anyai-key-modal p {
  margin: 0 0 1rem;
  font-size: 0.9rem;
  line-height: 1.5;
}
.anyai-key-modal label {
  display: block;
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 0.3rem;
  color: #374151;
}
.anyai-key-modal input[type="password"],
.anyai-key-modal input[type="text"] {
  width: 100%;
  padding: 0.55rem 0.65rem;
  border-radius: 6px;
  border: 1px solid #d1d5db;
  background: #f9fafb;
  font-size: 0.85rem;
  margin-bottom: 0.85rem;
}
.anyai-key-modal input[type="password"]:focus,
.anyai-key-modal input[type="text"]:focus {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
}
.anyai-key-modal .anyai-key-modal__error {
  color: #b91c1c;
  background: rgba(248, 113, 113, 0.12);
  border-radius: 6px;
  padding: 0.5rem 0.65rem;
  font-size: 0.78rem;
  margin-bottom: 0.9rem;
  display: none;
}
.anyai-key-modal .anyai-key-modal__error.is-visible {
  display: block;
}
.anyai-key-modal .anyai-key-modal__actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
  align-items: center;
}
.anyai-key-modal button {
  border: none;
  border-radius: 6px;
  font-size: 0.85rem;
  cursor: pointer;
  padding: 0.55rem 0.9rem;
}
.anyai-key-modal button[type="submit"] {
  background: #4f46e5;
  color: #fff;
}
.anyai-key-modal button[type="submit"][disabled] {
  opacity: 0.6;
  cursor: progress;
}
.anyai-key-modal__refresh {
  background: #e5e7eb;
  color: #111827;
}
.anyai-key-modal__persist {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.78rem;
  margin: -0.4rem 0 0.9rem;
  color: #4b5563;
}
.anyai-key-modal__persist input {
  width: auto;
  margin: 0;
}
`;
    document.head.appendChild(style);
  }

  function ensureOverlay() {
    if (state.overlay) return state.overlay;

    const overlay = document.createElement('div');
    overlay.className = 'anyai-loading-overlay';
    overlay.setAttribute('role', 'status');
    overlay.setAttribute('aria-live', 'polite');
    overlay.setAttribute('aria-hidden', 'true');

    const container = document.createElement('div');
    container.className = 'anyai-loading-overlay__container';

    const logo = document.createElement('img');
    logo.className = 'anyai-loading-overlay__logo';
    const customLogoPath = window.ANYAI_LOADING_LOGO_PATH;
    logo.src = typeof customLogoPath === 'string' && customLogoPath.trim()
      ? customLogoPath
      : '/static/anyai/assets/AnyAI_Logo_For_Loading.png';
    logo.alt = 'AnyAI Loading';
    logo.addEventListener('error', () => {
      if (logo.dataset.fallbackApplied === 'true') return;
      logo.dataset.fallbackApplied = 'true';
      logo.src = '/static/anyai/assets/AnyAI_Logo_For_Loading.png';
    });

    const text = document.createElement('div');
    text.className = 'anyai-loading-overlay__text';

    const label = document.createElement('span');
    label.className = 'anyai-loading-overlay__label';
    label.textContent = 'Now Loading';

    const dots = document.createElement('span');
    dots.className = 'anyai-loading-overlay__dots';
    dots.textContent = '';

    text.appendChild(label);
    text.appendChild(dots);
    container.appendChild(logo);
    container.appendChild(text);
    overlay.appendChild(container);

    function mount() {
      if (document.body && !document.body.contains(overlay)) {
        document.body.appendChild(overlay);
      }
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', mount, { once: true });
    } else {
      mount();
    }

    state.overlay = overlay;
    state.container = container;
    state.logo = logo;
    state.text = text;
    state.dots = dots;
    return overlay;
  }

  function updateDots(nextCount) {
    state.dotCount = nextCount;
    if (!state.dots) return;
    const visibleDots = '.'.repeat(state.dotCount);
    state.dots.textContent = visibleDots;
    if (state.text) {
      state.text.classList.remove('is-pulse');
      // force reflow to restart animation
      void state.text.offsetWidth; // eslint-disable-line no-void
      state.text.classList.add('is-pulse');
    }
  }

  function startCycle() {
    stopCycle();
    updateDots(0);
    state.interval = window.setInterval(() => {
      const next = (state.dotCount % 3) + 1;
      updateDots(next);
    }, 1000);
  }

  function stopCycle() {
    if (state.interval) {
      window.clearInterval(state.interval);
      state.interval = null;
    }
    updateDots(0);
  }

  function clearCompletionTimer() {
    if (state.completionTimer) {
      window.clearTimeout(state.completionTimer);
      state.completionTimer = null;
    }
  }

  function show() {
    const overlay = ensureOverlay();
    state.counter += 1;
    clearCompletionTimer();
    overlay.classList.remove('is-completing');
    overlay.setAttribute('aria-hidden', 'false');
    overlay.classList.add('is-visible');
    startCycle();
  }

  function hide(force) {
    if (!state.overlay) return;
    if (force) {
      state.counter = 0;
      stopCycle();
      state.overlay.classList.remove('is-visible');
      state.overlay.setAttribute('aria-hidden', 'true');
      return;
    }

    state.counter = Math.max(0, state.counter - 1);
    if (state.counter === 0) {
      stopCycle();
      const overlay = state.overlay;
      overlay.classList.add('is-completing');
      const target = state.container || overlay;
      const handleAnimationEnd = () => {
        clearCompletionTimer();
        overlay.classList.remove('is-visible');
        overlay.classList.remove('is-completing');
        overlay.setAttribute('aria-hidden', 'true');
        target.removeEventListener('animationend', handleAnimationEnd);
      };
      // Fallback in case animationend does not trigger
      clearCompletionTimer();
      state.completionTimer = window.setTimeout(() => {
        overlay.classList.remove('is-visible');
        overlay.classList.remove('is-completing');
        overlay.setAttribute('aria-hidden', 'true');
        clearCompletionTimer();
      }, 800);
      target.addEventListener('animationend', handleAnimationEnd, { once: true });
    }
  }

  async function fetchKeyStatus(force) {
    if (!force && keyState.ready === true) {
      return keyState.status;
    }
    try {
      const resp = await fetch('/settings/status', { cache: 'no-store' });
      if (!resp.ok) {
        throw new Error(`status ${resp.status}`);
      }
      const data = await resp.json();
      const keys = data && data.keys ? data.keys : {};
      const gemini = keys.gemini || {};
      const openai = keys.openai || {};
      keyState.status = {
        gemini: { ready: Boolean(gemini.ready) },
        openai: { ready: Boolean(openai.ready) },
      };
      keyState.ready = keyState.status.gemini.ready && keyState.status.openai.ready;
      return keyState.status;
    } catch (error) {
      console.warn('[AnyAILoading] Failed to fetch key status', error);
      keyState.status = null;
      keyState.ready = null;
      return null;
    }
  }

  function ensureKeyModal() {
    if (keyState.modal) {
      return keyState.modal;
    }
    ensureOverlay();
    ensureKeyStyles();
    const modal = document.createElement('div');
    modal.className = 'anyai-key-modal';

    const title = document.createElement('h2');
    title.textContent = 'APIキーの入力が必要です';

    const message = document.createElement('p');
    message.className = 'anyai-key-modal__message';

    const form = document.createElement('form');
    form.autocomplete = 'off';

    const gemLabel = document.createElement('label');
    gemLabel.textContent = 'Gemini APIキー';
    const gemInput = document.createElement('input');
    gemInput.type = 'password';
    gemInput.name = 'gemini_api_key';
    gemInput.placeholder = '例: AIzaSy...';

    const openLabel = document.createElement('label');
    openLabel.textContent = 'OpenAI APIキー';
    const openInput = document.createElement('input');
    openInput.type = 'password';
    openInput.name = 'openai_api_key';
    openInput.placeholder = '例: sk-...';

    const persistWrapper = document.createElement('label');
    persistWrapper.className = 'anyai-key-modal__persist';
    const persistCheckbox = document.createElement('input');
    persistCheckbox.type = 'checkbox';
    persistCheckbox.checked = true;
    persistCheckbox.name = 'persist';
    persistWrapper.appendChild(persistCheckbox);
    persistWrapper.appendChild(document.createTextNode('.env に保存して次回も利用します'));

    const errorBox = document.createElement('div');
    errorBox.className = 'anyai-key-modal__error';

    const actions = document.createElement('div');
    actions.className = 'anyai-key-modal__actions';

    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.className = 'anyai-key-modal__refresh';
    refreshBtn.textContent = '再チェック';

    const submitBtn = document.createElement('button');
    submitBtn.type = 'submit';
    submitBtn.textContent = '保存して続行';

    actions.appendChild(refreshBtn);
    actions.appendChild(submitBtn);

    form.appendChild(gemLabel);
    form.appendChild(gemInput);
    form.appendChild(openLabel);
    form.appendChild(openInput);
    form.appendChild(persistWrapper);
    form.appendChild(errorBox);
    form.appendChild(actions);

    modal.appendChild(title);
    modal.appendChild(message);
    modal.appendChild(form);

    if (state.overlay) {
      state.overlay.appendChild(modal);
    }

    keyState.modal = modal;
    keyState.messageEl = message;
    keyState.errorEl = errorBox;
    keyState.form = form;
    keyState.geminiInput = gemInput;
    keyState.openaiInput = openInput;
    keyState.persistInput = persistCheckbox;
    keyState.submitBtn = submitBtn;
    keyState.refreshBtn = refreshBtn;

    refreshBtn.addEventListener('click', async (event) => {
      event.preventDefault();
      setModalError('');
      setModalBusy(true);
      const status = await fetchKeyStatus(true);
      setModalBusy(false);
      updateKeyModal(status);
      if (status && status.gemini.ready && status.openai.ready) {
        finalizeKeyModal(true);
      }
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      setModalError('');
      const gemValue = keyState.geminiInput.value.trim();
      const openValue = keyState.openaiInput.value.trim();
      const needGemini = !keyState.status || !keyState.status.gemini || !keyState.status.gemini.ready;
      const needOpenai = !keyState.status || !keyState.status.openai || !keyState.status.openai.ready;
      if (needGemini && !gemValue) {
        setModalError('Gemini APIキーを入力してください。');
        keyState.geminiInput.focus();
        return;
      }
      if (needOpenai && !openValue) {
        setModalError('OpenAI APIキーを入力してください。');
        keyState.openaiInput.focus();
        return;
      }
      const formData = new FormData();
      if (gemValue) {
        formData.append('gemini_api_key', gemValue);
      }
      if (openValue) {
        formData.append('openai_api_key', openValue);
      }
      if (keyState.persistInput.checked) {
        formData.append('persist', 'true');
      }
      try {
        setModalBusy(true);
        const resp = await fetch('/settings', {
          method: 'POST',
          body: formData,
        });
        if (!resp.ok) {
          let detail = 'APIキーの保存に失敗しました。';
          try {
            const errJson = await resp.json();
            if (errJson && errJson.detail) {
              detail = Array.isArray(errJson.detail)
                ? errJson.detail.map((d) => d.msg || '').join(' ') || detail
                : String(errJson.detail);
            }
          } catch (err) {
            console.warn('[AnyAILoading] Failed to parse settings error', err);
          }
          throw new Error(detail);
        }
        keyState.geminiInput.value = '';
        keyState.openaiInput.value = '';
        const status = await fetchKeyStatus(true);
        updateKeyModal(status);
        if (status && status.gemini.ready && status.openai.ready) {
          finalizeKeyModal(true);
        } else {
          setModalError('一部のキーがまだ設定されていません。確認してください。');
        }
      } catch (err) {
        console.error('[AnyAILoading] Failed to save keys', err);
        setModalError(err.message || 'APIキーの保存に失敗しました。');
      } finally {
        setModalBusy(false);
      }
    });

    return modal;
  }

  function setModalError(message) {
    if (!keyState.errorEl) return;
    if (message) {
      keyState.errorEl.textContent = message;
      keyState.errorEl.classList.add('is-visible');
    } else {
      keyState.errorEl.textContent = '';
      keyState.errorEl.classList.remove('is-visible');
    }
  }

  function setModalBusy(isBusy) {
    if (keyState.submitBtn) {
      keyState.submitBtn.disabled = Boolean(isBusy);
    }
    if (keyState.refreshBtn) {
      keyState.refreshBtn.disabled = Boolean(isBusy);
    }
    if (keyState.geminiInput) {
      keyState.geminiInput.disabled = Boolean(isBusy);
    }
    if (keyState.openaiInput) {
      keyState.openaiInput.disabled = Boolean(isBusy);
    }
    if (keyState.persistInput) {
      keyState.persistInput.disabled = Boolean(isBusy);
    }
  }

  function updateKeyModal(status) {
    ensureKeyModal();
    keyState.status = status;
    let missingMessage = 'Gemini と OpenAI の APIキーを入力してください。';
    if (status) {
      const missing = [];
      if (!status.gemini || !status.gemini.ready) {
        missing.push('Gemini APIキー');
      }
      if (!status.openai || !status.openai.ready) {
        missing.push('OpenAI APIキー');
      }
      if (missing.length === 0) {
        missingMessage = '両方のキーが設定されました。処理を再開します。';
      } else if (missing.length === 1) {
        missingMessage = `${missing[0]} が未設定です。入力してください。`;
      } else {
        missingMessage = 'Gemini と OpenAI の APIキーが未設定です。入力してください。';
      }
    } else {
      missingMessage = 'APIキーの状態を取得できませんでした。キーを入力して保存してください。';
    }
    if (keyState.messageEl) {
      keyState.messageEl.textContent = missingMessage;
    }
  }

  function finalizeKeyModal(success) {
    if (success) {
      keyState.ready = true;
    }
    if (state.overlay) {
      state.overlay.classList.remove('has-key-prompt');
    }
    if (keyState.modal) {
      keyState.modal.classList.remove('is-open');
    }
    if (keyState.geminiInput) {
      keyState.geminiInput.disabled = false;
      keyState.geminiInput.value = '';
    }
    if (keyState.openaiInput) {
      keyState.openaiInput.disabled = false;
      keyState.openaiInput.value = '';
    }
    if (keyState.persistInput) {
      keyState.persistInput.disabled = false;
      keyState.persistInput.checked = true;
    }
    if (keyState.submitBtn) {
      keyState.submitBtn.disabled = false;
    }
    if (keyState.refreshBtn) {
      keyState.refreshBtn.disabled = false;
    }
    setModalError('');
    const resolver = keyState.resolve;
    keyState.resolve = null;
    if (keyState.overlayTemp) {
      hide(true);
    }
    keyState.overlayTemp = false;
    if (typeof resolver === 'function') {
      resolver(success !== false);
    }
  }

  async function promptForKeys(status) {
    const overlayWasVisible = state.counter > 0;
    if (!overlayWasVisible) {
      show();
      keyState.overlayTemp = true;
    } else {
      keyState.overlayTemp = false;
    }
    ensureKeyModal();
    updateKeyModal(status);
    if (state.overlay) {
      state.overlay.classList.add('has-key-prompt');
    }
    if (keyState.modal) {
      keyState.modal.classList.add('is-open');
    }
    window.setTimeout(() => {
      if (status && status.gemini && !status.gemini.ready && keyState.geminiInput) {
        keyState.geminiInput.focus();
      } else if (status && status.openai && !status.openai.ready && keyState.openaiInput) {
        keyState.openaiInput.focus();
      } else if (keyState.geminiInput) {
        keyState.geminiInput.focus();
      }
    }, 50);

    return new Promise((resolve) => {
      keyState.resolve = resolve;
    });
  }

  async function requireKeys(force) {
    if (!force && keyState.ready === true) {
      return true;
    }
    if (keyState.promise) {
      return keyState.promise;
    }
    keyState.promise = (async () => {
      const status = await fetchKeyStatus(force);
      if (status && status.gemini.ready && status.openai.ready) {
        keyState.ready = true;
        return true;
      }
      const result = await promptForKeys(status);
      keyState.promise = null;
      keyState.ready = Boolean(result);
      return keyState.ready;
    })();
    try {
      return await keyState.promise;
    } finally {
      keyState.promise = null;
    }
  }

  async function run(task) {
    await requireKeys(false);
    show();
    try {
      return await task();
    } finally {
      hide(false);
    }
  }

  async function wrap(promiseLike) {
    await requireKeys(false);
    show();
    try {
      return await promiseLike;
    } finally {
      hide(false);
    }
  }

  window.AnyAILoading = {
    show,
    hide,
    run,
    wrap,
    requireKeys,
    refreshKeys: () => fetchKeyStatus(true),
  };
})(window, document);
