;(function (window, document) {
  if (window.AnyAILoading) {
    return;
  }

  const state = {
    overlay: null,
    counter: 0,
  };

  function ensureOverlay() {
    if (state.overlay) return state.overlay;

    const overlay = document.createElement('div');
    overlay.className = 'anyai-loading-overlay';
    overlay.setAttribute('role', 'status');
    overlay.setAttribute('aria-live', 'polite');
    overlay.setAttribute('aria-hidden', 'true');

    const backdrop = document.createElement('div');
    backdrop.className = 'anyai-loading-overlay__backdrop';

    const panel = document.createElement('div');
    panel.className = 'anyai-loading-overlay__panel';

    const logo = document.createElement('img');
    logo.className = 'anyai-loading-overlay__logo';
    logo.src = '/static/anyai/assets/loading-logo.svg';
    logo.alt = 'Loading';

    const text = document.createElement('div');
    text.className = 'anyai-loading-overlay__text';
    const letters = Array.from('Loading...');
    letters.forEach((char, index) => {
      const span = document.createElement('span');
      span.textContent = char;
      span.style.setProperty('--anyai-wave-index', String(index));
      text.appendChild(span);
    });

    panel.appendChild(logo);
    panel.appendChild(text);
    overlay.appendChild(backdrop);
    overlay.appendChild(panel);

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
    return overlay;
  }

  function show() {
    const overlay = ensureOverlay();
    state.counter += 1;
    overlay.setAttribute('aria-hidden', 'false');
    overlay.classList.add('is-visible');
  }

  function hide(force) {
    if (!state.overlay) return;
    if (force) {
      state.counter = 0;
    } else {
      state.counter = Math.max(0, state.counter - 1);
    }
    if (state.counter === 0) {
      state.overlay.classList.remove('is-visible');
      state.overlay.setAttribute('aria-hidden', 'true');
    }
  }

  async function run(task) {
    show();
    try {
      return await task();
    } finally {
      hide(false);
    }
  }

  async function wrap(promiseLike) {
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
  };
})(window, document);
