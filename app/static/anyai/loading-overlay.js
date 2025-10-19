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
      logo.src = '/static/anyai/assets/AnyAI_logo.png';
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
