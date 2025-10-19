(function () {
  const sidebar = document.querySelector('.anyai-sidebar');
  if (!sidebar) return;

  const NAV_GROUPS = [
    {
      id: 'core',
      label: 'CORE',
      items: [
        { id: 'scoring', href: '/', text: 'Score', tooltip: 'AnyAI Scoring', icon: 'sparkles' },
        { id: 'cleansing', href: '/cleansing', text: 'Cleanse', tooltip: 'AnyAI Cleansing', icon: 'broom' },
        { id: 'interview', href: '/interview', text: 'Interview', tooltip: 'AnyAI Interview', icon: 'mic' }
      ]
    },
    {
      id: 'persona',
      label: 'PERSONA',
      items: [
        { id: 'persona', href: '/persona', text: 'Seeds', tooltip: 'Persona Seeds', icon: 'sprout' },
        { id: 'persona-dashboard', href: '/dashboard', text: 'Dashboard', tooltip: 'Persona Dashboard', icon: 'layout-dashboard' }
      ]
    },
    {
      id: 'video-suite',
      label: 'VIDEO SUITE',
      items: [
        { id: 'video-analysis', href: '/video-analysis', text: 'Analysis', tooltip: 'Video Analysis', icon: 'clapperboard' },
        { id: 'comment-enhancer', href: '/comment-enhancer', text: 'Enhancer', tooltip: 'Comment Enhancer', icon: 'message-circle' },
        { id: 'video-comment-review', href: '/video-comment-review', text: 'Review', tooltip: 'Video Comment Review', icon: 'file-text' },
        { id: 'kol-reviewer', href: '/kol-reviewer', text: 'KOL', tooltip: 'KOL Reviewer', icon: 'star' }
      ]
    }
  ];

  const FOOTER_ITEMS = [
    { id: 'settings', href: '/settings', text: 'Settings', tooltip: 'Settings', icon: 'settings' },
    { id: 'help', href: '/help', text: 'Help', tooltip: 'Help & Support', icon: 'help-circle' },
    { id: 'account', href: '/account', text: 'Account', tooltip: 'Account', icon: 'user' }
  ];

  const active = sidebar.dataset.activeTool || '';
  const root = document.createElement('div');
  root.className = 'anyai-sidebar-inner';

  const header = document.createElement('div');
  header.className = 'sidebar-top';

  const brand = document.createElement('a');
  brand.className = 'sidebar-brand';
  brand.href = '/';
  brand.setAttribute('aria-label', 'AnyAI ホーム');
  brand.innerHTML = `
    <span class="brand-icon"><img src="/static/anyai/assets/AnyAI_logo.png" alt="AnyAI ロゴ"></span>
    <span class="brand-text">AnyAI</span>
  `;

  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'sidebar-toggle';
  toggle.setAttribute('aria-label', 'サイドバーを折りたたむ');
  toggle.innerHTML = `
    <span class="sr-only">Toggle sidebar</span>
    <i data-lucide="chevrons-left" class="toggle-icon" aria-hidden="true"></i>
  `;

  header.append(brand, toggle);

  const nav = document.createElement('nav');
  nav.className = 'sidebar-nav';
  nav.setAttribute('aria-label', 'AnyAI ツール一覧');

  NAV_GROUPS.forEach((group) => {
    const groupEl = document.createElement('section');
    groupEl.className = 'nav-group';
    groupEl.dataset.groupId = group.id;

    const label = document.createElement('h2');
    label.className = 'nav-group-label';
    label.textContent = group.label;
    label.id = `sidebar-group-${group.id}`;

    const list = document.createElement('ul');
    list.className = 'nav-list';
    list.setAttribute('role', 'list');
    list.setAttribute('aria-labelledby', label.id);

    group.items.forEach((item) => {
      const li = document.createElement('li');
      const link = document.createElement('a');
      link.className = 'nav-item';
      link.href = item.href;
      link.dataset.tool = item.id;
      link.dataset.tooltip = item.tooltip;
      link.title = item.tooltip;
      link.innerHTML = `
        <span class="nav-icon" aria-hidden="true"><i data-lucide="${item.icon}"></i></span>
        <span class="nav-text">${item.text}</span>
      `;

      if (item.id === active) {
        link.classList.add('is-active');
        link.setAttribute('aria-current', 'page');
      }

      li.appendChild(link);
      list.appendChild(li);
    });

    groupEl.append(label, list);
    nav.appendChild(groupEl);
  });

  const footer = document.createElement('div');
  footer.className = 'sidebar-footer';

  FOOTER_ITEMS.forEach((item) => {
    const link = document.createElement('a');
    link.className = 'nav-item footer-item';
    link.href = item.href;
    link.dataset.tool = item.id;
    link.dataset.tooltip = item.tooltip;
    link.title = item.tooltip;
    link.innerHTML = `
      <span class="nav-icon" aria-hidden="true"><i data-lucide="${item.icon}"></i></span>
      <span class="nav-text">${item.text}</span>
    `;
    footer.appendChild(link);
  });

  root.append(header, nav, footer);
  sidebar.appendChild(root);

  const STORAGE_KEY = 'anyai.sidebar.collapsed';

  const getStoredCollapsed = () => {
    try {
      return window.localStorage.getItem(STORAGE_KEY) === 'true';
    } catch (error) {
      return false;
    }
  };

  const applyCollapsed = (collapsed, { store = true } = {}) => {
    sidebar.setAttribute('data-collapsed', collapsed ? 'true' : 'false');
    document.body.classList.toggle('sidebar-collapsed', collapsed);
    toggle.setAttribute('aria-pressed', collapsed ? 'true' : 'false');
    toggle.setAttribute('aria-label', collapsed ? 'サイドバーを展開する' : 'サイドバーを折りたたむ');
    if (store) {
      try {
        window.localStorage.setItem(STORAGE_KEY, collapsed ? 'true' : 'false');
      } catch (error) {
        /* noop */
      }
    }
  };

  const initialCollapsed = getStoredCollapsed();
  applyCollapsed(initialCollapsed, { store: false });

  toggle.addEventListener('click', () => {
    const next = !document.body.classList.contains('sidebar-collapsed');
    applyCollapsed(next);
  });

  const mediaQuery = window.matchMedia('(max-width: 960px)');
  const handleMedia = () => {
    if (mediaQuery.matches) {
      applyCollapsed(false, { store: false });
    } else {
      applyCollapsed(getStoredCollapsed(), { store: false });
    }
  };

  if (typeof mediaQuery.addEventListener === 'function') {
    mediaQuery.addEventListener('change', handleMedia);
  } else if (typeof mediaQuery.addListener === 'function') {
    mediaQuery.addListener(handleMedia);
  }
  handleMedia();

  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons({ attrs: { width: 20, height: 20, 'aria-hidden': 'true' } });
  }
})();
