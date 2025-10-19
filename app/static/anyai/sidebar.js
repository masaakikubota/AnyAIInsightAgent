(function () {
  const sidebar = document.querySelector('.anyai-sidebar');
  if (!sidebar) return;

  const NAV_GROUPS = [
    {
      id: 'core',
      label: 'CORE',
      items: [
        { id: 'scoring', href: '/', text: 'Score', tooltip: 'AnyAI Scoring', icon: 'sparkles' },
        { id: 'cleansing', href: '/cleansing', text: 'Cleanse', tooltip: 'AnyAI Cleansing', icon: 'wand-2' },
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
    <span class="brand-icon" aria-hidden="true"><img src="/static/anyai/assets/AnyAI_logo.png" alt=""></span>
    <span class="sr-only">AnyAI ホーム</span>
  `;
  header.append(brand);

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

  root.append(header, nav);
  sidebar.appendChild(root);

  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
})();
