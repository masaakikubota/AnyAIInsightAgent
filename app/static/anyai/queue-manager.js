;(function (window) {
  function noop() {}

  const QUEUE_ENDPOINTS = {
    state: '/queue-state',
    reorder: '/queue-reorder',
    remove: '/queue-remove',
    update: '/queue-update',
  };

  function toElement(target) {
    if (!target) return null;
    return typeof target === 'string' ? document.querySelector(target) : target;
  }

  function normaliseItem(item) {
    if (!item || !item.process_id) return null;
    const status = (item.status || 'queued').toLowerCase();
    let order = Number.isFinite(item.order) ? item.order : null;
    if (order === null) {
      order = status === 'running' ? 0 : 1000;
    }
    return {
      process_id: item.process_id,
      status,
      sheet_title: item.sheet_title || '',
      sheet_url: item.sheet_url || '',
      params: item.params || {},
      order,
      enqueued_at: item.enqueued_at || item.savedAt || Date.now(),
      job_type: item.job_type || item.target || '',
    };
  }

  function sortItems(items) {
    return [...items].sort((a, b) => {
      const ao = a.status === 'running' ? -1 : a.order ?? 9999;
      const bo = b.status === 'running' ? -1 : b.order ?? 9999;
      if (ao !== bo) return ao - bo;
      return (a.enqueued_at || 0) - (b.enqueued_at || 0);
    });
  }

  function createQueueManager(options) {
    const root = toElement(options && options.root);
    if (!root) {
      return {
        refresh: noop,
        addJob: noop,
        handleQueueEvent: noop,
        handleCompletion: noop,
      };
    }

    const log = (options && options.log) || noop;
    const jobType = (options && options.jobType) || null;
    const listEl = root.querySelector('[data-queue-list]');
    const emptyEl = root.querySelector('[data-queue-empty]');
    const detailsEl = root.querySelector('[data-queue-details]');
    const titleEl = root.querySelector('[data-queue-title]');
    const urlEl = root.querySelector('[data-queue-url]');
    const statusEl = root.querySelector('[data-queue-status]');
    const orderEl = root.querySelector('[data-queue-order]');
    const jsonEl = root.querySelector('[data-queue-json]');
    const jobTypeEl = root.querySelector('[data-queue-job-type]');
    const refreshBtns = root.querySelectorAll('[data-queue-refresh]');
    const moveUpBtn = root.querySelector('[data-queue-move-up]');
    const moveDownBtn = root.querySelector('[data-queue-move-down]');
    const removeBtn = root.querySelector('[data-queue-remove]');
    const applyBtn = root.querySelector('[data-queue-apply]');
    const toggleBtn = root.querySelector('[data-queue-toggle]');
    const panelBody = root.querySelector('[data-queue-body]') || root;

    if (applyBtn) {
      applyBtn.title = '現在のジョブ編集は未対応です';
      applyBtn.disabled = true;
    }

    let items = [];
    let selectedId = null;

    function setItems(next) {
      const previous = items;
      items = next
        .map((raw) => {
          const fallbackType =
            raw?.job_type || previous.find((itm) => itm.process_id === raw?.process_id)?.job_type || jobType || raw?.mode;
          const normalised = normaliseItem({ ...raw, job_type: fallbackType });
          return normalised;
        })
        .filter(Boolean)
        .filter((item) => !jobType || !item.job_type || item.job_type === jobType);
      renderList();
      if (selectedId) {
        const match = items.find((i) => i.process_id === selectedId);
        if (match) {
          renderDetails(match);
        } else {
          selectedId = null;
          renderDetails(null);
        }
      }
    }

    function upsertItem(item) {
      const existing = item && items.find((q) => q.process_id === item.process_id);
      const fallbackType = item?.job_type || existing?.job_type || jobType || item?.mode;
      const normalised = normaliseItem({ ...item, job_type: fallbackType });
      if (!normalised) return;
      if (jobType && normalised.job_type && normalised.job_type !== jobType) return;
      const idx = items.findIndex((q) => q.process_id === normalised.process_id);
      if (idx === -1) {
        items.push(normalised);
      } else {
        items[idx] = { ...items[idx], ...normalised, params: normalised.params };
      }
      renderList();
      if (!selectedId || selectedId === normalised.process_id) {
        selectedId = normalised.process_id;
        renderDetails(normalised);
      }
    }

    function renderList() {
      if (!listEl || !emptyEl) return;
      const sorted = sortItems(items);
      listEl.innerHTML = '';
      if (!sorted.length) {
        emptyEl.style.display = 'block';
        listEl.style.display = 'none';
        return;
      }
      emptyEl.style.display = 'none';
      listEl.style.display = 'flex';
      sorted.forEach((item) => {
        const li = document.createElement('li');
        li.className = 'queue-item';
        if (item.process_id === selectedId) {
          li.classList.add('is-active');
        }
        li.dataset.id = item.process_id;
        const statusClass = `queue-status-${item.status.replace(/[^a-z0-9]/g, '-')}`;
        const typeLabel = item.job_type ? `<span class="queue-item-type">${item.job_type}</span>` : '';
        li.innerHTML = `
          <div class="queue-item-main">
            <strong>${typeLabel ? `${typeLabel} ` : ''}${item.sheet_title || '(タイトル未取得)'}</strong>
            <span class="queue-status ${statusClass}">${item.status}</span>
          </div>
          <div class="queue-item-sub">
            <span>${item.sheet_url || ''}</span>
            <span>#${item.order != null ? item.order : '-'}</span>
          </div>
        `;
        listEl.appendChild(li);
      });
    }

    function renderDetails(item) {
      if (!detailsEl) return;
      if (!item) {
        detailsEl.hidden = true;
        return;
      }
      detailsEl.hidden = false;
      if (titleEl) titleEl.textContent = item.sheet_title || '(タイトル未取得)';
      if (urlEl) {
        urlEl.textContent = item.sheet_url || '(URL不明)';
        urlEl.href = item.sheet_url || '#';
        urlEl.rel = 'noreferrer noopener';
      }
      if (statusEl) {
        const statusClass = `queue-status queue-status-${item.status.replace(/[^a-z0-9]/g, '-')}`;
        statusEl.className = statusClass;
        statusEl.textContent = item.status;
      }
      if (orderEl) orderEl.textContent = item.order === 0 ? '実行中' : (item.order || '-');
      if (jsonEl) jsonEl.value = JSON.stringify(item.params || {}, null, 2);
      if (jobTypeEl) jobTypeEl.textContent = item.job_type || '-';
      const isRunning = item.status === 'running';
      if (moveUpBtn) moveUpBtn.disabled = isRunning;
      if (moveDownBtn) moveDownBtn.disabled = isRunning;
      if (removeBtn) removeBtn.disabled = isRunning;
      if (applyBtn) applyBtn.disabled = true;
    }

    function selectById(processId) {
      selectedId = processId;
      const selectedItem = items.find((item) => item.process_id === processId) || null;
      if (listEl) {
        listEl.querySelectorAll('.queue-item').forEach((li) => {
          li.classList.toggle('is-active', li.dataset.id === processId);
        });
      }
      renderDetails(selectedItem);
    }

    function refresh() {
      return fetch('/queue-state')
        .then((res) => {
          if (!res.ok) throw new Error(res.statusText || 'Failed to load queue state');
          return res.json();
        })
        .then((data) => {
          const items = Array.isArray(data) ? data : [];
          setItems(items);
        })
        .catch((error) => {
          log(`-> キュー状態の取得に失敗: ${error}`);
        });
    }

    function handleQueueEvent(payload) {
      if (!payload || !payload.process_id) return;
      if (jobType && payload.job_type && payload.job_type !== jobType) return;
      upsertItem(payload);
    }

    function addJob(job) {
      if (!job) return;
      handleQueueEvent({
        process_id: job.process_id,
        status: job.status || 'queued',
        sheet_title: job.sheet_title,
        sheet_url: job.sheet_url,
        params: job.params,
        order: job.order,
        job_type: job.job_type || jobType || job.mode,
      });
      selectById(job.process_id);
    }

    function handleCompletion(processId) {
      if (!processId) return;
      const existing = items.find((item) => item.process_id === processId);
      handleQueueEvent({
        process_id: processId,
        status: 'completed',
        job_type: existing?.job_type || jobType || existing?.mode,
      });
    }

    function move(direction) {
      if (!selectedId) return;
      const ordered = sortItems(items);
      const currentIdx = ordered.findIndex((item) => item.process_id === selectedId);
      if (currentIdx === -1) return;
      const delta = direction === 'up' ? -1 : 1;
      let newIndex = currentIdx + delta;
      if (newIndex < 0 || newIndex >= ordered.length) return;
      fetch(QUEUE_ENDPOINTS.reorder, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ process_id: selectedId, direction }),
      })
        .then((res) => {
          if (!res.ok) throw new Error(res.statusText || 'Failed to reorder');
          return res.json();
        })
        .then((payload) => {
          if (payload && Array.isArray(payload.queue)) {
            setItems(payload.queue);
            selectById(selectedId);
          } else {
            refresh().then(() => selectById(selectedId));
          }
        })
        .catch((error) => log(`-> 並び替えに失敗: ${error}`));
    }

    function removeSelected() {
      if (!selectedId) return;
      fetch(QUEUE_ENDPOINTS.remove, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ process_id: selectedId }),
      })
        .then((res) => {
          if (!res.ok) throw new Error(res.statusText || 'Failed to remove queue item');
          return res.json();
        })
        .then((payload) => {
          selectedId = null;
          renderDetails(null);
          if (payload && Array.isArray(payload.queue)) {
            setItems(payload.queue);
          } else {
            refresh();
          }
        })
        .catch((error) => log(`-> キューの削除に失敗: ${error}`));
    }

    function applyUpdates() {
      log('-> キュー更新は現在サポートされていません。', true);
    }

    function togglePanel() {
      if (!panelBody) return;
      panelBody.classList.toggle('is-collapsed');
      if (toggleBtn) {
        const collapsed = panelBody.classList.contains('is-collapsed');
        toggleBtn.textContent = collapsed ? '開く' : '折りたたむ';
      }
    }

    if (listEl) {
      listEl.addEventListener('click', (event) => {
        const target = event.target.closest('.queue-item');
        if (!target) return;
        selectById(target.dataset.id);
      });
    }

    refreshBtns.forEach((btn) => btn.addEventListener('click', refresh));
    if (moveUpBtn) moveUpBtn.addEventListener('click', () => move('up'));
    if (moveDownBtn) moveDownBtn.addEventListener('click', () => move('down'));
    if (removeBtn) removeBtn.addEventListener('click', removeSelected);
    if (applyBtn) applyBtn.addEventListener('click', applyUpdates);
    if (toggleBtn) toggleBtn.addEventListener('click', togglePanel);

    // Initial state
    setItems([]);
    refresh();

    return {
      refresh,
      addJob,
      handleQueueEvent,
      handleCompletion,
    };
  }

  window.AnyAIQueueManager = { init: createQueueManager };
})(window);
