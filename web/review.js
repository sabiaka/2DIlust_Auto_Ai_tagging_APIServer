(() => {
  const body = document.querySelector('#reviewRows');
  const search = document.querySelector('#reviewSearch');
  const category = document.querySelector('#reviewCategory');
  const summary = document.querySelector('#reviewSummary');
  const status = document.querySelector('#reviewStatus');
  const stats = document.querySelector('#reviewStats');
  const previous = document.querySelector('#reviewPrevious');
  const next = document.querySelector('#reviewNext');
  const page = document.querySelector('#reviewPage');
  const filters = document.querySelectorAll('[data-review-state]');
  const labels = { changed:'修正', new:'追加', review:'要確認', resolved:'手動確認済み', preserved:'原表記を保持', unchanged:'変更なし' };
  const categories = { general:'一般', character:'キャラクター', copyright:'作品', style:'画風・作者', meta:'メタ情報', rating:'レーティング' };
  let state = 'changes', offset = 0, data = null, controller = null, timer = null;
  const text = (tag, value, className = '') => {
    const element = document.createElement(tag);
    element.textContent = value;
    element.className = className;
    return element;
  };
  function metadata() {
    if (!data) return;
    const counts = data.counts;
    stats.replaceChildren(...[
      ['修正・追加', (counts.changed || 0)+(counts.new || 0)], ['要確認', counts.review || 0], ['手動確認済み', counts.resolved || 0],
    ].map(([label, count]) => {
      const card = text('div', '', 'review-stat');
      card.append(text('strong', count.toLocaleString()), text('span', label));
      return card;
    }));
    summary.textContent = `${data.total.toLocaleString()}件中 ${data.rows.length ? offset+1 : 0}〜${offset+data.rows.length}件を表示`;
    page.textContent = `1ページ50件`;
    previous.disabled = offset===0;
    next.disabled = offset+data.rows.length>=data.total;
  }
  async function load(reset = false) {
    if (reset) offset = 0;
    controller?.abort();
    controller = new AbortController();
    previous.disabled = next.disabled = true;
    summary.textContent = '読み込み中…';
    try {
      const params = new URLSearchParams({ q:search.value.trim(), state, category:category.value, offset:String(offset), limit:'50' });
      const response = await fetch(`/translation-review?${params}`, { signal:controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = await response.json();
      if (!data.rows.length && offset>0) { offset = Math.max(0,offset-50); return load(); }
      render();
      metadata();
    } catch (error) {
      if (error.name==='AbortError') return;
      status.textContent = `読み込みに失敗しました: ${error.message}`;
      summary.textContent = '読み込み失敗';
    }
  }
  function render() {
    body.replaceChildren();
    if (!data.rows.length) {
      const cell = text('td', 'この条件に一致するタグはありません。', 'empty-state');
      cell.colSpan = 4;
      const row = document.createElement('tr'); row.append(cell); body.append(row);
      return;
    }
    for (const row of data.rows) {
      const item = document.createElement('tr');
      item.dataset.tag = row.name;
      const nameCell = document.createElement('td');
      const reference = text('a', row.name, 'review-tag');
      reference.href = `https://danbooru.donmai.us/posts?tags=${encodeURIComponent(row.name)}`;
      reference.target = '_blank'; reference.rel = 'noreferrer'; reference.title = 'タグの参考画像を開く';
      const badge = text('span', labels[row.state], `review-badge state-${row.state}`);
      nameCell.append(reference, text('small', categories[row.category], 'review-category'), badge);
      const old = text('td', row.old_translation, 'review-old');
      const revised = document.createElement('td');
      revised.append(text('span', row.state==='resolved' ? row.current_translation : row.needs_review ? '未確定' : row.revised_translation));
      if (row.note) revised.append(text('small', row.note, 'review-note'));
      const edit = document.createElement('td');
      const input = document.createElement('input');
      input.type = 'text'; input.maxLength = 200; input.autocomplete = 'off';
      input.setAttribute('aria-label', `${row.name} の日本語訳`);
      input.value = row.needs_review && row.current_translation===row.name.replaceAll('_',' ') ? '' : row.current_translation;
      input.placeholder = '日本語訳を入力';
      const button = text('button', '保存して反映'); button.type = 'submit';
      const form = document.createElement('form'); form.className = 'review-edit';
      const feedback = text('small', '', 'review-feedback'); feedback.setAttribute('role','status');
      form.append(input, button); edit.append(form, feedback);
      form.addEventListener('submit', async event => {
        event.preventDefault();
        const value = input.value.trim();
        if (!value) { feedback.textContent = '日本語訳を入力してください。'; input.focus(); return; }
        button.disabled = true; input.disabled = true; button.textContent = '保存中…';
        feedback.textContent = '';
        try {
          const response = await fetch('/translation-review', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ name:row.name, japanese_name:value }) });
          const result = await response.json();
          if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
          data.counts[row.state] = Math.max(0,(data.counts[row.state] || 0)-1);
          data.counts.resolved = (data.counts.resolved || 0)+1;
          row.state = 'resolved'; row.needs_review = false; row.current_translation = value;
          badge.textContent = labels.resolved; badge.className = 'review-badge state-resolved';
          revised.replaceChildren(text('span', value), text('small','手動で確認した訳','review-note'));
          feedback.textContent = '保存済み・タグ付けに反映しました';
          status.textContent = `${row.name} → ${value} を反映しました。`;
          if (state==='review') {
            data.total--; data.rows = data.rows.filter(entry => entry!==row); item.remove();
            if (!data.rows.length) await load();
          }
          metadata();
        } catch (error) { feedback.textContent = `保存できませんでした: ${error.message}`; }
        finally { button.disabled = false; input.disabled = false; button.textContent = '保存して反映'; }
      });
      item.append(nameCell, old, revised, edit); body.append(item);
    }
  }
  filters.forEach(button => button.addEventListener('click', () => {
    state = button.dataset.reviewState;
    filters.forEach(other => other.classList.toggle('is-selected',other===button));
    status.textContent = ''; load(true);
  }));
  search.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => load(true),220); });
  category.addEventListener('change', () => load(true));
  previous.addEventListener('click', () => { offset = Math.max(0,offset-50); load(); });
  next.addEventListener('click', () => { offset += data.rows.length; load(); });
  document.querySelector('#reviewRefresh').addEventListener('click', () => load());
  document.addEventListener('viewchange', event => {
    if (event.detail==='translationReviewView') {
      history.replaceState(null,'','#translationReviewView');
      if (!data) load();
    }
  });
  if (location.hash==='#translationReviewView') document.querySelector('#reviewTab').click();
})();
