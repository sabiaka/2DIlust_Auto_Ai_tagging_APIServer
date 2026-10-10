(() => {
  const body = document.querySelector('#reviewRows'), scroll = document.querySelector('#reviewScroll');
  const index = document.querySelector('#reviewIndex'), position = document.querySelector('#reviewPosition');
  const search = document.querySelector('#reviewSearch'), category = document.querySelector('#reviewCategory'), order = document.querySelector('#reviewOrder');
  const summary = document.querySelector('#reviewSummary'), status = document.querySelector('#reviewStatus'), stats = document.querySelector('#reviewStats');
  const filters = document.querySelectorAll('[data-review-state]');
  const labels = { changed:'修正', new:'追加', review:'要確認', resolved:'手動確認済み', preserved:'原表記を保持', unchanged:'変更なし' };
  const categories = { general:'一般', character:'キャラクター', copyright:'作品', style:'画風・作者', meta:'メタ情報', rating:'レーティング' };
  const HEIGHT = 132, BUFFER = 8, drafts = new Map(), cache = new Map();
  let state = 'changes', data = null, controller, timer, frame, start = -1, end = -1;
  const text = (tag, value, cls = '') => {
    const element = document.createElement(tag); element.textContent = value; element.className = cls; return element;
  };
  function metadata() {
    const counts = data.counts;
    stats.replaceChildren(...[['修正・追加',(counts.changed||0)+(counts.new||0)],['要確認',counts.review||0],['手動確認済み',counts.resolved||0]].map(([label,count]) => {
      const card = text('div','','review-stat'); card.append(text('strong',count.toLocaleString()),text('span',label)); return card;
    }));
  }
  function buildIndex() {
    const sections = new Map();
    data.rows.forEach((row,i) => { if (!sections.has(row.letter)) sections.set(row.letter,i); });
    index.replaceChildren(...data.index_order.filter(letter => sections.has(letter)).map(letter => {
      const button = text('button',letter,'review-index-button'); button.type = 'button'; button.dataset.letter = letter;
      button.setAttribute('aria-label',letter+' のタグへ移動');
      button.addEventListener('click',() => { scroll.scrollTop = sections.get(letter)*HEIGHT; render(true); });
      return button;
    }));
  }
  function updatePosition() {
    const current = Math.min(data.rows.length-1,Math.floor(scroll.scrollTop/HEIGHT));
    const letter = data.rows[current]?.letter || '—'; position.textContent = letter;
    position.setAttribute('aria-label','現在の位置 '+letter);
    index.querySelectorAll('button').forEach(button => {
      const active = button.dataset.letter===letter; button.classList.toggle('is-current',active);
      if (active) button.setAttribute('aria-current','location'); else button.removeAttribute('aria-current');
    });
    summary.textContent = data.total.toLocaleString()+'件を1つの一覧で表示 · '+(data.rows.length ? current+1 : 0)+'〜'+Math.min(data.rows.length,current+Math.ceil(scroll.clientHeight/HEIGHT))+'件目';
  }
  function spacer(height) {
    const row = text('tr','','review-spacer'); row.style.height = height+'px'; row.setAttribute('aria-hidden','true'); return row;
  }
  function render(force = false) {
    if (!data) return;
    const first = Math.max(0,Math.floor(scroll.scrollTop/HEIGHT)-BUFFER);
    const last = Math.min(data.rows.length,first+Math.ceil(scroll.clientHeight/HEIGHT)+2*BUFFER);
    if (force || first!==start || last!==end) {
      const active = document.activeElement, selection = active instanceof HTMLInputElement ? [active.selectionStart,active.selectionEnd] : null;
      const fragment = document.createDocumentFragment();
      if (!data.rows.length) {
        const row = document.createElement('tr'), cell = text('td','この条件に一致するタグはありません。','empty-state'); cell.colSpan=4; row.append(cell); fragment.append(row);
      } else {
        fragment.append(spacer(first*HEIGHT));
        for (let i=first;i<last;i++) {
          const row = data.rows[i];
          if (!cache.has(row.name)) cache.set(row.name,createRow(row));
          const item = cache.get(row.name); item.setAttribute('aria-rowindex',String(i+2)); fragment.append(item);
        }
        fragment.append(spacer((data.rows.length-last)*HEIGHT));
      }
      body.replaceChildren(fragment); start=first; end=last;
      for (const [name,item] of cache) if (!item.isConnected) cache.delete(name);
      if (active?.isConnected && selection) { active.focus({preventScroll:true}); active.setSelectionRange(...selection); }
      body.closest('table').setAttribute('aria-rowcount',String(data.rows.length+1));
    }
    updatePosition();
  }
  async function load(reset = true) {
    controller?.abort(); controller = new AbortController(); summary.textContent = '読み込み中…';
    try {
      const params = new URLSearchParams({q:search.value.trim(),state,category:category.value,order:order.value,limit:'0'});
      const response = await fetch('/translation-review?'+params,{signal:controller.signal});
      if (!response.ok) throw new Error('HTTP '+response.status);
      data = await response.json(); cache.clear(); start=end=-1;
      if (reset) scroll.scrollTop=0;
      metadata(); buildIndex(); render(true);
    } catch (error) {
      if (error.name==='AbortError') return;
      status.textContent = '読み込みに失敗しました: '+error.message; summary.textContent='読み込み失敗';
    }
  }
  function createRow(row) {
    const item = text('tr','','review-data-row'); item.dataset.tag=row.name;
    const name = document.createElement('td'), reference = text('a',row.name,'review-tag');
    reference.href='https://danbooru.donmai.us/posts?tags='+encodeURIComponent(row.name); reference.target='_blank'; reference.rel='noreferrer'; reference.title=row.name;
    const badge = text('span',labels[row.state],'review-badge state-'+row.state);
    name.append(reference,text('small',categories[row.category],'review-category'),badge);
    const old = text('td',row.old_translation,'review-old'); old.title=row.old_translation;
    const revised = document.createElement('td');
    revised.append(text('span',row.state==='resolved' ? row.current_translation : row.needs_review ? '未確定' : row.revised_translation));
    if (row.note) revised.append(text('small',row.note,'review-note'));
    const edit = document.createElement('td'), input = document.createElement('input');
    input.type='text'; input.maxLength=200; input.autocomplete='off'; input.setAttribute('aria-label',row.name+' の日本語訳');
    input.value=drafts.get(row.name) ?? (row.needs_review && row.current_translation===row.name.replaceAll('_',' ') ? '' : row.current_translation);
    input.placeholder='日本語訳を入力'; input.addEventListener('input',() => drafts.set(row.name,input.value));
    const button = text('button','保存して反映'); button.type='submit';
    const form = text('form','','review-edit'), feedback = text('small','','review-feedback'); feedback.setAttribute('role','status');
    form.append(input,button); edit.append(form,feedback);
    form.addEventListener('submit',async event => {
      event.preventDefault(); const value=input.value.trim(), snapshot=data;
      if (!value) { feedback.textContent='日本語訳を入力してください。'; input.focus(); return; }
      button.disabled=input.disabled=true; button.textContent='保存中…'; feedback.textContent='';
      try {
        const response=await fetch('/translation-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:row.name,japanese_name:value})});
        const result=await response.json(); if (!response.ok) throw new Error(result.detail || 'HTTP '+response.status);
        snapshot.counts[row.state]=Math.max(0,(snapshot.counts[row.state]||0)-1); snapshot.counts.resolved=(snapshot.counts.resolved||0)+1;
        row.state='resolved'; row.needs_review=false; row.current_translation=value; drafts.delete(row.name);
        badge.textContent=labels.resolved; badge.className='review-badge state-resolved';
        revised.replaceChildren(text('span',value),text('small','手動で確認した訳','review-note'));
        feedback.textContent='保存済み・タグ付けに反映しました'; status.textContent=row.name+' → '+value+' を反映しました。';
        if (snapshot!==data) await load(false);
        else {
          if (state==='review') { data.rows=data.rows.filter(entry => entry!==row); data.total=data.rows.length; cache.delete(row.name); buildIndex(); render(true); }
          metadata(); updatePosition();
        }
      } catch (error) { feedback.textContent='保存できませんでした: '+error.message; }
      finally { button.disabled=input.disabled=false; button.textContent='保存して反映'; }
    });
    item.append(name,old,revised,edit); return item;
  }
  scroll.addEventListener('scroll',() => { if (!frame) frame=requestAnimationFrame(() => { frame=null; render(); }); },{passive:true});
  new ResizeObserver(() => render(true)).observe(scroll);
  filters.forEach(button => button.addEventListener('click',() => {
    state=button.dataset.reviewState; filters.forEach(other => other.classList.toggle('is-selected',other===button)); status.textContent=''; load();
  }));
  search.addEventListener('input',() => { clearTimeout(timer); timer=setTimeout(() => load(),220); });
  category.addEventListener('change',() => load()); order.addEventListener('change',() => load());
  document.querySelector('#reviewRefresh').addEventListener('click',() => load(false));
  document.addEventListener('viewchange',event => {
    if (event.detail==='translationReviewView') { history.replaceState(null,'','#translationReviewView'); if (!data) load(); else render(true); }
  });
  if (location.hash==='#translationReviewView') document.querySelector('#reviewTab').click();
})();
