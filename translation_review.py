"""Compare the original translations with revised and manual dictionary entries."""
import csv
import os
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

CATEGORIES = {'0':'general', '1':'style', '3':'copyright', '4':'character', '5':'meta', '9':'rating'}


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class TranslationReview:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.catalog_path = self.folder / 'translation_review_catalog.csv'
        self.revised_path = self.folder / 'tag_translations_ai.csv'
        self.manual_path = self.folder / 'tag_translations_extra.csv'
        self.lock = threading.RLock()

    @lru_cache(maxsize=1)
    def catalog(self):
        return {row['name']:row for row in read_csv(self.catalog_path)}

    def rows(self):
        revised = {row['name'].replace('_',' ').strip():row['japanese_name'] for row in read_csv(self.revised_path) if row.get('japanese_name','').strip()}
        manual = {row['name'].replace('_',' ').strip():row['japanese_name'] for row in read_csv(self.manual_path) if row.get('japanese_name','').strip()}
        for source in self.catalog().values():
            name = source['name']
            key = name.replace('_',' ').strip()
            old = source['old_translation'] or key
            proposed = revised.get(key, old)
            current = manual.get(key, proposed)
            pending = source['needs_review']=='1' and key not in manual
            if key in manual:
                state = 'resolved'
            elif pending:
                state = 'review'
            elif source['category']=='1' and proposed == name:
                state = 'preserved'
            elif not source['old_translation'] and key in revised and proposed != old:
                state = 'new'
            elif proposed != old:
                state = 'changed'
            else:
                state = 'unchanged'
            yield {'name':name, 'category':CATEGORIES.get(source['category'],source['category']), 'old_translation':old, 'revised_translation':proposed, 'current_translation':current, 'manual_translation':manual.get(key,''), 'state':state, 'needs_review':pending, 'note':source.get('note','') if pending else ''}

    def list_rows(self, query='', state='changes', category='', offset=0, limit=50):
        if state not in {'all','changes','changed','new','review','resolved','preserved','unchanged'}:
            raise ValueError('Unknown filter')
        if category and category not in set(CATEGORIES.values()):
            raise ValueError('Unknown category')
        query = query.strip().lower().replace('_',' ')
        all_rows = list(self.rows())
        counts = dict(Counter(row['state'] for row in all_rows))
        matched = []
        for row in all_rows:
            if state=='changes' and row['state'] not in {'changed','new','resolved'}:
                continue
            if state!='changes' and state!='all' and row['state']!=state:
                continue
            if category and row['category']!=category:
                continue
            text = ' '.join(str(row[k]) for k in ['name','old_translation','revised_translation','current_translation']).lower().replace('_',' ')
            if query and query not in text:
                continue
            matched.append(row)
        offset = max(0,offset)
        limit = max(1,min(limit,200))
        return {'rows':matched[offset:offset+limit], 'total':len(matched), 'offset':offset, 'limit':limit, 'counts':counts}

    def save(self, name, translation):
        if name not in self.catalog():
            raise KeyError(name)
        translation = translation.strip()
        if not translation or len(translation)>200 or '\n' in translation or '\r' in translation:
            raise ValueError('日本語訳を1〜200文字の1行で入力してください。')
        with self.lock:
            if self.manual_path.exists():
                with self.manual_path.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields = list(reader.fieldnames or ['name','japanese_name'])
                    records = list(reader)
            else:
                fields, records = ['name','japanese_name'], []
            for field in ['name','japanese_name']:
                if field not in fields:
                    fields.append(field)
            key = name.replace('_',' ').strip()
            found = False
            for record in records:
                if record.get('name','').replace('_',' ').strip()==key:
                    record['japanese_name'] = translation
                    found = True
            if not found:
                record = {field:'' for field in fields}
                record.update(name=name, japanese_name=translation)
                records.append(record)
            self.folder.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with NamedTemporaryFile('w', encoding='utf-8', newline='', dir=self.folder, delete=False) as stream:
                    temporary = Path(stream.name)
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)
                os.replace(temporary, self.manual_path)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
        return next(row for row in self.rows() if row['name']==name)
