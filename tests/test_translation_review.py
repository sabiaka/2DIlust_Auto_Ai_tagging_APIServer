import csv
import tempfile
import unittest
from pathlib import Path

from translation_review import TranslationReview


def write(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TranslationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        write(self.folder / 'translation_review_catalog.csv', ['name','category','old_translation','needs_review','note'], [
            {'name':'heart','category':'0','old_translation':'心臓','needs_review':'0','note':''},
            {'name':'unknown_tag','category':'0','old_translation':'','needs_review':'1','note':'要確認'},
            {'name':'unknown_character','category':'4','old_translation':'','needs_review':'1','note':'要確認'},
        ])
        write(self.folder / 'tag_translations_ai.csv', ['name','japanese_name'], [{'name':'heart','japanese_name':'ハート'}])
        write(self.folder / 'tag_translations_extra.csv', ['name','japanese_name','note'], [{'name':'other_tag','japanese_name':'既存の手動訳','note':'既存メタ情報'}])
        self.review = TranslationReview(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def test_compare_search_and_pagination(self):
        row = self.review.list_rows(query='心臓')['rows'][0]
        self.assertEqual((row['old_translation'],row['revised_translation']),('心臓','ハート'))
        pending = self.review.list_rows(state='review', limit=1, offset=1)
        self.assertEqual(pending['total'],2)
        self.assertEqual(pending['rows'][0]['name'],'unknown_character')
        self.assertEqual(self.review.list_rows(state='review', category='character')['total'],1)
        with self.assertRaises(ValueError):
            self.review.list_rows(state='invalid')

    def test_pending_save_persists_and_preserves_existing_entries(self):
        row = self.review.save('unknown_tag','新しい訳')
        self.assertEqual(row['state'],'resolved')
        self.assertFalse(row['needs_review'])
        self.assertEqual(self.review.list_rows(state='review')['total'],1)
        restored = TranslationReview(self.folder)
        saved = restored.list_rows(state='resolved')['rows'][0]
        self.assertEqual(saved['current_translation'],'新しい訳')
        with restored.manual_path.open(encoding='utf-8') as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(records[0]['note'],'既存メタ情報')
        self.assertEqual(records[0]['japanese_name'],'既存の手動訳')
        self.review.save('unknown_tag','修正した訳')
        self.assertEqual(self.review.list_rows(state='resolved')['total'],1)

    def test_invalid_save_does_not_change_file(self):
        before = self.review.manual_path.read_bytes()
        with self.assertRaises(KeyError):
            self.review.save('../missing','不正タグ')
        for value in ['', 'a\nb', '長'*201]:
            with self.assertRaises(ValueError):
                self.review.save('heart',value)
        self.assertEqual(before,self.review.manual_path.read_bytes())


if __name__=='__main__':
    unittest.main()
