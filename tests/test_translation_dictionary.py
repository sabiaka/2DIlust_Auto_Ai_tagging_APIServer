import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tagger


class TranslationDictionaryTests(unittest.TestCase):
    def test_revised_dictionary_and_user_override_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = [folder / name for name in ['base.csv', 'generated.csv', 'ai.csv', 'extra.csv']]
            for path, label in zip(paths, ['心臓', '旧訳', 'ハート', '好きな訳']):
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['name', 'japanese_name'])
                    writer.writerow(['heart', label])
                    writer.writerow(['heart_hands', '手でハート' if path.name == 'ai.csv' else ''])
            with patch.multiple(tagger, TAG_FILE=paths[0], GENERATED_TRANSLATION_FILES=[paths[1]], AI_TAG_TRANSLATION_FILE=paths[2], EXTRA_TAG_TRANSLATION_FILE=paths[3]):
                tagger.reload_translations()
                self.assertEqual(tagger._translate_tag('heart'), '好きな訳')
                self.assertEqual(tagger._translate_tag('heart hands'), '手でハート')
                paths[3].unlink()
                tagger.reload_translations()
                self.assertEqual(tagger._translate_tag('heart'), 'ハート')
                paths[2].unlink()
                tagger.reload_translations()
                self.assertEqual(tagger._translate_tag('heart'), '旧訳')
                self.assertEqual(tagger._translate_tag('unknown_tag'), 'unknown tag')
        tagger.reload_translations()


if __name__ == '__main__':
    unittest.main()
