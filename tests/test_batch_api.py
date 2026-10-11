import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from batch_api import router
from api_auth import ApiTokenMiddleware

class BatchTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.add_middleware(ApiTokenMiddleware, token='test-token')
        self.client = TestClient(app)
        self.headers = {'Authorization': 'Bearer test-token'}

    def test_partial_errors_and_ids(self):
        def predict(data):
            if data == b'broken': raise ValueError('broken')
            return ['日本語タグ']
        with patch('batch_api.predict_image', side_effect=predict):
            r = self.client.post('/tag-batch', headers=self.headers, files=[
                ('files', ('a.jpg', b'good', 'image/jpeg')),
                ('files', ('b.jpg', b'broken', 'image/jpeg')),
                ('files', ('c.mp4', b'video', 'video/mp4'))])
        self.assertEqual(r.status_code, 200)
        rows = r.json()['items']
        self.assertEqual([row['id'] for row in rows], ['a.jpg', 'b.jpg', 'c.mp4'])
        self.assertEqual(rows[0]['tags'], ['日本語タグ'])
        self.assertIn('error', rows[1])
        self.assertIn('error', rows[2])

    def test_bounds_and_auth(self):
        file = ('files', ('a.jpg', b'good', 'image/jpeg'))
        self.assertEqual(self.client.post('/tag-batch', files=[file]).status_code, 401)
        self.assertEqual(self.client.post('/tag-batch', headers=self.headers, files=[file, file]).status_code, 400)
        files = [('files', (str(i)+'.jpg', b'good', 'image/jpeg')) for i in range(13)]
        self.assertEqual(self.client.post('/tag-batch', headers=self.headers, files=files).status_code, 400)
        with patch('batch_api.MAX_BYTES', 3):
            self.assertEqual(self.client.post('/tag-batch', headers=self.headers, files=[file]).status_code, 413)
