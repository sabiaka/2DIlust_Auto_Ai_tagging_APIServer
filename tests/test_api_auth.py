import unittest
import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from api_auth import ApiTokenMiddleware


async def endpoint(request):
    return JSONResponse({"ok": True})


class ApiAuthTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, token, headers=None):
        app = ApiTokenMiddleware(Starlette(routes=[Route("/tag", endpoint, methods=["POST"])]), token)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/tag", headers=headers, files={"file": ("test.png", b"image")})

    async def test_lan_without_token_keeps_existing_behavior(self):
        self.assertEqual((await self.request("")).status_code, 200)

    async def test_public_requires_token(self):
        self.assertEqual((await self.request("secret")).status_code, 401)
        self.assertEqual((await self.request("secret", {"Authorization": "Bearer wrong"})).status_code, 401)
        self.assertEqual((await self.request("secret", {"Authorization": "Bearer secret"})).status_code, 200)

    async def test_duplicate_header_rejected(self):
        headers = [("Authorization", "Bearer secret"), ("Authorization", "Bearer wrong")]
        self.assertEqual((await self.request("secret", headers)).status_code, 401)
