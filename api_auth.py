"""Optional bearer authentication for deployments outside a private LAN."""
import hmac
from starlette.responses import JSONResponse


class ApiTokenMiddleware:
    def __init__(self, app, token=""):
        self.app = app
        self.expected = ("Bearer " + token).encode("utf-8") if token else None

    async def __call__(self, scope, receive, send):
        if self.expected and scope["type"] in {"http", "websocket"}:
            supplied = [value for key, value in scope.get("headers", []) if key.lower() == b"authorization"]
            if len(supplied) != 1 or not hmac.compare_digest(supplied[0], self.expected):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    response = JSONResponse({"detail": "API token required"}, status_code=401,
                                            headers={"WWW-Authenticate": "Bearer"})
                    await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
