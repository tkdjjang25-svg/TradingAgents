"""비밀번호 로그인 게이트웨이 (사내망 공개용).

구조:
  브라우저 ──▶ [이 게이트웨이 0.0.0.0:8501]  ──(로그인된 요청만)──▶ [Streamlit 127.0.0.1:8502]
- Streamlit 은 .streamlit/config.toml 에서 127.0.0.1 로만 열리므로 밖에서 직접 접속할 수 없다.
- 로그인하지 않은 요청: 페이지는 /login 으로 리다이렉트, 그 외(API·웹소켓)는 401.
- 비밀번호는 평문으로 두지 않는다. .env 의 APP_PASSWORD_HASH (scrypt 해시) 와 비교하고,
  비교는 hmac.compare_digest (상수 시간) 로 한다.
- 로그인 성공 시 서버 쪽 세션을 만들고 무작위 토큰을 HttpOnly + SameSite=Strict 쿠키로 준다.
- 같은 IP 에서 로그인에 연속 실패하면 잠시 막는다 (무차별 대입 방지).
- 게이트웨이는 Streamlit 으로 넘기는 요청에 내부 비밀 헤더를 붙이고, 앱(app.py)은 이 헤더가 없는
  접속(예: 같은 PC 에서 8502 로 직접 접속)을 거부한다.

사용:
  .venv\\Scripts\\python.exe auth_gateway.py set-password   # 비밀번호를 입력받아 해시를 .env 에 저장
  .venv\\Scripts\\python.exe auth_gateway.py serve          # Streamlit + 게이트웨이 함께 실행
"""
import argparse
import asyncio
import base64
import getpass
import hashlib
import hmac
import html
import logging
import os
import re
import secrets
import subprocess
import sys
import socket
import time
from pathlib import Path
from urllib.parse import quote
from typing import Dict, Optional, Tuple

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from dotenv import load_dotenv
from multidict import CIMultiDict
from yarl import URL

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

COOKIE_NAME = "ta_session"
FAIL_DELAY_SEC = 0.5  # 로그인 실패 응답을 늦춰 대입 속도를 줄인다
MAX_LOGIN_BODY = 8 * 1024
WS_RECHECK_SEC = 30  # 열린 웹소켓이 아직 유효한 세션인지 다시 확인하는 간격
GATEWAY_HEADER = "X-TA-Gateway"  # 게이트웨이 → Streamlit 내부 비밀 (매 실행마다 새로 만든다)
SESSION_HEADER = "X-TA-Session"  # 로그인 세션 식별자 (토큰 자체가 아니라 해시) → 작업 소유자 구분용
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length", "content-encoding",
}

log = logging.getLogger("auth_gateway")


# ---------------------------------------------------------------------------
# 비밀번호 해시 (scrypt, 표준 라이브러리)
# ---------------------------------------------------------------------------
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, maxmem=64 * 1024 * 1024, **_SCRYPT)
    b64 = lambda b: base64.b64encode(b).decode("ascii")  # noqa: E731
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """상수 시간 비교. 형식이 틀리면 False."""
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(hash_b64)
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt_b64), dklen=len(expected),
            n=int(n), r=int(r), p=int(p), maxmem=64 * 1024 * 1024,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


# ---------------------------------------------------------------------------
# 세션과 로그인 시도 제한
# ---------------------------------------------------------------------------
class SessionStore:
    """서버 메모리에 두는 세션. 쿠키에는 무작위 토큰만, 서버에는 토큰의 해시만 저장한다."""

    def __init__(self, ttl_sec: int):
        self.ttl = ttl_sec
        self._sessions: Dict[str, float] = {}

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode("ascii", "ignore")).hexdigest()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[self._key(token)] = time.time() + self.ttl
        return token

    def session_id(self, token: Optional[str]) -> Optional[str]:
        """유효한 토큰이면 세션 식별자(해시 앞부분), 아니면 None."""
        if not token:
            return None
        key = self._key(token)
        expires = self._sessions.get(key)
        if expires is None:
            return None
        if expires < time.time():
            self._sessions.pop(key, None)
            return None
        return key[:16]

    def revoke(self, token: Optional[str]) -> None:
        if token:
            self._sessions.pop(self._key(token), None)


class LoginLimiter:
    """IP 별 로그인 실패 횟수 제한: window 안에 max_failures 번 틀리면 lock 동안 차단."""

    def __init__(self, max_failures: int = 5, window_sec: int = 300, lock_sec: int = 300):
        self.max_failures, self.window, self.lock = max_failures, window_sec, lock_sec
        self._fails: Dict[str, list] = {}
        self._locked: Dict[str, float] = {}

    def locked_for(self, ip: str) -> int:
        until = self._locked.get(ip, 0)
        return max(0, int(until - time.time()))

    def fail(self, ip: str) -> None:
        now = time.time()
        recent = [t for t in self._fails.get(ip, []) if now - t < self.window] + [now]
        self._fails[ip] = recent
        if len(recent) >= self.max_failures:
            self._locked[ip] = now + self.lock
            self._fails[ip] = []

    def success(self, ip: str) -> None:
        self._fails.pop(ip, None)
        self._locked.pop(ip, None)


# ---------------------------------------------------------------------------
# 웹 앱
# ---------------------------------------------------------------------------
_LOGIN_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradingAgents 로그인</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f5f7fa;margin:0;display:flex;
min-height:100vh;align-items:center;justify-content:center}}
form{{background:#fff;padding:32px;border-radius:12px;box-shadow:0 2px 12px #0001;width:min(320px,90vw)}}
h1{{font-size:20px;margin:0 0 16px}} input{{width:100%;box-sizing:border-box;padding:10px;margin:8px 0 16px;
border:1px solid #ccd;border-radius:8px;font-size:15px}} button{{width:100%;padding:10px;border:0;border-radius:8px;
background:#ff4b4b;color:#fff;font-size:15px;cursor:pointer}} .err{{color:#c00;font-size:14px;margin:0 0 8px}}
</style></head><body>
<form method="post" action="/login" autocomplete="off">
<h1>📈 TradingAgents</h1>
{error}
<label for="pw">비밀번호</label>
<input id="pw" name="password" type="password" required autofocus>
<input type="hidden" name="next" value="{next}">
<button type="submit">로그인</button>
</form></body></html>"""


def _client_ip(request: web.Request) -> str:
    return request.remote or "unknown"


def _safe_next(value: str) -> str:
    """오픈 리다이렉트 방지: 같은 사이트 안의 경로만 허용.

    제어문자·공백(탭·개행 등)은 URL 처리 중 지워져 '/\\t/evil.com' 이 '//evil.com' 이 될 수
    있으므로 아예 거절하고, 정규화한 뒤에도 호스트·스킴이 없는 경로인지 다시 확인한다.
    """
    if not value or any(ord(c) <= 0x20 or ord(c) == 0x7F for c in value) or "\\" in value:
        return "/"
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    try:
        url = URL(value)
    except ValueError:
        return "/"
    if url.scheme or url.host or str(url).startswith("//"):
        return "/"
    return value


def _is_page_request(request: web.Request) -> bool:
    return request.method == "GET" and "text/html" in request.headers.get("Accept", "")


class Gateway:
    def __init__(self, password_hash: str, backend: str, gateway_secret: str,
                 session_ttl_sec: int = 12 * 3600, secure_cookie: bool = False):
        self.password_hash = password_hash
        self.backend = backend.rstrip("/")
        self.secret = gateway_secret
        self.sessions = SessionStore(session_ttl_sec)
        self.limiter = LoginLimiter()
        self.secure_cookie = secure_cookie
        self._http: Optional[ClientSession] = None
        self._sockets: Dict[str, set] = {}  # 세션별 열린 웹소켓 (로그아웃 시 끊기)

    # -- 앱 구성 -------------------------------------------------------------
    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=50 * 1024 * 1024)
        app.router.add_get("/login", self.login_page)
        app.router.add_post("/login", self.login_submit)
        app.router.add_post("/logout", self.logout)
        app.router.add_get("/logout", self.logout)
        app.router.add_route("*", "/{tail:.*}", self.proxy)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        app.middlewares.append(self._security_headers)
        return app

    async def _startup(self, app):
        self._http = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10), auto_decompress=False)

    async def _cleanup(self, app):
        if self._http:
            await self._http.close()

    @web.middleware
    async def _security_headers(self, request, handler):
        response = await handler(request)
        if isinstance(response, web.StreamResponse) and not isinstance(response, web.WebSocketResponse):
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "same-origin")
            response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return response

    # -- 로그인 ---------------------------------------------------------------
    def _render_login(self, error: str = "", next_path: str = "/", status: int = 200) -> web.Response:
        body = _LOGIN_PAGE.format(
            error=f'<p class="err">{html.escape(error)}</p>' if error else "",
            next=html.escape(_safe_next(next_path), quote=True),
        )
        resp = web.Response(text=body, content_type="text/html", status=status)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    async def login_page(self, request: web.Request) -> web.Response:
        if self.sessions.session_id(request.cookies.get(COOKIE_NAME)):
            raise web.HTTPFound(_safe_next(request.query.get("next", "/")))
        return self._render_login(next_path=request.query.get("next", "/"))

    def _same_origin(self, request: web.Request) -> bool:
        """로그인 CSRF 방지: 다른 사이트에서 보낸 POST 는 거절."""
        origin = request.headers.get("Origin")
        if not origin or origin == "null":
            return origin is None  # Origin 헤더가 아예 없는 옛 클라이언트는 허용, "null" 은 거절
        return re.sub(r"^https?://", "", origin) == request.host

    async def login_submit(self, request: web.Request) -> web.Response:
        ip = _client_ip(request)
        # 로그인 폼은 몇백 바이트다. 큰 본문은 읽기 전에 거절 (메모리 소모 공격 방지)
        if request.content_length is None or request.content_length > MAX_LOGIN_BODY:
            return web.Response(status=413, text="request too large")
        form = await request.post()
        next_path = str(form.get("next", "/"))
        if not self._same_origin(request):
            return self._render_login("잘못된 요청입니다.", next_path, status=403)
        wait = self.limiter.locked_for(ip)
        if wait:
            log.warning("로그인 차단 중인 IP %s (%s초 남음)", ip, wait)
            return self._render_login(f"로그인 시도가 너무 많습니다. {wait // 60 + 1}분 뒤 다시 시도하세요.", next_path, 429)
        password = str(form.get("password", ""))
        if not verify_password(password, self.password_hash):
            self.limiter.fail(ip)
            log.warning("로그인 실패: %s", ip)
            await asyncio.sleep(FAIL_DELAY_SEC)  # 자동화된 대입 속도를 늦춘다
            return self._render_login("비밀번호가 올바르지 않습니다.", next_path, status=401)
        self.limiter.success(ip)
        token = self.sessions.create()
        log.info("로그인 성공: %s", ip)
        resp = web.HTTPFound(_safe_next(next_path))
        resp.set_cookie(
            COOKIE_NAME, token, httponly=True, samesite="Strict", path="/",
            secure=self.secure_cookie or request.secure, max_age=self.sessions.ttl,
        )
        raise resp

    async def logout(self, request: web.Request) -> web.Response:
        token = request.cookies.get(COOKIE_NAME)
        await self._close_sockets(self.sessions.session_id(token))  # 열려 있는 탭의 연결도 즉시 끊는다
        self.sessions.revoke(token)
        resp = web.HTTPFound("/login")
        resp.del_cookie(COOKIE_NAME, path="/")
        raise resp

    # -- 프록시 ---------------------------------------------------------------
    def _backend_headers(self, request: web.Request, session_id: str) -> CIMultiDict:
        # 대소문자를 가리지 않고 걸러낸다 (x-ta-session 같은 위조 헤더, 쿠키 중복 등)
        drop = HOP_HEADERS | {"cookie", "origin", "forwarded", "x-real-ip",
                              GATEWAY_HEADER.lower(), SESSION_HEADER.lower()}
        # X-Forwarded-* 는 클라이언트가 마음대로 넣을 수 있고 Streamlit(uvicorn)은 127.0.0.1 에서 온 값을 믿는다
        headers = CIMultiDict(
            (k, v) for k, v in request.headers.items()
            if k.lower() not in drop and not k.lower().startswith("x-forwarded-")
        )
        cookies = [f"{k}={v}" for k, v in request.cookies.items() if k != COOKIE_NAME]  # 세션 토큰은 넘기지 않는다
        if cookies:
            headers["Cookie"] = "; ".join(cookies)
        headers[GATEWAY_HEADER] = self.secret
        headers[SESSION_HEADER] = session_id
        headers["Host"] = re.sub(r"^https?://", "", self.backend)
        if "Origin" in request.headers:  # 게이트웨이가 출처를 검사했으므로 Streamlit 에는 자기 주소로
            headers["Origin"] = self.backend
        return headers

    async def proxy(self, request: web.Request) -> web.StreamResponse:
        session_id = self.sessions.session_id(request.cookies.get(COOKIE_NAME))
        if not session_id:
            if _is_page_request(request):
                raise web.HTTPFound("/login?next=" + quote(_safe_next(request.path_qs), safe="/"))
            return web.json_response({"error": "unauthorized"}, status=401)
        is_ws = request.headers.get("Upgrade", "").lower() == "websocket"
        # 같은 PC 의 다른 포트 등에서 쿠키를 등에 업고 들어오는 요청(교차 출처 웹소켓 탈취 등) 차단
        if (is_ws or request.method not in ("GET", "HEAD")) and not self._origin_matches(request):
            return web.json_response({"error": "forbidden origin"}, status=403)
        if is_ws:
            return await self._proxy_ws(request, session_id)
        return await self._proxy_http(request, session_id)

    @staticmethod
    def _origin_matches(request: web.Request) -> bool:
        """Origin 이 있으면 게이트웨이 자신의 주소와 같아야 한다 (브라우저는 항상 보낸다)."""
        origin = request.headers.get("Origin")
        if origin is None:
            return True
        return re.sub(r"^https?://", "", origin) == request.host

    async def _proxy_http(self, request: web.Request, session_id: str) -> web.StreamResponse:
        body = await request.read()
        async with self._http.request(
            request.method, self.backend + request.path_qs, headers=self._backend_headers(request, session_id),
            data=body if body else None, allow_redirects=False,
        ) as upstream:
            resp = web.StreamResponse(status=upstream.status)
            for k, v in upstream.headers.items():
                if k.lower() not in HOP_HEADERS or k.lower() == "content-encoding":
                    resp.headers.add(k, v)
            await resp.prepare(request)
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    async def _proxy_ws(self, request: web.Request, session_id: str) -> web.WebSocketResponse:
        protocols = [p.strip() for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()]
        headers = self._backend_headers(request, session_id)
        for k in list(headers):
            if k.lower().startswith("sec-websocket"):
                headers.popall(k, None)
        url = "ws" + self.backend[len("http"):] + request.path_qs
        token = request.cookies.get(COOKIE_NAME)
        async with self._http.ws_connect(url, headers=headers, protocols=protocols, max_msg_size=0) as upstream:
            client = web.WebSocketResponse(protocols=[upstream.protocol] if upstream.protocol else (), max_msg_size=0)
            await client.prepare(request)
            self._sockets.setdefault(session_id, set()).add(client)

            async def pump(src, dst):
                async for msg in src:
                    if msg.type == WSMsgType.TEXT:
                        await dst.send_str(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await dst.send_bytes(msg.data)
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
                await dst.close()

            async def watch():
                # 로그아웃·만료된 세션의 열린 탭은 다음 확인 때 끊는다
                while not client.closed:
                    await asyncio.sleep(WS_RECHECK_SEC)
                    if not self.sessions.session_id(token):
                        await client.close(code=4001, message=b"session expired")
                        await upstream.close()
                        return

            watcher = asyncio.ensure_future(watch())
            try:
                await asyncio.gather(pump(client, upstream), pump(upstream, client), return_exceptions=True)
            finally:
                watcher.cancel()
                self._sockets.get(session_id, set()).discard(client)
            return client

    async def _close_sockets(self, session_id: Optional[str]) -> None:
        for ws in list(self._sockets.pop(session_id, set()) if session_id else []):
            await ws.close(code=4001, message=b"logged out")


# ---------------------------------------------------------------------------
# 명령
# ---------------------------------------------------------------------------
def _set_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        if re.match(rf"^\s*{re.escape(key)}\s*=", line):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out += ["", "# Login password hash (written by auth_gateway.py set-password)", f"{key}={value}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def cmd_set_password() -> None:
    first = getpass.getpass("새 비밀번호 (12자 이상): ")
    if len(first) < 12:
        sys.exit("비밀번호는 12자 이상이어야 합니다.")
    if getpass.getpass("한 번 더 입력: ") != first:
        sys.exit("두 비밀번호가 다릅니다.")
    _set_env_value(ENV_FILE, "APP_PASSWORD_HASH", hash_password(first))
    print(f"비밀번호 해시를 {ENV_FILE} 의 APP_PASSWORD_HASH 에 저장했습니다. (평문은 저장하지 않음)")
    print("실행 중인 서비스가 있으면 다시 시작해야 적용됩니다.")


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _listener_pids(port: int) -> Optional[set]:
    """127.0.0.1/0.0.0.0 에서 port 를 LISTEN 중인 프로세스 ID 들 (Windows netstat). 알 수 없으면 None."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].upper() == "LISTENING" and parts[1].endswith(f":{port}"):
            pids.add(int(parts[4]))
    return pids


def _parent_pid(pid: int) -> Optional[int]:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').ParentProcessId"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return int(out) if out.isdigit() else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def backend_owned_by(port: int, proc: subprocess.Popen) -> bool:
    """백엔드 포트를 듣는 프로세스가 우리 Streamlit(또는 venv 런처가 띄운 그 자식)뿐인지.

    Windows 에서는 SO_REUSEADDR 로 다른 프로세스가 같은 포트를 함께 열 수 있어서,
    그런 프로세스가 있으면 게이트웨이 비밀값과 로그인된 요청이 그쪽으로 갈 수 있다.
    확인할 수 없는 OS 에서는 True.
    """
    pids = _listener_pids(port)
    if pids is None:
        return True
    return bool(pids) and all(pid == proc.pid or _parent_pid(pid) == proc.pid for pid in pids)


def _wait_until_listening(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    """자식 Streamlit 이 포트를 열 때까지 기다린다. 먼저 죽으면 False."""
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if _port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def cmd_serve(host: str, port: int, backend_port: int) -> None:
    load_dotenv(ENV_FILE, override=True)
    password_hash = (os.getenv("APP_PASSWORD_HASH") or "").strip()
    if not password_hash.startswith("scrypt$"):
        sys.exit("APP_PASSWORD_HASH 가 없습니다. 먼저 `auth_gateway.py set-password` 로 비밀번호를 정하세요.")
    # 비밀값(.env)과 런타임 데이터(data/)는 이 계정·SYSTEM·관리자만 읽게 한다 (같은 PC 의 다른 계정 차단)
    import local_security
    from tradingagents.default_config import TRADINGAGENTS_HOME

    for target, ok in ((ENV_FILE, local_security.secure_file(ENV_FILE)),
                       (TRADINGAGENTS_HOME, local_security.secure_dir(Path(TRADINGAGENTS_HOME)))):
        if not ok:
            log.warning("권한을 제한하지 못했습니다: %s", target)

    # 백엔드 포트를 이미 다른 프로세스가 쓰고 있으면, 게이트웨이 비밀값과 로그인된 요청이 그쪽으로 간다
    if _port_in_use(backend_port):
        sys.exit(f"127.0.0.1:{backend_port} 를 이미 다른 프로그램이 쓰고 있습니다. 그 프로그램을 끄거나 --backend-port 를 바꾸세요.")
    secret = secrets.token_urlsafe(32)
    env = {**os.environ, "TA_GATEWAY_SECRET": secret, "PYTHONIOENCODING": "utf-8"}
    streamlit = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.address", "127.0.0.1", "--server.port", str(backend_port), "--server.headless", "true"],
        cwd=ROOT, env=env,
    )
    if not _wait_until_listening(backend_port, streamlit, timeout=90):
        streamlit.kill()
        sys.exit("Streamlit 이 시작되지 않았습니다. 위 로그를 확인하세요.")
    if not backend_owned_by(backend_port, streamlit):
        streamlit.kill()
        sys.exit(f"127.0.0.1:{backend_port} 를 다른 프로세스도 열고 있어 시작하지 않습니다 (가로채기 방지).")
    gateway = Gateway(
        password_hash, f"http://127.0.0.1:{backend_port}", secret,
        session_ttl_sec=int(float(os.getenv("APP_SESSION_HOURS", "12")) * 3600),
        secure_cookie=os.getenv("APP_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
    )
    app = gateway.make_app()

    async def stop_when_streamlit_exits(app):
        async def watch():
            ticks = 0
            while streamlit.poll() is None:
                await asyncio.sleep(2)
                ticks += 1
                if ticks % 15 == 0:  # 30초마다: 다른 프로세스가 백엔드 포트를 함께 열었으면 멈춘다
                    owned = await asyncio.get_running_loop().run_in_executor(
                        None, backend_owned_by, backend_port, streamlit)
                    if not owned:
                        log.error("백엔드 포트를 다른 프로세스가 열었습니다. 안전을 위해 게이트웨이를 멈춥니다.")
                        streamlit.terminate()
                        raise web.GracefulExit()
            log.error("Streamlit 이 종료되어 게이트웨이도 멈춥니다 (종료 코드 %s)", streamlit.returncode)
            raise web.GracefulExit()

        app["streamlit_watch"] = asyncio.create_task(watch())

    app.on_startup.append(stop_when_streamlit_exits)
    log.info("로그인 게이트웨이: http://%s:%s  →  Streamlit 127.0.0.1:%s", host, port, backend_port)
    try:
        web.run_app(app, host=host, port=port, print=None, access_log=None)
    finally:
        streamlit.terminate()
        try:
            streamlit.wait(timeout=10)
        except subprocess.TimeoutExpired:
            streamlit.kill()


def main(argv=None) -> None:
    load_dotenv(ENV_FILE, override=True)
    parser = argparse.ArgumentParser(description="TradingAgents 로그인 게이트웨이")
    parser.add_argument("command", choices=["serve", "set-password"])
    parser.add_argument("--host", default=os.getenv("APP_HOST", "0.0.0.0"), help="공개할 주소 (기본 0.0.0.0 = 사내망)")
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8501")))
    parser.add_argument("--backend-port", type=int, default=int(os.getenv("APP_BACKEND_PORT", "8502")))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "set-password":
        cmd_set_password()
    else:
        cmd_serve(args.host, args.port, args.backend_port)


if __name__ == "__main__":
    main()
