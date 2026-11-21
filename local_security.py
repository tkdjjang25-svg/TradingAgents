"""로컬 파일 권한 제한 (자격증명·런타임 데이터 보호).

Windows: 상속·기존 명시 권한을 모두 지우고 현재 계정(SID)·SYSTEM·Administrators 에게만
전체 권한을 준 뒤, 결과를 다시 읽어 다른 주체가 남아 있으면 실패로 본다.
그 외 OS: 폴더 0700, 파일 0600.

안전장치: 저장소 폴더 밖의 경로(예: TRADINGAGENTS_HOME 을 드라이브 루트로 잘못 지정)는
건드리지 않는다.
"""
import logging
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent
_SYSTEM_SID = "S-1-5-18"
_ADMINS_SID = "S-1-5-32-544"
log = logging.getLogger("local_security")


@lru_cache(maxsize=1)
def current_user_sid() -> Optional[str]:
    """이름이 아니라 SID 로 지정해야 같은 이름의 다른 계정(도메인/로컬)과 헷갈리지 않는다."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(S-1-5-[0-9-]+)", out)
    return match.group(1) if match else None


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def _icacls(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["icacls", *args], capture_output=True, text=True, timeout=60)


def _only_allowed(path: Path, sid: str) -> bool:
    """icacls /save 결과(SID 로 기록됨)에 허용한 세 주체 외에는 없는지 확인한다."""
    import tempfile

    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "data" if (REPO_ROOT / "data").exists() else None) as tmp:
        dump = Path(tmp) / "acl.txt"
        if _icacls([str(path), "/save", str(dump), "/Q"]).returncode != 0:
            return False
        text = dump.read_bytes().decode("utf-16", errors="ignore")
    sids = set(re.findall(r";;;(S-1-[0-9-]+|[A-Z]{2})\)", text))
    aliases = {"SY": _SYSTEM_SID, "BA": _ADMINS_SID}
    sids = {aliases.get(s, s) for s in sids}
    return bool(sids) and sids <= {sid, _SYSTEM_SID, _ADMINS_SID}


def secure_dir(path: Path) -> bool:
    """폴더와 그 안의 모든 파일을 소유자 전용으로. 실패하거나 저장소 밖이면 False."""
    path.mkdir(parents=True, exist_ok=True)
    if not _inside_repo(path):
        log.warning("저장소 밖의 경로라 권한을 바꾸지 않습니다: %s", path)
        return False
    if os.name != "nt":
        try:
            os.chmod(path, 0o700)
            for p in path.rglob("*"):
                os.chmod(p, 0o700 if p.is_dir() else 0o600)
            return True
        except OSError:
            return False
    sid = current_user_sid()
    if not sid:
        return False
    try:
        steps = [
            [str(path), "/reset", "/Q"],  # 기존 명시 권한 제거
            [str(path), "/inheritance:r", "/Q", "/grant:r",
             f"*{sid}:(OI)(CI)F", f"*{_SYSTEM_SID}:(OI)(CI)F", f"*{_ADMINS_SID}:(OI)(CI)F"],
        ]
        if any(path.iterdir()):
            steps.append([str(path / "*"), "/reset", "/T", "/Q"])  # 안의 것들은 위 폴더 권한만 물려받게
        for args in steps:
            if _icacls(args).returncode != 0:
                return False
        return _only_allowed(path, sid)
    except (OSError, subprocess.TimeoutExpired):
        return False


def secure_file(path: Path) -> bool:
    """파일 하나를 소유자 전용으로. 없으면 True(할 일 없음), 실패하거나 저장소 밖이면 False."""
    if not path.exists():
        return True
    if not _inside_repo(path):
        log.warning("저장소 밖의 경로라 권한을 바꾸지 않습니다: %s", path)
        return False
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
            return True
        except OSError:
            return False
    sid = current_user_sid()
    if not sid:
        return False
    try:
        for args in ([str(path), "/reset", "/Q"],
                     [str(path), "/inheritance:r", "/Q", "/grant:r",
                      f"*{sid}:F", f"*{_SYSTEM_SID}:F", f"*{_ADMINS_SID}:F"]):
            if _icacls(args).returncode != 0:
                return False
        return _only_allowed(path, sid)
    except (OSError, subprocess.TimeoutExpired):
        return False
