"""
长期记忆：用户画像（D1 长期记忆的落地件）。

跨会话记住用户的稳定偏好——「我只找广州的」「关键词用 Agent 开发」
「以后都用产品岗版简历」。存在 agent/data/user_profile.json 这一个文件里，
人可以直接打开看，出问题也好排查。

为什么不做成数据库表：画像就一条记录（单用户本地工具），
JSON 文件读起来 0 依赖、手改方便，比塞进 applications.db 更合适。

字段：
    target_cities     目标城市列表，如 ["广州"]
    target_keywords   偏好搜索关键词，如 ["Agent", "大模型"]
    resume_id         惯用简历版本（对应 storage.list_resumes 的 id）
    preferences       自由键值对，如 {"salary_min": "200/天", "job_type": "实习"}

API：
    load_profile() -> dict                    读整份画像（文件缺失/损坏都返回默认值）
    save_profile(profile) -> dict             整份写入（未知字段丢弃，保证结构干净）
    update_preference(key, value) -> dict     改单个字段（两个 React 工具用它）
"""
from __future__ import annotations

import json
import os
import re
import sys
import shutil
from pathlib import Path

# 允许 `python agent/user_profile.py` 直接跑自测：把仓库根目录放进 sys.path，
# 否则直接执行脚本时拿不到 shared 包（正常被 import 时这几行也无害）。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import ROOT_DIR  # noqa: E402
from shared.user_context import DEFAULT_USER_ID, get_current_user  # noqa: E402

# 画像路径（多用户）：默认 <repo_root>/agent/data/profiles/<user_id>.json，
# user_id 由 shared.user_context 的当前用户决定（没有登录态时是 'local'）。
#
# 兼容：显式设了环境变量 USER_PROFILE_PATH，或代码里直接把 PROFILE_PATH 赋成
# 一个 Path（老测试的做法）时走**单文件模式**，行为与改造前一字不变。
#
# 存量 agent/data/user_profile.json 没有归属信息，统一归给 DEFAULT_USER_ID
# （'local'）：首次解析路径时复制一份过去，**旧文件保留**，随时可回滚。
_ENV_PROFILE_PATH = os.getenv("USER_PROFILE_PATH", "").strip()
PROFILE_PATH = Path(_ENV_PROFILE_PATH) if _ENV_PROFILE_PATH else None

PROFILE_DIR = Path(os.getenv(
    "USER_PROFILE_DIR", str(ROOT_DIR / "agent" / "data" / "profiles")
))
LEGACY_PROFILE_PATH = Path(os.getenv(
    "USER_PROFILE_LEGACY_PATH", str(ROOT_DIR / "agent" / "data" / "user_profile.json")
))

_migrated = False


def _safe_user(user_id: str) -> str:
    """user_id → 安全的文件名（只留 [0-9A-Za-z._@-]，其余换成 _）"""
    return re.sub(r"[^0-9A-Za-z._@-]", "_", str(user_id or "")).strip("._") or "local"


def _migrate_legacy() -> None:
    """一次性迁移：把老的单文件画像复制给 DEFAULT_USER_ID（不删原文件）。"""
    global _migrated
    if _migrated:
        return
    _migrated = True
    try:
        target = PROFILE_DIR / f"{_safe_user(DEFAULT_USER_ID)}.json"
        if target.exists() or not LEGACY_PROFILE_PATH.is_file():
            return
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEGACY_PROFILE_PATH, target)
        print(f"[画像] 存量 {LEGACY_PROFILE_PATH.name} 已归给用户 "
              f"{DEFAULT_USER_ID}（原文件保留，可回滚）")
    except OSError as e:
        print(f"[画像] 存量迁移失败（忽略）：{type(e).__name__}: {e}")


def profile_path() -> Path:
    """当前用户的画像文件路径（单文件模式下就是 PROFILE_PATH）。"""
    if PROFILE_PATH is not None:
        return PROFILE_PATH
    _migrate_legacy()
    return PROFILE_DIR / f"{_safe_user(get_current_user())}.json"

# 顶层字段 → 默认值。preferences 是自由字典，其余三个有固定类型。
DEFAULT_PROFILE = {
    "target_cities": [],
    "target_keywords": [],
    "resume_id": None,
    "preferences": {},
}

LIST_FIELDS = ("target_cities", "target_keywords")
ALLOWED_KEYS = tuple(DEFAULT_PROFILE.keys())


def _empty_profile() -> dict:
    """返回一份全新的默认画像（深拷贝，避免调用方改到模块级默认值）"""
    return {
        "target_cities": [],
        "target_keywords": [],
        "resume_id": None,
        "preferences": {},
    }


def _norm_list(value) -> list:
    """把任意值归一成去重、去空、保序的字符串列表"""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        # 支持 "广州,深圳" / "广州、深圳" / "广州 深圳" 这类写法
        parts = re.split(r"[,，、;；/\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = [value]

    seen, result = set(), []
    for item in parts:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _norm_scalar(value):
    """resume_id 这类单值字段：空串/None 统一成 None，其余转字符串"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_profile(data) -> dict:
    """把任意 dict 规整成合法画像（未知字段丢弃，类型不对就修）"""
    profile = _empty_profile()
    if not isinstance(data, dict):
        return profile

    profile["target_cities"] = _norm_list(data.get("target_cities"))
    profile["target_keywords"] = _norm_list(data.get("target_keywords"))
    profile["resume_id"] = _norm_scalar(data.get("resume_id"))

    prefs = data.get("preferences")
    if isinstance(prefs, dict):
        profile["preferences"] = {
            str(k): v for k, v in prefs.items() if str(k).strip() and v is not None
        }
    return profile


def load_profile() -> dict:
    """读整份画像。

    文件不存在 → 返回默认画像（不建文件，第一次 save 时才落盘）；
    文件损坏（JSON 坏了/不是对象）→ 打印警告并返回默认画像，
    绝不因为画像坏了让 Agent 起不来。
    """
    path = profile_path()
    if not path.exists():
        return _empty_profile()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[画像] 读取失败（忽略，按默认画像处理）：{type(e).__name__}: {e}")
        return _empty_profile()

    return normalize_profile(raw)


def save_profile(profile) -> dict:
    """整份写入画像（父目录不存在就建），返回落盘后的内容"""
    normalized = normalize_profile(profile)
    path = profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再替换：中途崩了也不会留下半截 JSON 把画像读废
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)
    return normalized


def update_preference(key: str, value) -> dict:
    """改一个字段并落盘，返回更新后的整份画像。

    key 支持：
    - target_cities / target_keywords / resume_id（顶层字段）
    - preferences.<名字>（写进 preferences 字典，如 preferences.salary_min）
    - 其他任意 key：同样落进 preferences（Agent 传新偏好时不用先改代码）
    此外 preferences 整体传 dict 时会合并进现有 preferences。
    """
    key = str(key or "").strip()
    if not key:
        raise ValueError("preference key 不能为空")

    profile = load_profile()

    if key == "preferences" and isinstance(value, dict):
        profile["preferences"].update({str(k): v for k, v in value.items()})
        return save_profile(profile)

    if key in LIST_FIELDS:
        profile[key] = _norm_list(value)
        return save_profile(profile)

    if key == "resume_id":
        profile["resume_id"] = _norm_scalar(value)
        return save_profile(profile)

    # 顶层字段名写错（如 target_city）时也落进 preferences，避免静默丢失
    pref_key = key.split(".", 1)[1] if key.startswith("preferences.") else key
    if not pref_key:
        raise ValueError(f"preference key 不合法：{key!r}")
    profile["preferences"][pref_key] = value
    return save_profile(profile)


def get_preferences() -> dict:
    """读全部画像（React 工具 get_preferences 用，就是 load_profile 的别名）"""
    return load_profile()


def save_preference(key, value) -> dict:
    """React 工具 save_preference 的实现：写一个偏好并给出可读回执"""
    profile = update_preference(key, value)
    return {
        "saved": True,
        "key": key,
        "value": value,
        "profile": profile,
        "hint": "已记住该偏好，之后的搜索/匹配会自动带上。",
    }


def profile_to_prompt(profile: dict = None) -> str:
    """把画像压成一段进 system prompt 的文本；画像为空时返回空串。

    空画像不注入任何内容——免得每次对话都塞一段「暂无偏好」的废话占 token。
    """
    profile = normalize_profile(profile if profile is not None else load_profile())
    lines = []

    if profile["target_cities"]:
        lines.append(f"- 目标城市：{'、'.join(profile['target_cities'])}")
    if profile["target_keywords"]:
        lines.append(f"- 偏好关键词：{'、'.join(profile['target_keywords'])}")
    if profile["resume_id"]:
        lines.append(f"- 惯用简历：{profile['resume_id']}")
    for key, value in profile["preferences"].items():
        lines.append(f"- {key}：{value}")

    if not lines:
        return ""

    return (
        "【用户长期偏好（跨会话记住的画像）】\n"
        + "\n".join(lines)
        + "\n用户没特别说明时，搜索/匹配默认沿用上面的偏好；"
        "用户提出新的稳定偏好（如「我只找广州的」「以后都用产品岗版简历」）时，"
        "调用 save_preference 记住它。"
    )


if __name__ == "__main__":
    # 自测：走一遍 保存 → 读回 → 改偏好（默认写临时文件；加 --in-place 才动真实画像）
    import tempfile

    if "--in-place" not in sys.argv:
        PROFILE_PATH = Path(tempfile.mkdtemp(prefix="profile_selftest_")) / "user_profile.json"
    print(f"画像文件：{PROFILE_PATH}")
    print("初始画像：", load_profile())

    save_profile({
        "target_cities": ["广州", "广州", ""],
        "target_keywords": "Agent, 大模型",
        "preferences": {"salary_min": "200/天"},
    })
    print("写入后：", json.dumps(load_profile(), ensure_ascii=False))

    update_preference("preferences.house_type", "实习")
    update_preference("target_cities", ["广州", "深圳"])
    profile = load_profile()
    print("改偏好后：", json.dumps(profile, ensure_ascii=False))

    assert profile["target_cities"] == ["广州", "深圳"], profile
    assert profile["target_keywords"] == ["Agent", "大模型"], profile
    assert profile["preferences"]["salary_min"] == "200/天", profile
    assert profile["preferences"]["house_type"] == "实习", profile
    assert "目标城市：广州、深圳" in profile_to_prompt(profile)

    # 坏文件兜底
    PROFILE_PATH.write_text("{坏 JSON", encoding="utf-8")
    assert load_profile() == _empty_profile(), "坏文件应回退成默认画像"
    print("自测通过（含坏文件兜底）")
