"""
投递追踪管理器。
封装状态机和存储，提供统一 API。
"""
from agent import storage
from agent.state_machine import (
    validate_transition,
    get_status_label,
    is_terminal,
    STATUS,
)


def init():
    """初始化数据库"""
    storage.init_db()


def add_application(company, title, platform="", url="", notes=""):
    """添加投递记录"""
    app_id = storage.create_application(company, title, platform, url, notes)
    return app_id


def change_status(app_id: str, to_status: str, note: str = ""):
    """更新状态（会校验合法性）"""
    app = storage.get_application(app_id)
    if not app:
        raise ValueError(f"未找到记录：{app_id}")

    from_status = app["status"]
    validate_transition(from_status, to_status)
    storage.update_status(app_id, to_status, note)

    return (
        f"{app['company']} | {app['title']}："
        f"{get_status_label(from_status)} → {get_status_label(to_status)}"
    )


def show(app_id: str):
    """展示单条记录及时间线"""
    app = storage.get_application(app_id)
    if not app:
        print(f"未找到：{app_id}")
        return

    print(f"\n{'='*60}")
    print(f"【{app['company']}】{app['title']}")
    print(f"{'='*60}")
    print(f"平台：{app['platform']}")
    print(f"投递时间：{app['applied_at']}")
    print(f"当前状态：{get_status_label(app['status'])}")
    print(f"下次跟进：{app['next_follow_up'] or '未设置'}")
    print(f"备注：{app['notes'] or '无'}")
    print(f"\n状态时间线：")
    for e in storage.get_events(app_id):
        from_label = get_status_label(e["from_status"]) if e["from_status"] else "创建"
        to_label = get_status_label(e["to_status"])
        note = f"（{e['note']}）" if e["note"] else ""
        print(f"  {e['created_at']}  {from_label} → {to_label} {note}")


def list_all(status: str = None):
    """列出所有记录"""
    apps = storage.list_applications(status)
    if not apps:
        print("（无记录）")
        return

    print(f"\n{'='*70}")
    print(f"{'ID':<10}{'公司':<15}{'岗位':<20}{'状态':<10}")
    print(f"{'-'*70}")
    for a in apps:
        print(
            f"{a['id']:<10}{a['company']:<15}"
            f"{a['title'][:18]:<20}{get_status_label(a['status']):<10}"
        )


def stats():
    """统计各状态数量"""
    apps = storage.list_applications()
    counts = {s: 0 for s in STATUS}
    for a in apps:
        counts[a["status"]] += 1

    print(f"\n总投递数：{len(apps)}")
    print(f"{'='*40}")
    for status, label in STATUS.items():
        print(f"{label:<10} {counts[status]}")

    # 转化率
    if len(apps) > 0:
        viewed_count = sum(
            counts[s] for s in ["viewed", "interview", "interviewing", "offer", "accepted"]
        )
        interview_count = sum(
            counts[s] for s in ["interview", "interviewing", "offer", "accepted"]
        )
        print(f"\n简历被看率：{viewed_count / len(apps) * 100:.1f}%")
        print(f"约面率：{interview_count / len(apps) * 100:.1f}%")


if __name__ == "__main__":
    init()
    print("Tracker 初始化完成")