"""
投递状态机。
定义状态、合法转换、状态变更逻辑。
"""

# 所有状态
STATUS = {
    "applied": "已投递",
    "viewed": "HR已读",
    "interview": "约面中",
    "interviewing": "面试中",
    "offer": "已发Offer",
    "accepted": "已接受",
    "rejected": "被拒",
    "withdrawn": "主动放弃",
}

# 终态（不能再流转）
TERMINAL_STATES = {"accepted", "rejected", "withdrawn"}

# 合法转换：从某状态可以转到哪些状态
TRANSITIONS = {
    "applied": ["viewed", "rejected", "withdrawn"],
    "viewed": ["interview", "rejected", "withdrawn"],
    "interview": ["interviewing", "rejected", "withdrawn"],
    "interviewing": ["offer", "rejected", "withdrawn"],
    "offer": ["accepted", "rejected", "withdrawn"],
    "accepted": [],
    "rejected": [],
    "withdrawn": [],
}


def can_transition(from_status: str, to_status: str) -> bool:
    """判断从 from_status 转到 to_status 是否合法"""
    if from_status not in TRANSITIONS:
        return False
    return to_status in TRANSITIONS[from_status]


def get_status_label(status: str) -> str:
    """状态中文名"""
    return STATUS.get(status, status)


def is_terminal(status: str) -> bool:
    """是否终态"""
    return status in TERMINAL_STATES


def validate_transition(from_status: str, to_status: str):
    """校验转换，非法则抛异常"""
    if from_status not in STATUS:
        raise ValueError(f"未知状态：{from_status}")
    if to_status not in STATUS:
        raise ValueError(f"未知状态：{to_status}")
    if not can_transition(from_status, to_status):
        raise ValueError(
            f"非法转换：{get_status_label(from_status)} → {get_status_label(to_status)}"
        )


if __name__ == "__main__":
    # 测试合法转换
    print("已投递 → HR已读：", can_transition("applied", "viewed"))
    print("已投递 → Offer：", can_transition("applied", "offer"))

    # 测试非法转换
    try:
        validate_transition("applied", "offer")
    except ValueError as e:
        print("预期报错：", e)

    # 测试终态
    print("accepted 是终态：", is_terminal("accepted"))