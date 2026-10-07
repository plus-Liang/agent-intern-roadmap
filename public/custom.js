/* 侧边栏会话菜单加「置顶 / 取消置顶」—— Chainlit 2.12 的 DOM 注入。（.chainlit/config.toml: custom_js）
 *
 * 为什么只能注入：Chainlit 没有菜单扩展点，官方唯一的口子是 custom_js。
 * DOM 锚点（chainlit/frontend/dist/assets/index-*.js，编译产物里的字面量）：
 *   - 会话容器： f.jsx(att,{id:`thread-${G.id}`})        => id="thread-<threadId>"
 *   - 「…」按钮： id:"thread-options"                     => 全页多份、同名 id
 *   - 菜单项：   id:"rename-thread" / "delete-thread"     => 注入点（插在「重命名」前）
 *
 * 两个坑（都来自 Radix 的 DropdownMenu）：
 *   1. 菜单内容是 forceMount 的 —— 全页几十份同名 id，document.querySelector 拿到的是
 *      第一份（别的会话的），所以**必须**从当前这个「…」按钮出发去定位它那一份；
 *   2. 菜单内容可能被 Portal 到 body 下（trigger 和内容不在同一棵子树里），所以
 *      不能靠 parentElement 找兄弟，要用 trigger 的 aria-controls 反查内容元素的 id
 *      （Radix 在打开时才有 aria-controls，正好也当成「菜单已打开」的信号）。
 *
 * 触发时机：MutationObserver 盯 body，只在 data-state="open" 时注入 —— 不打开就不插，
 * 免得给全页每个会话都塞一个按钮；插入前先查 [data-dsh-pin]，同一份菜单不重复插。
 *
 * 点击后：POST /api/threads/{pin,unpin}（同源 cookie 认证，未登录 401），成功后
 * location.reload() —— 顺序完全由服务端（agent/data_layer.py 的 get_all_user_threads）
 * 决定，前端不自己重排，刷新一下最省事也最可靠。
 */
(function () {
  "use strict";

  var PIN_URL = "/api/threads/pin";
  var UNPIN_URL = "/api/threads/unpin";
  var PINNED_URL = "/api/threads/pinned";
  var ITEM_FLAG = "data-dsh-pin";
  var LABEL_PIN = "📌 置顶";
  var LABEL_UNPIN = "📌 取消置顶";

  //: 已置顶的 thread_id 集合；null = 还没问到服务端（先按「未置顶」显示）。
  var pinned = null;
  var scheduled = false;
  var pinnedAsked = false;

  /** 从「…」按钮反查它属于哪条会话：最近的祖先 [id^="thread-"]。
   *  注意要先走 parentElement —— 按钮自己的 id 就是 "thread-options"，
   *  直接 closest() 会先匹配到自己，取出来就是 "options"。 */
  function threadIdOf(trigger) {
    var parent = trigger.parentElement;
    var box = parent && parent.closest ? parent.closest('[id^="thread-"]') : null;
    if (!box || box.id.length <= "thread-".length) return "";
    return box.id.slice("thread-".length);
  }

  /** 定位「…」按钮对应的那一份菜单内容（找不齐就放弃这次注入，下轮 DOM 变化再来）。 */
  function contentOf(trigger, threadId) {
    var contentId = trigger.getAttribute("aria-controls");
    var content = contentId ? document.getElementById(contentId) : null;
    if (content) return content;

    // 兜底 1：菜单内容没被 Portal 出去时，它和按钮在同一个会话容器里。
    var box = document.getElementById("thread-" + threadId);
    if (box) {
      var rename = box.querySelector('[id="rename-thread"]');
      if (rename && rename.parentElement) return rename.parentElement;
    }

    // 兜底 2：全页正好只开着一个菜单时，那份打开着的内容就是它的。
    var openTriggers = document.querySelectorAll('[id="thread-options"][data-state="open"]');
    var openContents = document.querySelectorAll('[data-radix-menu-content][data-state="open"]');
    if (openTriggers.length === 1 && openContents.length === 1) return openContents[0];
    return null;
  }

  function makeItem(rename, threadId, label) {
    var item = document.createElement("div");
    item.setAttribute("role", "menuitem");
    item.setAttribute("tabindex", "-1");
    item.setAttribute(ITEM_FLAG, "1");
    item.className = rename.className;      // 复制「重命名」那一项：样式 / 深浅色 / 间距才对得上
    item.textContent = label;
    item.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      toggle(item, threadId, label);
    });
    return item;
  }

  function toggle(item, threadId, label) {
    if (item.dataset.busy) return;
    item.dataset.busy = "1";
    item.textContent = "处理中…";
    var willPin = !(pinned && pinned.has(threadId));

    fetch(willPin ? PIN_URL : UNPIN_URL, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: threadId })
    }).then(function (res) {
      if (res.ok) {
        location.reload();                  // 顺序交给服务端，整页刷新
        return null;
      }
      return res.json().catch(function () { return {}; }).then(function (data) {
        throw new Error((data && data.detail) || ("HTTP " + res.status));  // 401 / 上限 / 会话不存在
      });
    }).catch(function (err) {
      delete item.dataset.busy;
      item.textContent = label;
      window.alert("置顶操作失败：" + (err && err.message ? err.message : err));
    });
  }

  /** 扫描全页：把「打开着」的菜单补上「置顶 / 取消置顶」。 */
  function sweep() {
    var triggers = document.querySelectorAll('[id="thread-options"]');
    for (var i = 0; i < triggers.length; i++) {
      var trigger = triggers[i];
      if (trigger.getAttribute("data-state") !== "open") continue;
      var threadId = threadIdOf(trigger);
      if (!threadId) continue;
      var content = contentOf(trigger, threadId);
      if (!content) continue;
      var rename = content.querySelector('[id="rename-thread"]');
      if (!rename || !rename.parentElement) continue;

      var label = (pinned && pinned.has(threadId)) ? LABEL_UNPIN : LABEL_PIN;
      var existing = content.querySelector("[" + ITEM_FLAG + "]");
      if (existing) {                       // 幂等：同一份菜单只插一次，只更新文案
        if (!existing.dataset.busy && existing.textContent !== label) existing.textContent = label;
        continue;
      }
      content.insertBefore(makeItem(rename, threadId, label), rename);
    }
  }

  function schedule() {
    if (scheduled) return;
    scheduled = true;
    setTimeout(function () {
      scheduled = false;
      loadPinned();                       // 侧边栏冒出来了才去问（见 loadPinned 的说明）
      sweep();
    }, 50);                               // 攒一下再扫，别每个节点一次
  }

  /** 问一次「哪些会话已置顶」（只问一次，用来决定文案是「置顶」还是「取消置顶」）。
   *
   *  只在**侧边栏已经渲染出来**之后才发这个请求：登录页上还没有任何 [id^="thread-"]，
   *  那会儿发出去必然 401，浏览器控制台会留一条 Failed to load resource: 401 的红字。
   *  没登录本来也看不到菜单，等侧边栏出现再问最合适。 */
  function loadPinned() {
    if (pinnedAsked) return;
    if (!document.querySelector('[id^="thread-"]')) return;      // 登录页：先不问
    pinnedAsked = true;
    fetch(PINNED_URL, { credentials: "same-origin" }).then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status);          // 没登录 / 会话过期：当成「都没置顶」
      return res.json();
    }).then(function (data) {
      pinned = new Set((data && data.thread_ids) || []);
      sweep();
    }).catch(function () {
      pinned = new Set();
    });
  }

  function start() {
    if (!document.body) {                 // custom_js 可能在 <head> 里就跑了
      document.addEventListener("DOMContentLoaded", start);
      return;
    }
    new MutationObserver(schedule).observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-state", "aria-controls"]
    });
    loadPinned();
    sweep();
  }

  start();
})();
