# 内置中文字体

`wqy-microhei.ttc` —— 文泉驿微米黑（WenQuanYi Micro Hei），随本仓库分发，
供 `agent/tools/pdf_export.py` 生成中文 PDF 简历时内嵌使用。

## 为什么必须随仓库带一份

精简容器镜像（`python:3.12-slim`）里**一个中文字体都没有**，也装不了
reportlab/fpdf2。只靠系统字体探测时 `find_cjk_font()` 返回 `None`，
渲染降级到 Helvetica（Base14 + WinAnsi），所有中文都变成 `?`。

## 来源与许可

- 上游：<https://github.com/anthonyfok/fonts-wqy-microhei>（文泉驿官方打包镜像）
- 许可：**GPL v3 + 字体例外条款**（font exception）——
  字体本体可随任何程序分发，不传染该程序自身代码。
- 已核对覆盖度：GB2312 一/二级汉字 + ASCII + 常用中文标点共 **6990 字符，缺 0 个**。
- 体积 5.2 MB；生成的 PDF 只内嵌**实际用到的字形子集**（`_subset_font`），
  单份简历 PDF 约 20~60 KB，与字体本体大小无关。

## 其它可用字体

`PDF_CJK_FONT` 环境变量可指向任意 TrueType 字体覆盖内置字体
（必须是 TrueType 轮廓，`.ttf`/`.ttc`；**不支持 CFF/OTF**，
因为 `agent/tools/pdf_export.py` 的纯标准库渲染器自带的是 TrueType 解析器）。
