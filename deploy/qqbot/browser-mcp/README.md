# 两个群共用的浏览器 MCP

使用一个官方 Playwright MCP 容器，两个 MaiBot 实例分别通过 Streamable HTTP 连接 `http://maibot-browser-mcp:8931/mcp`。服务只加入已有的 `maibot-two-groups_bot` Docker 网络，不发布宿主机端口。镜像固定到已验证的摘要。

在服务主机保存 `compose.yaml` 后运行 `docker compose -f compose.yaml up -d browser`。将 `bot_config.overlay.toml` 中的服务条目追加到每个实例已有的 `mcp.servers` 中，保留原有服务。浏览器无需模型 API key，Exa 与 GPT 搜索继续使用各自的配置。

`--isolated` 让不同 MCP 客户端使用独立的浏览器上下文；未启用共享上下文。两个群的页面与 Cookie 分开，同一 MaiBot 进程中的主、子 Agent 仍共用该进程的 MCP 连接，同群浏览器任务按提示词依次执行。浏览器上下文在连接结束或服务重启后清除。

工具由 MCP 目录提供，包括 `browser_navigate`、`browser_wait_for`、`browser_snapshot`、`browser_evaluate`、标签页与截图。打开页面后等待正文出现，再从最新快照或可见 DOM 读取内容。网站的登录要求、访问权限和内容状态按实际页面处理。

Pi 会话固定 System、工具与执行绑定。给已有实例增加 MCP 服务时，需要停止该实例的唯一会话写入者，使用 Pi SessionManager 追加更新后的 `maibot.contract`（完整工具列表与绑定），保留原有文件前缀，再启动服务。仅修改 TOML 后直接重启会引发 `SESSION_CONTRACT_CHANGED`。
