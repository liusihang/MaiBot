# Decisions 群聊参与入口

是否交给主 Agent 现在默认使用 **OpenAI Decisions API**，不是让 chat completion
输出一个 JSON。已锁定的 `@earendil-works/pi-coding-agent@1.4.0` 实际含有
`ModelRuntime.classify()`，其 `@earendil-works/pi-ai` 原生 `openai-decisions`
执行 `POST {base_url}/decisions`。Python 只整理业务证据、调用 Pi RPC 和记账。
主 Agent 仍是每群独立、持续的 Pi AgentSession，自主决定工具和 `reply`；没有
Planner → Replyer，也没有入口生成的立场、口吻或回复计划。

## 配置与兼容

在 bot 配置的 `[chat]` 中（未指定时也是下列默认值）：

```toml
[chat]
participation_mode = "decisions" # 显式设为 "chat" 保留旧 Pi 结构化聊天判断
participation_model = "group-helper" # model_config 中已有别名
participation_timeout_seconds = 30
```

`fast_model` 任务与 `participation_model` 别名复用现有路由/鉴权。Decisions 当前只
接受精确 `model_identifier="gpt-6-luna"`，provider 必须是 `openai` 或
`openai_responses`；其他模型/协议请求前明确失败，不猜别名、不切模型。
同一 Luna 别名的记忆、媒体描述等 chat 调用仍走原来 API，classifier 注册是独立的。

**模型名正确不代表网关支持 Decisions。** 配置的 base URL 必须实际提供
`/decisions`，一般为 `https://api.openai.com/v1` 或明确支持该协议的现有代理。
官方端点需要 OpenAI API key；ChatGPT 登录凭据不等于 Decisions API 权限。
需要固定出口时，可在参与判断的专属模型中设置
`extra_params.pi.decisionsProxyUrl="http://proxy-host:port"`。它只代理该模型的
Decisions 请求，不改进程环境变量、其他模型或工具的网络路由。每次请求结束或
取消后关闭代理连接；API 密钥仍通过到官方端点的 TLS 隧道发送。
404、401/403 等会明确记录错误并把本批交给主 Agent，不冒充判断成功。
仓库改动不替换实际凭据、不部署、不进行收费模型调用；启用前需在自己的账户/网关
确认可用性。缺少配置字段的旧配置也将采用 `decisions` 默认值；需要保留原行为应
显式选择 `chat`，不能依赖自动回退。

chat 的 reasoning、temperature、stream、工具、max_tokens 等字段不会传给
Decisions；未识别的 body 参数明确拒绝。现有 `strict` 容量策略在无法证明分类
操作容量时拒绝，不伪装成 chat 的输出 token 预算证明；默认 `audit` 不承诺硬上限。

## 请求与判断边界

- 一个 `choice` 问题 `group_participation`，可选值严格为
  `engage` / `ignore` / `uncertain`。`confidence` 和三项概率保留为诊断，不添加未校准阈值。
- 参与策略来自中英日对齐模板；保留原政治话题等约束。近期 24 条历史、本批消息、
  真实作者/机器人标记、源时间/接收时间、引用对象和当前判断时间都放在 user 的
  `input_json.data` 中。群友文字或 JSON 中的 `role`/`instructions` 不是控制指令。
- 保留原有单消息 1600 字符的首尾节选；不是摘要写回，不替代源消息。本批不按条数
  砍尾。`input_json` 超过官方 32 容器层、10000 值或 256 KiB 时，请求前报错，
  交给主 Agent；不会为了通过 API 验证而丢消息。
- 明确 @、提及、引用机器人、已有任务、焦点模式等仍直接进入主 Agent；普通批次
  保持每群最多每 30 秒启动一次判断。
- 只有合法 `ignore` 阻止本次主 Agent 执行；`engage` 和 `uncertain` 均允许主 Agent
  再决定，包括主动 stop。拒答 `type: refusal`、响应畸形、网络/HTTP 错误和超时
  是**失败**而非 ignore，也不切回 chat。日志/阶段显示安全错误码和 HTTP 状态，
  不记录上游错误正文、URL 查询串或密钥。主动取消继续向上传播。

源消息依旧先持久入库并追加 Pi 线程，再判断是否执行；负判断不会删除源材料。
判断时又到新批次时，既有循环追加新消息并放行主 Agent，不将旧 ignore 用于新批。
冷却、会话恢复、原生压缩和后台工具回执逻辑未改。

## 缓存、计数与错误

原生 Pi 1.4 的 Decisions 序列化早于 GA JSON/cache 字段；这里用官方 `onPayload`
hook 补 `input_json` 和可选参数，共享 provider fetch 继续负责路由、鉴权、禁止跳转
和取消。不会实现另一条 Python HTTP 路径。所有请求 `maxRetries=0`。

`model.decide` RPC 的可选 `options` 为 `timeoutMs`、`promptCacheKey`、
`promptCacheOptions`、`safetyIdentifier`。缓存参数也可放在模型
`extra_params.body` 的官方 snake_case 字段中。cache mode 支持 `implicit`/`explicit`，
TTL 仅 `30m`；key ≤64 字符，safety identifier ≤128 字符。未设置 key 时使用真实
会话 ID 的摘要分区，不把 QQ 群号/原文作为缓存键。默认服务端隐式缓存；不人为重写
主线程或追加缓存断点。显式 cache key 的复用由配置者负责，缓存不是会话历史。

回执保留经过字段白名单清洗的 `providerUsage`，另规范为 Pi
`input`（未缓存部分）、`cacheRead`、`cacheWrite`、`output`、`totalTokens`。
业务账本将三种 input 相加恰好一次。合法拒答/答案畸形仍保留上游真实 usage，
标记失败；HTTP/网络失败没有 token 回执时不估造 token。
现有业务费用统计使用配置模型单价，不是 OpenAI 发票；别名同时用于 chat 时，
不能据此声称已精确区分 Decisions 费率或 cache-write 费用。本变更不修改线上单价。

## 离线验证

```sh
npm --prefix runtime/pi_agent ci --ignore-scripts --no-audit --no-fund
npm --prefix runtime/pi_agent test
uv run pytest -q pytests/maisaka/test_participation.py pytests/services/test_pi_harness_bridge.py
```

Node 测试使用真实 Pi SDK 和 localhost HTTP，断言 `/v1/decisions`、请求结构、
数据注入边界、三态/拒答、严格响应、缓存记账、401/403/404/429/5xx、无重试、
超时/取消、群隔离、请求前校验和密钥清洗；Python 验证业务路由、持久输入和
判断期间新消息的保留。它们不证明真实账户开通、代理支持或分类质量。

官方依据（核对日期 2026-10-30）：
[Decisions guide](https://developers.openai.com/api/docs/guides/decisions) ·
[Create a decision](https://developers.openai.com/api/reference/resources/decisions/methods/create)。
