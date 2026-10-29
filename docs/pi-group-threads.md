# Pi 持续群线程（完整 harness，未部署）

本次重构使用官方 `@earendil-works/pi-coding-agent` **1.4.0** 的
`createAgentSession`、`AgentSession`、`SessionManager`，而不是只使用底层
`Agent` 并回调 Python 模型。详细合同见：

- [运行与持久化协议](../runtime/pi_agent/README.md)
- [全入口迁移、验证覆盖与诚实边界](pi-harness-migration.md)

## 当前行为

每个真实群/私聊单独持久 SDK 会话；Focus 只调度，不合并模型历史。System、工具目录
与原有模型/人格保持固定；旧内容、顺序、原生 reasoning/replay、调用/结果和参考在
成功显式压缩前不改写。新消息、时间、来源版本只追加；重复消息按全部 archive 去重，
参考按当前可见版本去重。Luna 忽略不意味着丢掉消息，人物事实仍按十分钟批次抽取。

所有仓库内模型调用统一到 Node Pi 官方 provider 或明确的非原生能力适配器。
Python 只保留业务源、记忆/画像/黑话检索、工具权限和 QQ outbox 回执，不再保存另一套
权威模型历史。普通回复直接发送主 Agent 的 `reply.text`，不经过 Planner→Replyer。
默认 coding read/bash/edit/write、工作区资源提示与语义任务规划字段均不注入。

显式 SDK 压缩只有成功同步落盘才替换可见历史；自动高水位维护只能压已完成旧前缀，
新请求/图片/参考原文尾保留。失败、取消和并发入站不裁丢。摘要附带 scoped archive
句柄；工具未知副作用保持暂停，不自动重发。子 Agent 仅获明确只读能力，结论附主线程
可读取的公开原始来源，隐藏推理/签名不作为证据外泄。

## 运行条件与预算

- Linux、Node ≥22.19.0、util-linux `flock`、可靠本地持久文件系统与 fsync。
- `npm ci --prefix runtime/pi_agent --ignore-scripts` 安装精确锁定依赖。
- `pi_agent_enabled` 仅保留旧配置解析兼容；关闭它不会恢复旧 Python 模型循环。
- 默认 `audit` 允许未知容量，但**不保证硬上下文上限**。显式 `strict` 仍严格拒绝
  未知证明或已知超限，不伪造容量/计数器，不 clamp 输出、不换模型、不无变化重试。
- 保留 Terra high / 25600 与 Luna max；未知窗口使用 SDK 的 `contextWindow:0`。

旧版 Python Pi-adapter 历史不会静默伪造成 SDK 原生历史；保留旧数据供审查。
实际配置文件、凭据与线上服务未变更。Docker 依赖已更新，但未声称完成生产镜像验证。

## 外部能力边界

仓库中的 Qwen 图片 embedding 插件已使用权限化 Pi capability；实际生产 Gemini
绘图插件全源码缺失，补丁合同不是完整 pipeline 迁移或上线证明。外部 ACP 代理内部
模型调用不由 Pi 代管。没有真实模型自然度、缓存命中或 QQ 投递验证，不表示可直接上线。

保留 `9b9e953262cabc75457ffae75272833eee145fe0` 与 PR #1
`ca54ba79286e2d9bdf501609ed5d710bc8e3849e` 祖先；不修改 PR #1，不 merge/deploy。
