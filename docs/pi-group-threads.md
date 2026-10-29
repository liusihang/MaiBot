# Pi 持续群线程（新 PR，未部署）

## 基线与实现边界

基于 `9b9e953262cabc75457ffae75272833eee145fe0`，保留人物事实十分钟抽取；合入未合并 PR #1 的 `ca54ba79286e2d9bdf501609ed5d710bc8e3849e`。本分支不是线上容器版本，不修改 PR #1，不表示已合并或部署。

实际循环来自官方 [earendil-works/pi](https://github.com/earendil-works/pi) 的 `@earendil-works/pi-agent-core` **1.4.0**。Node sidecar 使用真实 `Agent`，不是重新命名 Python while loop。Python 提供原有模型路由、工具和持久化；不启用 Pi coding CLI 的 shell/任意文件默认工具。模型和人格配置未替换；群聊仍由 Luna 判断入口，主 Agent 直接提供 `reply.text`，普通回复不交第二个模型重写。

## 启用前置条件（不是“打开开关即上线”）

1. Node.js ≥ 22.19；在仓库根目录运行 `npm ci --prefix runtime/pi_agent --ignore-scripts`。Dockerfile 已加入锁定 npm 依赖构建阶段；本次未实际构建/部署 Docker 镜像。
2. `chat.pi_agent_enabled = true`（新配置默认值）。关闭时显式使用旧循环；不会在 Pi 缺失、预算未知或 replay 不兼容时偷偷退回旧实现。
3. **核实实际接入端容量与计数合约**。当前模板未声明经核实的上下文窗口，Pi 的严格准入因此会暂停请求。这是部署阻塞条件，不是模型质量问题；不能填写从模型名字猜出的数值。
4. 对精确 provider / endpoint / model 配置 `models.request_admission`：能力来源 `capability_source`、共享 `context_window`（或独立输入/输出上限）、输出参数是否包含 reasoning、必要余量，以及可信全 payload `input_counter`。最终 HTTP 请求的输入上界 + 最大输出 + 余量必须满足声明的容量。固定鉴权账号/租户也属于能力核实范围；更换凭据或租户须重新核实，transport hash 不证明租户能力。缓存命中不会减少计入的输入。
5. 原生 opaque reasoning、图片及隐式 gateway 上下文不能用字符数冒充精确 token。内置 UTF-8 bytes 上界只接受已声明可用的纯文本协议；这不适用于当前含 opaque replay 的部署路线。需要由接入端实现并注册权威计数/上界适配器，覆盖工具 schema、原生状态和多模态全部输入。
6. 当前模板有 `X-OpenAI-Internal-Codex-Responses-Lite` 自定义路由 header；此类路线还必须核实并声明精确 transport 合约。注册计数器并核实后，填写 `transport_contract_fingerprint`：用 `calculate_transport_fingerprint` 对精确 route 及非认证路由元数据生成 SHA256 合约值；不会自动填入，凭据不写入日志或传给计数器。修改实际 route/header/query 后旧能力声明不得继续复用。未知路线、插件 RPC/ACP 或无法验证的计量不得声称已满足容量保证。

保留现用 Terra high / 25600 和 Luna max；不要通过降低输出、换模型、关闭 reasoning 或删旧历史绕过拒绝。

## 历史与缓存

- 使用 host 真实 `session_id` 隔离持久 SQLite 主线程；压缩前保持已提交内容及顺序。新批次、时间、参考资料仅追加尾部。
- System/工具声明保持稳定。模型返回的原始 ContextItems、provider scope、opaque replay bytes 与调度用 Pi projection 分开；发送下一次请求时用原始 items，而非根据 projection 重造。
- 正在工具执行的群消息留在 host 队列，于协议完整边界追加。Luna 的 ignore 仅禁止启动主模型，不禁止消息入账。
- 自动参考绕过旧会话 TTL 渲染缓存，仅依据新批次、参与者和现有范围约束检索。每条来源/版本单独去重；更新追加 supersedes，查无结果不删除旧历史。
- 默认最多 3 条 / 2400 字的自动参考是保守初始值，未用真实聊天确认最佳数量。未知来源/其他群原文拒绝自动注入。首版只做可靠身份去重，不做可能混淆否定和时间的语义“同事件”合并；详细往事仍可显式查。
- 缓存友好指稳定前缀与实际请求结构，不承诺服务端命中率或费用下降。

## 工具、子线程与失败

工具按顺序执行，assistant/tool 事件持久化 ACK 后才推进。执行前保存意图；已确认回执可恢复；崩溃或取消后的未知副作用不会自动再执行。必须先核对外部真实结果再恢复，不能凭错误字符串推断“未发送”。有副作用工具的失败若没有 host 可信的“未执行”证明，会保守暂停；少数纯参数/前置校验失败也可能因此需要核实，而不是自主重试。取消导致孤儿调用时线程暂停，不能继续发送不完整的 provider 协议。

复杂任务可调用 `subagent`；普通闲聊/短查询不强制委派。子线程只获得明确只读、host 验证的能力。中间过程保留在子线程；父线程得到结论与持久 evidence handle，使用 `read_subagent` 分页回查公开来源/工具原文。内部 native reasoning/replay 不作为公开资料交给父模型。

压缩是显式维护事务：完整工具回合结束后暂停 Pi，摘要成功且版本仍可提交才替换较早历史；system/工具声明保持不变，原文可归档回查。摘要失败/取消不改变旧头；并发后缀保留。SQLite 提交后 sidecar ACK 不确定必须暂停并从账本恢复，不能把旧状态写回覆盖新状态。压缩后注入去重与仍可见的原文版本对齐，不永久封锁主动读取。

## 验证与仍需授权的工作

离线测试使用真实安装的 Pi Agent + 模拟 provider/tool，不调用付费模型、不发送 QQ 消息。覆盖原生 replay 完整性、下一次请求 prefix、搜索→阅读→回复、双群隔离、Luna 跳过、来源版本增量、子线程回查、工具未知结果、摘要失败/取消/并发和最终请求准入。具体最终结果以 PR 中的执行记录为准。

已知边界：为保持旧消息引用目标可用，host `_chat_history` 镜像目前仍驻内存增长；模型上下文和持久账本通过显式压缩管理。新增工具声明会增量追加；当前压缩器不会跨越中途的工具声明，遇到该边界会安全暂停，后续需专门的声明保留压缩支持。归档读取当前提供公开文本/工具原文，原生图片仍保存在账本中，但尚无压缩后图片本体重注入工具。本文启用步骤针对源码/Docker；wheel 打包与运行环境另需验证。

未验证：真实聊天质量/最佳自动参考预算、当前 gateway 全 payload token 计数、真实服务端缓存命中、线上 Docker、QQ 实际投递。上线前需按实际消息到达顺序回放失败聊天，固定模型/人格/记忆快照，不使用未来消息或后来学习的记忆；完成容量合约后再另行授权部署或群内测试。

## 本次离线执行记录

- `pytest pytests/maisaka pytests/test_request_admission.py pytests/test_provider_stop_reason.py pytests/A_memorix_test/test_person_fact_batches.py`：579 项集成回归通过（12 个既有弃用警告）；最后冻结工具合约增量另跑 runner 回归，结果见 PR。
- `npm test --prefix runtime/pi_agent`：21 项通过，包含实际 Node JSONL 子进程。
- `ruff check`（新增 Pi 模块与测试、准入模块与测试）及 `git diff --check`：通过。
- 旧循环 fixture 显式关闭 Pi 以继续验证 legacy；direct reply 测试关闭随机错别字副作用使其确定性验证正文不由另一个模型改写。
