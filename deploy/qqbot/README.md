# QQ 群聊定制版

这份部署资料对应本仓库的 QQ 单 Agent、群聊参与判断、记忆修复和本地插件。
代码以 MaiBot 1.3.18 为基础，定制差异来自原先基于 1.3.15 的两个群聊实例。

## 核心行为

- 主 Agent 自己理解消息、调用工具并写正文；`reply(msg_id, text)` 直接发送正文，监控记录 `replyer_llm_calls=0`。
- `subagent(task, wait)` 在独立 Pi 线程完成复杂查询或动作，继承同一套工具。主 Agent 自选等待或后台执行；后台完成后结果持久追加并唤醒本群线程。`read_subagent(handle, wait=true)` 可等待已派发任务，原始证据继续按群归属读取。
- 插件明确调用回复生成服务时，保留原来的生成入口和 1.3.18 富文本补充编排。
- `group-helper` 承担语义参与判断，只返回 `engage/ignore/uncertain`；各聊天最多每 30 秒启动一次判断。@、提及、引用机器人和已有工具任务直接放行。冷却中的普通消息留在队列，判断失败交给主 Agent。
- 摘要按连续消息窗口写入，以真实身份和保存回执推进 v2 检查点。显式记忆提供用户原消息作为证据；人物事实不使用引用、转发或机器人回答代替自述。
- 自动人物事实从新接收的用户消息入队，每个聊天最多每 600 秒启动一批，多人合并抽取；单批最多 100 条、约 24000 字符，溢出留待下一窗口。消息去重、空结果和调用时间保存在 `data/person_fact_writeback.sqlite3`，重启继续处理；模型输出先暂存，写入失败复用结果，不再次调用模型。显式 `remember_memory` 仍即时执行，历史文件导入和参与判断间隔保持不变。
- 人物画像由模型选择已有证据 ID，正文取回原证据；撤回事实不重新进入画像，已确认别名与当前昵称一起保留。
- 回复参考资料标记来源；拆句逻辑保留句末问号、感叹号和省略号。是否拆句仍由每个实例自己的配置控制。

## 本地插件

`plugins/` 保存的是可安装的插件源码，不会因为检出本仓库而自动启用。
将需要的目录复制到实例自己的 `plugins/` 目录，再把对应 `config.example.toml`
复制为 `config.toml` 并填写部署参数。

| 目录 | 用途 |
| --- | --- |
| `gpt_web_search` | 默认搜索、搜索续读和互联网图片发送 |
| `qwen3vl_image_provider` | 注册 `qwen3vl_image` Client，连接 Qwen 图片向量服务 |
| `replyer_autonomy` | 单 Agent 工具约定、动态北京时间和表达选择历史 |
| `group_proactive` | 冷场时唤醒现有 Agent；北京时间 00:00–09:00 不主动开题 |

QQ 收发继续使用 [SnowLuma Adapter](https://github.com/Mai-with-u/MaiBot-SnowLuma-Adapter)。
此处不复制未修改的第三方适配器。绘图继续使用
[Gemini Drawer](https://github.com/namchengwuqiu/gemini_drawer)。

`patches/gemini_drawer-image-edit.patch` 保存现用 1.15.7 绘图插件的图生图修复，
包含 `mode=edit`、真实参考消息 ID、按顺序解析多张参考图和多文件 edit 请求。
在对应版本的绘图插件目录先执行 `git apply --check <补丁绝对路径>`，通过后再应用。
保留插件原有 MIT 许可和作者信息。

## 提示词和配置

`custom_prompts/zh-CN/` 保存中文部署提示词，包括主 Agent、参与判断和记忆摘要。
马斯克的身份、关系与参与动机已按角色设计调整，主提示词和参与判断提供英、日文对应版本。
两个群已使用同一份当前提示词与人格配置。`maisaka_replyer.prompt` 供显式生成入口使用，普通群聊正文不经过它。
称呼优先级、按场合决定篇幅、政治话题限制及现有工具能力继续保留。
身份设计、来源与隔离回放的限制见 [ROLEPLAY.md](ROLEPLAY.md)。
内置普通/Focus 提示词的中文、英文和日文工具约定已同步为 `reply(msg_id, text)`。

- `bot_config.overlay.toml`：基础设置；其中旧人格、表达注入与拆句设置由当前 overlay 覆盖。
- `bot_config.group-a.overlay.toml`：当前两个群共用的差异设置，文件保留原名。启用 Pi 与发言，关闭自动拆句，使用当前人格。
- `model_config.template.toml`：主模型 Terra/high/25600，辅助模型 Luna/max，
  Qwen 文本/图片嵌入及各任务分配。
- `model_config.group-a.overlay.toml`：当前两个群的主 Agent 使用同济分组的
  `DeepSeek-V4.1-Flash`，推理 max、输出预算 25600；保留原 Terra 模型项供学习等任务使用，Luna 路由不变。
  每个实例的 `tongji-campus` 使用各自绑定同济分组的专属 key，不改变原 key 的分组。

这些配置文件是部署模板。已有实例应保留原配置中的端点、密钥、账号、群绑定和路径，
按表合并相关设置；不要用 overlay 整文件替换现有配置。
持续 Pi 会话切换模型时，先停止该群写入并保存恢复点，再按新配置解析模型路由，
追加新的契约及路由变更记录。保留原 JSONL 前缀，不能只改配置后删除或重建聊天历史。
DeepSeek 兼容模式把网关返回的 `reasoning` 在请求发送时映射到
`reasoning_content`，保留思考原文和工具配对，不修改 SDK 原生消息。
模型模板中的 `gateway.example.com` 和空 API key 必须替换为自己的服务配置。
`a_memorix.filter.chats` 的 `group:REPLACE_GROUP_ID` 必须替换为该实例的群号。
主动插件的 `group_id`、`session_id`、`bot_user_id` 使用同一实例的实际值。
大果训练营（376080499）已从旧 Python 主循环升级到与没变群相同的 Pi 镜像，并恢复发言。
升级前对该实例的 config、plugins、data 和 depends-data 做停机备份，移除 13 个旧源文件挂载；
聊天数据库、记忆、向量、图片、群绑定和插件配置使用大果群自己的目录。
该实例此前没有 SDK 原生线程，新 Pi 线程在收到首批实际消息时建立，不复制另一群的会话或伪造旧 SDK 历史。
共享 NapCat 和没变群未在这次同步中重启。
模型模板已适配 1.3.18 配置格式；原 EXA 联网服务保留无凭证定义。
如保留 EXA 功能分配，需要填写其服务密钥；仅使用 GPT 搜索插件时可清空该原生功能分配。
旧的私聊频率字段已移除，注意力漂移设置按新版拆为群聊和私聊，关闭状态保持。

自定义提示词复制到实例数据目录的 `custom_prompts/`。
每个群继续使用独立的数据目录，保留聊天历史、记忆、向量、人物画像和插件状态。
源码仓库不包含这些运行数据、凭证或原始群聊。

## A_memorix 补丁归属

按 `src/A_memorix/MODIFICATION_POLICY.md`，本次保留三个小范围本地补丁：
向量库管理动作注册、人物画像证据与撤回处理、画像文本证据选择说明。
它们用于保持现有部署的接口兼容和纠正人物记忆行为，尚未同步到
`A-Dawn/A_memorix` 的 `MaiBot_branch`。后续同步 A_memorix 时需核对这些差异，
避免重新引入已修复的误记或漏注册问题。

## 验证

相关回归位于 `pytests/maisaka/`、`pytests/A_memorix_test/` 和 `pytests/qqbot/`。
测试替换模型与发送边界，覆盖直接正文不调用 Replyer、参与判断冷却、保存回执、
消息证据归属、人物别名和插件消息行为，不向 QQ 发消息。
通过这些测试不代表已经验证长期自然聊天质量或完成生产升级。
