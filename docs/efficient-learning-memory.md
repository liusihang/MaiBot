# 学习、记忆与表情包检索改造

本改造面向 Pi 单 Agent 群聊链路，不恢复 Planner → Replyer 双阶段生成。
学习是从聊天中提取带来源的资料并检索使用，不修改模型权重。

## 验收目标

1. 表达库真正进入 Pi 回复上下文。按群配置和当前语境提供少量可选表达，
   由主 Agent 决定是否自然使用，不要求套用口头禅。
2. 表达与黑话共享一批真实消息和一次候选提取调用。累计 100 条新消息，
   或最早待处理消息等待 30 分钟，先到先处理；计时到期不依赖下一条消息。
3. 表达审核使用来源消息本身、前 10 条及后 10 条实际消息。
   这是最多 21 条记录，其中 20 条为邻近上下文；窗口不足明确标注。
   来源作者与引用作者必须区分，机器人、工具结果和记忆注入不成为学习来源。
4. 人物记忆保留来源和更正关系，区分长期事实与临时状态，优先复用已有字段。
5. 摘要、事件和画像复用可追溯证据，减少对已分类材料的重复模型处理。
   一段摘要不一定是一个事件，不能仅凭段落数量跳过事件识别。
6. 表情包通过 embedding 查询返回最多 3 个真实候选，主 Agent 明确选择后发送。
   无候选或选择无效时返回明确结果，不随机发送，也不再调用表情选择子 Agent。

## 公开记忆实践的参考边界

- [ChatGPT Memory](https://help.openai.com/en/articles/8590148-memory-in-chatgpt)
  公开说明记忆可更正、删除及控制；没有公开内部抽取调度或数据库结构。
- [Mem0 Add](https://docs.mem0.ai/core-concepts/memory-operations/add) 与
  [Update](https://docs.mem0.ai/core-concepts/memory-operations/update)
  区分新增与显式更新。不能假定每次新增自动完成所有冲突消解。
- [Letta Memory Blocks](https://docs.letta.com/v1-sdk/memory/memory-blocks) 与
  [Archival Memory](https://docs.letta.com/v1-sdk/memory/archival-memory)
  展示小型常驻记忆和按需检索历史的分层思路。
- [LangGraph Memory](https://docs.langchain.com/oss/python/concepts/memory)
  讨论事实集合、画像、后台更新及频率的权衡。

本项目借鉴这些公开原则，不声称复制 GPT 内部实现，不引入外部记忆托管服务，
不把群聊上传到上述项目。

## 验证边界

使用离线测试和模拟模型回执验证调度、来源、作用域、错误处理与恢复。
真实聊天中的表达适切性、记忆准确率、embedding 检索质量和费用变化需要部署后
在获准的数据上评估，测试通过不代表这些指标已实测改善。

本 PR 不授权合并、部署、替换线上配置或进行收费模型调用。
