# 起步模板使用边界

- `label_studio_ready.xml`：Label Studio 的三视角决策帧分类配置草稿；一个任务对应一只手臂的一种候选事件。
- `tasks.example.json`：结构示例。没有真实媒体或人工标签，不能直接作为训练数据。
- 完整设计见 `../LEROBOT_ANNOTATION_PLAN.md`。

部署适配器之后，将 JSON 中媒体路径替换成标注服务实际可访问的 URL，并将 UUID、frame_index、evidence 范围替换为统一帧索引中的值，再导入任务。相对 `/media/` 路径在这里是占位符，不是 Label Studio 已配置的存储接口。

三张图必须来自同一决策时刻的相机映射；相机缺失时单独记录 view mask，不用黑图冒充完整观测。若 DSE 使用历史或本体状态，添加截止到 t 的历史媒体/状态字段，并修改配置显示相同的信息。当前模板仅适用于当前帧试标。

分类结果绑定到 cam_high 是 Label Studio 的结果归属约定；标签语义使用三路视角共同判断，导出时将它归属整个 decision，而不是解释为只对头部图有效。

导出时用保留的 sample_id 查回原始索引，读取 readiness/reasons/note，生成 decisions 表中的新修订。未提交任务保持 missing；显式 unknown 保留为 unknown；没有复核的结果不能标为 adjudicated。当前未实现该导入导出程序。

已完成的检查限于 XML/JSON 语法及字段对应；尚未在 Label Studio 运行时验证配置，也未解码或上传任何视频。
