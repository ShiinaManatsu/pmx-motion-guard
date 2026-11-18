# DeepSeek 映射试验

2026-11-18，通过用户提供的临时凭据访问官方 API。GET /models 返回 deepseek-flash 与 deepseek-v4-pro。本次使用 deepseek-flash。

## 实际结果

- 自由映射的两组输出存在错误。fufu 的骨盆及脊柱选择偏离既有映射。长离把肩部映射到手腕，重复骨骼 ID，并出现左右映射错误。
- 固定本地规则已确定的 22 个角色后，两组输出均保留锁定映射。骨骼 ID 存在性和唯一性检查通过。长离的 spine3 保持 null，返回上半身2到颈部中点的虚拟关节建议。
- 最后一轮 fufu 用量为 26906 输入、471 输出 token。长离为 9856 输入、773 输出 token。正式请求应缩减到歧义区域与必要祖先，提高复用率。
- 返回文本也存在细节错误，例如将 21 个非空角色描述为 20 个，以及对虚拟关节提出额外确认措辞。插件应解析受校验的结构化字段，虚拟关节坐标由本地数据计算。

## 传输与参数

Python urllib 请求遇到 TLS EOF。PowerShell 请求可成功完成。首轮默认 thinking 消耗全部 2500 个完成 token，未返回可用映射。显式 thinking.type=disabled 后得到有效 JSON。连接曾发生一次强制关闭，后续请求成功。

测试输出与模型骨架数据保存在被 Git 忽略的 test_outputs 和 mapping_test_input.json。仓库脚本从环境变量 DEEPSEEK_API_KEY 读取凭据。实际成功请求使用 PowerShell，Python 客户端传输兼容性仍需处理。

## 验证范围

这是接口连通性和两种已知 PMX 骨架的映射试验。自动未知骨架识别、映射缓存、NLA 烘焙、进度界面和 Alembic 导出尚处设计阶段。
