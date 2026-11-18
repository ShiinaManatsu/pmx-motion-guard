# PMX Motion Guard

显示名称和发布包统一使用 PMX Motion Guard。扩展内部 ID 沿用 pmx_retarget_guard，已有安装、偏好设置、工程属性与缓存路径继续复用。

Blender 5.3 / Windows x64 的 PMX 动画穿插修正插件。

## 使用

通过 Blender 偏好设置中的从磁盘安装选择 release 目录中的 ZIP。3D 视图 N 面板打开 Motion Guard，选择 PMX 主网格，点击分析模型。检查骨骼映射和身体材质，再选择当前帧或动画范围处理。

偏好设置提供持久缓存路径、DeepSeek 地址、模型和密码显示的 API key。默认缓存位于 Blender DATAFILES/pmx_retarget_guard。映射同时保存在网格对象属性与磁盘指纹缓存。标准 PMX 使用本地识别，DeepSeek 可补充缺失映射。

骨骼模式在原骨架上输出 Motion Guard · correction NLA Replace 层。已有活动 Action 自动推入 Original motion 层，原有 NLA 轨道保留。Motion Guard · baseline 层记录源动画未覆盖的静态属性，保证静音修正层后恢复原姿态。修正片段内同时覆盖骨骼约束影响值，避免已烘焙约束重复求值。面板切换按钮控制修正层静音。表面模式输出可播放 PC2 网格缓存与独立烘焙骨架。导出 Alembic 保存最终网格形变，FBX 保存配套骨架动画。时间轴帧数沿用 Blender 工程。

结果对象的切换按钮控制原始与修正对象的视口显示。渲染或导出前选择所需对象及可见性。缓存路径随工程保存，移动工程时需携带缓存目录或已导出的 Alembic。

## 当前求解行为

0.2 将采集、独立求解、动画写入分为三个阶段。Blender 主线程采集最终骨骼矩阵，独立 Python 进程加载 NPZ 数组和角色包进行数值计算，最后批量写入 Action 并生成 NLA 片段。处理中可以取消，求解进程会随任务取消终止。各阶段分别估算时间。

骨骼模式使用固定角色包和连续求解器，包含最多五帧前瞻。输出以已求值的原始全部骨骼矩阵为基础，通过语义区域修正还原骨骼局部动画，移除结果骨架约束。当前支持完整缩放继承、局部位置及旋转继承的骨架，其他继承设置在处理前报告。代理骨架与原 PMX 的独立位移存在近似，采集报告记录 FK 误差。

表面模式采集已求值顶点至磁盘映射数组。独立进程逐帧运行姿态局部求解和表面压缩，直接写 PC2，并生成附件骨架数据。原始网格动画、形态键和修改器形变通过采集保留，拓扑需要固定。表面路径保留逐帧求解和修正时间滤波。

独立进程默认使用 Blender 自带 Python，需要 NumPy。偏好设置可指定其他 Python。任务目录包含 input.json、input.npz、character.npz、result.npz、worker.log 和报告。表面模式另存 vertices.npy、bases.npy、surface.npz 与 motion.pc2。已有采集数据可使用以下命令重新求解。

```powershell
& 'D:/Games/SteamLibrary/steamapps/common/Blender/5.3/python/bin/python.exe' ./offline_worker.py '<任务缓存目录>'
```

Alembic 与 FBX 导出通过 Blender 导出器运行。独立表面求解直接生成 PC2。

6390 帧 full-test.blend 的后台测试总用时 43.08 秒，采集 19.01 秒，独立求解 6.70 秒，骨骼通道写入 14.89 秒。抽查四帧的输出骨架矩阵最大元素误差为 2.09e-6。实际交互会受到工程修改器、插件和视口求值开销影响。连续序列当前使用一个有状态求解器，默认整段处理。

Alembic 记录每帧最终网格，FBX 包含全部烘焙骨骼。FBX 重导入测试使用 anim_offset=0，Blender FBX 导入器默认偏移为 1 帧。

长动作的采集数据和网格缓存会占用磁盘空间。已有结果保留为独立对象，便于比较多组参数。骨骼修正以已求值动画为输入，采集前关闭需要排除的实时物理效果。

## 来源

Retarget Guard 上游提交 37c3f63e3ab8ae1c098749929b4e1da7f6431c08，MIT 许可证见 vendor/RETARGET_GUARD_LICENSE。Windows 移植包含 DLL 导出与单调计时兼容。发布包包含预编译 x64 DLL。
