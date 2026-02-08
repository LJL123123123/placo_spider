import numpy as np
import placo

class SpiderCompensation:
    def __init__(self, robot, leg_foot_name_map):
        """
        初始化 SpiderCompensation
        :param robot: placo.RobotWrapper 实例 (应与 IK 共享)
        :param leg_foot_name_map: 字典，映射腿名到足端 Frame 名 (例如 {'LF': 'LF_foot', ...})
        """
        self.robot = robot
        self.leg_foot_name_map = leg_foot_name_map

        # === 参数配置 ===
        self.mu = 0.8               # 摩擦系数
        self.weight_forces = 1e-4   # 力分布正则化权重 (越小越允许力分布不均，但也越精确满足动力学)
        self.com_weight = 1e1       # CoM 任务权重 (保持重心)
        self.trunk_weight = 1e3     # 躯干姿态任务权重

    def compute(self, support_legs: list, target_com=None, target_rot=None):
        """
        计算动力学补偿
        :param support_legs: 当前处于支撑相的腿名列表 (例如 ['LF', 'RH'])
        :param target_com: (可选) 期望的 CoM 位置 [x, y, z]，默认为当前 CoM
        :param target_rot: (可选) 期望的基座旋转矩阵 (3x3)，默认为当前基座姿态
        :return: (tau_ff, contact_forces)
                 tau_ff: 关节前馈力矩 (numpy array)
                 contact_forces: 字典 {leg_name: [fx, fy, fz]}
        """
        # 1. 创建动力学求解器 (每次计算重新创建以清除旧约束)
        solver = placo.DynamicsSolver(self.robot)
        
        # 屏蔽浮动基的“驱动”，因为它是欠驱动的，必须通过接触力平衡
        # 求解器会自动处理前6维的动力学方程
        solver.mask_fbase(True) 

        # 2. 添加任务：维持身体状态
        # (动力学求解器需要任务来决定‘为了保持什么状态而施加力’)
        
        # A. CoM 任务 (如果未指定目标，则维持当前实际 CoM)
        if target_com is None:
            target_com = self.robot.com_world()
        com_task = solver.add_com_task(target_com)
        com_task.configure("com_stabilize", "soft", self.com_weight)

        # B. 躯干姿态任务 (维持当前姿态或目标姿态)
        if target_rot is None:
            target_rot = self.robot.get_T_world_frame("base_link")[:3, :3]
        ori_task = solver.add_orientation_task("base_link", target_rot)
        ori_task.configure("trunk_stabilize", "soft", self.trunk_weight)

        # 3. 添加接触 (Contacts)
        contact_tasks = [] # 保持引用防止被回收
        active_contacts = {}

        for leg_name in support_legs:
            if leg_name not in self.leg_foot_name_map:
                continue
                
            foot_frame = self.leg_foot_name_map[leg_name]
            
            # 获取当前足端位置 (锁定在地面)
            # 注意：这里使用 robot 当前的运动学位置
            T_world_foot = self.robot.get_T_world_frame(foot_frame)
            foot_pos = T_world_foot[:3, 3]

            # 创建位置任务 (作为接触点的锚点)
            foot_task = solver.add_position_task(foot_frame, foot_pos)
            foot_task.configure(f"{leg_name}_contact_lock", "hard", 1.0)
            contact_tasks.append(foot_task)

            # 创建点接触 (PointContact)
            contact = solver.add_point_contact(foot_task)
            contact.mu = self.mu
            contact.unilateral = True        # 开启单向力约束 (Fz >= 0)
            contact.weight_forces = self.weight_forces # 最小化力的范数 (实现均匀分布)
            
            active_contacts[leg_name] = contact

        # 4. 求解
        # solve(False) 表示只求解力矩和加速度，不进行积分更新状态
        result = solver.solve(False)

        # 5. 提取结果
        tau_ff = np.zeros(self.robot.model.nv - 6) # 默认全0
        contact_results = {}

        if result.success:
            # 提取关节力矩 (去除前6维浮动基)
            # result.tau 包含了满足上述任务 + 重力补偿 + 接触力平衡 所需的总力矩
            tau_ff = result.tau[6:]
            
            # 提取各腿的接触力 (Wrench)
            for leg, contact in active_contacts.items():
                contact_results[leg] = contact.wrench
        else:
            print("[SpiderCompensation] Dynamics solver failed to solve!")
            # 失败时可以使用 fallback: 纯重力补偿 (static gravity)
            # tau_ff = self.robot.generalized_gravity()[6:]

        return tau_ff, contact_results