#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
import math
from collections import deque, defaultdict
import time

class BodyPoseSubscriber(Node):
    def __init__(self):
        super().__init__('body_pose_subscriber')
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_mono2d_body_detection',
            self.listener_callback,
            10
        )
        # 每个目标的历史记录（动作队列）
        self.action_history = {}  # track_id -> deque of actions
        # 每个目标上次触发动作的时间（用于切换滞后）
        self.last_trigger_time = {}  # track_id -> timestamp
        # 每个目标上次触发的动作
        self.last_triggered_action = {}
        # 窗口长度
        self.history_len = 10
        # 动作切换最小间隔（秒），防止频繁切换
        self.switch_cooldown = 2.0

        # 用于减少“缺少关键点”警告的频率
        self.missing_keypoints_count = defaultdict(int)
        self.last_warn_time = time.time()
        self.warn_interval = 5.0  # 每5秒打印一次汇总

        self.get_logger().info('人体姿态订阅已启动（优化版），等待检测结果...')

    def listener_callback(self, msg):
        current_ids = set()
        now = time.time()

        # --- 1. 收集所有有效目标 ---
        valid_targets = []
        for target in msg.targets:
            track_id = target.track_id
            current_ids.add(track_id)

            # 提取关键点
            points_holder = target.points
            keypoints = None
            for pset in points_holder:
                if 'body_kps' in pset.type.lower():
                    keypoints = pset.point
                    break
            if keypoints is None or len(keypoints) < 17:
                self.missing_keypoints_count[track_id] += 1
                continue  # 跳过该目标

            # 转为 (x,y) 列表
            pts = [(p.x, p.y) for p in keypoints[:17]]
            # 可选：对坐标进行简单平滑（移动平均），但这里先不引入额外状态
            valid_targets.append((track_id, pts))

        # --- 2. 定期打印缺失统计（避免刷屏） ---
        if now - self.last_warn_time > self.warn_interval:
            if self.missing_keypoints_count:
                total_missing = sum(self.missing_keypoints_count.values())
                self.get_logger().warn(f'近 {self.warn_interval}s 内共有 {total_missing} 次缺失关键点（涉及 {len(self.missing_keypoints_count)} 个目标）')
            self.missing_keypoints_count.clear()
            self.last_warn_time = now

        # 如果没有有效目标，清理过期记录并返回
        if not valid_targets:
            self._cleanup_stale_targets(current_ids)
            return

        # --- 3. 只保留一个目标（可选：选择最近出现的） ---
        # 按 track_id 排序，选择最小的（或你可以改为选择最先出现的）
        valid_targets.sort(key=lambda x: x[0])
        track_id, pts = valid_targets[0]  # 只处理第一个目标
        # 如果希望处理所有目标，可以注释掉本段，并用循环处理

        # --- 4. 动作识别 ---
        action = self._recognize_action(pts)

        # --- 5. 防抖 + 切换滞后 ---
        if track_id not in self.action_history:
            self.action_history[track_id] = deque(maxlen=self.history_len)
            self.last_trigger_time[track_id] = now

        self.action_history[track_id].append(action)

        # 判断是否达到稳定状态（队列满且全部相同）
        hist = self.action_history[track_id]
        if len(hist) == self.history_len and all(a == hist[0] for a in hist):
            stable_action = hist[0]
            if stable_action is not None:
                # 检查是否与上次触发相同，且冷却时间已过
                last_action = self.last_triggered_action.get(track_id)
                last_time = self.last_trigger_time.get(track_id, 0)
                if (last_action != stable_action) and (now - last_time > self.switch_cooldown):
                    # 触发新动作
                    self.last_triggered_action[track_id] = stable_action
                    self.last_trigger_time[track_id] = now
                    self.get_logger().info(f'目标 {track_id} 稳定动作: {self._action_name(stable_action)}')
                    self._handle_action(track_id, stable_action)

        # --- 6. 清理消失的目标 ---
        self._cleanup_stale_targets(current_ids)

    def _cleanup_stale_targets(self, current_ids):
        """清除不在画面中的目标记录"""
        for tid in list(self.action_history.keys()):
            if tid not in current_ids:
                self.get_logger().info(f'目标 {tid} 已消失')
                del self.action_history[tid]
                if tid in self.last_trigger_time:
                    del self.last_trigger_time[tid]
                if tid in self.last_triggered_action:
                    del self.last_triggered_action[tid]

    def _action_name(self, value):
        mapping = {
            'raise_hand': '举手',
            'wave': '挥手',
            'squat': '蹲下',
            'raise_both': '双手举起',
        }
        return mapping.get(value, value)

    def _handle_action(self, track_id, action):
        if action == 'raise_hand':
            self.get_logger().info(f'[动作] 目标{track_id} 举手 🙋')
        elif action == 'wave':
            self.get_logger().info(f'[动作] 目标{track_id} 挥手 👋')
        elif action == 'squat':
            self.get_logger().info(f'[动作] 目标{track_id} 蹲下 🧎')
        elif action == 'raise_both':
            self.get_logger().info(f'[动作] 目标{track_id} 双手举起 🙌')

    # ========== 动作识别（与之前相同，但阈值可调） ==========
    def _recognize_action(self, pts):
        if len(pts) < 17:
            return None

        # 提取关键点（按COCO索引）
        nose = pts[0]
        left_shoulder = pts[5]
        right_shoulder = pts[6]
        left_elbow = pts[7]
        right_elbow = pts[8]
        left_wrist = pts[9]
        right_wrist = pts[10]
        left_hip = pts[11]
        right_hip = pts[12]
        left_knee = pts[13]
        right_knee = pts[14]
        left_ankle = pts[15]
        right_ankle = pts[16]

        def dist(p1, p2):
            return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

        # 判断手是否高于肩膀（阈值可调，此处为20像素）
        left_hand_high = left_wrist[1] < left_shoulder[1] - 20
        right_hand_high = right_wrist[1] < right_shoulder[1] - 20

        # 判断手臂是否水平（挥手）
        left_arm_horizontal = abs(left_wrist[0] - left_shoulder[0]) > abs(left_wrist[1] - left_shoulder[1]) * 1.5
        right_arm_horizontal = abs(right_wrist[0] - right_shoulder[0]) > abs(right_wrist[1] - right_shoulder[1]) * 1.5

        # 判断蹲下（膝盖低于髋部50像素以上）
        squat_left = left_knee[1] > left_hip[1] + 50
        squat_right = right_knee[1] > right_hip[1] + 50
        squat = squat_left and squat_right

        # 动作判定
        if left_hand_high and right_hand_high:
            return 'raise_both'
        elif left_hand_high and not right_hand_high:
            if left_arm_horizontal:
                return 'wave'
            else:
                return 'raise_hand'
        elif right_hand_high and not left_hand_high:
            if right_arm_horizontal:
                return 'wave'
            else:
                return 'raise_hand'
        elif squat:
            return 'squat'
        else:
            return None

def main(args=None):
    rclpy.init(args=args)
    node = BodyPoseSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
