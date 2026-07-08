#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
import math
from collections import defaultdict
import time

class TrackState:
    def __init__(self, track_id):
        self.track_id = track_id
        # 深蹲
        self.squat_phase = 'up'
        self.squat_count = 0
        self.squat_total_time = 0.0
        self.squat_start_time = None

        # 箭步蹲
        self.lunge_phase = 'up'
        self.lunge_count = 0
        self.lunge_total_time = 0.0
        self.lunge_start_time = None
        self.lunge_down_pts = None
        self.lunge_stable_frames = 0

        # 开合跳（套用原推举逻辑）
        self.jj_state = 'down'          # 'down'=手臂未举，'up'=手臂举起
        self.jj_count = 0
        self.jj_total_time = 0.0
        self.jj_cycle_start = None      # 手臂举起的开始时间

        # 俯卧撑
        self.pushup_phase = 'up'
        self.pushup_count = 0
        self.pushup_total_time = 0.0
        self.pushup_start_time = None

        # 平板支撑
        self.plank_active = False
        self.plank_start_time = None
        self.plank_count = 0
        self.plank_total_time = 0.0

        # 哑铃弓步蹲
        self.dlunge_stable_frames = 0
        self.dlunge_start_time = None
        self.dlunge_count = 0
        self.dlunge_total_time = 0.0

        # 防抖
        self.last_triggered_action = None
        self.last_trigger_time = 0.0
        self.action_cooldown = 1.0

        # 质量记录
        self.quality_records = defaultdict(list)

class BodyPoseSubscriber(Node):
    def __init__(self):
        super().__init__('body_pose_subscriber')
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_mono2d_body_detection',
            self.listener_callback,
            10
        )
        self.track_states = {}
        self.get_logger().info('多动作识别+评分节点已启动')

    def listener_callback(self, msg):
        current_ids = set()
        now = time.time()
        for target in msg.targets:
            tid = target.track_id
            current_ids.add(tid)
            points_holder = target.points
            keypoints = None
            for pset in points_holder:
                if 'body_kps' in pset.type.lower():
                    keypoints = pset.point
                    break
            if keypoints is None or len(keypoints) < 17:
                continue
            pts = [(p.x, p.y) for p in keypoints[:17]]
            if tid not in self.track_states:
                self.track_states[tid] = TrackState(tid)
            state = self.track_states[tid]
            action, duration, score, issues = self._recognize_exclusive(state, pts, now)
            if action:
                if action != state.last_triggered_action:
                    if now - state.last_trigger_time > state.action_cooldown:
                        state.last_triggered_action = action
                        state.last_trigger_time = now
                        state.quality_records[action].append((score, issues))
                        self._handle_action(state, action, duration, score, issues)
            else:
                state.last_triggered_action = None
        self._cleanup_stale_targets(current_ids)

    def _cleanup_stale_targets(self, current_ids):
        for tid in list(self.track_states.keys()):
            if tid not in current_ids:
                state = self.track_states[tid]
                now = time.time()
                if state.plank_active and state.plank_start_time is not None:
                    dur = now - state.plank_start_time
                    state.plank_total_time += dur
                    state.plank_count += 1
                if state.lunge_start_time is not None:
                    state.lunge_start_time = None
                    state.lunge_down_pts = None
                    state.lunge_phase = 'up'
                if state.dlunge_start_time is not None:
                    dur = now - state.dlunge_start_time
                    state.dlunge_total_time += dur
                    state.dlunge_count += 1
                # 开合跳若中途中断，丢弃该次
                state.jj_state = 'down'
                state.jj_cycle_start = None

                summary = []
                for action in ['深蹲','箭步蹲','开合跳','俯卧撑','平板支撑','哑铃弓步蹲']:
                    count = getattr(state, f'{self._action_attr(action)}_count', 0)
                    total_time = getattr(state, f'{self._action_attr(action)}_total_time', 0.0)
                    if count > 0:
                        recs = state.quality_records.get(action, [])
                        if recs:
                            scores = [r[0] for r in recs]
                            avg_score = sum(scores) / len(scores)
                            min_score = min(scores)
                            issue_counter = defaultdict(int)
                            for r in recs:
                                for iss in r[1]:
                                    issue_counter[iss] += 1
                            issues_str = ', '.join(f'{k}({v}次)' for k,v in issue_counter.items())
                            quality_str = f'均分:{avg_score:.0f} 最低:{min_score}'
                            if issues_str:
                                quality_str += f' 常见问题:{issues_str}'
                            summary.append(f'{action} {count}次 {total_time:.1f}s {quality_str}')
                        else:
                            summary.append(f'{action} {count}次 {total_time:.1f}s')
                if summary:
                    self.get_logger().info(f'目标 {tid} 已消失，运动记录: {", ".join(summary)}')
                else:
                    self.get_logger().info(f'目标 {tid} 已消失，无有效运动记录')
                del self.track_states[tid]

    def _action_attr(self, action):
        mapping = {
            '深蹲': 'squat', '箭步蹲': 'lunge', '开合跳': 'jj',
            '俯卧撑': 'pushup', '平板支撑': 'plank', '哑铃弓步蹲': 'dlunge'
        }
        return mapping.get(action, action)

    def _angle(self, p1, p2, p3):
        v1 = (p1[0]-p2[0], p1[1]-p2[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm = math.hypot(*v1) * math.hypot(*v2)
        if norm == 0: return 180
        return math.degrees(math.acos(max(-1, min(1, dot/norm))))

    def _body_y_std(self, pts):
        l_sho, r_sho = pts[5], pts[6]
        l_hip, r_hip = pts[11], pts[12]
        l_kne, r_kne = pts[13], pts[14]
        l_ank, r_ank = pts[15], pts[16]
        y_vals = [l_sho[1], r_sho[1], l_hip[1], r_hip[1], l_kne[1], r_kne[1], l_ank[1], r_ank[1]]
        mean = sum(y_vals) / len(y_vals)
        return (sum((y-mean)**2 for y in y_vals) / len(y_vals)) ** 0.5

    def _evaluate_quality(self, action, pts):
        """
        严格评分：从100分开始，对每个不规范之处扣分，最低0分。
        返回 (score, issues)，issues 是字符串列表。
        """
        score = 100
        issues = []

        l_sho, r_sho = pts[5], pts[6]
        l_elb, r_elb = pts[7], pts[8]
        l_wri, r_wri = pts[9], pts[10]
        l_hip, r_hip = pts[11], pts[12]
        l_kne, r_kne = pts[13], pts[14]
        l_ank, r_ank = pts[15], pts[16]

        mid_sho = ((l_sho[0]+r_sho[0])/2, (l_sho[1]+r_sho[1])/2)
        mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
        shoulder_width = abs(l_sho[0] - r_sho[0])

        dx = mid_sho[0] - mid_hip[0]
        dy = mid_sho[1] - mid_hip[1]
        if dy != 0:
            trunk_angle = abs(math.degrees(math.atan2(dx, -dy)))
        else:
            trunk_angle = 90

        def angle(p1, p2, p3):
            return self._angle(p1, p2, p3)

        if action == '深蹲':
            left_knee = angle(l_hip, l_kne, l_ank)
            right_knee = angle(r_hip, r_kne, r_ank)
            avg_knee = (left_knee + right_knee) / 2
            if avg_knee > 100:
                score -= 25
                issues.append(f"下蹲幅度不足(膝角{avg_knee:.0f}°)")
            if trunk_angle > 25:
                score -= 20
                issues.append(f"躯干过度前倾({trunk_angle:.0f}°)")
            knee_width = abs(l_kne[0] - r_kne[0])
            ank_width = abs(l_ank[0] - r_ank[0])
            if ank_width > 0 and knee_width / ank_width < 0.8:
                score -= 15
                issues.append("膝盖内扣")

        elif action == '箭步蹲':
            if l_ank[0] < r_ank[0]:
                front_knee = angle(l_hip, l_kne, l_ank)
                back_knee = angle(r_hip, r_kne, r_ank)
            else:
                front_knee = angle(r_hip, r_kne, r_ank)
                back_knee = angle(l_hip, l_kne, l_ank)
            if front_knee < 80 or front_knee > 110:
                score -= 25
                issues.append(f"前腿膝角异常({front_knee:.0f}°，应为80-110°)")
            if back_knee < 170:
                score -= 20
                issues.append(f"后腿未伸直({back_knee:.0f}°，应>170°)")
            if trunk_angle > 15:
                score -= 20
                issues.append(f"躯干前倾({trunk_angle:.0f}°，应<15°)")
            hip_center_x = (l_hip[0] + r_hip[0]) / 2
            foot_center_x = (l_ank[0] + r_ank[0]) / 2
            if abs(hip_center_x - foot_center_x) > 30:
                score -= 10
                issues.append("重心偏移")

        elif action == '开合跳':
            # 只检查手臂质量（腿是否张开不再扣分）
            hands_high = l_wri[1] < l_sho[1] - 15 and r_wri[1] < r_sho[1] - 15
            left_straight = angle(l_sho, l_elb, l_wri) > 140
            right_straight = angle(r_sho, r_elb, r_wri) > 140
            if not (hands_high and left_straight and right_straight):
                score -= 10
                issues.append("手臂上举不充分或未伸直")

        elif action == '俯卧撑':
            y_std = self._body_y_std(pts)
            if y_std > 25:
                score -= 25
                issues.append("身体未呈直线")
            left_elbow = angle(l_sho, l_elb, l_wri)
            right_elbow = angle(r_sho, r_elb, r_wri)
            avg_elbow = (left_elbow + right_elbow) / 2
            if avg_elbow > 90:
                score -= 20
                issues.append(f"下放幅度不足(肘角{avg_elbow:.0f}°)")
            hand_width = abs(l_wri[0] - r_wri[0])
            if hand_width < shoulder_width * 0.8 or hand_width > shoulder_width * 2.0:
                score -= 10
                issues.append("双手间距异常")

        elif action == '平板支撑':
            y_std = self._body_y_std(pts)
            if y_std > 20:
                score -= 30
                issues.append("腰部塌陷或拱起")
            horiz_delta_l = abs(l_elb[0] - l_sho[0])
            horiz_delta_r = abs(r_elb[0] - r_sho[0])
            if horiz_delta_l > shoulder_width * 0.15 or horiz_delta_r > shoulder_width * 0.15:
                score -= 20
                issues.append("肘部未在肩正下方")

        elif action == '哑铃弓步蹲':
            if l_ank[0] < r_ank[0]:
                front_knee = angle(l_hip, l_kne, l_ank)
                back_knee = angle(r_hip, r_kne, r_ank)
            else:
                front_knee = angle(r_hip, r_kne, r_ank)
                back_knee = angle(l_hip, l_kne, l_ank)
            if front_knee < 80 or front_knee > 110:
                score -= 20
                issues.append(f"前腿膝关节角度异常({front_knee:.0f}°)")
            if back_knee < 160:
                score -= 10
                issues.append("后腿未伸直")
            if trunk_angle > 20:
                score -= 15
                issues.append(f"躯干前倾({trunk_angle:.0f}°)")
            left_straight = angle(l_sho, l_elb, l_wri) > 160
            right_straight = angle(r_sho, r_elb, r_wri) > 160
            if not (left_straight and right_straight):
                score -= 15
                issues.append("手臂未伸直上举")

        score = max(0, score)
        return score, issues

    def _recognize_exclusive(self, state, pts, now):
        l_sho, r_sho = pts[5], pts[6]
        l_elb, r_elb = pts[7], pts[8]
        l_wri, r_wri = pts[9], pts[10]
        l_hip, r_hip = pts[11], pts[12]
        l_kne, r_kne = pts[13], pts[14]
        l_ank, r_ank = pts[15], pts[16]

        hip_y = (l_hip[1] + r_hip[1]) / 2.0
        sho_y = (l_sho[1] + r_sho[1]) / 2.0
        shoulder_width = abs(l_sho[0] - r_sho[0])

        # ---- 地面动作优先 ----
        y_std = self._body_y_std(pts)
        body_horiz = y_std < 30 and abs(sho_y - hip_y) < 40

        # 俯卧撑
        if body_horiz:
            avg_elbow = (self._angle(l_sho,l_elb,l_wri) + self._angle(r_sho,r_elb,r_wri))/2
            if avg_elbow < 90 and state.pushup_phase == 'up':
                state.pushup_phase = 'down'
                state.pushup_start_time = now
            elif avg_elbow > 150 and state.pushup_phase == 'down' and state.pushup_start_time is not None:
                dur = now - state.pushup_start_time
                state.pushup_total_time += dur
                state.pushup_count += 1
                state.pushup_start_time = None
                state.pushup_phase = 'up'
                score, issues = self._evaluate_quality('俯卧撑', pts)
                return '俯卧撑', None, score, issues
        else:
            state.pushup_phase = 'up'
            state.pushup_start_time = None

        # 平板支撑
        elbow_low = (l_elb[1] > l_sho[1] + 20 or r_elb[1] > r_sho[1] + 20)
        is_plank = body_horiz and elbow_low
        if is_plank:
            if not state.plank_active:
                state.plank_active = True
                state.plank_start_time = now
        else:
            if state.plank_active:
                if state.plank_start_time is not None:
                    dur = now - state.plank_start_time
                    if dur >= 2.0:
                        state.plank_total_time += dur
                        state.plank_count += 1
                        state.plank_active = False
                        state.plank_start_time = None
                        score, issues = self._evaluate_quality('平板支撑', pts)
                        return '平板支撑', dur, score, issues
                state.plank_active = False
                state.plank_start_time = None

        # ---- 站立动作特征 ----
        hands_high = (l_wri[1] < l_sho[1] - 30 and r_wri[1] < r_sho[1] - 30)
        elbows_ext = (self._angle(l_sho,l_elb,l_wri) > 150 and self._angle(r_sho,r_elb,r_wri) > 150)
        hands_wide = abs(l_wri[0] - r_wri[0]) > shoulder_width * 1.8
        legs_wide = abs(l_ank[0] - r_ank[0]) > shoulder_width * 1.5

        # 弓步特征
        foot_dist = abs(l_ank[0] - r_ank[0])
        if l_ank[0] < r_ank[0]:
            fh,fk,fa = l_hip,l_kne,l_ank
            bh,bk,ba = r_hip,r_kne,r_ank
        else:
            fh,fk,fa = r_hip,r_kne,r_ank
            bh,bk,ba = l_hip,l_kne,l_ank
        f_knee_ang = self._angle(fh,fk,fa)
        b_knee_ang = self._angle(bh,bk,ba)
        mid_sho = ((l_sho[0]+r_sho[0])/2, (l_sho[1]+r_sho[1])/2)
        mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
        dx, dy = mid_sho[0]-mid_hip[0], mid_sho[1]-mid_hip[1]
        vert_ang = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 90

        is_lunge = (
            foot_dist > 120 and
            70 < f_knee_ang < 120 and
            b_knee_ang > 160 and
            vert_ang < 20 and
            abs(l_hip[1] - r_hip[1]) < 30
        )

        # ===== 动作优先级（开合跳最高） =====
        # 1. 开合跳（套用原站姿哑铃推举相位，但不要求腿是否张开，仅防止与弓步冲突）
        if hands_high and elbows_ext and not is_lunge:
            if state.jj_state == 'down':
                state.jj_state = 'up'
                state.jj_cycle_start = now
        else:
            if state.jj_state == 'up' and state.jj_cycle_start is not None:
                dur = now - state.jj_cycle_start
                state.jj_total_time += dur
                state.jj_count += 1
                state.jj_cycle_start = None
                state.jj_state = 'down'
                score, issues = self._evaluate_quality('开合跳', pts)
                return '开合跳', None, score, issues

        # 2. 哑铃弓步蹲
        if is_lunge and hands_high and elbows_ext:
            if state.dlunge_start_time is None:
                state.dlunge_start_time = now
                state.dlunge_stable_frames = 0
            state.dlunge_stable_frames += 1
            if state.dlunge_stable_frames >= 30:
                dur = now - state.dlunge_start_time
                state.dlunge_total_time += dur
                state.dlunge_count += 1
                state.dlunge_start_time = now
                state.dlunge_stable_frames = 0
                score, issues = self._evaluate_quality('哑铃弓步蹲', pts)
                return '哑铃弓步蹲', dur, score, issues
        else:
            state.dlunge_start_time = None
            state.dlunge_stable_frames = 0

        # 3. 箭步蹲（纯弓步，无举臂）
        if is_lunge and not hands_high:
            state.lunge_stable_frames += 1
        else:
            state.lunge_stable_frames = 0

        if state.lunge_stable_frames >= 5:
            if state.lunge_phase == 'up':
                state.lunge_phase = 'down'
                state.lunge_start_time = now
                state.lunge_down_pts = pts.copy()
        else:
            if state.lunge_phase == 'down':
                if state.lunge_start_time is not None:
                    dur = now - state.lunge_start_time
                    if dur >= 0.5:
                        state.lunge_total_time += dur
                        state.lunge_count += 1
                        if state.lunge_down_pts is not None:
                            score, issues = self._evaluate_quality('箭步蹲', state.lunge_down_pts)
                        else:
                            score, issues = 0, ["未保存姿态"]
                        state.lunge_phase = 'up'
                        state.lunge_start_time = None
                        state.lunge_down_pts = None
                        return '箭步蹲', dur, score, issues
                    else:
                        state.lunge_start_time = None
                        state.lunge_down_pts = None
                state.lunge_phase = 'up'

        # 4. 深蹲（最后触发）
        other_active = (
            state.jj_state == 'up' or          # 开合跳激活时阻止深蹲
            state.dlunge_start_time is not None or
            state.lunge_phase == 'down' or
            state.plank_active
        )
        if not other_active and not hands_high and not hands_wide and not legs_wide:
            feet_narrow = abs(l_ank[0] - r_ank[0]) < shoulder_width * 1.5
            upright = vert_ang < 25
            squat_left = l_kne[1] > l_hip[1] + 50
            squat_right = r_kne[1] > r_hip[1] + 50
            is_squat = squat_left and squat_right and feet_narrow and upright
            if is_squat and state.squat_phase == 'up':
                state.squat_phase = 'down'
                state.squat_start_time = now
            elif not is_squat and state.squat_phase == 'down' and state.squat_start_time is not None:
                dur = now - state.squat_start_time
                state.squat_total_time += dur
                state.squat_count += 1
                state.squat_start_time = None
                state.squat_phase = 'up'
                score, issues = self._evaluate_quality('深蹲', pts)
                return '深蹲', None, score, issues
        else:
            state.squat_phase = 'up'
            state.squat_start_time = None

        return None, None, 0, []

    def _handle_action(self, state, action, duration, score, issues):
        issues_str = ', '.join(issues) if issues else ''
        score_str = f'{score}分' if score > 0 else ''
        if issues_str:
            score_str += f' {issues_str}'
        extra = f' 本次 {duration:.1f}s' if duration is not None else ''
        self.get_logger().info(f'目标 {state.track_id} {action} [{score_str}]{extra}')

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
