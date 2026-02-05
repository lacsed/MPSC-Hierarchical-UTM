import math
from dataclasses import dataclass

from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import String


def yaw_to_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


def sat(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def move_towards(curr, target, max_delta):
    if curr < target:
        return min(curr + max_delta, target)
    return max(curr - max_delta, target)


def split_suffix_id(ev):
    if "_" not in ev:
        return ev, None
    base, suf = ev.rsplit("_", 1)
    if suf.isdigit():
        return base, int(suf)
    return ev, None


@dataclass
class BatteryModel:
    voltage_nom: float = 22.2
    capacity_Wh: float = 180.0
    i_base: float = 1.8
    i_vgain: float = 2.5
    i_wgain: float = 1.2


class UAVHardware:
    def __init__(
        self,
        node,
        *,
        entity_name,
        agent_id,
        graph_positions,
        set_state_client,
        event_topic="/event",
        speed_mps=3.0,
        vspeed_mps=1.0,
        yaw_rate_max_rps=1.2,
        accel_mps2=2.0,
        waypoint_tol_m=0.25,
        clearance_m=0.0,
        alt_offset_m=0.0,
        work_time_s=2.0,
        charge_time_s=5.0,
        battery_model=None,
        low_batt_threshold=0.40,
        ground_z=0.0,
        g_mps2=9.81,
        terminal_vz_mps=12.0,
        init_pose=None,  # (x,y,z,yaw)
        snap_on_take=True,
    ):
        self.node = node
        self.entity_name = str(entity_name)
        self.agent_id = int(agent_id)
        self.pos = graph_positions
        self.cli_set = set_state_client

        self.pub_event = self.node.create_publisher(String, event_topic, 50)

        self.speed = float(speed_mps)
        self.vspeed = float(vspeed_mps)
        self.yaw_rate_max = float(yaw_rate_max_rps)
        self.accel = float(accel_mps2)
        self.tol = float(waypoint_tol_m)

        self.clearance = float(clearance_m)
        self.alt_offset = float(alt_offset_m)

        self.work_time_s = float(work_time_s)
        self.charge_time_s = float(charge_time_s)

        self.batt = battery_model if battery_model is not None else BatteryModel()
        self.soc = 1.0
        self.low_batt_threshold = float(low_batt_threshold)
        self._low_batt_sent = False
        self._last_batt_ts = self._now()

        self.ground_z = float(ground_z)
        self.g = float(g_mps2)
        self.terminal_vz = float(terminal_vz_mps)
        self.vz = 0.0

        if init_pose is None:
            any_node = next(iter(self.pos.keys()))
            x, y, z = self._pos_xyz(any_node)
            self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), 0.0
        else:
            self.x, self.y, self.z, self.yaw = map(float, init_pose)

        self.mode = "IDLE"
        self.current_node = self._nearest_node()

        # Snap initial XY to the nearest node to avoid rejecting the first command
        nx, ny, nz = self._pos_xyz(self.current_node)
        self.x, self.y = float(nx), float(ny)
        self.z = max(float(self.z), float(nz) + self.clearance + self.alt_offset)

        self.snap_on_take = bool(snap_on_take)

        self._v_cmd = 0.0

        self.edge_u = None
        self.edge_v = None
        self.edge_len = 0.0
        self.edge_s = 0.0
        self._ax = self._ay = self._az = 0.0
        self._bx = self._by = self._bz = 0.0

        self._action_node = None
        self._action_t = 0.0

    def on_event(self, ev):
        ev = str(ev or "").strip()
        if not ev:
            return False

        base, eid = split_suffix_id(ev)
        if eid is None or eid != self.agent_id:
            return False

        if base.startswith("edge_take::"):
            u, v = self._parse_edge_take(base)
            return self._cmd_take_edge(u, v)

        if base.startswith("work_start::"):
            n = base.split("work_start::", 1)[1]
            return self._cmd_work_start(n)

        if base.startswith("charge_start::"):
            n = base.split("charge_start::", 1)[1]
            return self._cmd_charge_start(n)

        return False

    def step(self, dt):
        dt = float(dt)
        if dt <= 0.0 or self.mode == "STOPPED":
            return

        now = self._now()

        if self.mode == "FALLING":
            self._fall_step(dt)
            return

        if self.mode == "MOVING":
            v_meas, yaw_rate = self._move_edge_step(dt)
            self._battery_update(v_meas, yaw_rate, now)
            self._check_battery_empty()
            return

        if self.mode == "WORKING":
            self._action_t += dt
            self._battery_update(0.0, 0.0, now)
            self._check_battery_empty()
            if self._action_t >= self.work_time_s:
                n = self._action_node
                self.mode = "IDLE"
                self._action_node = None
                self._action_t = 0.0
                self._emit(f"work_end::{n}")
            return

        if self.mode == "CHARGING":
            self._action_t += dt
            if self.charge_time_s > 1e-6:
                self.soc = min(1.0, self.soc + dt / self.charge_time_s)
            if self._action_t >= self.charge_time_s:
                n = self._action_node
                self.soc = 1.0
                self._low_batt_sent = False
                self.mode = "IDLE"
                self._action_node = None
                self._action_t = 0.0
                self._emit(f"charge_end::{n}")
            return

        if self.mode == "IDLE":
            if self.current_node in self.pos:
                _, _, nz = self._pos_xyz(self.current_node)
                z_tgt = float(nz) + self.clearance + self.alt_offset
                self.z = move_towards(self.z, z_tgt, self.vspeed * dt)
            self._battery_update(0.0, 0.0, now)
            self._check_battery_empty()
            return

    def send_pose(self):
        pose = Pose()
        pose.position = Point(x=float(self.x), y=float(self.y), z=float(self.z))
        pose.orientation = yaw_to_quaternion(float(self.yaw))

        state = EntityState()
        state.name = self.entity_name
        state.pose = pose
        state.twist = Twist()
        state.reference_frame = "world"

        req = SetEntityState.Request()
        req.state = state
        self.cli_set.call_async(req)

    def _cmd_take_edge(self, u, v):
        if self.mode != "IDLE":
            return False
        if u not in self.pos or v not in self.pos:
            return False

        ux, uy, uz = self._pos_xyz(u)

        # Accept command even if spawn is not exactly on the node
        if self.snap_on_take:
            self.x, self.y = float(ux), float(uy)
        else:
            if math.hypot(self.x - float(ux), self.y - float(uy)) > max(self.tol, 0.35):
                return False

        self.edge_u = u
        self.edge_v = v

        self._ax, self._ay, self._az = float(ux), float(uy), float(uz)
        bx, by, bz = self._pos_xyz(v)
        self._bx, self._by, self._bz = float(bx), float(by), float(bz)

        self.edge_len = math.hypot(self._bx - self._ax, self._by - self._ay)
        if self.edge_len <= 1e-9:
            return False

        self.edge_s = 0.0
        self._v_cmd = 0.0
        self.mode = "MOVING"
        self.current_node = u
        self.z = max(self.z, self._az + self.clearance + self.alt_offset)
        return True

    def _cmd_work_start(self, n):
        if self.mode != "IDLE":
            return False
        if n != self.current_node:
            return False
        self.mode = "WORKING"
        self._action_node = n
        self._action_t = 0.0
        return True

    def _cmd_charge_start(self, n):
        if self.mode != "IDLE":
            return False
        if n != self.current_node:
            return False
        self.mode = "CHARGING"
        self._action_node = n
        self._action_t = 0.0
        return True

    def _move_edge_step(self, dt):
        v_des = self.speed
        self._v_cmd = move_towards(self._v_cmd, v_des, self.accel * dt)

        ds = (self._v_cmd * dt) / max(1e-9, self.edge_len)
        s0 = self.edge_s
        self.edge_s = min(1.0, self.edge_s + ds)

        self.x = self._ax + self.edge_s * (self._bx - self._ax)
        self.y = self._ay + self.edge_s * (self._by - self._ay)

        yaw_des = math.atan2(self._by - self._ay, self._bx - self._ax)
        dyaw = yaw_des - self.yaw
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi

        max_dyaw = self.yaw_rate_max * dt
        yaw_step = sat(dyaw, -max_dyaw, max_dyaw)
        yaw_rate = yaw_step / max(1e-6, dt)
        self.yaw += yaw_step

        z_ref = self._az + self.edge_s * (self._bz - self._az)
        z_tgt = float(z_ref) + self.clearance + self.alt_offset
        self.z = move_towards(self.z, z_tgt, self.vspeed * dt)

        if self.edge_s >= 1.0 - 1e-9:
            self.x, self.y = float(self._bx), float(self._by)
            self.current_node = self.edge_v

            u, v = self.edge_u, self.edge_v
            self.edge_u = None
            self.edge_v = None
            self.edge_len = 0.0
            self.edge_s = 0.0
            self._v_cmd = 0.0
            self.mode = "IDLE"

            self._emit(f"edge_release::{u}::{v}")

        v_meas = (self.edge_len * (self.edge_s - s0)) / max(1e-6, dt)
        return v_meas, yaw_rate

    def _battery_update(self, v, yaw_rate, now_s):
        dt = max(1e-3, now_s - self._last_batt_ts)
        self._last_batt_ts = now_s

        prev = self.soc
        power_W = (
            (self.batt.i_base + self.batt.i_vgain * abs(v) + self.batt.i_wgain * abs(yaw_rate))
            * self.batt.voltage_nom
        )
        used_Wh = power_W * (dt / 3600.0)
        self.soc = max(0.0, self.soc - used_Wh / max(1e-9, self.batt.capacity_Wh))

        if (not self._low_batt_sent) and prev > self.low_batt_threshold and self.soc <= self.low_batt_threshold:
            self._low_batt_sent = True
            self._emit("battery_low")

    def _check_battery_empty(self):
        if self.soc > 0.0:
            return
        if self.mode in ("FALLING", "STOPPED"):
            return
        self._emit("battery_empty")
        self.mode = "FALLING"
        self.vz = 0.0

    def _fall_step(self, dt):
        self.vz = max(-self.terminal_vz, self.vz - self.g * dt)
        self.z += self.vz * dt
        if self.z <= self.ground_z:
            self.z = self.ground_z
            self.mode = "STOPPED"
            self._emit("crashed")

    def _emit(self, base):
        msg = String()
        msg.data = f"{base}_{self.agent_id}"
        self.pub_event.publish(msg)

    def _parse_edge_take(self, base):
        rest = base.split("edge_take::", 1)[1]
        parts = rest.split("::")
        if len(parts) != 2:
            return "", ""
        return parts[0], parts[1]

    def _nearest_node(self):
        best = ""
        best_d2 = 1e30
        for nid in self.pos.keys():
            x, y, _z = self._pos_xyz(nid)
            dx = self.x - float(x)
            dy = self.y - float(y)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = nid
        return best

    def _pos_xyz(self, nid):
        p = self.pos[nid]
        if len(p) >= 3:
            return float(p[0]), float(p[1]), float(p[2])
        return float(p[0]), float(p[1]), 0.0

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9
