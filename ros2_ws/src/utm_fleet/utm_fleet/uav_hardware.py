import math
from dataclasses import dataclass

from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import String


def yaw_to_quaternion(yaw):
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(0.5 * float(yaw)),
        w=math.cos(0.5 * float(yaw)),
    )


def sat(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def move_towards(curr, target, max_delta):
    if curr < target:
        return min(curr + max_delta, target)
    return max(curr - max_delta, target)


@dataclass
class BatteryModel:
    voltage_nom: float = 22.2
    capacity_Wh: float = 180.0
    i_base: float = 1.8
    i_vgain: float = 2.5
    i_wgain: float = 1.2


class UAVHardware:
    """
    Pure physical execution layer.

    This class does not own DES logic and does not consume DES events.
    It only exposes physical actions and emits uncontrollable events when
    those actions complete.

    Public actions:
        - start_move(u, v)
        - start_pick(provider_node)
        - start_deliver(client_node)
        - start_charge(station_node)

    Emitted uncontrollable events:
        - edge_release::<u>::<v>_<id>
        - work_end::<provider>::SUPPLIER_<id>
        - work_end::<client>::CLIENT_<id>
        - charge_end::<station>_<id>
        - battery_low_<id>
    """

    def __init__(
        self,
        node,
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
        pickup_time_s=2.0,
        delivery_time_s=2.0,
        charge_time_s=5.0,
        battery_model=None,
        low_batt_threshold=0.40,
        ground_z=0.0,
        g_mps2=9.81,
        terminal_vz_mps=12.0,
        init_pose=None,
        snap_on_move=True,
        local_event_callback=None,
        battery_log_period_s=0.0,
    ):
        self.node = node
        self.entity_name = str(entity_name)
        self.agent_id = int(agent_id)
        self.pos = graph_positions
        self.cli_set = set_state_client

        self.pub_event = self.node.create_publisher(String, event_topic, 50)
        self.local_event_callback = local_event_callback

        self.speed = float(speed_mps)
        self.vspeed = float(vspeed_mps)
        self.yaw_rate_max = float(yaw_rate_max_rps)
        self.accel = float(accel_mps2)
        self.tol = float(waypoint_tol_m)

        self.clearance = float(clearance_m)
        self.alt_offset = float(alt_offset_m)

        self.pickup_time_s = float(pickup_time_s)
        self.delivery_time_s = float(delivery_time_s)
        self.charge_time_s = float(charge_time_s)

        self.batt = battery_model if battery_model is not None else BatteryModel()
        self.soc = 1.0
        self.low_batt_threshold = float(low_batt_threshold)
        self._low_batt_sent = False
        self._last_batt_ts = self._now()
        self._battery_log_period_s = float(battery_log_period_s)
        self._last_battery_log_ts = self._last_batt_ts

        self.ground_z = float(ground_z)
        self.g = float(g_mps2)
        self.terminal_vz = float(terminal_vz_mps)
        self.vz = 0.0

        if init_pose is None:
            any_node = next(iter(self.pos.keys()))
            x, y, z = self._pos_xyz(any_node)
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.yaw = 0.0
        else:
            self.x, self.y, self.z, self.yaw = map(float, init_pose)

        self.mode = "IDLE"
        self.current_node = self._nearest_node()

        px, py, pz = self._pos_xyz(self.current_node)
        self.x = float(px)
        self.y = float(py)
        self.z = max(float(self.z), float(pz) + self.clearance + self.alt_offset)

        self.snap_on_move = bool(snap_on_move)
        self._v_cmd = 0.0

        self.edge_u = None
        self.edge_v = None
        self.edge_len = 0.0
        self.edge_s = 0.0
        self._ax = 0.0
        self._ay = 0.0
        self._az = 0.0
        self._bx = 0.0
        self._by = 0.0
        self._bz = 0.0

        self._action_node = None
        self._action_t = 0.0
        self._action_duration_s = 0.0
        self._action_end_event_base = None

    # ------------------------------------------------------------------
    # public status API
    # ------------------------------------------------------------------

    def is_busy(self):
        return self.mode in ("MOVING", "PICKING", "DELIVERING", "CHARGING", "FALLING")

    def is_idle(self):
        return self.mode == "IDLE"

    def mode_str(self):
        return str(self.mode)

    def current_node_id(self):
        return self.current_node

    def restore_full_battery(self):
        self.soc = 1.0
        self._low_batt_sent = False
        self._last_batt_ts = self._now()

    def send_pose(self):
        pose = Pose()
        pose.position = Point(
            x=float(self.x),
            y=float(self.y),
            z=float(self.z),
        )
        pose.orientation = yaw_to_quaternion(self.yaw)

        state = EntityState()
        state.name = self.entity_name
        state.pose = pose
        state.twist = Twist()
        state.reference_frame = "world"

        req = SetEntityState.Request()
        req.state = state
        self.cli_set.call_async(req)

    # ------------------------------------------------------------------
    # public action API
    # ------------------------------------------------------------------

    def start_move(self, u, v):
        u = str(u)
        v = str(v)

        if self.mode != "IDLE":
            return False

        if u not in self.pos or v not in self.pos:
            return False

        ux, uy, uz = self._pos_xyz(u)

        if self.snap_on_move:
            self.x = float(ux)
            self.y = float(uy)
        else:
            dist = math.hypot(self.x - float(ux), self.y - float(uy))
            if dist > max(self.tol, 0.35):
                return False

        bx, by, bz = self._pos_xyz(v)
        edge_len = math.hypot(float(bx) - float(ux), float(by) - float(uy))
        if edge_len <= 1e-9:
            return False

        self.edge_u = u
        self.edge_v = v

        self._ax = float(ux)
        self._ay = float(uy)
        self._az = float(uz)

        self._bx = float(bx)
        self._by = float(by)
        self._bz = float(bz)

        self.edge_len = float(edge_len)
        self.edge_s = 0.0
        self._v_cmd = 0.0

        self.mode = "MOVING"
        self.current_node = u
        self.z = max(self.z, self._az + self.clearance + self.alt_offset)
        return True

    def start_pick(self, provider_node):
        provider_node = str(provider_node)
        return self._start_timed_action(
            node_id=provider_node,
            mode_name="PICKING",
            duration_s=self.pickup_time_s,
            end_event_base=f"work_end::{provider_node}::SUPPLIER",
        )

    def start_deliver(self, client_node):
        client_node = str(client_node)
        return self._start_timed_action(
            node_id=client_node,
            mode_name="DELIVERING",
            duration_s=self.delivery_time_s,
            end_event_base=f"work_end::{client_node}::CLIENT",
        )

    def start_charge(self, station_node):
        station_node = str(station_node)

        if self.mode != "IDLE":
            return False

        if station_node != self.current_node:
            return False

        self.mode = "CHARGING"
        self._action_node = station_node
        self._action_t = 0.0
        self._action_duration_s = self.charge_time_s
        self._action_end_event_base = f"charge_end::{station_node}"
        return True

    # ------------------------------------------------------------------
    # main simulation step
    # ------------------------------------------------------------------

    def step(self, dt):
        dt = float(dt)
        if dt <= 0.0:
            return

        if self.mode == "STOPPED":
            return

        now = self._now()

        if self.mode == "FALLING":
            self._fall_step(dt)
            self._battery_maybe_log(now)
            return

        if self.mode == "MOVING":
            v_meas, yaw_rate = self._move_step(dt)
            self._battery_update(v_meas, yaw_rate, now)
            self._check_battery_empty()
            self._battery_maybe_log(now)
            return

        if self.mode in ("PICKING", "DELIVERING"):
            self._action_t += dt
            self._battery_update(0.0, 0.0, now)
            self._check_battery_empty()

            if self.mode == "FALLING":
                self._battery_maybe_log(now)
                return

            if self._action_t >= self._action_duration_s:
                end_event = self._action_end_event_base
                self._clear_action()
                self.mode = "IDLE"
                self._emit_uncontrollable(end_event)

            self._battery_maybe_log(now)
            return

        if self.mode == "CHARGING":
            self._action_t += dt

            if self.charge_time_s > 1e-9:
                self.soc = min(1.0, self.soc + dt / self.charge_time_s)

            if self._action_t >= self._action_duration_s:
                end_event = self._action_end_event_base
                self.restore_full_battery()
                self._clear_action()
                self.mode = "IDLE"
                self._emit_uncontrollable(end_event)

            self._battery_maybe_log(now)
            return

        if self.mode == "IDLE":
            if self.current_node in self.pos:
                _, _, nz = self._pos_xyz(self.current_node)
                z_tgt = float(nz) + self.clearance + self.alt_offset
                self.z = move_towards(self.z, z_tgt, self.vspeed * dt)

            self._battery_update(0.0, 0.0, now)
            self._check_battery_empty()
            self._battery_maybe_log(now)
            return

    # ------------------------------------------------------------------
    # internal action helpers
    # ------------------------------------------------------------------

    def _start_timed_action(self, node_id, mode_name, duration_s, end_event_base):
        if self.mode != "IDLE":
            return False

        if node_id != self.current_node:
            return False

        self.mode = str(mode_name)
        self._action_node = str(node_id)
        self._action_t = 0.0
        self._action_duration_s = float(duration_s)
        self._action_end_event_base = str(end_event_base)
        return True

    def _clear_action(self):
        self._action_node = None
        self._action_t = 0.0
        self._action_duration_s = 0.0
        self._action_end_event_base = None

    # ------------------------------------------------------------------
    # motion
    # ------------------------------------------------------------------

    def _move_step(self, dt):
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
            dyaw -= 2.0 * math.pi
        while dyaw < -math.pi:
            dyaw += 2.0 * math.pi

        max_dyaw = self.yaw_rate_max * dt
        yaw_step = sat(dyaw, -max_dyaw, max_dyaw)
        yaw_rate = yaw_step / max(1e-6, dt)
        self.yaw += yaw_step

        z_ref = self._az + self.edge_s * (self._bz - self._az)
        z_tgt = float(z_ref) + self.clearance + self.alt_offset
        self.z = move_towards(self.z, z_tgt, self.vspeed * dt)

        if self.edge_s >= 1.0 - 1e-9:
            self.x = float(self._bx)
            self.y = float(self._by)
            self.current_node = self.edge_v

            u = self.edge_u
            v = self.edge_v

            self.edge_u = None
            self.edge_v = None
            self.edge_len = 0.0
            self.edge_s = 0.0
            self._v_cmd = 0.0
            self.mode = "IDLE"

            self._emit_uncontrollable(f"edge_release::{u}::{v}")

        v_meas = (self.edge_len * (self.edge_s - s0)) / max(1e-6, dt)
        return v_meas, yaw_rate

    # ------------------------------------------------------------------
    # battery
    # ------------------------------------------------------------------

    def _battery_update(self, v, yaw_rate, now_s):
        dt = max(1e-3, now_s - self._last_batt_ts)
        self._last_batt_ts = now_s

        if self.mode == "CHARGING":
            return

        prev = self.soc

        power_w = (
            self.batt.i_base
            + self.batt.i_vgain * abs(float(v))
            + self.batt.i_wgain * abs(float(yaw_rate))
        ) * self.batt.voltage_nom

        used_wh = power_w * (dt / 3600.0)
        self.soc = max(0.0, self.soc - used_wh / max(1e-9, self.batt.capacity_Wh))

        if (not self._low_batt_sent) and prev > self.low_batt_threshold and self.soc <= self.low_batt_threshold:
            self._low_batt_sent = True
            self._emit_uncontrollable("battery_low")

    def _check_battery_empty(self):
        if self.soc > 0.0:
            return

        if self.mode in ("FALLING", "STOPPED"):
            return

        self.mode = "FALLING"
        self.vz = 0.0

    def _battery_maybe_log(self, now_s):
        if self._battery_log_period_s <= 0.0:
            return

        if (now_s - self._last_battery_log_ts) < self._battery_log_period_s:
            return

        self._last_battery_log_ts = now_s
        try:
            self.node.get_logger().info(
                "agent=%d mode=%s soc=%.4f node=%s"
                % (
                    self.agent_id,
                    self.mode,
                    float(self.soc),
                    str(self.current_node),
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # fall
    # ------------------------------------------------------------------

    def _fall_step(self, dt):
        self.vz = max(-self.terminal_vz, self.vz - self.g * dt)
        self.z += self.vz * dt

        if self.z <= self.ground_z:
            self.z = self.ground_z
            self.mode = "STOPPED"

    # ------------------------------------------------------------------
    # event emission
    # ------------------------------------------------------------------

    def _emit_uncontrollable(self, base):
        full = "%s_%d" % (str(base), self.agent_id)

        if self.local_event_callback is not None:
            try:
                self.local_event_callback(full)
            except Exception as e:
                try:
                    self.node.get_logger().warning(
                        "local_event_callback failed: %s" % str(e)
                    )
                except Exception:
                    pass

        msg = String()
        msg.data = full
        self.pub_event.publish(msg)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

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