import math
<<<<<<< HEAD
import time
=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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
<<<<<<< HEAD

        # Gazebo pose-update protection.
        # This prevents visual flickering caused by flooding
        # /gazebo/set_entity_state with async service calls.
        pose_rate_hz=10.0,
        pose_precision=4,
        resend_idle_pose_s=1.0,
=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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
<<<<<<< HEAD
        self._skip_battery_update_once = False
        self.vertiport_nodes = set()
=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966

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

<<<<<<< HEAD
        # --------------------------------------------------------------
        # Gazebo SetEntityState back-pressure
        # --------------------------------------------------------------
        self.pose_rate_hz = max(1e-6, float(pose_rate_hz))
        self._pose_period_wall_s = 1.0 / self.pose_rate_hz
        self._pose_precision = int(pose_precision)
        self._resend_idle_pose_s = max(0.0, float(resend_idle_pose_s))

        self._pose_future = None
        self._last_pose_wall_s = -1e30
        self._last_forced_idle_pose_wall_s = -1e30
        self._last_sent_pose_key = None
        self._last_pose_error_log_wall_s = -1e30

=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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

<<<<<<< HEAD
    def _is_vertiport_node(self, node_id):
        node_id = str(node_id)
        return node_id in getattr(self, "vertiport_nodes", set()) or "VERTIPORT" in node_id.upper()

=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
    def restore_full_battery(self):
        self.soc = 1.0
        self._low_batt_sent = False
        self._last_batt_ts = self._now()

<<<<<<< HEAD
    def send_pose(self, force=False):
        """
        Send UAV pose to Gazebo with throttling and back-pressure.

        The previous implementation called call_async() at every timer tick.
        With several UAVs, that can accumulate pending SetEntityState requests,
        making Gazebo display old poses late. This appears visually as flickering,
        blinking, or delayed movement.
        """
        now_wall = time.monotonic()

        # If the previous async call is still pending, do not enqueue another one.
        if self._pose_future is not None:
            if not self._pose_future.done():
                return

            # Consume completed future and log errors at low frequency.
            try:
                self._pose_future.result()
            except Exception as exc:
                if now_wall - self._last_pose_error_log_wall_s > 1.0:
                    self._last_pose_error_log_wall_s = now_wall
                    try:
                        self.node.get_logger().warning(
                            "agent=%d SetEntityState failed for entity='%s': %s"
                            % (self.agent_id, self.entity_name, str(exc))
                        )
                    except Exception:
                        pass
            finally:
                self._pose_future = None

        pose_key = (
            round(float(self.x), self._pose_precision),
            round(float(self.y), self._pose_precision),
            round(float(self.z), self._pose_precision),
            round(float(self.yaw), self._pose_precision),
        )

        pose_changed = pose_key != self._last_sent_pose_key

        # While idle, avoid repeatedly sending the exact same pose.
        if self.mode == "IDLE" and not pose_changed and not force:
            if self._resend_idle_pose_s <= 0.0:
                return

            if now_wall - self._last_forced_idle_pose_wall_s < self._resend_idle_pose_s:
                return

            self._last_forced_idle_pose_wall_s = now_wall

        # Global rate limit.
        if not force and (now_wall - self._last_pose_wall_s) < self._pose_period_wall_s:
            return

=======
    def send_pose(self):
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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
<<<<<<< HEAD

        self._pose_future = self.cli_set.call_async(req)
        self._last_pose_wall_s = now_wall
        self._last_sent_pose_key = pose_key
=======
        self.cli_set.call_async(req)
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966

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
<<<<<<< HEAD
        ux = float(ux)
        uy = float(uy)
        uz = float(uz)

        if self.snap_on_move:
            self.x = ux
            self.y = uy
        else:
            # Use horizontal tolerance for deciding whether the UAV is located
            # at the source vertex. This keeps the check compatible with
            # graph nodes that differ only by altitude.
            dist_xy = math.hypot(self.x - ux, self.y - uy)
            if dist_xy > max(self.tol, 0.35):
                return False

        bx, by, bz = self._pos_xyz(v)
        bx = float(bx)
        by = float(by)
        bz = float(bz)

        dx = bx - ux
        dy = by - uy
        dz = bz - uz

        edge_len_xy = math.hypot(dx, dy)
        edge_len_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        # IMPORTANT:
        # The previous implementation used only the horizontal distance
        # math.hypot(dx, dy). Therefore, vertical or same-(x,y) inter-layer
        # edges were rejected as zero-length edges. This made the MPSC/UTM
        # select free higher-altitude edges, receive a grant, and then fail at
        # the hardware layer, producing apparent gridlock at the vertex.
        if edge_len_3d <= 1e-9:
=======

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
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
            return False

        self.edge_u = u
        self.edge_v = v

<<<<<<< HEAD
        self._ax = ux
        self._ay = uy
        self._az = uz

        self._bx = bx
        self._by = by
        self._bz = bz

        self.edge_len = float(edge_len_3d)
        self.edge_len_xy = float(edge_len_xy)
        self.edge_dz = float(dz)
=======
        self._ax = float(ux)
        self._ay = float(uy)
        self._az = float(uz)

        self._bx = float(bx)
        self._by = float(by)
        self._bz = float(bz)

        self.edge_len = float(edge_len)
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
        self.edge_s = 0.0
        self._v_cmd = 0.0

        self.mode = "MOVING"
        self.current_node = u
<<<<<<< HEAD

        # Start exactly at the graph altitude of the source node. This is
        # required for smooth vertical/inter-layer transitions.
        self.z = uz + self.clearance + self.alt_offset

        # Force one immediate pose update at the start of motion.
        self.send_pose(force=True)
=======
        self.z = max(self.z, self._az + self.clearance + self.alt_offset)
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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
<<<<<<< HEAD

            if getattr(self, "_skip_battery_update_once", False):
                self._skip_battery_update_once = False
            else:
                self._battery_update(v_meas, yaw_rate, now)

=======
            self._battery_update(v_meas, yaw_rate, now)
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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

<<<<<<< HEAD
            if self._is_vertiport_node(self.current_node):
                self.restore_full_battery()
                self._battery_maybe_log(now)
                return

=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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
<<<<<<< HEAD
        old_edge_len = float(self.edge_len)

        # Bound the path-progress speed by both the horizontal cruise speed
        # and the configured vertical speed.  For a purely vertical edge this
        # makes the vehicle climb/descend at vspeed instead of the horizontal
        # cruise speed.
        dz_total = abs(float(self._bz) - float(self._az))
        if dz_total > 1e-9:
            v_z_limited = self.vspeed * max(float(self.edge_len), 1e-9) / dz_total
            v_des = min(float(self.speed), float(v_z_limited))
        else:
            v_des = float(self.speed)

=======
        v_des = self.speed
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
        self._v_cmd = move_towards(self._v_cmd, v_des, self.accel * dt)

        ds = (self._v_cmd * dt) / max(1e-9, self.edge_len)
        s0 = self.edge_s
        self.edge_s = min(1.0, self.edge_s + ds)

        self.x = self._ax + self.edge_s * (self._bx - self._ax)
        self.y = self._ay + self.edge_s * (self._by - self._ay)

<<<<<<< HEAD
        # For purely vertical/inter-layer edges, dx=dy=0 and the yaw is
        # undefined. Keep the previous yaw instead of forcing atan2(0,0).
        dx = self._bx - self._ax
        dy = self._by - self._ay
        if math.hypot(dx, dy) > 1e-9:
            yaw_des = math.atan2(dy, dx)
            dyaw = yaw_des - self.yaw

            while dyaw > math.pi:
                dyaw -= 2.0 * math.pi
            while dyaw < -math.pi:
                dyaw += 2.0 * math.pi

            max_dyaw = self.yaw_rate_max * dt
            yaw_step = sat(dyaw, -max_dyaw, max_dyaw)
            yaw_rate = yaw_step / max(1e-6, dt)
            self.yaw += yaw_step
        else:
            yaw_rate = 0.0

        # Interpolate altitude directly along the selected 3D graph edge.
        # This avoids rejecting or visually stalling same-(x,y) inter-layer
        # edges. The graph edge itself defines the admissible vertical motion.
        z_ref = self._az + self.edge_s * (self._bz - self._az)
        self.z = float(z_ref) + self.clearance + self.alt_offset

        v_meas = (old_edge_len * (self.edge_s - s0)) / max(1e-6, dt)
=======
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
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966

        if self.edge_s >= 1.0 - 1e-9:
            self.x = float(self._bx)
            self.y = float(self._by)
<<<<<<< HEAD
            self.z = float(self._bz) + self.clearance + self.alt_offset
=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
            self.current_node = self.edge_v

            u = self.edge_u
            v = self.edge_v

            self.edge_u = None
            self.edge_v = None
            self.edge_len = 0.0
<<<<<<< HEAD
            self.edge_len_xy = 0.0
            self.edge_dz = 0.0
=======
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
            self.edge_s = 0.0
            self._v_cmd = 0.0
            self.mode = "IDLE"

<<<<<<< HEAD
            if self._is_vertiport_node(v):
                self.restore_full_battery()
                self._skip_battery_update_once = True

            # Force final pose before emitting completion.
            self.send_pose(force=True)
            self._emit_uncontrollable(f"edge_release::{u}::{v}")

=======
            self._emit_uncontrollable(f"edge_release::{u}::{v}")

        v_meas = (self.edge_len * (self.edge_s - s0)) / max(1e-6, dt)
>>>>>>> 5adce261dd757e3d93e0c03c34300c7fe91ec966
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