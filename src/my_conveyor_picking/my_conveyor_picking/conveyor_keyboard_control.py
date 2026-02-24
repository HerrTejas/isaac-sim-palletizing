"""Conveyor belt keyboard control node.

Control multiple conveyor belts simultaneously using keyboard arrow keys:
- Left Arrow: Start all conveyors
- Right Arrow: Stop all conveyors
- q: Quit
"""

import curses
import rclpy
from rclpy.node import Node
from isaac_ros2_messages.srv import SetPrimAttribute


class ConveyorKeyboardControl(Node):

    DEFAULT_PRIM_PATHS = [
        "/World/ConveyorBelts/ConveyorTrack/ConveyorBeltGraph",
        "/World/ConveyorBelts/ConveyorTrack_01/ConveyorBeltGraph",
    ]
    DEFAULT_VELOCITY = 0.4
    CONVEYOR_VELOCITY_OFF = 0.0

    def __init__(self):
        super().__init__("conveyor_keyboard_control")

        self.declare_parameter("conveyor.prim_paths", self.DEFAULT_PRIM_PATHS)
        self.declare_parameter("conveyor.velocity", self.DEFAULT_VELOCITY)

        self._prim_paths = self.get_parameter("conveyor.prim_paths").value
        self._velocity_on = self.get_parameter("conveyor.velocity").value

        self._set_prim_client = self.create_client(
            SetPrimAttribute, "/set_prim_attribute"
        )

        self._conveyor_running = False
        self._waiting_for_service = True

        self.get_logger().info("Conveyor Keyboard Control initialized")
        self.get_logger().info(f"Controlling {len(self._prim_paths)} conveyors")
        self.get_logger().info("Waiting for /set_prim_attribute service...")

    def wait_for_service(self, timeout_sec: float = 5.0) -> bool:
        """Wait for the SetPrimAttribute service to be available.

        Args:
            timeout_sec: Timeout in seconds.

        Returns:
            True if service is available, False otherwise.
        """
        if self._set_prim_client.wait_for_service(timeout_sec=timeout_sec):
            self._waiting_for_service = False
            self.get_logger().info("Service available!")
            return True
        self.get_logger().warn("Service not available")
        return False

    def set_conveyor_velocity(self, velocity: float) -> None:
        """Set the conveyor belt velocity for all conveyors.

        Args:
            velocity: Target velocity for the conveyor belts.
        """
        if self._waiting_for_service:
            return

        for prim_path in self._prim_paths:
            request = SetPrimAttribute.Request()
            request.path = prim_path
            request.attribute = "graph:variable:Velocity"
            request.value = str(velocity)
            self._set_prim_client.call_async(request)

        self._conveyor_running = velocity > 0

    def start_conveyor(self) -> None:
        """Start the conveyor belt."""
        self.set_conveyor_velocity(self._velocity_on)

    def stop_conveyor(self) -> None:
        """Stop the conveyor belt."""
        self.set_conveyor_velocity(self.CONVEYOR_VELOCITY_OFF)


def run_curses(stdscr, node: ConveyorKeyboardControl) -> None:
    """Run the curses-based keyboard interface.

    Args:
        stdscr: Curses standard screen.
        node: The conveyor control node.
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.clear()

    def draw_screen():
        stdscr.clear()
        stdscr.addstr(0, 0, "=== Conveyor Keyboard Control ===")
        stdscr.addstr(1, 0, f"Controlling {len(node._prim_paths)} conveyors")
        stdscr.addstr(3, 0, "Controls:")
        stdscr.addstr(4, 2, "<- (Left Arrow)  : START conveyors")
        stdscr.addstr(5, 2, "-> (Right Arrow) : STOP conveyors")
        stdscr.addstr(6, 2, "q                : Quit")
        stdscr.addstr(8, 0, "-" * 40)

        status = "RUNNING" if node._conveyor_running else "STOPPED"
        color = curses.A_BOLD
        stdscr.addstr(10, 0, f"Conveyors Status: {status}", color)

        if node._waiting_for_service:
            stdscr.addstr(12, 0, "Waiting for Isaac Sim service...", curses.A_DIM)
        else:
            stdscr.addstr(12, 0, "Connected to Isaac Sim", curses.A_DIM)

        stdscr.refresh()

    if not node.wait_for_service(timeout_sec=10.0):
        stdscr.addstr(14, 0, "ERROR: Could not connect to Isaac Sim!")
        stdscr.addstr(15, 0, "Press any key to exit...")
        stdscr.nodelay(False)
        stdscr.getch()
        return

    draw_screen()

    while rclpy.ok():
        try:
            key = stdscr.getch()

            if key == ord("q") or key == ord("Q"):
                node.stop_conveyor()
                break
            elif key == curses.KEY_LEFT:
                node.start_conveyor()
                draw_screen()
            elif key == curses.KEY_RIGHT:
                node.stop_conveyor()
                draw_screen()

            rclpy.spin_once(node, timeout_sec=0.05)

        except KeyboardInterrupt:
            node.stop_conveyor()
            break


def main(args=None):
    rclpy.init(args=args)
    node = ConveyorKeyboardControl()

    try:
        curses.wrapper(run_curses, node)
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        node.stop_conveyor
        node.destroy_node
        rclpy.shutdown()

if __name__ == "__main__":
    main()