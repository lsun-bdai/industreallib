import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaRobotState
import numpy as np
import math
import logging
import tf_transformations as tr
from geometry_msgs.msg import Pose, PoseStamped
from spatialmath.base import rt2tr, qunit, q2r, r2q, qslerp, qeye
from bdai_msgs.msg import CartesianImpedanceGain
class FrankaArmStateClient:
    def __init__(self, 
                 ):
        self.node = Node("franka_arm_state_client")
        # Joint related states
        self.joint_names = FrankaConstants.JOINT_NAMES
        self.joint_states = None
        self.robot_state = None
        self.q = None
        self.dq = None
        self.arm_q = None
        self.ee_pose = None
        self.reset_joint_target = FrankaConstants.JOINT_NAMES
        self.reset_joint_target_values = np.zeros(len(self.reset_joint_target))

        # End effector related states
        self.pos = None
        self.force = None
        self.torque = None
        self.cartesian_stiffness = None
        self.cartesian_damping = None

        # Gripper related states
        self.gripper_states = None
        self.gripper_q = None
        # state subscriber
        self.state_sub = self.node.create_subscription(
            FrankaRobotState,
            "/franka_robot_state_broadcaster/robot_state",
            self._set_robot_state,
            10,
        )
        self.arm_joint_state_sub = self.node.create_subscription(
            JointState, "/fr3/joint_states", self._joint_state_callback, 1
        )
        self.gripper_sub = self.node.create_subscription(
            JointState, "/fr3_gripper/joint_states", self._gripper_callback, 1000
        )
        self.cartesian_impedance_gain_sub = self.node.create_subscription(
            CartesianImpedanceGain, "/franka_cartesian_impedance_controller/gains",
            self._cartesian_stiffness_callback, 10
        )

    def _gripper_callback(self, msg: JointState):
        self.gripper_pos = np.array([msg.position[0], msg.position[1]])

    def _joint_state_callback(self, msg: JointState):
        self.arm_q = np.array(msg.position[: len(self.reset_joint_target)])

    def _set_robot_state(self, msg: FrankaRobotState):
        # base frame o: fr3_link0
        # end effector frame ee: fr3_link8 if no hand, else fr3_hand_tcp
        # For now assume fr3_hand is attached, base to tip pose
        is_arr = isinstance(msg.o_t_ee, np.ndarray)
        if is_arr:
            pose_b2t = msg.o_t_ee.reshape(4, 4).T
        else:
            pose_b2t = msg_to_se3(msg.o_t_ee)

        position = pose_b2t[:3, 3]
        quat = r2q(pose_b2t[:3, :3], order="sxyz")
        pq_b2t = np.concatenate([position, quat])
        self.ee_pose = pq_b2t

        if is_arr:
            self.dq = msg.dq
            self.q = msg.q
        else:
            self.dq = msg.measured_joint_state.velocity
            self.q = msg.measured_joint_state.position

        if is_arr:
            self.force = msg.k_f_ext_hat_k[:3]
            self.torque = msg.k_f_ext_hat_k[3:]
        else:
            force = msg.k_f_ext_hat_k.wrench.force
            self.force = [force.x, force.y, force.z]
            torque = msg.k_f_ext_hat_k.wrench.torque
            self.torque = [torque.x, torque.y, torque.z]
        self.robot_state = msg

    def _cartesian_stiffness_callback(self, msg: CartesianImpedanceGain):
        self.cartesian_stiffness = msg.stiffness
        self.cartesian_damping = msg.damping

    def get_ee_pose(self):
        return self.ee_pose

    def get_joint_positions(self):
        return self.q

    def get_joint_velocities(self):
        return self.dq

    def get_joint_efforts(self):
        # TODO: Implement this method
        return None

    def get_robot_state(self):
        return self.robot_state

    def get_gripper_positions(self):
        return self.gripper_pos

    def get_gripper_velocities(self):
        return None  # Gripper velocities are not stored in the current implementation

    def get_gripper_efforts(self):
        return None  # Gripper efforts are not stored in the current implementation

class FrankaConstants:
    '''
    Contains default robot values, as well as robot limits.
    All units are in SI. 
    '''

    LOGGING_LEVEL = logging.INFO

    EMPTY_SENSOR_VALUES = [0]

    # translational stiffness, rotational stiffness
    DEFAULT_FORCE_AXIS_TRANSLATIONAL_STIFFNESS = 600
    DEFAULT_FORCE_AXIS_ROTATIONAL_STIFFNESS = 20

    # buffer time
    DEFAULT_TERM_BUFFER_TIME = 0.2

    HOME_JOINTS = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]

    # See https://frankaemika.github.io/docs/control_parameters.html
    JOINT_NAMES = ["fr3_joint1",
                   "fr3_joint2",
                   "fr3_joint3",
                   "fr3_joint4",
                   "fr3_joint5",
                   "fr3_joint6",
                   "fr3_joint7"]

    JOINT_LIMITS_MIN = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
    JOINT_LIMITS_MAX = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]

    DEFAULT_POSE_THRESHOLDS = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    DEFAULT_JOINT_THRESHOLDS = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

    GRIPPER_WIDTH_MAX = 0.08
    GRIPPER_WIDTH_MIN = 0
    GRIPPER_MAX_FORCE = 60

    MAX_LIN_MOMENTUM = 20
    MAX_ANG_MOMENTUM = 2
    MAX_LIN_MOMENTUM_CONSTRAINED = 100

    DEFAULT_FRANKA_INTERFACE_TIMEOUT = 10
    ACTION_WAIT_LOOP_TIME = 0.001

    GRIPPER_CMD_SLEEP_TIME = 0.2

    DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
    DEFAULT_D_GAINS = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
    DEFAULT_TRANSLATIONAL_STIFFNESSES = [600.0, 600.0, 600.0]
    DEFAULT_ROTATIONAL_STIFFNESSES = [50.0, 50.0, 50.0]

    DEFAULT_JOINT_IMPEDANCES = [3000, 3000, 3000, 2500, 2500, 2000, 2000]
    DEFAULT_CARTESIAN_IMPEDANCES = [3000, 3000, 3000, 300, 300, 300]

    DEFAULT_LOWER_TORQUE_THRESHOLDS_ACCEL = [20.0,20.0,18.0,18.0,16.0,14.0,12.0]
    DEFAULT_UPPER_TORQUE_THRESHOLDS_ACCEL = [120.0,120.0,120.0,118.0,116.0,114.0,112.0]
    DEFAULT_LOWER_TORQUE_THRESHOLDS_NOMINAL = [20.0,20.0,18.0,18.0,16.0,14.0,12.0]
    DEFAULT_UPPER_TORQUE_THRESHOLDS_NOMINAL = [120.0,120.0,118.0,118.0,116.0,114.0,112.0]

    DEFAULT_LOWER_FORCE_THRESHOLDS_ACCEL = [10.0,10.0,10.0,10.0,10.0,10.0]
    DEFAULT_UPPER_FORCE_THRESHOLDS_ACCEL = [120.0,120.0,120.0,125.0,125.0,125.0]
    DEFAULT_LOWER_FORCE_THRESHOLDS_NOMINAL = [10.0,10.0,10.0,10.0,10.0,10.0]
    DEFAULT_UPPER_FORCE_THRESHOLDS_NOMINAL = [120.0,120.0,120.0,125.0,125.0,125.0]

    DH_PARAMS = np.array([[0, 0.333, 0, 0],
                        [0, 0, -np.pi/2, 0],
                        [0, 0.316, np.pi/2, 0],
                        [0.0825, 0, np.pi/2, 0],
                        [-0.0825, 0.384, -np.pi/2, 0],
                        [0, 0, np.pi/2, 0],
                        [0.088, 0, np.pi/2, 0],
                        [0, 0.107, 0, 0],
                        [0, 0.1034, 0, 0]])

# Helper Functions ##########################################################
def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    )
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)
            )
        )
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g


###############################################################################

if __name__ == "__main__":
    rclpy.init()
    client = FrankaArmStateClient()
    rclpy.spin(client.node)
    rclpy.shutdown()    