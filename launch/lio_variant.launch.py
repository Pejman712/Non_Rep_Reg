from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TUM_PATH = "/u/97/habibip1/unix/ros2_ws/src/regnonrep/tum/lio_odom.tum"


def _setup(context):
    # Generic launch for any lio_base.py variant.
    #   exe : the variant executable (e.g. lio_gen_lio.py)
    #   cfg : full path to the dataset config YAML (keyed under `lio_node:`)
    #   prefilter[/_voxel/_sor] : optional overrides for the light prefilter, so the
    #     web bench can toggle it without editing the YAML.  Appended AFTER cfg so
    #     they override the file's values.
    exe = LaunchConfiguration("exe").perform(context)
    cfg = LaunchConfiguration("cfg").perform(context)
    pf = LaunchConfiguration("prefilter").perform(context).strip().lower()
    pf_voxel = LaunchConfiguration("prefilter_voxel").perform(context).strip()
    pf_range = LaunchConfiguration("prefilter_range").perform(context).strip()
    pf_ror = LaunchConfiguration("prefilter_ror").perform(context).strip().lower()
    pf_ror_tau = LaunchConfiguration("prefilter_ror_tau").perform(context).strip()
    pf_sor = LaunchConfiguration("prefilter_sor").perform(context).strip().lower()
    save_maps = LaunchConfiguration("save_maps").perform(context).strip().lower()
    map_out = LaunchConfiguration("map_out").perform(context).strip()
    overlay = LaunchConfiguration("params_overlay").perform(context).strip()

    params = [cfg]
    if pf in ("on", "true", "1"):
        ov = {"prefilter_enable": True}
        if pf_voxel:
            try:
                ov["prefilter_voxel"] = float(pf_voxel)
            except ValueError:
                pass
        if pf_range:
            try:
                ov["prefilter_range_max"] = float(pf_range)
            except ValueError:
                pass
        if pf_ror in ("on", "true", "1"):
            ov["prefilter_ror"] = True
        elif pf_ror in ("off", "false", "0"):
            ov["prefilter_ror"] = False
        if pf_ror_tau:
            try:
                ov["prefilter_ror_tau"] = float(pf_ror_tau)
            except ValueError:
                pass
        if pf_sor in ("on", "true", "1"):
            ov["prefilter_sor"] = True
        elif pf_sor in ("off", "false", "0"):
            ov["prefilter_sor"] = False
        params.append(ov)
    elif pf in ("off", "false", "0"):
        params.append({"prefilter_enable": False})

    # optional parameter-overlay YAML (parameter-sweep campaigns): appended LAST so
    # its values override both the base cfg and the prefilter overrides above.
    if overlay:
        params.append(overlay)

    nodes = [
        Node(package="regnonrep", executable=exe, name="lio_node",
             parameters=params, output="screen"),
    ]
    # nrlio owns/loop-closes the evaluated TUM itself (writes TUM_PATH live +
    # corrected at shutdown), so we must NOT also start odom_to_tum for it — two
    # writers on the same file would race and clobber the loop-closed result.
    if "nrlio" not in exe:
        nodes.append(
            Node(package="regnonrep", executable="odom_to_tum.py",
                 name="odom_to_tum_fused", output="screen",
                 parameters=[{"odom_to_tum": {
                     "enabled": True, "odom_topic": "/lio/odom",
                     "output_path": TUM_PATH, "flush_every_n": 10,
                     "use_msg_time": True, "append": False,
                 }}]))
    # opt-in map saver: accumulate /lio/cloud_world → .pcd for offline MapEval/MME
    if save_maps in ("on", "true", "1") and map_out:
        nodes.append(
            Node(package="regnonrep", executable="map_saver.py",
                 name="map_saver", output="screen",
                 parameters=[{"out": map_out, "voxel": 0.05, "save_period_s": 15.0}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("exe"),
        DeclareLaunchArgument("cfg"),
        DeclareLaunchArgument("prefilter", default_value=""),
        DeclareLaunchArgument("prefilter_voxel", default_value=""),
        DeclareLaunchArgument("prefilter_range", default_value=""),
        DeclareLaunchArgument("prefilter_ror", default_value=""),
        DeclareLaunchArgument("prefilter_ror_tau", default_value=""),
        DeclareLaunchArgument("prefilter_sor", default_value=""),
        DeclareLaunchArgument("save_maps", default_value=""),
        DeclareLaunchArgument("map_out", default_value=""),
        DeclareLaunchArgument("params_overlay", default_value=""),
        OpaqueFunction(function=_setup),
    ])
