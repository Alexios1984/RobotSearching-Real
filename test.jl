# ======================================
# --- Initial Setup ---
# ======================================

# --- Julia Imports ---

using PythonCall
using StaticArrays
using LinearAlgebra
using VMRobotControl
using Base.Threads
using MeshIO 
using VMRobotControl: robot_ndof, Transform, Rigid, FramePoint, ReferenceCoord, CoordDifference, TanhSpring, LinearDamper, GaussianSpring
using FileIO, UUIDs, MeshIO
using Dates
using Rotations
using Printf

# --- ROS 2 Python Modules ---

const rclpy = pyimport("rclpy")
const Node = pyimport("rclpy.node").Node


# --- ROS 2 Message Types ---

const JointState = pyimport("sensor_msgs.msg").JointState
const Float64MultiArray = pyimport("std_msgs.msg").Float64MultiArray
const CameraInfo = pyimport("sensor_msgs.msg").CameraInfo
const VmcRobotState = pyimport("vmc_interfaces.msg").VmcRobotState
const VmcObstacles = pyimport("vmc_interfaces.msg").VmcObstacles
const VmcTarget = pyimport("vmc_interfaces.msg").VmcTarget
const VmcControlConfig = pyimport("vmc_interfaces.msg").VmcControlConfig
const VmcSystemLog = pyimport("vmc_interfaces.msg").VmcSystemLog
const Point = pyimport("geometry_msgs.msg").Point
const Vector3 = pyimport("geometry_msgs.msg").Vector3


# --- Python Path Setup ---

sys = pyimport("sys")
if haskey(ENV, "PYTHONPATH")
    for path in split(ENV["PYTHONPATH"], ":")
        if !isempty(path) && !(path in sys.path)
            sys.path.append(path)
        end
    end
end


# -- Configuration Constants ---

const N_REPULSORS = 100             # Number of obstacle repulsors distributed on the objects

const COLLISION_LINKS =[            # Links to consider for collision avoidance
        "fr3_hand_tcp", 
        "fr3_link7", 
        "fr3_link6", 
        "fr3_link5", 
        "fr3_link4", 
        "fr3_link3", 
        "fr3_link2", 
        "fr3_link1"
]


const DESIRED_JOINT_ORDER = [       # Desired joint order for state array
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7"
]

const IGNORED_FLOOR_LINKS = [       # Links to ignore for collision avoidance (e.g., floor) 
    "fr3_link0", 
    "fr3_link1", 
    "fr3_link2"
]

const EXTRA_POINTS_CONFIG = [       # Extra points on the robot to be repulsed for improved precision in holes passing
    # Link 5
    ("fr3_link5", SVector(0.0, 0.11, 0.0),    "coll_L5_front"),
    ("fr3_link5", SVector(0.0, 0.11, -0.11),  "coll_L5_corner"),
    ("fr3_link5", SVector(0.0, 0.0, -0.22),   "coll_L5_back"),
    
    # Link 3
    ("fr3_link3", SVector(0.09, 0.09, 0.01),  "coll_L3_sideA"),
    ("fr3_link3", SVector(0.09, -0.09, 0.01), "coll_L3_sideB"),
    
    # Link 7
    ("fr3_link7", SVector(0.0, 0.0, -0.06),   "coll_L7_back"),
    
    # Hand TCP
    ("fr3_hand_tcp", SVector(0.0, 0.08, -0.07), "coll_HAND_left"),
    ("fr3_hand_tcp", SVector(0.0, -0.08, -0.07),"coll_HAND_right"),

    # Fingers
    ("fr3_hand_tcp", SVector(0.0, 0.05, -0.0), "left_finger"),
    ("fr3_hand_tcp", SVector(0.0, -0.05, -0.0),"right_finger")
]
const TOTAL_COLLISION_FRAMES = vcat(COLLISION_LINKS, [cfg[3] for cfg in EXTRA_POINTS_CONFIG])


const DIST_NOSE = 0.25              # Distance from camera frame to nose point along Z axis

const CAMERA_OFFSET = Transform(    # Camera mounted on the robot's end-effector has a fixed offset
    SVector(0.095, 0.0, -0.05), 
    Rotor(RotZ(pi/2))
) 

const verbose = false               # Flag to activate the verbose prints    

const FLOOR_Z_LEVEL = 0.01          # Height at which the floor is supposed to be


# --- Shared Channels for Communication ---

state_channel = Channel{Vector{Float64}}(1)         # Buffer for joint states
target_channel = Channel{Any}(2)                # Buffer for target 
obstacles_channel = Channel{Any}(2)     # Buffer for obstacles


# ======================================
# --- VMC & Robot Setup ---
# ======================================

try
    FileIO.add_format(format"DAE", (), ".dae", [:DigitalAssetExchangeFormatIO => UUID("43182933-f65b-495a-9e05-4d939cea427d")])
catch
end
cfg = URDFParserConfig(;suppress_warnings=true) # This is just to hide warnings about unsupported URDF features
module_path = joinpath(splitpath(splitdir(pathof(VMRobotControl))[1])[1:end-1])
robot = parseURDF(joinpath(module_path, "URDFs/franka_description/urdfs/fr3_franka_hand(copy).urdf"), cfg)

add_gravity_compensation!(robot, VMRobotControl.DEFAULT_GRAVITY)


# -- Virtual Mechanism System --

vms = VirtualMechanismSystem("franka_impedance_control", robot)

offset = 0.5
stiffness_limits = 20.0

upper_L = Float64[]
lower_L = Float64[]

joint_limits = cfg.joint_limits
for i in 1:7
    jname = "fr3_joint$i"
    limits = joint_limits[jname]
    push!(upper_L, limits.upper-offset)
    push!(lower_L, limits.lower+offset)
    add_coordinate!(robot, JointSubspace("fr3_joint$i");    id="JointValue$i")
end

# ======================================
# --- VMC Creation ---
# ======================================

root = root_frame(vms.robot)

add_coordinate!(robot, FramePoint("fr3_hand_tcp", SVector(0.0, 0.0, 0.0)); id="TCP position")                                          # TCP Position Coordinate

# --- Camera Frame & Attractor --- 

add_frame!(robot, "camera_frame")
add_joint!(robot, Rigid(CAMERA_OFFSET); parent="fr3_hand_tcp", child="camera_frame", id="J_Camera_Mount")       # Camera Mount Joint (Displaced Rigidly)

add_coordinate!(robot, FrameOrigin("camera_frame"); id="camera_position")                                       # Camera Position Coordinate
add_coordinate!(robot, FramePoint("camera_frame", SVector(0.0, 0.0, DIST_NOSE)); id="camera_nose")              # Camera Nose Coordinate

add_coordinate!(vms, ReferenceCoord(Ref(SVector(0.0, 0.4, 0.4))); id="target_attractor")                        # Target Attractor Coordinate (Front to Workspace)
# add_coordinate!(vms, ReferenceCoord(Ref(SVector(0.4, 0.0, 0.4))); id="target_attractor")                        # Target Attractor Coordinate (Front to PC)


# --- Attraction to Target ---

add_coordinate!(vms, CoordDifference("target_attractor", ".robot.camera_nose"); id="cam_to_target_error")

CAM_TO_TARG_STIFF = 0.01
CAM_TO_TARG_MAXFORCE = 0.01
CAM_TO_TARG_DAMPING = 0.01
add_component!(vms, TanhSpring("cam_to_target_error"; max_force=CAM_TO_TARG_MAXFORCE, stiffness=CAM_TO_TARG_STIFF); id="attraction_spring")
add_component!(vms, LinearDamper(CAM_TO_TARG_DAMPING * identity(3), "cam_to_target_error"); id="attraction_damper")


# --- Obstacle Repulsors for Camera ---

const REPULSION_MAXFORCE = -0.01
const REPULSION_WIDTH = 0.04

for i in 1:N_REPULSORS

    # Camera Repulsor Coordinate (Initialized Far Away)
    rep_id = "active_repulsor_$i"
    add_coordinate!(vms, ReferenceCoord(Ref(SVector(99.0, 99.0, 99.0))); id=rep_id)

    # Camera Repulsion Error 
    err_id = "camera_repulsion_error_$i"
    add_coordinate!(vms, CoordDifference(rep_id, ".robot.camera_position"); id=err_id)
    
    # Camera Repulsion Spring
    add_component!(vms, GaussianSpring(err_id; max_force=REPULSION_MAXFORCE, width=REPULSION_WIDTH); id="repulsor_spring_$i")
end


# Creation of virtual frames for additional collision points
for (parent, offset, fname) in EXTRA_POINTS_CONFIG
    add_frame!(robot, fname)
    add_joint!(robot, Rigid(Transform(offset)); parent=parent, child=fname, id="J_Fix_$(fname)")
end

# --- Obstacle Repulsors for Body (Floor + Obstacles) ---

const REPULSION_FLOOR_MAXFORCE = -2.0
const REPULSION_FLOOR_WIDTH = 0.05

for link_name in TOTAL_COLLISION_FRAMES

    add_coordinate!(robot, FrameOrigin(link_name); id="$(link_name)_pos")

    for i in 1:N_REPULSORS

        # Body Repulsor Error
        repulsor_id = "active_repulsor_$i"
        error = "$(link_name)_repulsion_error_$i"
        add_coordinate!(vms, CoordDifference(repulsor_id, ".robot.$(link_name)_pos"); id=error)

        # Body Repulsion Spring
        sping_id = "$(link_name)_link_repulsor_spring_$i"
        add_component!(vms, GaussianSpring(error; max_force=REPULSION_MAXFORCE, width=REPULSION_WIDTH); id=sping_id)
    end

    # Repulsor on each link for FLOOR
    shadow_id = "floor_shadow_$(link_name)"

    add_coordinate!(vms, ReferenceCoord(Ref(SVector(0.0, 0.0, FLOOR_Z_LEVEL))); id=shadow_id)

    floor_error_id = "floor_error_$(link_name)"
    add_coordinate!(vms, CoordDifference(shadow_id, ".robot.$(link_name)_pos"); id=floor_error_id)

    if !(link_name in IGNORED_FLOOR_LINKS)
        add_component!(vms, GaussianSpring(floor_error_id; max_force=REPULSION_FLOOR_MAXFORCE, width=REPULSION_FLOOR_WIDTH); id="FloorSpring_$(link_name)")
    end

end

println("✅ VMS Built.")



# ======================================
# --- Control Logic Functions ---
# ======================================

function f_setup(cache)

    # IDs for coordinate updates
    target_id = get_compiled_coordID(cache, "target_attractor")
    repulsor_ids = [get_compiled_coordID(cache, "active_repulsor_$i") for i in 1:N_REPULSORS]

    # Frame ID for Telemetry (to send pose to Python)
    cam_frame_id = get_compiled_frameID(cache, ".robot.camera_frame")


    # --- Link Coordinates for Velocity Check & Floor Repulsion ---

    link_coords_ids = []
    floor_repulsor_ids = String[]

    for name in TOTAL_COLLISION_FRAMES
        try
            shadow_id = "floor_shadow_$(name)"
            push!(floor_repulsor_ids, shadow_id)
            
            cid = get_compiled_coordID(cache, ".robot.$(name)_pos")
            push!(link_coords_ids, cid)
        catch
            println("Warning: Coordinate for $name not found, skipping velocity check.")
        end
    end

    floor_repulsors_handle = [cache[get_compiled_coordID(cache, fl_rep_id)].coord_data.val for fl_rep_id in floor_repulsor_ids]

    joint_ids = [get_compiled_coordID(cache, ".robot.JointValue$i") for i in 1:7]

    return (target_id, repulsor_ids, cam_frame_id, link_coords_ids, floor_repulsors_handle, joint_ids)
end

function f_control(cache, t, args, dt)

    (target_id, repulsor_ids, _, link_coords_ids, floor_repulsors_handle, joint_ids) = args

    u_robot = Float64[]

    # Non-blocking check for new target data
    if isready(target_channel)

        new_target = take!(target_channel) 
        # println("🎯 New Target received: $new_target")
                
        println("t: $(round(t, digits=2))")
        
        # --- Update Target ---
        
        cache[target_id].coord_data.val[] = SVector(new_target[1], new_target[2], new_target[3])
    
    end

    # Non-blocking check for new obstacle data
    if isready(obstacles_channel)

        obstacles = take!(obstacles_channel)

        # --- Update Repulsors ---

        for i in 1:N_REPULSORS

            # If we have an obstacle, set its position 
            if i <= length(obstacles)
                cache[repulsor_ids[i]].coord_data.val[] = obstacles[i]
            
            # Otherwise, move it far away
            else
                cache[repulsor_ids[i]].coord_data.val[] = SVector(99.0, 99.0, 99.0)
            end
        end

        # --- Update Floor Repulsors Positions ---

        for (i, link_compiled_coord_id) in enumerate(link_coords_ids)

            link_pos = configuration(cache, link_compiled_coord_id)

            floor_repulsors_handle[i][] = SVector(link_pos[1], link_pos[2], FLOOR_Z_LEVEL)

        end

    end

    # Limit Avoidance

    current_joint_vals = [configuration(cache, id) for id in joint_ids]

    for i in 1:7

        val = current_joint_vals[i][1]
        
        # 1. Controllo Limite Inferiore
        if val < lower_L[i]
            # Il giunto è troppo basso -> spingiamo verso il positivo
            force_lower = stiffness_limits * (lower_L[i] - val)
        else
            # Siamo sopra il minimo -> nessuna forza dal limite inferiore
            force_lower = 0.0
        end

        # 2. Controllo Limite Superiore
        if val > upper_L[i]
            # Il giunto è troppo alto -> spingiamo verso il negativo (upper - val sarà negativo)
            force_upper = stiffness_limits * (upper_L[i] - val)
        else
            # Siamo sotto il massimo -> nessuna forza dal limite superiore
            force_upper = 0.0
        end 

        # Somma le componenti (una delle due sarà sempre 0, a meno che i limiti non siano invertiti)
        push!(u_robot, (force_lower + force_upper))

    end 

    return u_robot

end


"""
Controller with ROS 2 Integration and Debugging
"""
function ros_vm_controller(
        vms,
        qᵛ;
        gravity=VMRobotControl.DEFAULT_GRAVITY,
        f_setup=f_setup,
        f_control=f_control,
        E_max=500.0, # Increased specifically for debugging
    )

    GC.enable(false)
    rclpy.init()
    node = nothing

    try

        println("DEBUG [0]: Initializing ROS Node...")
        node = Node("vmc_controller_julia_debug")


        # --- Publisher ---

        # Torque Publisher
        torque_topic = "/NS_1/julia_torque_controller/external_torques"
        torque_publisher = node.create_publisher(Float64MultiArray, torque_topic, 10)
        
        # State Publisher
        state_publisher = node.create_publisher(VmcRobotState, "/vmc/robot_state", 10)
        
        # Deadlock Publisher 
        deadlock_pub = node.create_publisher(Float64MultiArray, "/vmc/deadlock_data", 10)
        
        println("DEBUG [1.1]: Publishers created.")
        
        
        # --- Subscribers ---

        # Joint State Subscriber
        py_callback_wrapper = (pymsg) -> joint_state_callback(pymsg, state_channel)
        sub = node.create_subscription(JointState, "NS_1/franka/joint_states", py_callback_wrapper, 10)

        # Target & Obstacles Subscriber
        target_callback_wrapper = (target_msg) -> target_callback(target_msg, target_channel)
        sub_target = node.create_subscription(VmcTarget, "/vmc/target_point", target_callback_wrapper, 10)

        obstacles_callback_wrapper = (obstacle_msg) -> obstacles_callback(obstacle_msg, obstacles_channel)
        sub_obstacles = node.create_subscription(VmcObstacles, "/vmc/active_obstacles", obstacles_callback_wrapper, 10)

        println("DEBUG [1.2]: Subscribers created.")

        
        # --- ROS 2 Spin Task ---

        println("DEBUG [2]: Starting ROS 2 spin task...")
        ros_task = @async try
            rclpy.spin(node)
        catch e
            println("CRITICAL ERROR IN ROS TASK:")
            showerror(stdout, e, catch_backtrace())
        end

        # Initialize Control Cache with first received joint state
        println("DEBUG [3]: Waiting for initial joint state data...")
        control_cache = new_control_cache(vms, qᵛ, gravity)
    
        let 
            NDOF = robot_ndof(control_cache)

            if !isready(state_channel)
                println("DEBUG [3.1]: Channel empty, blocking wait for message...")
            end
            latest_state_array = take!(state_channel)
            println("DEBUG [3.2]: Data received!")

            qʳ = latest_state_array[1:NDOF]
            q̇ʳ = zeros(eltype(control_cache), NDOF)
            
            control_step!(control_cache, 0.0, qʳ, q̇ʳ) 
            @info "Initial joint state set in control cache."
        end


        # --- Setup Function ---

        println("DEBUG [4]: Calling f_setup...")
        args = nothing
        try
            args = f_setup(control_cache)
            println("DEBUG [4.1]: f_setup successful. Args type: $(typeof(args))")
        catch e
            println("\n🔴 CRITICAL ERROR INSIDE f_setup:")
            showerror(stdout, e, catch_backtrace())
            println("\n")
            rethrow(e)
        end
        
        # Initialize the arguments for the f_setup
        local target_id, rep_ids, cam_frame_id, link_coords_ids

        # Unpack args explicitly to check structure
        println("DEBUG [5]: Unpacking args...")
        try
            (target_id, rep_ids, cam_frame_id, link_coords_ids) = args
            println("DEBUG [5.1]: Unpacking OK. Camera Frame ID found: $cam_frame_id")
        catch e
            println("\n🔴 CRITICAL ERROR UNPACKING ARGS (Check f_setup return):")
            showerror(stdout, e, catch_backtrace())
            rethrow(e)
        end


        # --- Control Function ---
        
        control_func! = let control_cache=control_cache, args=args, node=node, 
                            torque_publisher=torque_publisher, state_publisher=state_publisher, deadlock_pub=deadlock_pub,
                            cam_frame_id=cam_frame_id, link_coords_ids=link_coords_ids
            
            function control_func!(t, dt)

                NDOF = robot_ndof(control_cache)

                # --- Debug Blocking ---

                # Print before blocking in input
                if verbose
                    if mod(round(Int, t*100), 100) == 0 
                        print("⌛ Waiting for ROS msg... ")
                        Base.flush(stdout) 
                    end
                end

                latest_state_array = take!(state_channel)       # Possible Cause of Block

                # Print after blocking in input
                if verbose
                    if mod(round(Int, t*100), 100) == 0
                        println("✅ Got it!")
                        Base.flush(stdout)
                    end
                end

                qʳ = latest_state_array[1:NDOF]
                q̇ʳ = latest_state_array[NDOF+1:2*NDOF]
                
                # Logic Step
                u_robot = f_control(control_cache, t, args, dt) 
                
                # Physics Step
                desired_torques = control_step!(control_cache, t, qʳ, q̇ʳ) 

                # Limit Avoidance
                desired_torques += u_robot
                
                # ========================
                # ---- Publish Torques ---
                # ========================

                # Apply Safety Clamp
                SAFETY_LIMIT = 3.0    
                
                if any(isnan, desired_torques) || any(isinf, desired_torques)
                    println("\n💀 CRITICAL: Found NaN/Inf torque at time t=$t! Sending Zeros.")
                    desired_torques = zeros(7)
                else
                    # Clamp only if torques are valid
                    desired_torques = clamp.(desired_torques, -SAFETY_LIMIT, SAFETY_LIMIT)
                end

                pymsg = Float64MultiArray()
                pymsg.data = pylist(desired_torques)
                torque_publisher.publish(pymsg)


                # ========================
                # --- Publish State ---
                # ========================

                try
                    
                    # Get Camera Transform
                    cam_tf = VMRobotControl.get_transform(control_cache, cam_frame_id)
                    
                    # Build State Message
                    msg_state = VmcRobotState()
                    msg_state.header.stamp = node.get_clock().now().to_msg()
                    
                    # -- Camera Pose --
                    msg_state.cam_pose.position.x = cam_tf.origin[1]
                    msg_state.cam_pose.position.y = cam_tf.origin[2]
                    msg_state.cam_pose.position.z = cam_tf.origin[3]
                    
                    
                    # -- Camera Orientation (Quaternion) --

                    try
                        
                        # Extract Rotor (Quaternion) from Transform
                        r_val = cam_tf.rotor
                        
                        # Scalar part of the quaternion
                        w_val = r_val.scalar
                        
                        # This is the vector part (bivector) of the quaternion
                        b_val = r_val.bivector 
                        
                        # Construct Quaternion explicitly: (w, x, y, z)
                        quat = QuatRotation(w_val, b_val[1], b_val[2], b_val[3])

                        # Assign to message
                        msg_state.cam_pose.orientation.x = quat.x
                        msg_state.cam_pose.orientation.y = quat.y
                        msg_state.cam_pose.orientation.z = quat.z
                        msg_state.cam_pose.orientation.w = quat.w

                    catch e

                        println("\n🔴 ERROR ROTATION: $e")
                        msg_state.cam_pose.orientation.w = 1.0 
                        msg_state.cam_pose.orientation.x = 0.0
                        msg_state.cam_pose.orientation.y = 0.0
                        msg_state.cam_pose.orientation.z = 0.0

                    end
                    
                    msg_state.joint_positions = pylist(qʳ)
                    msg_state.joint_velocities = pylist(q̇ʳ)
                    
                    # Publish State Message
                    state_publisher.publish(msg_state)
                
                catch e
                    println("ERROR publishing state: $e")
                end


                # --- Compute Metrics for Deadlock ---
                
                should_print = mod(round(Int, t*1000), 500) == 0    # Print every 0.5 seconds


                # 1. Compute and Print Link Velocities

                vel_treshold = 0.02

                max_link_vel = 0.0              # We use just the maximum velocity among all the links ones to represent the velocity situation

                if verbose
                    if should_print
                        println("\n🔍 --- DEBUG DEADLOCK (t=$(round(t, digits=2))) ---")
                        println("   🏎️  LINK VELOCITIES (Treshold = 0.02):")
                    end
                end

                if isempty(link_coords_ids) && should_print && verbose
                    println("      ❌ ERROR! There are no link coordinates for velocity check.")
                end

                for (i, cid) in enumerate(link_coords_ids)

                    # Take velocity vector for this link
                    v_vec = VMRobotControl.velocity(control_cache, cid)
                    v_norm = norm(v_vec)

                    # Update the maximum velocity to update the value to send to the python node
                    if v_norm > max_link_vel
                        max_link_vel = v_norm
                    end

                    if verbose
                        if should_print
                            marker = v_norm <= vel_treshold ? "🔴" : "  "
                            @printf("      %s Link %d: %.4f m/s\n", marker, i, v_norm)
                        end
                    end
                end


                # 2. Compute and Print Joint Torques

                torque_treshold = 0.2

                current_torques = desired_torques[1:7]                              # First 7 are joint torques

                max_joint_torque = maximum(abs.(current_torques))               # We use just the maximum torque among all the joints ones to represent the torques situation

                if should_print && verbose

                    println("   💪 JOINT TORQUES (Treshold = 0.20):")
                    for i in 1:7
                        tau = abs(current_torques[i])
                        marker = tau <= torque_treshold ? "🔴" : "  "
                        @printf("      %s J%d:     %.4f Nm\n", marker, i, tau)
                    end
                                        
                    if max_link_vel <= vel_treshold #=&& max_joint_torque <= torque_treshold=#
                        println("   💀 STATUS: Deadlock detected.")
                    else
                        println("   🟢 STATUS: No Deadlock")
                    end

                    println("--------------------------------------------------")

                    # Force to print in the terminal
                    Base.flush(stdout)
                end



                # ========================
                # --- Publish Deadlock ---
                # ========================

                dl_msg = Float64MultiArray()
                dl_msg.data = pylist([max_link_vel, max_joint_torque])
                deadlock_pub.publish(dl_msg)

                return false
            end
        end

        (E = VMRobotControl.stored_energy(control_cache)) > E_max && error("Initial stored energy exceeds $(E_max)J, was $(E)J")

        
        
        # --- Main Control Loop ---
        
        t = 0.0
        dt = 0.001 
        println("DEBUG [6]: Starting main control loop...")
        
        while !istaskdone(ros_task)
            try
                # Debug check
                if istaskfailed(ros_task)
                    println("\n💀 Fatal Error: ROS node suddenly died!")
                    fetch(ros_task)
                    break
                end

                control_func!(t, dt)
                t = t + dt

            catch e
                println("\n🔴 CRITICAL ERROR INSIDE CONTROL LOOP at t=$t:")
                showerror(stdout, e, catch_backtrace())
                println("\n")
                rethrow(e)
            end
        end
        
        println("DEBUG [7]: Loop exited. ROS Task Done? $(istaskdone(ros_task))")
        if istaskdone(ros_task)
            println("ROS Task Result/Error:")
            try fetch(ros_task) catch e showerror(stdout, e) end
        end

    catch e
        if e isa InterruptException
            println("\nCaught Interrupt. Shutting down...")
        else
            println("\n🔴 CRITICAL TOP LEVEL ERROR:")
            showerror(stdout, e, catch_backtrace())
        end
    finally
        println("Destroying node and shutting down rclpy.")
        if !isnothing(node)
            node.destroy_node()
        end
        rclpy.shutdown()
    end
end


# ======================================
# --- Callback Functions ---
# ======================================

"""
Callback for Joint State Messages
"""
function joint_state_callback(pymsg, state_channel::Channel{Vector{Float64}})
    
    local names::Vector{String}
    local positions::Vector{Float64}
    local velocities::Vector{Float64}
    
    # 1. Convert ALL required data from Python types
    names = pyconvert(Vector{String}, pymsg.name)
    positions = pyconvert(Vector{Float64}, pymsg.position)
    velocities = pyconvert(Vector{Float64}, pymsg.velocity)
    
    try
        # 2. Create lookup maps (Dictionaries)
        #    This maps each joint name to its current position and velocity
        pos_map = Dict(zip(names, positions))
        vel_map = Dict(zip(names, velocities))

        # 3. Pre-allocate the 14-element array
        state_array = Vector{Float64}(undef, 14)

        # 4. Fill the array in the correct order
        for (i, joint_name) in enumerate(DESIRED_JOINT_ORDER)
            # Check if the joint exists in our map (prevents KeyErrors)
            if !haskey(pos_map, joint_name) || !haskey(vel_map, joint_name)
                @warn "Joint '$joint_name' not found in received message. Skipping."
                return
            end

            # Assign position
            state_array[i] = pos_map[joint_name]
            
            # Assign velocity (offset by 7)
            state_array[i + 7] = vel_map[joint_name]
        end
        put!(state_channel, state_array)

    catch e
        println("Error in callback while creating array:")
        showerror(stdout, e)
    end
end



function target_callback(pymsg, target_channel::Channel{Any})
    try
        # Extract Target Attractor (GeometryMsgs/Point -> SVector)
        tx = pyconvert(Float64, pymsg.target_position.x)
        ty = pyconvert(Float64, pymsg.target_position.y)
        tz = pyconvert(Float64, pymsg.target_position.z)

        put!(target_channel, SVector(tx, ty, tz))

    catch e
        println("Error in target callback:")
        showerror(stdout, e)
    end
end

function obstacles_callback(pymsg, obstacles_channel::Channel{Any})
    try
        obs_vec = Vector{SVector{3, Float64}}()
        py_list = pymsg.obstacles
        
        for single_obstacle in py_list
            ox = pyconvert(Float64, single_obstacle.x)
            oy = pyconvert(Float64, single_obstacle.y)
            oz = pyconvert(Float64, single_obstacle.z)
            push!(obs_vec, SVector(ox, oy, oz))
        end
        
        put!(obstacles_channel, obs_vec)

    catch e
        println("Error obstacles callback: $e")
    end
end

cvms = compile(vms)
qᵛ = Float64[]
ros_vm_controller(cvms, qᵛ; f_control, f_setup, E_max=30.0)
