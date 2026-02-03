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
const VmcControlTarget = pyimport("vmc_interfaces.msg").VmcControlTarget


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
        "fr3_link0",
        "fr3_link1",
        "fr3_link2",
        "fr3_link3",
        "fr3_link4",
        "fr3_link5",
        "fr3_link6",
        "fr3_link7",
        "fr3_hand_tcp"
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

const DIST_NOSE = 0.25              # Distance from camera frame to nose point along Z axis

const CAMERA_OFFSET = Transform(    # Camera mounted on the robot's end-effector has a fixed offset
    SVector(0.095, 0.0, -0.05), 
    Rotor(RotZ(pi/2))
) 

const verbose = false               # Flag to activate the verbose prints               


# --- Shared Channels for Communication ---

state_channel = Channel{Vector{Float64}}(1)         # Buffer for joint states
target_channel = Channel{Any}(2)                    # Buffer for target and obstacles


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

# joint_limits = cfg.joint_limits
#     for (i, τ_coulomb) in zip(1:7, [5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0])
#         jname = "fr3_joint$i"
#         cid = "J$i"
#         add_coordinate!(robot, JointSubspace(jname); id=cid)
#         limits = joint_limits[jname]
#         if !isnothing(limits) && !isnothing(limits.lower) && !isnothing(limits.upper)
#             add_deadzone_springs!(robot, 80.0, (limits.lower+0.1, limits.upper-0.1), cid)
#         end
#         add_component!(robot, TanhDamper(τ_coulomb, 1e-1, cid); id="JDamp$i")
#         # add_component!(robot, LinearDamper(3.0, cid); id="ViscousDamp$i")
#     end


# -- Virtual Mechanism System --

vms = VirtualMechanismSystem("franka_impedance_control", robot)



# ======================================
# --- VMC Creation ---
# ======================================

root = root_frame(vms.robot)

add_coordinate!(robot, FrameOrigin("fr3_hand_tcp"); id="TCP position")                                          # TCP Position Coordinate

# --- Camera Frame & Attractor --- 

add_frame!(robot, "camera_frame")
add_joint!(robot, Rigid(CAMERA_OFFSET); parent="fr3_hand_tcp", child="camera_frame", id="J_Camera_Mount")       # Camera Mount Joint (Displaced Rigidly)

add_coordinate!(robot, FrameOrigin("camera_frame"); id="camera_position")                                       # Camera Position Coordinate
add_coordinate!(robot, FramePoint("camera_frame", SVector(0.0, 0.0, DIST_NOSE)); id="camera_nose")              # Camera Nose Coordinate

add_coordinate!(vms, ReferenceCoord(Ref(SVector(0.4, 0.0, 0.4))); id="target_attractor")                        # Target Attractor Coordinate


# --- Attraction to Target ---

add_coordinate!(vms, CoordDifference("target_attractor", ".robot.camera_nose"); id="cam_to_target_error")

add_component!(vms, TanhSpring("cam_to_target_error"; max_force=15.0, stiffness=200.0); id="attraction_spring")
add_component!(vms, LinearDamper(10.0 * identity(3), "cam_to_target_error"); id="attraction_damper")


# --- Obstacle Repulsors for Camera ---

for i in 1:N_REPULSORS

    # Camera Repulsor Coordinate (Initialized Far Away)
    rep_id = "active_repulsor_$i"
    add_coordinate!(vms, ReferenceCoord(Ref(SVector(99.0, 99.0, 99.0))); id=rep_id)

    # Camera Repulsion Error 
    err_id = "camera_repulsion_error_$i"
    add_coordinate!(vms, CoordDifference(rep_id, ".robot.camera_position"); id=err_id)
    
    # Camera Repulsion Spring
    add_component!(vms, GaussianSpring(err_id; max_force=-10.0, width=0.1); id="repulsor_spring_$i")
end



# --- Obstacle Repulsors for Body ---

for link_name in COLLISION_LINKS

    add_coordinate!(robot, FrameOrigin(link_name); id="$(link_name)_pos")

    for i in 1:N_REPULSORS

        # Body Repulsor Error
        repulsor_id = "active_repulsor_$i"
        error = "$(link_name)_repulsion_error_$i"
        add_coordinate!(vms, CoordDifference(repulsor_id, ".robot.$(link_name)_pos"); id=error)

        # Body Repulsion Spring
        sping_id = "$(link_name)_link_repulsor_spring_$i"
        add_component!(vms, GaussianSpring(error; max_force=-15.0, width=0.08); id=sping_id)
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


    # --- Link Coordinates for Velocity Check ---

    link_names = [
        "fr3_hand_tcp", "fr3_link7", "fr3_link6", "fr3_link5", 
        "fr3_link4", "fr3_link3", "fr3_link2", "fr3_link1"
    ]

    link_coords_ids = []

    for name in link_names
        try
            cid = get_compiled_coordID(cache, ".robot.$(name)_pos")
            push!(link_coords_ids, cid)
        catch
            println("Warning: Coordinate for $name not found, skipping velocity check.")
        end
    end
    
    return (target_id, repulsor_ids, cam_frame_id, link_coords_ids)
end

function f_control(cache, t, args, dt)

    (target_id, repulsor_ids, _, _) = args

    # Non-blocking check for new target data
    if isready(target_channel)

        # Get Data from Target & Obstacles Channel: Expecting Tuple (Target, Obstacles list)
        data = take!(target_channel) 

        new_target = data[1]        # New Target Attractor
        obstacles = data[2]         # New Obstacles List
        
        # println("🎯 New Target Voxel: [x=$(round(new_target[1], digits=3)), y=$(round(new_target[2], digits=3)), z=$(round(new_target[3], digits=3))]")
        println("t: $(round(t, digits=2))")
        
        # -- Update Target ---
        
        cache[target_id].coord_data.val[] = SVector(new_target[1], new_target[2], new_target[3])
        

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

    end
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
        sub_target = node.create_subscription(VmcControlTarget, "/vmc/target_obstacles", target_callback_wrapper, 10)

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
                            torque_publisher=torque_publisher, state_publisher=state_publisher,
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
                f_control(control_cache, t, args, dt) 
                
                # Physics Step
                desired_torques = control_step!(control_cache, t, qʳ, q̇ʳ) 
                
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

                (all_torques, _) = VMRobotControl.get_generalized_force(control_cache)

                current_torques = all_torques[1:7]                              # First 7 are joint torques

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
        tx = pyconvert(Float64, pymsg.target_attractor.x)
        ty = pyconvert(Float64, pymsg.target_attractor.y)
        tz = pyconvert(Float64, pymsg.target_attractor.z)

        # Create Target SVector
        target_vec = SVector(tx, ty, tz)
        
        # Extract Obstacles (GeometryMsgs/Point[] -> Vector{SVector})
        obs_vec = Vector{SVector{3, Float64}}()
        py_obstacles = pymsg.active_obstacles
        
        for py_pt in py_obstacles
            ox = pyconvert(Float64, py_pt.x)
            oy = pyconvert(Float64, py_pt.y)
            oz = pyconvert(Float64, py_pt.z)
            push!(obs_vec, SVector(ox, oy, oz))
        end
        
        # Put tuple (Target, Obstacles) into channel
        put!(target_channel, (target_vec, obs_vec))

    catch e
        println("Error in target callback:")
        showerror(stdout, e)
    end
end

cvms = compile(vms)
qᵛ = Float64[]
ros_vm_controller(cvms, qᵛ; f_control, f_setup, E_max=30.0)


