# ==============================================================================
#   MACRO SECTION 1: IMPORTS & INFRASTRUCTURE
#   Dependencies, libraries, and basic robot configuration.
# ==============================================================================

using GeometryBasics: Vec3f, Point3f, TriangleFace, Rect3f
using GLMakie
using LinearAlgebra, StaticArrays
using FileIO, UUIDs, MeshIO
using Printf
using VMRobotControl
using OrdinaryDiffEq
using VMRobotControl: Transform, FramePoint, CoordDifference, CoordNorm, ConstCoord, Prismatic, Rigid, ReferenceCoord, GaussianSpring, LinearSpring, LinearDamper, TanhSpring
using DifferentialEquations
using Rotations
using Random

# Include utility scripts
include("utils.jl")
include("params.jl")

# Enable DAE format for Franka meshes
try
    FileIO.add_format(format"DAE", (), ".dae", [:DigitalAssetExchangeFormatIO => UUID("43182933-f65b-495a-9e05-4d939cea427d")])
catch
end

# ------------------------------------------------------------------------------
#   Micro Section: Robot URDF & Joint 
# ------------------------------------------------------------------------------

cfg = URDFParserConfig(;suppress_warnings=true)
module_path = joinpath(splitpath(splitdir(pathof(VMRobotControl))[1])[1:end-1])
robot = parseURDF(joinpath(module_path, "URDFs/franka_description/urdfs/fr3_franka_hand_cam.urdf"), cfg)

add_gravity_compensation!(robot, VMRobotControl.DEFAULT_GRAVITY)
joint_limits = cfg.joint_limits

for (i, τ_coulomb) in zip(1:7, [5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0])
    β = 1e-1
    limits = joint_limits["fr3_joint$i"]
    isnothing(limits) && continue
    @assert ~isnothing(limits.lower) && ~isnothing(limits.upper)
    add_coordinate!(robot, JointSubspace("fr3_joint$i"); id="J$i")
    add_deadzone_springs!(robot, 50.0, (limits.lower+0.2, limits.upper-0.2), "J$i")
    add_component!(robot, TanhDamper(τ_coulomb, β, "J$i"); id="JointDamper$i")
end

ORDERED_LINK_NAMES = ["fr3_hand_tcp", "fr3_link7", "fr3_link6", "fr3_link5", "fr3_link4", "fr3_link3", "fr3_link2", "fr3_link1", "fr3_link0"]

vms = VirtualMechanismSystem("gripper_exploration_system", robot)

savepath = "./AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.mp4";

# ==============================================================================
#   MACRO SECTION 2: ENVIRONMENT & VOXEL GRID
#   Workspace definition, Voxel structure, and Obstacle generation.
# ==============================================================================

# Initialize Grid and Obstacles
const GLOBAL_VOXEL_GRID = build_voxel_grid_3d()
@info "$(N_VOXELS_X)x$(N_VOXELS_Y)x$(N_VOXELS_Z) grid successfully created."


const ALL_OBSTACLES = generate_random_obstacles_3d(N_OBSTACLES, is_random_3d, GLOBAL_VOXEL_GRID) # Mutates GLOBAL_VOXEL_GRID

@info "Creating global occupancy set for occlusion..."
const ALL_OCCUPIED_VOXELS = create_occupancy_set_3d(ALL_OBSTACLES)
@info "$(length(ALL_OCCUPIED_VOXELS)) voxels are occupied in total."

# ==============================================================================
#   MACRO SECTION 3: VIRTUAL MECHANISM SYSTEM (VMS) SETUP
#   Definition of Robot Frames, Virtual cam, and Impedance Control.
# ==============================================================================

# ------------------------------------------------------------------------------
#   Micro Section: Virtual Mechanism for the Robot
# ------------------------------------------------------------------------------

add_frame!(robot, "camera_frame")
add_joint!(robot, Rigid(Transform(CAMERA_OFFSET)); parent="fr3_hand_tcp", child="camera_frame", id="J_Camera_Mount")

# Coordinates 'camera pos' 
add_coordinate!(robot, FrameOrigin("camera_frame"); id="camera_position")

# Coordinates 'camera_nose' 
add_coordinate!(robot, FramePoint("camera_frame", SVector(0.0, 0.0, DIST_NOSE)); id="camera_nose")

# Target point (Riferimento esplorazione)
add_coordinate!(vms, ReferenceCoord(Ref(SVector(0.0, 0.0, 0.0))); id="target_attractor")

# ------------------------------------------------------------------------------
#   Micro Section: Repulsive Fields & Steering
# ------------------------------------------------------------------------------

@info "Setting up $N_OBSTACLES dynamic obstacle repulsors..."

REPLUSOR_POS_DEFAULT = SVector(99.0, 99.0, 99.0) # Placeholder

for i in 1:N_OBSTACLES
    repulsor_id = "active_repulsor_$i"
    add_coordinate!(vms, ReferenceCoord(Ref(REPLUSOR_POS_DEFAULT)); id=repulsor_id)

    # VIRTUAL cam to OBSTACLES
    error_id = "camera_repulsion_error_$i"
    add_coordinate!(vms, CoordDifference(repulsor_id, ".robot.camera_position"); id=error_id)
    add_component!(vms, GaussianSpring(error_id; max_force=REPLUSOR_MAX_FORCE, width=REPLUSOR_WIDTH); id="CamRepulsorSpring_$i")

end

# ------------------------------------------------------------------------------
#   Micro Section: Bosy Collision Avoidance
# ------------------------------------------------------------------------------
@info "Setting up full-body collision avoidance springs (Obstacle + Floor)..."

IGNORED_FLOOR_LINKS = ["fr3_link0", "fr3_link1", "fr3_link2"]

for link_name in ORDERED_LINK_NAMES
    
    add_coordinate!(robot, FrameOrigin(link_name); id="$(link_name)_pos")

    # Repulsors on each link for OBSTACLES
    for i in 1:N_OBSTACLES
        
        repulsor_id = "active_repulsor_$i" 

        error_id = "$(link_name)_repulsion_error_$i"
        
        add_coordinate!(vms, CoordDifference(repulsor_id, ".robot.$(link_name)_pos"); id=error_id)
        
        spring_id = "$(link_name)_RepulsorSpring_$i"
        add_component!(vms, GaussianSpring(error_id; max_force=REPLUSOR_MAX_FORCE, width=REPLUSOR_WIDTH+0.03); id=spring_id)
    end

    
    # Repulsor on each link for FLOOR
    shadow_id = "floor_shadow_$(link_name)"

    add_coordinate!(vms, ReferenceCoord(Ref(SVector(0.0, 0.0, FLOOR_Z_LEVEL))); id=shadow_id)

    floor_error_id = "floor_error_$(link_name)"
    add_coordinate!(vms, CoordDifference(shadow_id, ".robot.$(link_name)_pos"); id=floor_error_id)
    
    if !(link_name in IGNORED_FLOOR_LINKS)
        add_component!(vms, GaussianSpring(floor_error_id; max_force=FLOOR_REPULSION_MAX_FORCE, width=FLOOR_REPULSION_WIDTH); id="FloorSpring_$(link_name)")

        # add_component!(vms, LinearDamper(FLOOR_DAMPING, floor_error_id); id="FloorDamper_$(link_name)")

    end
end
# ------------------------------------------------------------------------------
#   Micro Section: Movement Springs & Attractors
# ------------------------------------------------------------------------------

add_coordinate!(vms, CoordDifference("target_attractor", ".robot.camera_nose"); id="cam_to_target_error")

add_component!(vms, TanhSpring("cam_to_target_error"; stiffness=SPRING_ATTRACTION_GAIN, max_force=MAX_FORCE_SPRING); id="Spring_Error")
add_component!(vms, LinearDamper(DAMPER_ATTRACTION_GAIN, "cam_to_target_error"); id="Damper_Error")

# ==============================================================================
#   MACRO SECTION 4: EXPLORATION LOGIC (THE "BRAIN")
#   Geometric constants, State definition, and Main Logic Function.
# ==============================================================================

# ------------------------------------------------------------------------------
#   Micro Section: FOV Geometry
# ------------------------------------------------------------------------------

@info "Defining FOV pyramid geometry..."
FOV_HEIGHT = 0.50
FOV_V_DEG = 52.0
FOV_H_DEG = 87.0
fov_v_rad = deg2rad(FOV_V_DEG)
fov_h_rad = deg2rad(FOV_H_DEG)
half_width_at_base = FOV_HEIGHT * tan(fov_v_rad / 2)
half_height_at_base = FOV_HEIGHT * tan(fov_h_rad / 2)

tip = Point3f(0.0, 0.0, 0.0)
p1 = Point3f( half_width_at_base,  half_height_at_base, FOV_HEIGHT)
p2 = Point3f(-half_width_at_base,  half_height_at_base, FOV_HEIGHT)
p3 = Point3f(-half_width_at_base, -half_height_at_base, FOV_HEIGHT)
p4 = Point3f( half_width_at_base, -half_height_at_base, FOV_HEIGHT)
LOCAL_FOV_VERTICES = [tip, p1, p2, p3, p4]

FOV_FACES = [
    TriangleFace(2, 3, 4), TriangleFace(2, 4, 5), # Base
    TriangleFace(1, 2, 3), TriangleFace(1, 3, 4), TriangleFace(1, 4, 5), TriangleFace(1, 5, 2) # Sides
]

# ------------------------------------------------------------------------------
#   Micro Section: Logic Function
# ------------------------------------------------------------------------------

"""
The ONE main "brain" function.
Called by both physics and animation loops.
"""
function run_exploration_logic!(
        state::ExplorationState,
        grid::Array{Voxel, 3}, 
        cache, t, args, is_animation
    )
    
    (
    hand_frame_id, 
    target_ref_handle, 
    repulsor_handles_safety,
    nose_id
    ) = args

    # 0. --- SENSING ---
    T_cam = get_transform(cache, hand_frame_id) 

    camera_world_pos = T_cam.origin
    
    fov_verts = map(v -> T_cam * SVector{3, Float64}(v), LOCAL_FOV_VERTICES)
    fov_normals = get_fov_normals(fov_verts)

    voxels_to_prune = Tuple{Int, Int, Int}[]

    for (i, j, k) in state.unknown_voxels

        voxel = grid[i, j, k]
        
        if voxel.state != 0
            push!(voxels_to_prune, (i, j, k))
            continue
        end

        if is_in_fov_3d(voxel.world_center, fov_verts, fov_normals)

            if !is_occluded_3d(camera_world_pos, voxel, ALL_OCCUPIED_VOXELS, grid)

                if (i, j, k) in ALL_OCCUPIED_VOXELS
                    voxel.state = 2 # Occupied
                else
                    voxel.state = 1 # Free
                end

                push!(voxels_to_prune, (i, j, k))

            end
        end
    end 

    
    for coords in voxels_to_prune
        delete!(state.unknown_voxels, coords)
    end 

    # 0.1 --- ZONE UNBANNING LOGIC (Check 30%) ---

    # Check if any ignored zone has become "known enough" to be unbanned
    zones_to_keep = IgnoredZone[]
    
    for zone in state.active_ignored_zones
        
        # Count how many voxels in this zone are currently NOT unknown (state != 0)
        known_count = 0
        for (zi, zj, zk) in zone.coords
            if grid[zi, zj, zk].state != 0
                known_count += 1
            end
        end

        discovery_ratio = known_count / zone.total_count

        if discovery_ratio >= ZONE_UNBAN_THRESHOLD
            # UNBAN: Remove these coords from the blacklist set
            # @info "🔓 Zone unbanned! ($(round(discovery_ratio*100, digits=1))% discovered)"
            for coords in zone.coords
                delete!(state.temporarily_ignored, coords)
            end
            # We do NOT add this zone to zones_to_keep, so it gets dropped
        else
            # Keep the zone active
            push!(zones_to_keep, zone)
        end
    end
    state.active_ignored_zones = zones_to_keep

    # 1. --- TARGET VALIDITY CHECK ---
    
    if !is_animation    # because in animation the target is already given by the log diary, so we need to avoid computing the target
        if state.current_target != (-1, -1, -1) && !(state.current_target in state.unknown_voxels)
            state.current_target = (-1, -1, -1)
        end
    end 

    # 2. --- DISCOVERY RATE CHECK ---

    if !is_animation && (t - state.last_discovery_check_t > DISCOVERY_CHECK_INTERVAL)
        
        current_unknown_count = length(state.unknown_voxels)
        voxels_found = state.unknown_count_last_check - current_unknown_count
        dt_check = t - state.last_discovery_check_t
        
        # Rate calculation
        discovery_rate = (voxels_found / dt_check) / max(1, current_unknown_count)
        
        # Controllo: Se ho ancora un target attivo (quindi NON l'ho trovato sopra)
        # e sono lento, allora è un problema.
        if state.current_target != (-1, -1, -1) && current_unknown_count > 0
            if discovery_rate < MIN_DISCOVERY_RATE_THRESHOLD
                # @info "⚠️ LOW RATE. Triggering Deadlock Logic."
                state.deadlock_detected = true
            end
        end

        state.last_discovery_check_t = t
        state.unknown_count_last_check = current_unknown_count
    end

    # 3. --- DEADLOCK AVOIDANCE ---

    if state.deadlock_detected && state.current_target != (-1, -1, -1)
        
        # 1. Identify the culprit
        (ti, tj, tk) = state.current_target
        state.last_deadlock_coords = state.current_target
        
        # 2. CREATE NEIGHBORHOOD ZONE
        zone_coords = Tuple{Int, Int, Int}[]
        
        extent = DEADLOCK_ZONE_EXTENT
        for di in -extent:extent, dj in -extent:extent, dk in -extent:extent
            ni, nj, nk = ti + di, tj + dj, tk + dk
            
            # Check bounds
            if ni >= 1 && ni <= N_VOXELS_X &&
               nj >= 1 && nj <= N_VOXELS_Y &&
               nk >= 1 && nk <= N_VOXELS_Z
               
               if grid[ni, nj, nk].state == 0
                   push!(zone_coords, (ni, nj, nk))
                   push!(state.temporarily_ignored, (ni, nj, nk))
               end

            end
        end
        
        # 4. Create and store the IgnoredZone object
        if !isempty(zone_coords)
            new_zone = IgnoredZone(zone_coords, length(zone_coords))
            push!(state.active_ignored_zones, new_zone)
            # @info "⛔ Deadlock at $state.current_target. Blacklisted zone of $(length(zone_coords)) voxels."
        end

        # 5. Reset Target and Flag
        state.current_target = (-1, -1, -1)
        state.deadlock_detected = false
    end

    # 4. --- THINKING (Targeting) ---
    target_world_pos = SVector(0.0, 0.0, 0.0)

    if state.current_target == (-1, -1, -1)

        best_target = find_closest_unknown_target_3d(
            camera_world_pos, 
            state.unknown_voxels, 
            grid, 
            state.temporarily_ignored,
            state.last_deadlock_coords
        )
        

        if best_target == (-1, -1, -1) && !isempty(state.temporarily_ignored)
            
            state.last_deadlock_coords = (-1, -1, -1)

            # @info "\n♻️  No easy targets left. Retrying previously ignored targets.\n"
            empty!(state.temporarily_ignored)
            
            best_target = find_closest_unknown_target_3d(
                camera_world_pos, 
                state.unknown_voxels, 
                grid, 
                state.temporarily_ignored,
                state.last_deadlock_coords
            )
        end
        
        state.current_target = best_target
    end

    if isempty(state.unknown_voxels) && isempty(state.temporarily_ignored)
        if !state.all_voxels_seen
             
             state.all_voxels_seen = true
        end
        return (SVector(0.0, 0.0, 0.0), (-1,-1, -1), voxels_to_prune, SVector{3, Float64}[])
    end
    
    
    if state.current_target == (-1, -1 ,-1)
        # --- DEBUG REPORT UNICO ---
        io = IOBuffer()
        println(io, "\n🛑 CRITICAL STOP: TARGET SELECTION FAILED!")
        println(io, "==============================================================")
        
        # 1. LISTA IGNORATI (Blacklist)
        n_ignored = length(state.temporarily_ignored)
        ignored_sorted = sort(collect(state.temporarily_ignored))
        
        println(io, "🚫 1. IGNORED TARGETS (Blacklist) [Count: $n_ignored]")
        if n_ignored > 0
            # Li stampiamo tutti in una riga o blocco leggibile
            println(io, ignored_sorted)
        else
            println(io, "   (None)")
        end
        println(io, "--------------------------------------------------------------")

        # 2. LISTA SCONOSCIUTI RIMASTI (Quelli che vorrebbe vedere ma non raggiunge)
        n_unknown = length(state.unknown_voxels)
        unknown_sorted = sort(collect(state.unknown_voxels))
        
        println(io, "📦 2. UNKNOWN VOXELS (Remaining) [Count: $n_unknown]")
        if n_unknown > 0
            # Stampiamo la lista completa
            println(io, unknown_sorted)
        else
            println(io, "   (None - Exploration actually complete?)")
        end
        println(io, "==============================================================\n")
        
        # Stampa tutto in un colpo solo
        print(String(take!(io)))
        # ---------------------------
        @info "No valid target found, exploration complete."
        state.all_voxels_seen = true
        return (SVector(0.0, 0.0, 0.0), (-1,-1,-1), voxels_to_prune, SVector{3, Float64}[])

    else 
        # We have a target
        current_target_voxel = grid[state.current_target...]
        target_world_pos = current_target_voxel.world_center
        target_coords = current_target_voxel.grid_coords

        # Update the physical target reference
        target_ref_handle[] = target_world_pos
    end
    

    # 5. --- THINKING (Avoidance) ---
    visible_obstacles_list = ObstacleState[]


    # Update safety repulsors (all N obstacles)
    for (i, obs) in enumerate(ALL_OBSTACLES)

        # Visibility logic: if I see one obstacle voxel then I see the obstacle center
        if !obs.is_visible
            for (vx, vy, vz) in obs.voxels
                if grid[vx, vy, vz].state == 2
                    obs.is_visible = true 
                    break
                end
            end
        end

        if obs.is_visible
            # Update the safety spring (Body & Camera)
            repulsor_handles_safety[i][] = obs.center_pos
            push!(visible_obstacles_list, obs)

        else
            repulsor_handles_safety[i][] = REPLUSOR_POS_DEFAULT
        end
    end

    all_visible_obst_pos = SVector{3, Float64}[]
    for obs in visible_obstacles_list
         push!(all_visible_obst_pos, obs.center_pos)
    end


    # Return forces AND the list of changed voxels
    return (target_world_pos, target_coords, voxels_to_prune, all_visible_obst_pos)
end


# ==============================================================================
#   MACRO SECTION 5: PHYSICS SIMULATION LOOP
#   Callbacks and ODE Solver execution.
# ==============================================================================

"""
Called ONCE before the simulation starts.
Gathers all the "handles" (IDs, indices, Refs).
"""
function f_setup_physics(cache)
    @info "Physics setup initiated..."
    
    # 1. Initialize the exploration state
    unknown_set = Set{Tuple{Int, Int, Int}}()
    for i in 1:N_VOXELS_X, j in 1:N_VOXELS_Y, k in 1:N_VOXELS_Z
        GLOBAL_VOXEL_GRID[i, j, k].state = 0 
        push!(unknown_set, (i, j, k))
    end

    println("\n📦 --- VOXEL GRID CONFIGURATION ---")
    @printf("   🔹 Voxel Size X: %.4f cm\n", VOXEL_SIZE_X*100)
    @printf("   🔹 Voxel Size Y: %.4f cm\n", VOXEL_SIZE_Y*100)
    @printf("   🔹 Voxel Size Z: %.4f cm\n", VOXEL_SIZE_Z*100)
    @printf("   🔹 Grid Dims   : %d x %d x %d\n", N_VOXELS_X, N_VOXELS_Y, N_VOXELS_Z)
    @printf("   🔹 Total Voxels: %d\n", N_VOXELS_TOTAL)
    println("------------------------------------\n")

    println("\n📦 --- ALGORITHM PARAMETERS ---")
    if USE_FRONTALITY_HEURISTIC 
        @printf("   🔹 Weight Frontality: %.1f m\n", W_FRONTALITY)
    end
    if USE_RELEVANCE_HEURISTIC
        @printf("   🔹 Weight Relevance : %.1f m\n", W_RELEVANCE)
    end

    @printf("   🔹 Weight Distance  : %.1f m\n", W_DISTANCE)
    println("------------------------------------\n")
    @printf("   🔹 Deadlock Distance Trigger: %.1f N\n", DEADLOCK_TRIGGER)
    @printf("   🔹 Deadlock Distance Radius : %.2f cm\n", DEADLOCK_IGNORE_RADIUS*100)
    println("------------------------------------\n")

    # ---------------------------------

    empty!(FULL_SIMULATION_LOG)

    state = ExplorationState(unknown_set, 
        (N_VOXELS_X, 1, N_VOXELS_Z ÷ 2), 
        0.0, 
        false, 
        -1, 
        Set{Tuple{Int, Int, Int}}(), 
        false, 
        0.0, 
        false, 
        (-1, -1, -1),
        0.0,
        length(unknown_set),
        IgnoredZone[],
        0.0,
        0.0,
        0
    )

    # Manipulator Springs ID + Frame Coords ID + Floor Repulsor IDs
    link_springs_map = []
    link_frame_coord = []
    spring_ids = []
    floor_repulsor_ids = String[]
    for link_name in ORDERED_LINK_NAMES
        # Raccogliamo tutte le molle (una per ostacolo) associate a questo link
        
        for i in 1:N_OBSTACLES
            # Nota: Assicurati che i nomi delle molle nel tuo URDF/VMS matchino questo pattern
            push!(spring_ids, get_compiled_componentID(cache, "$(link_name)_RepulsorSpring_$i"))
        end
        push!(link_springs_map, (link_name, spring_ids))
        push!(link_frame_coord, (link_name, get_compiled_coordID(cache, ".robot.$(link_name)_pos")))
        push!(floor_repulsor_ids, "floor_shadow_$(link_name)")
    end

    floor_shadow_handles = [cache[get_compiled_coordID(cache, fid)].coord_data.val for fid in floor_repulsor_ids]


    # Joint IDs Handle
    joint_ids_handle = []
    for i in 1:7
        # We named them "J1", "J2"... in the setup loop
        push!(joint_ids_handle, ("J$i", get_compiled_coordID(cache, ".robot.J$i")))
    end

    # Get handles
    handles = (
        get_compiled_frameID(cache, ".robot.camera_frame"), 
        cache[get_compiled_coordID(cache, "target_attractor")].coord_data.val, 
        [cache[get_compiled_coordID(cache, "active_repulsor_$i")].coord_data.val for i in 1:N_OBSTACLES],
        get_compiled_coordID(cache, ".robot.camera_nose"),
        [get_compiled_componentID(cache, "CamRepulsorSpring_$i") for i in 1:N_OBSTACLES],
        get_compiled_componentID(cache, "Spring_Error"),
        [get_compiled_componentID(cache, "fr3_hand_tcp_RepulsorSpring_$i") for i in 1:N_OBSTACLES],
        link_springs_map,
        link_frame_coord,
        joint_ids_handle,
        floor_shadow_handles
    )

    is_animation = false
    
    @info "Physics setup complete. $(length(unknown_set)) unknown voxels."
    return (state, handles, is_animation)
end

"""
Called at EVERY time-step of the physics simulation.
"""
function f_control_physics(cache, t, args, extra)
    
    t_start_function = time_ns()

    # 1. Unpack state and handles
    state, control_args, is_animation = args
    (hand_frame_id, target_ref_handle, repulsor_handles_safety, nose_id, repulsion_spring_ids, attraction_spring_id, tcp_repulsion_ids, link_springs_map, link_frame_coord, joint_ids_handle, floor_shadow_handles) = control_args
    
    # 2. Calculate dt
    current_dt = t - state.last_t
    state.debug_step_count += 1
    state.last_t = t
    
    # 3. Run the "brain"
    t_logic_start = time_ns()
    (target_world_pos, target_coords, voxels_to_prune, all_visible_obst_pos) = run_exploration_logic!(state, GLOBAL_VOXEL_GRID, cache, t, control_args, is_animation)
    t_logic_end = time_ns() 

    state.debug_timer_logic += (t_logic_end - t_logic_start) / 1e9 # Convert to seconds
    
    # Completion Bar
    percent = floor(Int, (t / duration) * 100)

    now_wall = time()

    if percent > state.last_printed_percent || (now_wall - state.last_console_update_t > 0.25)
        
        bar_width = 30 
        filled = floor(Int, bar_width * (percent / 100))
        empty = bar_width - filled
        
        bar_str = "█"^filled * "░"^empty

        avg_brain = 0.0
        if state.debug_step_count > 0
            avg_brain = (state.debug_timer_logic / state.debug_step_count) * 1000
        end

        line_bar  = "⏳ Simulation: [$bar_str] $percent% (t=$(round(t, digits=1))s)"

        dt_str = @sprintf("%.1e", current_dt)
        line_perf = "⏱️ Average Time: - Brain: $(round(avg_brain, digits=3))ms | dt: $dt_str s"

        if state.last_printed_percent == -1
            println(line_bar)
            print(line_perf)
        else
            print("\e[1A\r\e[2K$line_bar\n\e[2K$line_perf")
        end

        state.last_printed_percent = percent
        state.last_console_update_t = now_wall

    end


    # --- LOG DIARY UPDATE FOR ANIMATION ---
    # 1. Catch the change in state
    voxel_updates = [(i, j, k, GLOBAL_VOXEL_GRID[i, j, k].state) for (i, j, k) in voxels_to_prune]

    # 2. Visible obstacles
    visible_indices = [i for (i, obs) in enumerate(ALL_OBSTACLES) if obs.is_visible]

    ignored_vec = collect(state.temporarily_ignored)

    # -- Compute Velocities of Links ---
    link_vel_data = Vector{Tuple{String, SVector{3, Float64}}}()
    for (name, fcoord) in link_frame_coord
        # velocity(cache, fid) returns the linear velocity of the frame origin
        v_link = velocity(cache, fcoord)
        push!(link_vel_data, (name, v_link))
    end

    # =========================================================================
    # --- NEW: JOINT TORQUES ONLY ---
    # =========================================================================
    
    # 1. Recuperiamo solo le forze (la velocità non la chiamiamo nemmeno)
    (all_torques_robot, _) = VMRobotControl.get_generalized_force(cache)
    
    joint_torques_data = Vector{Tuple{String, Float64}}()
    
    for i in 1:7
        # Salviamo solo Nome e Torque
        push!(joint_torques_data, ("J$i", all_torques_robot[i]))
    end
    # =========================================================================

    # -------------------------------------------------------------------------
    # --- DEADLOCK DETECTION LOGIC ---
    # -------------------------------------------------------------------------
    # Condition: All Links Velocity < EPS  AND  All Joint Torques < EPS
    
    # Verifica Velocità (tutti i link devono essere fermi)
    all_links_stopped = all(norm(v) < DEADLOCK_VEL_EPS for (_, v) in link_vel_data)

    # Verifica Torques (tutti i giunti devono avere coppia bassa/nulla)
    all_joints_relaxed = all(abs(tau) < DEADLOCK_TORQUE_EPS for (_, tau) in joint_torques_data)

    # Trigger
    if all_links_stopped  && all_joints_relaxed && state.current_target != (-1, -1, -1)
        # Se siamo fermi, rilassati, ma abbiamo ancora un target attivo -> Deadlock (minimo locale)
        state.deadlock_detected = true
        # @info "💀 Deadlock Detected at t=$(round(t, digits=2))s (Low Vel & Low Torque)"
    end
    # -------------------------------------------------------------------------

    # ==========================================================================
    # --- UPDATE FLOOR SHADOWS ---
    # ==========================================================================
    
    for (i, (link_name, fid)) in enumerate(link_frame_coord)
        
        # 1. Ottieni la trasformata corrente del link 'i'
        link_pos = configuration(cache, fid)
        
        # 2. Aggiorna l'ombra corrispondente (indice 'i')
        # La mettiamo esattamente sotto il link (X, Y) ma inchiodata al pavimento (Z)
        floor_shadow_handles[i][] = SVector(link_pos[1], link_pos[2], FLOOR_Z_LEVEL)
        
    end

    # 3. Create Snapshot
    step_data = SimStepData(
        t,

        target_coords,
        target_world_pos,     
        voxel_updates,
        visible_indices,
        state.deadlock_detected,
        ignored_vec,

        joint_torques_data,

        link_vel_data,

        length(state.unknown_voxels)
    )

    # Insert the data
    push!(FULL_SIMULATION_LOG, step_data)
    # --------------------------------------    

    if state.all_voxels_seen && !state.success_printed
         state.success_printed = true
         @info "\n🎉 All voxels have been explored at t:$(round(t, digits=1))s\n"
    end
end

# ------------------------------------------------------------------------------
#   Micro Section: Simulation Execution
# ------------------------------------------------------------------------------

@info "Starting simulation..."

tspan = (0.0, duration)

# Initial joint angles
q_initial = ([0.0, -pi/4, 0.0, -3*pi/4, 0.0, pi/2, pi/4], Float64[]) # Robot joint angle, vm joint angles
qvel_initial = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float64[]) # Robot joint velocity, vm joint velocities

# Compile and Solve
g = VMRobotControl.DEFAULT_GRAVITY
compiled_vms = compile(vms)
dcache = new_dynamics_cache(compiled_vms)

prob = get_ode_problem(dcache, g, q_initial, qvel_initial, tspan; f_setup=f_setup_physics, f_control=f_control_physics)

@info "Running ODE solver for $duration seconds..."
sol = solve(prob, Rodas5P(autodiff=false); maxiters=1e7, abstol=1e-6, reltol=1e-6, dtmin=1e-3, force_dtmin=true)
# sol = solve(prob, Tsit5(); maxiters=1e7, abstol=1e-6, reltol=1e-6, dtmax=0.005)

@info "\nSimulation completed."

# ==============================================================================
#   MACRO SECTION 6: VISUALIZATION & ANIMATION
#   Replay of the simulation with visual cues.
# ==============================================================================

@info "Initializing animation state..."

# 1. Define colors
COLOR_UNKNOWN = RGBAf(0.5, 0.5, 0.5, 0.2)
COLOR_FREE = RGBAf(0.0, 0.0, 0.0, 0.0)
COLOR_OCCUPIED = RGBAf(1.0, 0.0, 0.0, 0.8)
COLOR_TARGET = RGBAf(1.0, 1.0, 0.0, 0.3)
COLOR_IGNORED = RGBAf(0.0, 0.0, 1.0, 0.4) 

# 2. Observables
plotting_t = Observable(0.0)
plotting_kcache = Observable(new_kinematics_cache(compiled_vms))

voxel_colors_ANIMATION = Observable(fill(COLOR_UNKNOWN, N_VOXELS_TOTAL))

cam_pos_ANIMATION = Observable(Point3f(0))

nose_arrow_origin_ANIMATION = Observable([Point3f(0)])
nose_arrow_vec_ANIMATION    = Observable([Vec3f(0)])

cam_spring_lines_ANIMATION = Observable(Point3f[])

nose_to_target_ANIMATION = Observable(Point3f[])

nose_spring_line_ANIMATION = Observable([Point3f(0)])

floor_spring_lines_ANIMATION = Observable(Point3f[])

# Static voxel positions for plotting
voxel_centers_display = [GLOBAL_VOXEL_GRID[i,j,k].world_center + DISPLAY_OFFSET for i in 1:N_VOXELS_X, j in 1:N_VOXELS_Y, k in 1:N_VOXELS_Z][:]

# HUD
completion_text_ANIMATION = Observable("Exploration: 0.0%")
time_text_ANIMATION = Observable("Time: 0.00 s")
warning_vel_ANIMATION = Observable("")
warning_tau_ANIMATION = Observable("")
# ------------------------------------------------------------------------------
#   Micro Section: Animation Logic
# ------------------------------------------------------------------------------

GLOBAL_VOXEL_GRID_ANIMATION = deepcopy(GLOBAL_VOXEL_GRID)

"""
Updates only the GLMakie Observables.
"""
function update_animation_observables!(
        cache,
        hand_frame_id,
        nose_id,
        target_world_pos,
        target_coords,
        all_visible_obst_pos,
        ignored_list_vec,
        link_pos_coords
    )

    ignored_set = Set(ignored_list_vec)

    # 1. Colors
    new_colors = copy(voxel_colors_ANIMATION[])

    for i in 1:N_VOXELS_X
        for j in 1:N_VOXELS_Y
            for k in 1:N_VOXELS_Z

                state = GLOBAL_VOXEL_GRID_ANIMATION[i, j, k].state
                
                idx = i + (j-1)*N_VOXELS_X + (k-1)*(N_VOXELS_X*N_VOXELS_Y)

                if state == 0
                    new_colors[idx] = COLOR_UNKNOWN
                elseif state == 1
                    new_colors[idx] = COLOR_FREE
                elseif state == 2
                    new_colors[idx] = COLOR_OCCUPIED
                end

                if (i, j, k) in ignored_set
                    new_colors[idx] = COLOR_IGNORED
                end

                # Evidenzia il target corrente in giallo
                if (i, j, k) == target_coords && target_coords != (-1, -1, -1)
                    new_colors[idx] = COLOR_TARGET
                end

            end
        end
    end
    
    voxel_colors_ANIMATION[] = new_colors
    
    # 2. cam 
    T_cam = get_transform(cache, hand_frame_id) 
    camera_world_pos = T_cam.origin
    cam_pos_ANIMATION[] = Point3f(camera_world_pos)

    # 3. Nose to Target
    current_nose_pos = Point3f(configuration(cache, nose_id))
    nose_spring_line_ANIMATION[] = [current_nose_pos, Point3f(target_world_pos)]

    # 4. Direction Line
    T_cam = get_transform(cache, hand_frame_id) 
    
    heading_vec = T_cam.rotor * SVector(0.0, 0.0, 1.0)
    
    arrow_vector = Vec3f(heading_vec * DIST_NOSE)

    nose_arrow_origin_ANIMATION[] = [camera_world_pos]
    nose_arrow_vec_ANIMATION[]    = [Vec3f(heading_vec * DIST_NOSE)]

    # 5. cam to Obstacles
    spring_segments = Point3f[]
    for obs_pos in all_visible_obst_pos

        if obs_pos[3] < 50.0 

            push!(spring_segments, camera_world_pos)
            push!(spring_segments, Point3f(obs_pos))

        end
    end
    cam_spring_lines_ANIMATION[] = spring_segments

    # 6. Floor Shadows to Obstacles
    floor_segments = Point3f[]

    IGNORED_FLOOR_LINKS = ["fr3_link0", "fr3_link1", "fr3_link2"]

    # For each link, draw springs to visible obstacles till the treshold imposed
    VISUAL_THRESHOLD = FLOOR_REPULSION_WIDTH * 3.5

    for (link_name, fid) in link_pos_coords

        if link_name in IGNORED_FLOOR_LINKS
            continue
        end

        # Recuperiamo la posizione del link usando configuration() 
        pos = configuration(cache, fid)
        
        # Check altezza
        if pos[3] < VISUAL_THRESHOLD
            # Punto sul Link
            push!(floor_segments, Point3f(pos))
            # Punto Ombra sul Pavimento (z = FLOOR_Z_LEVEL)
            push!(floor_segments, Point3f(pos[1], pos[2], FLOOR_Z_LEVEL))
        end
    end

    floor_spring_lines_ANIMATION[] = floor_segments


end

function f_setup_animation(cache)
    # Reset animation grid
    for i in 1:N_VOXELS_X, j in 1:N_VOXELS_Y, k in 1:N_VOXELS_Z
        GLOBAL_VOXEL_GRID_ANIMATION[i, j, k].state = 0
    end

    for obs in ALL_OBSTACLES
        obs.is_visible = false
    end

    unknown_set_anim = Set{Tuple{Int, Int, Int}}()
    for i in 1:N_VOXELS_X, j in 1:N_VOXELS_Y, k in 1:N_VOXELS_Z
        push!(unknown_set_anim, (i, j, k))
    end
    

    state_anim = ExplorationState(
        unknown_set_anim, 
        (N_VOXELS_X, 1, N_VOXELS_Z ÷ 2), 
        0.0, 
        false, 
        -1, 
        Set{Tuple{Int, Int, Int}}(), 
        false, 
        -1.0, 
        false, 
        (-1, -1, -1), 
        0.0, 
        length(unknown_set_anim),
        IgnoredZone[],
        0.0,
        0.0,
        0
    )

    voxel_colors_ANIMATION[] = fill(COLOR_UNKNOWN, N_VOXELS_TOTAL)
    
    link_force_map = []
    link_pos_coords = []

    for link_name in ORDERED_LINK_NAMES
        
        spring_ids = []
        for i in 1:N_OBSTACLES
            push!(spring_ids, get_compiled_componentID(cache, "$(link_name)_RepulsorSpring_$i"))
        end
        
        push!(link_force_map, (link_name, spring_ids))
        push!(link_pos_coords, (link_name, get_compiled_coordID(cache, ".robot.$(link_name)_pos")))
    end

    
    # --------------------------------

    # Return handles (same as physics)
    handles = (
        get_compiled_frameID(cache, ".robot.camera_frame"),     
        cache[get_compiled_coordID(cache, "target_attractor")].coord_data.val, 
        [cache[get_compiled_coordID(cache, "active_repulsor_$i")].coord_data.val for i in 1:N_OBSTACLES],
        get_compiled_coordID(cache, ".robot.camera_nose"),
        [get_compiled_componentID(cache, "CamRepulsorSpring_$i") for i in 1:N_OBSTACLES],
        get_compiled_componentID(cache, "Spring_Error"),
        link_force_map,
        link_pos_coords
    )

    is_animation = true

    log_replay_index = Ref(0)

    return (state_anim, handles, is_animation, log_replay_index)
end


"""
Animation Loop: Replays physics data from FULL_SIMULATION_LOG.
"""
function f_control_animation(cache, t, args, extra)
    # 1. Unpack arguments (Requires log_idx_ref from setup return)
    state_anim, control_args, is_animation, log_idx_ref = args
    (hand_frame_id, target_ref_handle, repulsor_handles_safety, nose_id, _, _, _, link_pos_coords) = control_args

    state_anim.last_t = t

    # Safety check: if log is empty, do nothing
    if isempty(FULL_SIMULATION_LOG)
        return
    end

    # 2. Find the target snapshot based on time 't'
    # searchsortedlast finds the last entry where entry.t <= t
    target_idx = searchsortedlast(FULL_SIMULATION_LOG, t, by = x -> x isa SimStepData ? x.t : x)    
    
    # If t is before the first log entry, wait
    if target_idx == 0
        return
    end

    # 3. "Time Machine": Apply cumulative voxel updates
    current_idx = log_idx_ref[]

    # Handle rewind/loop (e.g., video restarts): Reset grid and index
    if target_idx < current_idx
        current_idx = 0
        for i in 1:N_VOXELS_X, j in 1:N_VOXELS_Y, k in 1:N_VOXELS_Z
            GLOBAL_VOXEL_GRID_ANIMATION[i, j, k].state = 0
        end
    end

    # Catch-up loop: Apply ALL voxel changes from last frame up to current target time
    # This ensures we don't miss any voxel discovery between animation frames
    while current_idx < target_idx
        current_idx += 1
        snap = FULL_SIMULATION_LOG[current_idx]
        
        for (i, j, k, new_state) in snap.voxel_updates
            GLOBAL_VOXEL_GRID_ANIMATION[i, j, k].state = new_state
        end
    end
    
    # Save current index for the next frame
    log_idx_ref[] = current_idx

    # 4. Retrieve Data Snapshot
    snap = FULL_SIMULATION_LOG[target_idx]

    # 5. Update Visuals (Observables)
    # Reconstruct visible obstacle positions using indices from the log
    visible_obst_pos = [ALL_OBSTACLES[i].center_pos for i in snap.visible_obstacle_indices]

    update_animation_observables!(
        cache, 
        hand_frame_id,
        nose_id,
        snap.target_world_pos, 
        snap.target_coords,
        visible_obst_pos,
        snap.ignored_list,
        link_pos_coords
    )

    # 6. Update HUD Observables
    state_anim.deadlock_detected = snap.deadlock_active
    
    # HUD Update
    n_unknown = count(v -> v.state == 0, GLOBAL_VOXEL_GRID_ANIMATION)
    completion_perc = 100.0 * (1.0 - (n_unknown / N_VOXELS_TOTAL))
    completion_text_ANIMATION[] = @sprintf("Exploration: %.1f%%\nUnknown Voxels: %d", completion_perc, n_unknown)
    time_text_ANIMATION[] = @sprintf("Time: %.2f s", t)

    # --- LOGICA AVVISI A COMPARSA ---
    
    # 1. Controllo Velocità
    max_vel_val = 0.0
    if !isempty(snap.link_velocities)
        max_vel_val = maximum(norm(v) for (_, v) in snap.link_velocities)
    end

    if max_vel_val < DEADLOCK_VEL_EPS
        # La scritta appare solo se siamo sotto la soglia
        warning_vel_ANIMATION[] = "⚠️ ALL LINKS STOPPED (< $(DEADLOCK_VEL_EPS))"
    else
        # Altrimenti scompare
        warning_vel_ANIMATION[] = ""
    end

    # 2. Controllo Torques
    max_tau_val = 0.0
    if !isempty(snap.joint_torques)
        max_tau_val = maximum(abs(tau) for (_, tau) in snap.joint_torques)
    end

    if max_tau_val < DEADLOCK_TORQUE_EPS
        # La scritta appare solo se siamo sotto la soglia
        warning_tau_ANIMATION[] = "⚠️ ALL JOINTS RELAXED (< $(DEADLOCK_TORQUE_EPS))"
    else
        # Altrimenti scompare
        warning_tau_ANIMATION[] = ""
    end

    # 6. PRINT TELEMETRY
    PRINT_INTERVAL = 0.0
    
    if (t - state_anim.last_printed_time) >= PRINT_INTERVAL
        
        io = IOBuffer()
        println(io, "📊 --- TELEMETRY (t=$(round(t, digits=2)) s) ---")
        
        # A. TARGET
        @printf(io, "   🎯 Target Coords: [%d, %d, %d]\n\n   📦 Unknown: %-4d | 🚫 Ignored: %-4d\n\n", snap.target_coords..., snap.count_unknown, length(snap.ignored_list))

        # D. ROBOT LINK VELOCITIES
        println(io, "🏎️ --- ROBOT LINK VELOCITIES ---")
        for (name, v_link) in snap.link_velocities
             short_name = replace(name, "fr3_" => "")
             @printf(io, "   🚀 %-10s:        [%.2f, %.2f, %.2f] | Mag: %.5f m/s\n", short_name, v_link..., norm(v_link))
        end
        println(io, "")

        # E. JOINT TORQUES ---
        println(io, "🔧 --- JOINT TORQUES ---")
        # snap.joint_torques contains tuples (Name, Torque, Velocity)
        for (name, tau) in snap.joint_torques
             @printf(io, "   ⚙️ %-5s: %.5f Nm\n", name, tau)
        end
        println(io, "")


        if snap.deadlock_active
            println(io, "\n💀 DEADLOCK ACTIVE")
        end
        println(io, "=========================================================")

        

        # F. IGNORED TARGETS (REINSERITO!)
        println(io, "\n--- 🚫 TEMPORARILY IGNORED TARGETS ---")
        n_ignored = length(snap.ignored_list)
        if n_ignored == 0
            println(io, "   (None - System clear)")
        else
            sorted_list = sort(snap.ignored_list)
            MAX_SHOW = 15
            counts = 0
            for voxel in sorted_list
                if counts < MAX_SHOW
                    @printf(io, "   ❌ [%d, %d, %d]\n", voxel...)
                    counts += 1
                else
                    remaining = n_ignored - MAX_SHOW
                    println(io, "   ... and $remaining more voxels banned.")
                    break
                end
            end
        end
        println(io, "=========================================================")

        # Console Overwrite Logic
        msg_str = String(take!(io))
        if state_anim.last_printed_time != -1.0
             num_lines = count(==('\n'), msg_str)
             print("\e[$(num_lines)A\e[1G\e[J") 
        end
        print(msg_str)
        
        state_anim.last_printed_time = t
    end
end

# ------------------------------------------------------------------------------
#   Micro Section: GLMakie Scene Construction
# ------------------------------------------------------------------------------

@info "Setting up GLMakie scene..."
fig = Figure(size = (900, 800))
display(fig) 
ls = LScene(fig[1, 1])

# 1. Cam
cam = cam3d!(ls, camera=:perspective, center=false, fov = 45)
cam.lookat[] = [0.2, 0.7, 0.8] 
cam.eyeposition[] = [2.0, -0.2, 1.2]

# 2. Robot
robotvisualize!(ls, plotting_kcache;)

# 3. Memory Map
meshscatter!(ls, voxel_centers_display, marker = Rect3f(Point3f(-1.0), Point3f(1.0)), markersize = Vec3f(VOXEL_SIZE_X*0.9, VOXEL_SIZE_Y*0.9, VOXEL_SIZE_Z*0.9), color = voxel_colors_ANIMATION, shading = NoShading, label="Memory Map")

# 4. FOV Pyramid
global_fov_vertices_obs = map(plotting_kcache) do kcache
    cam_transform_id = get_compiled_frameID(kcache, ".robot.camera_frame")
    cam_transform = get_transform(kcache, cam_transform_id)
    map(v -> cam_transform * SVector{3, Float64}(v), LOCAL_FOV_VERTICES)
end
mesh!(ls, global_fov_vertices_obs, FOV_FACES; color=(:grey, 0.1), transparency=true, shading=NoShading, label="Camera FOV")

# 5. Virtual cam
meshscatter!(ls, cam_pos_ANIMATION; marker = Rect3f(Vec3f(-0.1), Vec3f(0.10)), color = :black, shading = NoShading, label = "Virtual cam")


# 7. cam Axes
hand_frame_id = get_compiled_frameID(plotting_kcache[], ".robot.camera_frame")
cam_transform_obs = map(c -> get_transform(c, hand_frame_id), plotting_kcache)

cam_origin = map(T -> [Point3f(T.origin)], cam_transform_obs)
cam_x_axis = map(T -> [Vec3f(T.rotor * SVector(1.0, 0.0, 0.0))], cam_transform_obs)
cam_y_axis = map(T -> [Vec3f(T.rotor * SVector(0.0, 1.0, 0.0))], cam_transform_obs)
cam_z_axis = map(T -> [Vec3f(T.rotor * SVector(0.0, 0.0, 1.0))], cam_transform_obs)

axis_length, axis_width = 0.05, 0.01
arrows!(ls, cam_origin, cam_x_axis, lengthscale = axis_length, arrowsize = Vec3f(axis_width, axis_width, 0.015), color=:red)
arrows!(ls, cam_origin, cam_y_axis, lengthscale = axis_length, arrowsize = Vec3f(axis_width, axis_width, 0.015), color=:green)
arrows!(ls, cam_origin, cam_z_axis, lengthscale = axis_length, arrowsize = Vec3f(axis_width, axis_width, 0.015), color=:blue)

# 8. Nose Direction
arrows!(ls, 
    nose_arrow_origin_ANIMATION, 
    nose_arrow_vec_ANIMATION;
    color = :red,
    linewidth = 0.001,       # Spessore asta
    arrowsize = 0.003,       # Dimensione punta
    lengthscale = 1.0,      # Scala 1:1 col vettore calcolato
    label = "Heading"
)

# 9. HUD
text!(
    fig.scene,                     
    completion_text_ANIMATION;      
    position = (20, 20),           
    space = :pixel,                
    align = (:left, :bottom),       
    fontsize = 20,
    font = :bold,
    color = :grey,
    glowwidth = 1.5,                
    glowcolor = :yellow
)
text!(
    fig.scene,
    time_text_ANIMATION;
    position = (0.87, 0.95),  # 2% da sinistra, 95% dal basso (quindi in alto)
    space = :relative,        # Usa coordinate relative (0-1) invece di pixel
    align = (:left, :top),    # Allineamento rispetto al punto di posizione
    fontsize = 15,            # Bello grande
    font = :bold,
    color = :grey
)
# 1. Avviso Velocità
text!(
    fig.scene,
    warning_vel_ANIMATION;
    position = (0.98, 0.85),  # In alto a destra
    space = :relative,        
    align = (:right, :top),    
    fontsize = 16,            
    font = :bold,
    color = :red,
    glowwidth = 1.0, glowcolor = :white # Glow per renderlo leggibile su sfondo scuro
)

# 2. Avviso Torque (subito sotto quello della velocità)
text!(
    fig.scene,
    warning_tau_ANIMATION;
    position = (0.98, 0.80),  # Un po' più in basso (0.80)
    space = :relative,        
    align = (:right, :top),    
    fontsize = 16,            
    font = :bold,
    color = :red,
    glowwidth = 1.0, glowcolor = :white
)

# 10. cam-Obs Springs
linesegments!(ls, 
    cam_spring_lines_ANIMATION;
    color = (:blue, 0.5), # Blu semi-trasparente
    linewidth = 1,
    linestyle = :dash,
    label = "Repulsive Springs"
)

# 11. cam-Obs Springs
linesegments!(ls, 
    nose_spring_line_ANIMATION;
    color = (:green, 0.5), # Blu semi-trasparente
    linewidth = 1,
    linestyle = :dash,
    label = "Attraction Springs"
)

# --- FLOOR REPULSION LINES ---
linesegments!(ls, 
    floor_spring_lines_ANIMATION;
    color = :red,          # Rosso pericolo!
    linewidth = 2,         # Un po' più spesse
    linestyle = :dash,     # Tratteggiate
    label = "Floor Protection"
)


# Legend
Legend(fig[1, 1], ls; merge=true, tellwidth=false, halign=:left, valign=:top)



# ==============================================================================
# 12. Grid Indices Labels (Visual Aid)
# ==============================================================================

# Definiamo un margine per staccare le scritte dal bordo dei voxel
MARGIN_OFFSET = 0.05 

# --- ASSE X (Indici 1 -> N_X) ---
# Posizioniamo i numeri lungo il bordo "sotto" e "sinistra"
x_labels_pos = [
    Point3f(
        WS_MIN_X + (i-0.5) * VOXEL_SIZE_X,  # Centro X del voxel i-esimo
        WS_MIN_Y - MARGIN_OFFSET,             # Un po' fuori su Y
        WS_MIN_Z                              # Alla base Z
    ) 
    for i in 1:N_VOXELS_X
]
text!(ls, x_labels_pos; 
    text = string.(1:N_VOXELS_X), 
    color=:black, 
    fontsize=14, 
    align=(:center, :top) # Allineati centrati rispetto al voxel
)

# --- ASSE Y (Indici 1 -> N_Y) ---
y_labels_pos = [
    Point3f(
        WS_MAX_X + MARGIN_OFFSET,             # Un po' fuori su X
        WS_MIN_Y + (j -0.5) * VOXEL_SIZE_Y,  # Centro Y del voxel j-esimo
        WS_MIN_Z                              # Alla base Z
    ) 
    for j in 1:N_VOXELS_Y
]
text!(ls, y_labels_pos; 
    text = string.(1:N_VOXELS_Y), 
    color=:black, 
    fontsize=14, 
    align=(:right, :center) 
)

# --- ASSE Z (Indici 1 -> N_Z) ---
z_labels_pos = [
    Point3f(
        WS_MIN_X - MARGIN_OFFSET,             # Un po' fuori su X
        WS_MIN_Y - MARGIN_OFFSET,             # Un po' fuori su Y (angolo esterno)
        WS_MIN_Z + (k -0.5) * VOXEL_SIZE_Z   # Centro Z del voxel k-esimo
    ) 
    for k in 1:N_VOXELS_Z
]
text!(ls, z_labels_pos; 
    text = string.(1:N_VOXELS_Z), 
    color=:black, 
    fontsize=14, 
    align=(:right, :center)
)

# Aggiungiamo anche delle linee guida (assi) per far capire meglio dove sono i numeri
lines!(ls, 
    [Point3f(WS_MIN_X, WS_MIN_Y-MARGIN_OFFSET/2, WS_MIN_Z), Point3f(WS_MAX_X, WS_MIN_Y-MARGIN_OFFSET/2, WS_MIN_Z)], 
    color=:black, linewidth=1
)
lines!(ls, 
    [Point3f(WS_MAX_X+MARGIN_OFFSET/2, WS_MIN_Y, WS_MIN_Z), Point3f(WS_MAX_X+MARGIN_OFFSET/2, WS_MAX_Y, WS_MIN_Z)], 
    color=:black, linewidth=1
)
lines!(ls, 
    [Point3f(WS_MIN_X-MARGIN_OFFSET/2, WS_MIN_Y-MARGIN_OFFSET/2, WS_MIN_Z), Point3f(WS_MIN_X-MARGIN_OFFSET/2, WS_MIN_Y-MARGIN_OFFSET/2, WS_MAX_Z)], 
    color=:black, linewidth=1
)


# ------------------------------------------------------------------------------
#   Micro Section: Render Execution
# ------------------------------------------------------------------------------

@info "Rendering animation... this may take a while."
VMRobotControl.animate_robot_odesolution(
    fig, 
    sol, 
    plotting_kcache, 
    savepath; 
    t=plotting_t, 
    fps=60, 
    fastforward=1.0,
    f_setup=f_setup_animation, f_control=f_control_animation
);

@info "Animation saved to: $savepath"