import torch
import pymongo
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve_continuous_are

# --- CONFIGURATION ---
MDB_URI = "mongodb://localhost:27017/"
DB_NAME = "mas_research"
RAW_COLL_NAME = "raw_simulations"      # Raw simulation data
PROCESSED_COLL_NAME = "processed_data" # Processed training data

# Agent types and their configurations
AGENT_TYPES = {
    "single_integrator": {
        "state_dim": 4,  # [pos_x, pos_y, vel_x, vel_y]
        "control_dim": 2,
        "dynamics": "kinematic",
        "safety_threshold": 0.5,
        "max_velocity": 1.0
    },
    "double_integrator": {
        "state_dim": 6,  # [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        "control_dim": 2,
        "dynamics": "dynamic",
        "safety_threshold": 0.7,
        "max_velocity": 2.0
    },
    "quadrotor": {
        "state_dim": 12, # [pos_x,y,z, vel_x,y,z, att_roll,pitch,yaw, ang_vel_x,y,z]
        "control_dim": 4, # [thrust, roll_cmd, pitch_cmd, yaw_cmd]
        "dynamics": "nonlinear",
        "safety_threshold": 1.0,
        "max_velocity": 3.0
    }
}

# Obstacle configuration
NUM_OBSTACLES = 5
OBSTACLE_RADIUS = 0.5
COLLISION_DISTANCE = 0.3

# Connect to MongoDB
client = pymongo.MongoClient(MDB_URI)
db = client[DB_NAME]
raw_collection = db[RAW_COLL_NAME]
processed_collection = db[PROCESSED_COLL_NAME]

# --- ETL PIPELINE CLASSES ---
class ETLPipeline:
    """
    Extract, Transform, Load pipeline for multi-agent simulation data.
    """
    def __init__(self):
        self.extracted_data = []
        self.transformed_data = []

    def extract(self, agent_types=None, num_simulations=50):
        """
        Extract: Generate and store raw simulation data for different agent types.
        """
        print("🔄 EXTRACT PHASE: Generating raw simulation data...")

        if agent_types is None:
            agent_types = list(AGENT_TYPES.keys())

        raw_collection.delete_many({})  # Clear old raw data

        for agent_type in agent_types:
            print(f"  → Generating {num_simulations} simulations for {agent_type}")
            self._generate_agent_simulations(agent_type, num_simulations)

        print(f"✓ Extracted {raw_collection.count_documents({})} raw simulation snapshots")

    def transform(self, agent_types=None, feature_engineering=True):
        """
        Transform: Process raw data into training-ready format.
        """
        print("🔄 TRANSFORM PHASE: Processing raw data...")

        if agent_types is None:
            agent_types = list(AGENT_TYPES.keys())

        processed_collection.delete_many({})  # Clear old processed data

        for agent_type in agent_types:
            print(f"  → Transforming data for {agent_type}")
            self._transform_agent_data(agent_type, feature_engineering)

        print(f"✓ Transformed {processed_collection.count_documents({})} processed training samples")

    def load(self, output_formats=["mongodb"], export_path="./data/"):
        """
        Load: Export processed data to different formats.
        """
        print("🔄 LOAD PHASE: Exporting processed data...")

        for fmt in output_formats:
            if fmt == "mongodb":
                print("  → Data already in MongoDB")
            elif fmt == "csv":
                self._export_to_csv(export_path)
            elif fmt == "pytorch":
                self._export_to_pytorch(export_path)

        print("✓ Load phase completed")

    def _generate_agent_simulations(self, agent_type, num_simulations):
        """Generate raw simulation data for a specific agent type."""
        config = AGENT_TYPES[agent_type]

        for sim_id in range(num_simulations):
            # Generate obstacles
            obstacles = generate_obstacles(NUM_OBSTACLES)

            # Generate agents based on type
            num_agents = np.random.randint(40, 50)
            positions = np.random.rand(num_agents, 2) * 10

            if agent_type == "single_integrator":
                # [pos_x, pos_y, vel_x, vel_y]
                velocities = np.random.rand(num_agents, 2) * config["max_velocity"]
                states = np.hstack([positions, velocities])

            elif agent_type == "double_integrator":
                # [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
                velocities = np.random.rand(num_agents, 2) * config["max_velocity"]
                accelerations = np.random.rand(num_agents, 2) * 0.5
                states = np.hstack([positions, velocities, accelerations])

            elif agent_type == "quadrotor":
                # [pos_x,y,z, vel_x,y,z, att_roll,pitch,yaw, ang_vel_x,y,z]
                positions_3d = np.hstack([positions, np.random.rand(num_agents, 1) * 5])
                velocities_3d = np.random.rand(num_agents, 3) * config["max_velocity"]
                attitudes = np.random.rand(num_agents, 3) * 2*np.pi  # Euler angles
                angular_velocities = np.random.rand(num_agents, 3) * 1.0
                states = np.hstack([positions_3d, velocities_3d, attitudes, angular_velocities])

            # Compute safety scores
            safety_scores, collision_flags = self._compute_agent_safety(states, obstacles, agent_type)
            num_collisions = np.sum(collision_flags)
            avg_safety = np.mean(safety_scores)

            # Build graph adjacency
            edge_index = self._build_adjacency(positions, connection_radius=3.0)
            avg_degree = edge_index.shape[1] / num_agents if num_agents > 0 else 0

            # Store raw data
            doc = {
                "timestamp": datetime.now(),
                "agent_type": agent_type,
                "simulation_id": sim_id,
                "meta": {
                    "num_agents": num_agents,
                    "avg_degree": avg_degree,
                    "num_collisions": int(num_collisions),
                    "avg_safety_score": float(avg_safety),
                    "collision_risk": "high" if num_collisions > 5 else "low"
                },
                "obstacles": obstacles.tolist(),
                "raw_data": {
                    "states": states.tolist(),
                    "safety_labels": safety_scores.tolist(),
                    "collision_flags": collision_flags.tolist()
                }
            }

            raw_collection.insert_one(doc)

    def _compute_agent_safety(self, states, obstacles, agent_type):
        """Compute safety scores based on agent type and dynamics."""
        num_agents = len(states)
        safety_scores = np.ones(num_agents)
        collision_flags = np.zeros(num_agents, dtype=bool)

        config = AGENT_TYPES[agent_type]

        # Extract positions based on agent type
        if agent_type == "single_integrator":
            positions = states[:, :2]  # [pos_x, pos_y]
        elif agent_type == "double_integrator":
            positions = states[:, :2]  # [pos_x, pos_y]
        elif agent_type == "quadrotor":
            positions = states[:, :3]  # [pos_x, pos_y, pos_z] - full 3D for quadrotors

        # Agent-to-agent collisions
        for i in range(num_agents):
            min_dist_to_agent = float('inf')
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_dist_to_agent = min(min_dist_to_agent, dist)

                    if dist < COLLISION_DISTANCE:
                        collision_flags[i] = True
                        safety_scores[i] = 0.0

            # Reduce safety based on proximity
            if min_dist_to_agent < config["safety_threshold"]:
                safety_scores[i] = min(safety_scores[i], min_dist_to_agent / config["safety_threshold"])

        # Agent-to-obstacle collisions (project obstacles to 3D for quadrotors)
        for i in range(num_agents):
            min_dist_to_obstacle = float('inf')
            for obstacle in obstacles:
                if agent_type == "quadrotor":
                    # Project 2D obstacles to 3D (assume obstacles span full Z range)
                    obstacle_3d = np.array([obstacle[0], obstacle[1], positions[i][2]])  # Same Z as agent
                    dist = np.linalg.norm(positions[i] - obstacle_3d)
                else:
                    dist = np.linalg.norm(positions[i][:2] - obstacle)  # 2D distance for ground agents

                min_dist_to_obstacle = min(min_dist_to_obstacle, dist)

                if dist < (OBSTACLE_RADIUS + COLLISION_DISTANCE):
                    collision_flags[i] = True
                    safety_scores[i] = 0.0

            # Reduce safety based on obstacle proximity
            if min_dist_to_obstacle < config["safety_threshold"]:
                safety_scores[i] = min(safety_scores[i], min_dist_to_obstacle / config["safety_threshold"])

        return safety_scores, collision_flags

    def _build_adjacency(self, positions, connection_radius=3.0):
        """Build graph adjacency matrix based on spatial proximity."""
        num_agents = len(positions)
        edge_start = []
        edge_end = []

        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < connection_radius:
                        edge_start.append(i)
                        edge_end.append(j)

        return torch.tensor([edge_start, edge_end], dtype=torch.long)

    def _transform_agent_data(self, agent_type, feature_engineering=True):
        """Transform raw data for a specific agent type."""
        # Query raw data for this agent type
        raw_docs = list(raw_collection.find({"agent_type": agent_type}))

        for doc in raw_docs:
            raw_data = doc["raw_data"]
            states = np.array(raw_data["states"])
            safety_labels = np.array(raw_data["safety_labels"])

            # Build graph adjacency from positions (use appropriate dimensions)
            if agent_type == "quadrotor":
                positions = states[:, :3]  # 3D positions for quadrotors
            else:
                positions = states[:, :2]  # 2D positions for ground agents

            edge_index = self._build_adjacency(positions[:, :2], connection_radius=3.0)  # Use XY projection for adjacency

            # Feature engineering (keep original state dimensions)
            if feature_engineering:
                # Add derived features but keep original state dimensions
                if agent_type == "quadrotor":
                    velocity_magnitudes = np.linalg.norm(states[:, 3:6], axis=1, keepdims=True)  # 3D velocity
                else:
                    velocity_magnitudes = np.linalg.norm(states[:, 2:4], axis=1, keepdims=True)  # 2D velocity

                relative_distances = self._compute_relative_distances(positions[:, :2]).reshape(-1, 1)  # Use XY projection

                # Combine original states with derived features
                processed_states = np.column_stack([
                    states,  # Original state features
                    velocity_magnitudes,
                    relative_distances
                ])
            else:
                processed_states = states

            # Store processed data
            processed_doc = {
                "timestamp": datetime.now(),
                "agent_type": agent_type,
                "original_id": doc["_id"],
                "meta": doc["meta"],
                "processed_data": {
                    "x": processed_states.tolist(),
                    "edge_index": edge_index.tolist(),
                    "safety_labels": safety_labels.tolist()
                }
            }

            processed_collection.insert_one(processed_doc)

    def _compute_relative_distances(self, positions):
        """Compute relative distances to k-nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(5, len(positions)), algorithm='ball_tree').fit(positions)
        distances, _ = nbrs.kneighbors(positions)
        return distances[:, 1:].mean(axis=1)  # Average distance to 4 nearest neighbors

    def _normalize_features(self, states, agent_type):
        """Normalize features based on agent type."""
        config = AGENT_TYPES[agent_type]

        # Simple min-max normalization per feature
        normalized = np.zeros_like(states, dtype=np.float32)

        for i in range(states.shape[1]):
            feature_min = states[:, i].min()
            feature_max = states[:, i].max()
            if feature_max > feature_min:
                normalized[:, i] = (states[:, i] - feature_min) / (feature_max - feature_min)
            else:
                normalized[:, i] = 0.5  # Constant features

        return normalized

    def _export_to_csv(self, export_path):
        """Export processed data to CSV format."""
        import os
        os.makedirs(export_path, exist_ok=True)

        # Export by agent type
        for agent_type in AGENT_TYPES.keys():
            docs = list(processed_collection.find({"agent_type": agent_type}))

            if not docs:
                continue

            # Flatten data for CSV
            csv_data = []
            for doc in docs:
                processed = doc["processed_data"]
                x = processed["x"]
                safety_labels = processed["safety_labels"]

                for i, (features, safety) in enumerate(zip(x, safety_labels)):
                    row = {
                        "agent_type": agent_type,
                        "node_id": i,
                        "safety_label": safety,
                        **{f"feature_{j}": feat for j, feat in enumerate(features)}
                    }
                    csv_data.append(row)

            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(export_path, f"{agent_type}_training_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"  → Exported {len(csv_data)} samples to {csv_path}")

    def _export_to_pytorch(self, export_path):
        """Export processed data to PyTorch format."""
        import os
        os.makedirs(export_path, exist_ok=True)

        # Export by agent type
        for agent_type in AGENT_TYPES.keys():
            docs = list(processed_collection.find({"agent_type": agent_type}))

            if not docs:
                continue

            pytorch_data = []
            for doc in docs:
                processed = doc["processed_data"]
                x = torch.tensor(processed["x"])
                edge_index = torch.tensor(processed["edge_index"], dtype=torch.long)
                safety_labels = torch.tensor(processed["safety_labels"])

                data = Data(x=x, edge_index=edge_index, y=safety_labels)
                pytorch_data.append(data)

            # Save PyTorch data
            torch_path = os.path.join(export_path, f"{agent_type}_dataset.pt")
            torch.save(pytorch_data, torch_path)
            print(f"  → Exported {len(pytorch_data)} graphs to {torch_path}")

# --- PART 1: THE GNN MODEL ---

# --- PART 1: THE GNN MODEL ---
class SafetyGNN(torch.nn.Module):
    """
    A simple GNN that predicts a 'safety score' for each agent 
    based on its neighbors' positions.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SafetyGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node features (Position X, Position Y, Velocity X, Velocity Y)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- PART 1.5: OPTIMAL CONTROLLER ---
def compute_lqr_control(current_state, goal_position, dt=0.1, agent_type="double_integrator"):
    """
    Compute LQR optimal control for different agent types.
    """
    if agent_type == "double_integrator":
        # Double integrator dynamics: x_dot = [0 1; 0 0]*x + [0; 1]*u
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])

        # LQR cost matrices
        Q = np.array([[10, 0], [0, 1]])  # State cost (position, velocity)
        R = np.array([[0.1]])  # Control cost

        # Solve continuous-time algebraic Riccati equation
        P = solve_continuous_are(A, B, Q, R)

        # LQR gain: K = R^-1 * B^T * P
        K = np.linalg.inv(R) @ B.T @ P

        # Current state relative to goal
        position_error = current_state[:2] - goal_position
        velocity_error = current_state[2:4]

        state_error = np.array([position_error[0], velocity_error[0]])  # X direction
        u_x = -K @ state_error

        state_error = np.array([position_error[1], velocity_error[1]])  # Y direction
        u_y = -K @ state_error

        return np.array([u_x[0], u_y[0]])

    elif agent_type == "quadrotor":
        # Treat quadrotor as 3D double integrator for simplicity
        # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        # Control: [ax, ay, az, tau_phi, tau_theta, tau_psi]

        # Extract position and velocity
        pos = current_state[:3]
        vel = current_state[3:6]

        # Simple PD control towards goal
        kp = 1.0  # Position gain (increased)
        kd = 0.5  # Velocity gain (increased)

        # Position errors
        pos_error = goal_position - pos

        # Desired accelerations (clamped to reasonable values)
        desired_acc = kp * pos_error - kd * vel
        desired_acc = np.clip(desired_acc, -2.0, 2.0)  # Limit acceleration

        # For quadrotor control, map to thrust and simple attitude commands
        g = 9.81
        m = 1.0

        # Total thrust (counter gravity + desired Z acceleration)
        total_thrust = m * (g + desired_acc[2])
        total_thrust = np.clip(total_thrust, 0.1, 20.0)  # Reasonable thrust limits

        # Simple attitude commands (proportional to desired horizontal acceleration)
        # Limit attitude angles to prevent instability
        max_angle = 0.3  # radians (~17 degrees)
        phi_desired = np.clip(desired_acc[1] / g, -max_angle, max_angle)
        theta_desired = np.clip(-desired_acc[0] / g, -max_angle, max_angle)
        psi_desired = 0.0

        # Attitude control torques (simple proportional control)
        current_phi, current_theta, current_psi = current_state[6:9]
        kp_att = 0.5  # Reduced attitude gain
        tau_phi = kp_att * (phi_desired - current_phi)
        tau_theta = kp_att * (theta_desired - current_theta)
        tau_psi = kp_att * (psi_desired - current_psi)

        return np.array([desired_acc[0], desired_acc[1], desired_acc[2], 0.0])  # [ax, ay, az, dummy]


def compute_optimal_control(agent_type, current_state, goal_position):
    """
    Compute optimal control action to drive agent towards goal.
    Uses LQR for double integrator, proportional control for others.
    """
    config = AGENT_TYPES[agent_type]

    if agent_type == "single_integrator":
        # Direct velocity control towards goal
        position = current_state[:2]
        direction = goal_position - position
        distance = np.linalg.norm(direction)

        if distance > 0.1:  # If not at goal
            velocity_cmd = direction / distance * config["max_velocity"] * 0.8  # Scale down for safety
        else:
            velocity_cmd = np.zeros(2)

        return velocity_cmd

    elif agent_type == "double_integrator":
        # Use LQR controller
        return compute_lqr_control(current_state, goal_position, agent_type=agent_type)

    elif agent_type == "quadrotor":
        # Use LQR controller
        return compute_lqr_control(current_state, goal_position, agent_type=agent_type)

# --- PART 1.6: SAFETY CONTROLLER ---
class SafetyController(torch.nn.Module):
    """
    Neural network safety controller that modifies optimal actions to ensure safety.
    Takes optimal action + current state + safety prediction and outputs safe action.
    """
    def __init__(self, state_dim, control_dim, hidden_dim=64):
        super(SafetyController, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim

        # Input: [state, optimal_action, safety_score]
        input_dim = state_dim + control_dim + 1

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, control_dim)
        )

    def forward(self, state, optimal_action, safety_score):
        """
        state: current agent state
        optimal_action: action from optimal controller
        safety_score: predicted safety score (0-1)
        """
        # Concatenate inputs
        inputs = torch.cat([state, optimal_action, safety_score.unsqueeze(-1)], dim=-1)

        # Compute safe action modification
        action_modification = self.net(inputs)

        # Apply safety filter: if unsafe, modify action; if safe, keep optimal
        # Use safety score to blend between optimal and modified action
        safe_action = optimal_action + action_modification * (1.0 - safety_score)

        return safe_action

# --- OBSTACLE & COLLISION UTILITIES ---
def generate_obstacles(num_obstacles=NUM_OBSTACLES, grid_size=10):
    """Generate random obstacle positions in the grid."""
    obstacles = np.random.rand(num_obstacles, 2) * grid_size
    return obstacles

def compute_collision_risk(positions, velocities, obstacles, safety_threshold=0.5):
    """
    Compute per-agent safety scores based on:
    1. Distance to nearest agent
    2. Distance to nearest obstacle
    3. Whether trajectory leads to collision

    Returns: safety_scores (0=unsafe, 1=safe), collision_flags
    """
    num_agents = len(positions)
    safety_scores = np.ones(num_agents)  # Start all safe
    collision_flags = np.zeros(num_agents, dtype=bool)
    
    # 1. Check agent-to-agent collisions
    for i in range(num_agents):
        min_dist_to_agent = float('inf')
        for j in range(num_agents):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist_to_agent = min(min_dist_to_agent, dist)
                
                # Flag collision if too close
                if dist < COLLISION_DISTANCE:
                    collision_flags[i] = True
                    safety_scores[i] = 0.0
        
        # Reduce safety score based on proximity
        if min_dist_to_agent != float('inf') and min_dist_to_agent < safety_threshold:
            safety_scores[i] = min(safety_scores[i], min_dist_to_agent / safety_threshold)
    
    # 2. Check agent-to-obstacle collisions
    for i in range(num_agents):
        min_dist_to_obstacle = float('inf')
        for obstacle in obstacles:
            dist = np.linalg.norm(positions[i] - obstacle)
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)
            
            # Flag collision if too close to obstacle
            if dist < (OBSTACLE_RADIUS + COLLISION_DISTANCE):
                collision_flags[i] = True
                safety_scores[i] = 0.0
        
        # Reduce safety score based on obstacle proximity
        if min_dist_to_obstacle != float('inf') and min_dist_to_obstacle < safety_threshold:
            safety_scores[i] = min(safety_scores[i], min_dist_to_obstacle / safety_threshold)
    
    return safety_scores, collision_flags

# --- PART 2: DATA GENERATION & STORAGE ---
def generate_and_store_simulation(num_steps=50, agent_types=None):
    """
    Generate multi-agent simulations for different agent types using ETL pipeline.
    """
    if agent_types is None:
        agent_types = list(AGENT_TYPES.keys())

    etl = ETLPipeline()
    etl.extract(agent_types=agent_types, num_simulations=num_steps)
    etl.transform(agent_types=agent_types)
    etl.load(output_formats=["mongodb", "csv", "pytorch"])

# --- PART 3: SMART RETRIEVAL (The 'Data Science' Part) ---
def retrieve_training_data(agent_types=None, difficulty="hard"):
    """
    Retrieve processed training data for specific agent types and difficulty levels.
    """
    if agent_types is None:
        agent_types = list(AGENT_TYPES.keys())

    print(f"\nQuerying processed data for {agent_types} agents ({difficulty} difficulty)...")

    # Build query based on agent types and difficulty
    query = {"agent_type": {"$in": agent_types}}

    if difficulty == "hard":
        query["meta.avg_degree"] = {"$gt": 2.0}
    elif difficulty == "easy":
        query["meta.avg_degree"] = {"$lte": 2.0}

    pipeline = [
        {"$match": query},
        {"$project": {"processed_data": 1, "agent_type": 1, "meta": 1, "_id": 0}}
    ]

    results = list(processed_collection.aggregate(pipeline))
    print(f"Found {len(results)} training samples across {len(agent_types)} agent types")

    return results

def get_agent_type_statistics():
    """
    Get statistics about stored data by agent type.
    """
    pipeline = [
        {"$group": {
            "_id": "$agent_type",
            "count": {"$sum": 1},
            "avg_safety": {"$avg": "$meta.avg_safety_score"},
            "avg_collisions": {"$avg": "$meta.num_collisions"},
            "avg_degree": {"$avg": "$meta.avg_degree"}
        }},
        {"$sort": {"count": -1}}
    ]

    stats = list(raw_collection.aggregate(pipeline))

    print("\n📊 Agent Type Statistics:")
    print("-" * 60)
    for stat in stats:
        print(f"Agent: {stat['_id']}")
        print(f"  Samples: {stat['count']}")
        print(".3f")
        print(".1f")
        print(".2f")
        print()

    return stats

# --- PART 4: TRAINING LOOP ---
def train_gnn(agent_types=None, difficulty="hard", epochs=25, return_test_losses=False, test_data=None):
    """
    Train GNN on specific agent types and difficulty levels.
    """
    if agent_types is None:
        agent_types = list(AGENT_TYPES.keys())

    # 1. Initialize separate models for each agent type
    models = {}
    optimizers = {}

    for agent_type in agent_types:
        if agent_type == "quadrotor":
            input_dim = AGENT_TYPES[agent_type]["state_dim"] + 2  # +2 for derived features (velocity mag, relative dist)
        else:
            input_dim = AGENT_TYPES[agent_type]["state_dim"] + 2  # +2 for derived features
        models[agent_type] = SafetyGNN(in_channels=input_dim, hidden_channels=32, out_channels=1)
        optimizers[agent_type] = torch.optim.Adam(models[agent_type].parameters(), lr=0.01)

    # 2. Get Training Data
    train_data = retrieve_training_data(agent_types, difficulty)

    if not train_data:
        print("No training data found. Run ETL pipeline first.")
        return

    print(f"\n🚀 Training GNN on {len(train_data)} samples from {agent_types}")

    training_history = {
        "epochs": [],
        "train_losses": [],
        "test_losses": [],
        "agent_type_stats": {agent: [] for agent in agent_types}
    }

    for epoch in range(epochs):
        total_train_loss = 0
        agent_losses = {agent: [] for agent in agent_types}

        # Training phase
        for doc in train_data:
            agent_type = doc["agent_type"]
            processed = doc["processed_data"]

            x = torch.tensor(processed["x"])
            edge_index = torch.tensor(processed["edge_index"], dtype=torch.long)
            safety_labels = torch.tensor(processed["safety_labels"], dtype=torch.float)

            # Use the appropriate model for this agent type
            model = models[agent_type]
            optimizer = optimizers[agent_type]

            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)

            # Train to predict safety scores
            loss = F.mse_loss(out.squeeze(), safety_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            agent_losses[agent_type].append(loss.item())

        # Calculate per-agent statistics
        avg_train_loss = total_train_loss / len(train_data)
        agent_stats = {}
        for agent_type in agent_types:
            if agent_losses[agent_type]:
                agent_stats[agent_type] = np.mean(agent_losses[agent_type])
            else:
                agent_stats[agent_type] = 0.0

        training_history["epochs"].append(epoch + 1)
        training_history["train_losses"].append(round(avg_train_loss, 4))

        # Evaluate test loss if requested
        if return_test_losses and test_data:
            test_loss = evaluate_test_loss(agent_types[0], models[agent_types[0]], test_data)
            training_history["test_losses"].append(round(test_loss, 4))
        else:
            training_history["test_losses"].append(0.0)

        for agent_type in agent_types:
            training_history["agent_type_stats"][agent_type].append(round(agent_stats[agent_type], 4))

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Test Loss: {training_history['test_losses'][-1]:.4f} | " +
              " | ".join([f"{agent}: {agent_stats[agent]:.4f}" for agent in agent_types]))

    # Save models
    for agent_type, model in models.items():
        model_name = f"gnn_model_{agent_type}_{difficulty}.pth"
        torch.save(model.state_dict(), model_name)
        print(f"✓ Model for {agent_type} saved to '{model_name}'")

    # Save training history
    history_name = f"training_history_{'_'.join(agent_types)}_{difficulty}.json"
    with open(history_name, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Training history saved to '{history_name}'")

    # Create visualization
    plot_training_results(training_history, agent_types)

    return models, training_history

def plot_training_results(history, agent_types):
    """
    Plot training loss and per-agent statistics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epochs = history["epochs"]

    # Plot 1: Overall Training Loss
    axes[0].plot(epochs, history["train_losses"], 'o-', label='Overall Loss', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)

    # Plot 2: Per-Agent Loss Comparison
    colors = ['#F18F01', '#A23B72', '#C73E1D', '#0B6623']
    for i, agent_type in enumerate(agent_types):
        if agent_type in history["agent_type_stats"]:
            color = colors[i % len(colors)]
            axes[1].plot(epochs, history["agent_type_stats"][agent_type],
                        'o-', label=agent_type, linewidth=2, markersize=6, color=color)

    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[1].set_title('Per-Agent Type Training Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300, bbox_inches='tight')
    print("✓ Plot saved to 'training_results.png'")
    plt.close()

# --- PART 5: SAFETY CONTROLLER TRAINING ---
def generate_optimal_trajectories(agent_type, num_trajectories=10, trajectory_length=20):
    """
    Generate trajectories using optimal controller for training safety controller.
    Returns trajectories with states, optimal actions, and safety classifications.
    """
    config = AGENT_TYPES[agent_type]
    trajectories = []

    for traj_id in range(num_trajectories):
        # Initialize random start and goal positions
        start_pos = np.random.rand(2 if agent_type != "quadrotor" else 3) * 8 + 1  # Avoid edges
        goal_pos = np.random.rand(2 if agent_type != "quadrotor" else 3) * 8 + 1

        # Initialize state based on agent type
        if agent_type == "single_integrator":
            state = np.concatenate([start_pos, np.zeros(2)])  # [pos_x, pos_y, vel_x, vel_y]
        elif agent_type == "double_integrator":
            state = np.concatenate([start_pos, np.zeros(4)])  # [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        elif agent_type == "quadrotor":
            state = np.concatenate([start_pos, np.zeros(9)])  # [pos, vel, att, ang_vel]

        # Generate obstacles
        obstacles = generate_obstacles(NUM_OBSTACLES)

        trajectory = {
            "agent_type": agent_type,
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "obstacles": obstacles,
            "states": [],
            "optimal_actions": [],
            "safety_scores": [],
            "is_safe_trajectory": True
        }

        # Simulate trajectory
        dt = 0.1  # Time step
        min_safety_score = 1.0

        for step in range(trajectory_length):
            # Store current state
            trajectory["states"].append(state.copy())

            # Compute optimal action
            optimal_action = compute_optimal_control(agent_type, state, goal_pos)
            trajectory["optimal_actions"].append(optimal_action)

            # Simulate one step (simplified dynamics)
            state = simulate_dynamics(agent_type, state, optimal_action, dt)

            # Compute safety score
            # Use 2D positions for obstacle collision checking (project quadrotor positions)
            positions_2d = state[:2] if agent_type != "quadrotor" else state[:2]  # Use XY projection for obstacles
            safety_score, collision_flags = compute_collision_risk(
                positions_2d.reshape(1, -1), np.zeros((1, 2)), obstacles,
                config["safety_threshold"]
            )
            safety_score = safety_score[0]
            trajectory["safety_scores"].append(safety_score)
            min_safety_score = min(min_safety_score, safety_score)

            # Check if trajectory becomes unsafe
            if safety_score < 0.5 or collision_flags[0]:
                trajectory["is_safe_trajectory"] = False

            # Check if reached goal
            pos = state[:2] if agent_type != "quadrotor" else state[:3]
            if np.linalg.norm(pos - goal_pos) < 0.5:
                break

        trajectories.append(trajectory)

    return trajectories

def simulate_dynamics(agent_type, state, action, dt):
    """
    Simple forward simulation of agent dynamics.
    """
    new_state = state.copy()

    if agent_type == "single_integrator":
        # Velocity control: state = [pos_x, pos_y, vel_x, vel_y]
        new_state[2:4] = action  # Set velocity
        new_state[:2] += action * dt  # Integrate position

    elif agent_type == "double_integrator":
        # Acceleration control: state = [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        new_state[4:6] = action  # Set acceleration
        new_state[2:4] += action * dt  # Integrate velocity
        new_state[:2] += new_state[2:4] * dt  # Integrate position

    elif agent_type == "quadrotor":
        # Simplified quadrotor: direct velocity control for goal reaching
        # action = [ax, ay, az, dummy] - desired accelerations
        # state = [pos_x,y,z, vel_x,y,z, roll,pitch,yaw, ang_vel_x,y,z]

        ax, ay, az, _ = action

        # Scale down accelerations for stability and treat as velocity commands
        vx_cmd, vy_cmd, vz_cmd = np.clip([ax, ay, az], -2.0, 2.0)

        # Direct velocity control (kinematic model)
        new_state[3:6] = np.array([vx_cmd, vy_cmd, vz_cmd])  # Set velocity directly

        # Integrate position
        new_state[:3] += new_state[3:6] * dt

        # Attitude dynamics (damped)
        new_state[6:9] *= 0.95  # Damp attitudes
        new_state[9:12] *= 0.9  # Damp angular velocities

    return new_state

def train_safety_controller(agent_types=None, epochs=20):
    """
    Train safety controller to modify optimal actions for safety.
    """
    if agent_types is None:
        agent_types = list(AGENT_TYPES.keys())

    print(f"\n🛡️ Training Safety Controller for {agent_types}")

    # Initialize safety controllers and load trained GNN models
    safety_controllers = {}
    gnn_models = {}
    optimizers = {}

    for agent_type in agent_types:
        config = AGENT_TYPES[agent_type]

        # Initialize safety controller
        safety_controllers[agent_type] = SafetyController(
            state_dim=config["state_dim"],
            control_dim=config["control_dim"]
        )
        optimizers[agent_type] = torch.optim.Adam(safety_controllers[agent_type].parameters(), lr=0.001)

        # Load trained GNN model for safety prediction
        try:
            gnn_model = SafetyGNN(
                in_channels=config["state_dim"] + 2,  # + velocity mag + relative dist
                hidden_channels=32,
                out_channels=1
            )
            gnn_model.load_state_dict(torch.load(f"gnn_model_{agent_type}_hard.pth"))
            gnn_model.eval()
            gnn_models[agent_type] = gnn_model
            print(f"✓ Loaded GNN model for {agent_type}")
        except FileNotFoundError:
            print(f"⚠️ GNN model for {agent_type} not found. Train GNN first.")
            return

    training_history = {
        "epochs": [],
        "safety_losses": [],
        "agent_stats": {agent: [] for agent in agent_types}
    }

    for epoch in range(epochs):
        total_safety_loss = 0
        agent_losses = {agent: [] for agent in agent_types}

        for agent_type in agent_types:
            safety_controller = safety_controllers[agent_type]
            gnn_model = gnn_models[agent_type]
            optimizer = optimizers[agent_type]
            config = AGENT_TYPES[agent_type]

            # Generate fresh trajectories for this epoch
            trajectories = generate_optimal_trajectories(agent_type, num_trajectories=5)

            for trajectory in trajectories:
                optimizer.zero_grad()

                # Process trajectory steps
                for step in range(len(trajectory["states"]) - 1):
                    current_state = torch.tensor(trajectory["states"][step], dtype=torch.float)
                    optimal_action = torch.tensor(trajectory["optimal_actions"][step], dtype=torch.float)

                    # Get safety prediction from GNN
                    # Build graph for single agent (simplified)
                    positions = current_state[:2].unsqueeze(0) if agent_type != "quadrotor" else current_state[:3].unsqueeze(0)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop

                    # Create node features (simplified)
                    if agent_type == "quadrotor":
                        features = torch.cat([
                            current_state.unsqueeze(0),
                            torch.norm(current_state[3:6]).unsqueeze(0).unsqueeze(0),  # velocity mag
                            torch.tensor([[0.0]])  # relative dist (simplified)
                        ], dim=1)
                    else:
                        features = torch.cat([
                            current_state.unsqueeze(0),
                            torch.norm(current_state[2:4]).unsqueeze(0).unsqueeze(0),  # velocity mag
                            torch.tensor([[0.0]])  # relative dist (simplified)
                        ], dim=1)

                    with torch.no_grad():
                        safety_pred = gnn_model(features, edge_index).squeeze()

                    # Safety controller forward pass
                    safe_action = safety_controller(current_state, optimal_action, safety_pred)

                    # Loss: encourage safe actions when unsafe, keep optimal when safe
                    target_safety = trajectory["safety_scores"][step + 1]  # Next step safety

                    if target_safety < 0.5:  # If next state would be unsafe
                        # Loss encourages action that improves safety
                        safety_loss = (1.0 - target_safety) * torch.norm(safe_action - optimal_action)**2
                    else:
                        # When safe, keep actions similar to optimal
                        safety_loss = 0.1 * torch.norm(safe_action - optimal_action)**2

                    safety_loss.backward()
                    total_safety_loss += safety_loss.item()
                    agent_losses[agent_type].append(safety_loss.item())

                optimizer.step()

        # Record training progress
        training_history["epochs"].append(epoch + 1)
        training_history["safety_losses"].append(total_safety_loss / sum(len(agent_losses[a]) for a in agent_types))
        for agent_type in agent_types:
            training_history["agent_stats"][agent_type].append(
                np.mean(agent_losses[agent_type]) if agent_losses[agent_type] else 0.0
            )

        print(f"Epoch {epoch+1} | Safety Loss: {training_history['safety_losses'][-1]:.4f}")

    # Save safety controllers
    for agent_type, controller in safety_controllers.items():
        controller_path = f"safety_controller_{agent_type}.pth"
        torch.save(controller.state_dict(), controller_path)
        print(f"✓ Safety controller for {agent_type} saved to '{controller_path}'")

    # Save training history
    history_path = f"safety_controller_training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Safety controller training history saved to '{history_path}'")

    return safety_controllers, training_history

# --- PART 7: INTERACTIVE TRAINING & VISUALIZATION ---
def select_environment():
    """
    Interactive environment selection for training.
    """
    print("\n🤖 Multi-Agent Safety Controller Training")
    print("=" * 50)
    print("Available Agent Types:")
    for i, agent_type in enumerate(AGENT_TYPES.keys(), 1):
        config = AGENT_TYPES[agent_type]
        print(f"{i}. {agent_type.upper()}")
        print(f"   State dim: {config['state_dim']}, Control dim: {config['control_dim']}")
        print(f"   Dynamics: {config['dynamics']}, Max velocity: {config['max_velocity']}")

    # For demo purposes, auto-select quadrotor
    # In real usage, uncomment the input lines below
    print("\n[DEMO MODE] Auto-selecting: QUADROTOR")
    return "quadrotor"

    # Uncomment for interactive mode:
    # while True:
    #     try:
    #         choice = input("\nSelect agent type (1-3): ").strip()
    #         agent_types = list(AGENT_TYPES.keys())
    #         idx = int(choice) - 1
    #         if 0 <= idx < len(agent_types):
    #             selected_agent = agent_types[idx]
    #             print(f"✓ Selected: {selected_agent.upper()}")
    #             return selected_agent
    #         else:
    #             print("Invalid choice. Please select 1-3.")
    #     except ValueError:
    #         print("Please enter a number.")

def evaluate_test_loss(agent_type, model, test_data):
    """
    Evaluate test loss on held-out data.
    """
    model.eval()
    total_test_loss = 0
    num_samples = 0

    with torch.no_grad():
        for doc in test_data:
            if doc["agent_type"] != agent_type:
                continue

            processed = doc["processed_data"]
            x = torch.tensor(processed["x"])
            edge_index = torch.tensor(processed["edge_index"], dtype=torch.long)
            safety_labels = torch.tensor(processed["safety_labels"], dtype=torch.float)

            out = model(x, edge_index)
            loss = F.mse_loss(out.squeeze(), safety_labels)

            total_test_loss += loss.item()
            num_samples += 1

    return total_test_loss / num_samples if num_samples > 0 else 0.0

def create_simulation_video(agent_type, optimal_trajectory, safe_trajectory, obstacles, filename="simulation.mp4"):
    """
    Create a video visualization of the simulation.
    """
    try:
        import matplotlib.animation as animation
        from matplotlib.patches import Circle
    except ImportError:
        print("⚠️ matplotlib.animation not available. Skipping video creation.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Set up the plot
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{agent_type.upper()} Agent Safety Control Simulation')
    ax.grid(True, alpha=0.3)

    # Plot obstacles
    obstacle_patches = []
    for obstacle in obstacles:
        circle = Circle(obstacle, OBSTACLE_RADIUS, color='red', alpha=0.7, label='Obstacle')
        ax.add_patch(circle)
        obstacle_patches.append(circle)

    # Plot trajectories
    optimal_line, = ax.plot([], [], 'r--', linewidth=2, label='Optimal Trajectory', alpha=0.7)
    safe_line, = ax.plot([], [], 'b-', linewidth=3, label='Safe Trajectory')

    # Agent markers
    optimal_marker, = ax.plot([], [], 'ro', markersize=8, label='Optimal Agent')
    safe_marker, = ax.plot([], [], 'bo', markersize=10, label='Safe Agent')

    # Goal marker
    goal_pos = safe_trajectory[-1][:2] if agent_type != "quadrotor" else safe_trajectory[-1][:2]
    goal_marker, = ax.plot([goal_pos[0]], [goal_pos[1]], 'g*', markersize=15, label='Goal')

    ax.legend()

    def animate(frame):
        # Update trajectory lines
        optimal_positions = np.array([s[:2] for s in optimal_trajectory[:frame+1]])
        safe_positions = np.array([s[:2] for s in safe_trajectory[:frame+1]])

        if len(optimal_positions) > 1:
            optimal_line.set_data(optimal_positions[:, 0], optimal_positions[:, 1])
        if len(safe_positions) > 1:
            safe_line.set_data(safe_positions[:, 0], safe_positions[:, 1])

        # Update agent positions
        if frame < len(optimal_trajectory):
            opt_pos = optimal_trajectory[frame][:2]
            optimal_marker.set_data([opt_pos[0]], [opt_pos[1]])

        if frame < len(safe_trajectory):
            safe_pos = safe_trajectory[frame][:2]
            safe_marker.set_data([safe_pos[0]], [safe_pos[1]])

        return optimal_line, safe_line, optimal_marker, safe_marker

    # Create animation
    frames = max(len(optimal_trajectory), len(safe_trajectory))
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=True)

    # Save video
    try:
        anim.save(filename, writer='ffmpeg', fps=5, dpi=100)
        print(f"✓ Simulation video saved to '{filename}'")
    except Exception as e:
        print(f"⚠️ Could not save video: {e}")
        print("Make sure ffmpeg is installed: brew install ffmpeg")

    plt.close()

def plot_comprehensive_results(training_history, test_losses, agent_type):
    """
    Create comprehensive plots: training loss, test loss, and comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epochs = training_history["epochs"]

    # Plot 1: Training vs Test Loss
    axes[0].plot(epochs, training_history["train_losses"], 'b-o', linewidth=2, markersize=6, label='Training Loss')
    axes[0].plot(epochs, test_losses, 'r-s', linewidth=2, markersize=6, label='Test Loss')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{agent_type.upper()} - Training vs Test Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)

    # Plot 2: Safety Controller Training Loss
    if "safety_losses" in training_history:
        axes[1].plot(epochs, training_history["safety_losses"], 'g-^', linewidth=2, markersize=6, label='Safety Loss')
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Safety Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('Safety Controller Training Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(epochs)

    plt.tight_layout()
    plot_filename = f"comprehensive_results_{agent_type}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive results plot saved to '{plot_filename}'")
    plt.close()

def run_interactive_training():
    """
    Main interactive training pipeline.
    """
    # 1. Environment Selection
    selected_agent = select_environment()

    # 2. ETL Pipeline for selected agent
    print(f"\n🔄 Running ETL Pipeline for {selected_agent}...")
    generate_and_store_simulation(num_steps=15, agent_types=[selected_agent])

    # 3. Split data for train/test before training
    print(f"\n📊 Preparing data split for {selected_agent}...")
    all_data = retrieve_training_data([selected_agent], "hard")

    # Simple 80/20 split
    train_size = int(0.8 * len(all_data))
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    # 4. Train GNN Safety Predictor with test evaluation
    print(f"\n🚀 Training GNN Safety Predictor for {selected_agent}...")
    gnn_models, gnn_history = train_gnn(
        agent_types=[selected_agent],
        difficulty="hard",
        epochs=12,
        return_test_losses=True,
        test_data=test_data
    )

    # 5. Train Safety Controller
    print(f"\n🛡️ Training Safety Controller for {selected_agent}...")
    safety_controllers, safety_history = train_safety_controller([selected_agent], epochs=10)

    # Combine histories for plotting
    combined_history = gnn_history.copy()
    # Pad safety losses to match GNN epochs if needed
    safety_epochs = len(safety_history["safety_losses"])
    gnn_epochs = len(gnn_history["epochs"])
    if safety_epochs < gnn_epochs:
        # Pad with last value
        padding = [safety_history["safety_losses"][-1]] * (gnn_epochs - safety_epochs)
        combined_history["safety_losses"] = safety_history["safety_losses"] + padding
    else:
        combined_history["safety_losses"] = safety_history["safety_losses"][:gnn_epochs]

    # 6. Create comprehensive plots
    print(f"\n📈 Creating comprehensive plots...")
    plot_comprehensive_results(combined_history, combined_history["test_losses"], selected_agent)

    # 7. Demonstrate and create video
    print(f"\n🎬 Running multi-agent simulation demonstration...")
    agents, obstacles = demonstrate_multi_agent_safety(
        num_agents=20,
        num_obstacles=15,
        agent_type=selected_agent,
        max_steps=200,
        goal_tolerance=1.0
    )

    print(f"\n🎥 Creating multi-agent simulation video...")
    if selected_agent == "quadrotor":
        create_3d_multi_agent_video(agents, obstacles, selected_agent, f"3d_multi_agent_simulation_{selected_agent}.mp4")
    else:
        create_multi_agent_video(agents, obstacles, selected_agent, f"multi_agent_simulation_{selected_agent}.mp4")

    print(f"\n✅ Training complete for {selected_agent.upper()}!")
    print("Generated files:")
    print(f"  - GNN Model: gnn_model_{selected_agent}_hard.pth")
    print(f"  - Safety Controller: safety_controller_{selected_agent}.pth")
    print(f"  - Comprehensive Plot: comprehensive_results_{selected_agent}.png")
    print(f"  - Multi-Agent Simulation Video: {'3d_' if selected_agent == 'quadrotor' else ''}multi_agent_simulation_{selected_agent}.mp4")
    print(f"  - Training Histories: *_training_history.json")

def demonstrate_multi_agent_safety(num_agents=20, num_obstacles=15, agent_type="double_integrator", max_steps=200, goal_tolerance=1.0):
    """
    Demonstrate the safety controller with multiple agents and many obstacles.
    Runs until all agents reach their goals or max_steps is reached.
    """
    print(f"\n🛡️ Demonstrating Multi-Agent Safety Controller")
    print("=" * 60)
    print(f"Agents: {num_agents} | Obstacles: {num_obstacles} | Type: {agent_type}")
    print(f"Max Steps: {max_steps} | Goal Tolerance: {goal_tolerance}")

    # Load trained models
    config = AGENT_TYPES[agent_type]

    # Load GNN safety predictor
    gnn_model = SafetyGNN(
        in_channels=config["state_dim"] + 2,
        hidden_channels=32,
        out_channels=1
    )
    gnn_model.load_state_dict(torch.load(f"gnn_model_{agent_type}_hard.pth"))
    gnn_model.eval()

    # Load safety controller
    safety_controller = SafetyController(
        state_dim=config["state_dim"],
        control_dim=config["control_dim"]
    )
    safety_controller.load_state_dict(torch.load(f"safety_controller_{agent_type}.pth"))
    safety_controller.eval()

    # Generate multi-agent scenario
    print(f"\n🎯 Generating {num_agents}-agent scenario...")

    # Generate random start and goal positions
    np.random.seed(42)  # For reproducible results
    grid_size = 12

    # Generate agents with start/goal pairs
    agents = []
    for i in range(num_agents):
        # Ensure minimum separation between start positions
        while True:
            start_pos = np.random.rand(2) * grid_size
            if all(np.linalg.norm(start_pos - np.array([a['start'][:2] for a in agents])) > 1.5 for a in agents):
                break

        # Generate goal position (ensure it's not too close to start)
        if agent_type == "quadrotor":
            # 3D goals for quadrotor
            start_pos_3d = np.concatenate([start_pos, [2.0]])  # Add default Z=2.0 for start
            while True:
                goal_pos = np.random.rand(3) * np.array([grid_size, grid_size, 6.0]) + np.array([0, 0, 1.0])  # Z from 1-7
                if np.linalg.norm(goal_pos[:2] - start_pos) > 4.0 and abs(goal_pos[2] - start_pos_3d[2]) > 1.0:
                    break
        else:
            # 2D goals for other agents
            while True:
                goal_pos = np.random.rand(2) * grid_size
                if np.linalg.norm(goal_pos - start_pos) > 4.0:
                    break

        # Initialize state based on agent type
        if agent_type == "single_integrator":
            state = np.concatenate([start_pos, np.zeros(2)])
        elif agent_type == "double_integrator":
            state = np.concatenate([start_pos, np.zeros(4)])
        elif agent_type == "quadrotor":
            # [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r] - 12 states total
            start_pos_3d = np.concatenate([start_pos, [2.0]])  # Add Z=2.0
            state = np.concatenate([start_pos_3d, np.zeros(9)])

        agents.append({
            'id': i,
            'start': start_pos,
            'goal': goal_pos,
            'state': state,
            'optimal_trajectory': [state.copy()],
            'safe_trajectory': [state.copy()]
        })

    # Generate obstacles
    obstacles = generate_obstacles(num_obstacles, grid_size)

    print(f"✓ Generated {num_agents} agents and {num_obstacles} obstacles")

    # Simulation parameters
    dt = 0.1

    # Run simulation until all agents reach goals or max steps reached
    print(f"\n🚀 Running simulation until all agents reach goals (max {max_steps} steps)...")

    step = 0
    all_reached_goals = False

    while not all_reached_goals and step < max_steps:
        step += 1

        # Check if all agents have reached their goals
        all_reached_goals = True
        for agent in agents:
            current_pos = agent['state'][:len(agent['goal'])]  # Match goal dimensions
            distance_to_goal = np.linalg.norm(current_pos - agent['goal'])
            if distance_to_goal > goal_tolerance:
                all_reached_goals = False
                break

        if all_reached_goals:
            break

        # Print progress every 10 steps
        if step % 10 == 0 or step == 1:
            goal_counts = sum(1 for agent in agents if np.linalg.norm(agent['state'][:len(agent['goal'])] - agent['goal']) <= goal_tolerance)
            print(f"Step {step}/{max_steps} - {goal_counts}/{num_agents} agents at goal", end='\r')

        # Update each agent
        for agent in agents:
            state = agent['state']

            # Compute LQR optimal action toward goal
            optimal_action = compute_optimal_control(agent_type, state, agent['goal'])

            # Build graph for GNN (multi-agent scenario)
            all_positions = np.array([a['state'][:2] for a in agents])
            all_velocities = np.array([a['state'][2:4] if len(a['state']) > 3 else np.zeros(2) for a in agents])

            # Create edges (fully connected for simplicity)
            num_agents_total = len(agents)
            edge_index = []
            for i in range(num_agents_total):
                for j in range(num_agents_total):
                    if i != j:
                        edge_index.extend([[i, j], [j, i]])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()

            # Create node features
            features = []
            for i, a in enumerate(agents):
                agent_state = a['state']
                if agent_type == "quadrotor":
                    vel_norm = np.linalg.norm(agent_state[3:6])
                    features.append(np.concatenate([agent_state, [vel_norm, 0.0]]))
                else:
                    vel_norm = np.linalg.norm(agent_state[2:4]) if len(agent_state) > 3 else 0.0
                    features.append(np.concatenate([agent_state, [vel_norm, 0.0]]))

            features = torch.tensor(np.array(features), dtype=torch.float)

            # Get safety prediction
            with torch.no_grad():
                safety_scores = gnn_model(features, edge_index).squeeze()

            # Get safe action for this agent
            agent_safety = safety_scores[agent['id']] if num_agents_total > 1 else safety_scores

            # For now, use optimal action directly (disable safety controller)
            safe_action = torch.tensor(optimal_action, dtype=torch.float)

            # Debug: print safety info for first agent
            if agent['id'] == 0 and step % 20 == 0:
                print(f"Agent 0: safety_score={agent_safety:.3f}, opt_action={optimal_action}, safe_action={safe_action.numpy()}")

            # Simulate trajectories
            optimal_state = simulate_dynamics(agent_type, state, optimal_action, dt)
            safe_state = simulate_dynamics(agent_type, state, safe_action.detach().numpy(), dt)

            # Store trajectories
            agent['optimal_trajectory'].append(optimal_state)
            agent['safe_trajectory'].append(safe_state)

            # Update state for next iteration (using safe action)
            agent['state'] = safe_state

    # Print final status
    if all_reached_goals:
        print(f"\n✅ All agents reached their goals in {step} steps!")
    else:
        goal_counts = sum(1 for agent in agents if np.linalg.norm(agent['state'][:len(agent['goal'])] - agent['goal']) <= goal_tolerance)
        print(f"\n⚠️ Simulation ended after {step} steps - {goal_counts}/{num_agents} agents reached goals")

    # Compute final safety statistics
    print(f"\n📊 Computing safety statistics...")

    optimal_safety_scores = []
    safe_safety_scores = []
    goal_reach_stats = []

    for agent in agents:
        optimal_positions = np.array([s[:2] for s in agent['optimal_trajectory']])
        safe_positions = np.array([s[:2] for s in agent['safe_trajectory']])

        # Compute safety for optimal trajectory
        opt_safety, _ = compute_collision_risk(
            optimal_positions,
            np.zeros((len(optimal_positions), 2)),
            obstacles,
            config["safety_threshold"]
        )
        optimal_safety_scores.extend(opt_safety)

        # Compute safety for safe trajectory
        safe_safety, _ = compute_collision_risk(
            safe_positions,
            np.zeros((len(safe_positions), 2)),
            obstacles,
            config["safety_threshold"]
        )
        safe_safety_scores.extend(safe_safety)

        # Check final distance to goal
        final_pos = agent['safe_trajectory'][-1][:len(agent['goal'])]  # Match goal dimensions
        goal_distance = np.linalg.norm(final_pos - agent['goal'])
        goal_reach_stats.append(goal_distance)

    optimal_safety_scores = np.array(optimal_safety_scores)
    safe_safety_scores = np.array(safe_safety_scores)
    goal_reach_stats = np.array(goal_reach_stats)

    print(f"\n🎯 Results:")
    print(f"Safety - Optimal: min={optimal_safety_scores.min():.3f}, avg={optimal_safety_scores.mean():.3f}")
    print(f"Safety - Safe: min={safe_safety_scores.min():.3f}, avg={safe_safety_scores.mean():.3f}")
    print(f"Safety improvement: {safe_safety_scores.mean() - optimal_safety_scores.mean():.3f}")
    print(f"Goal reaching: {sum(g < goal_tolerance for g in goal_reach_stats)}/{len(agents)} agents within {goal_tolerance}m")
    print(f"Average goal distance: {goal_reach_stats.mean():.3f}m")

    return agents, obstacles


def create_3d_multi_agent_video(agents, obstacles, agent_type="quadrotor", filename="3d_multi_agent_simulation.mp4"):
    """
    Create a 3D video animation of the multi-agent quadrotor safety simulation.
    """
    print(f"\n🎥 Creating 3D multi-agent simulation video...")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set up the plot
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_zlim(0, 8)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Z Position', fontsize=12)
    ax.set_title(f'3D Multi-Agent Quadrotor Safety Controller Simulation ({len(agents)} agents, {len(obstacles)} obstacles)', fontsize=14, fontweight='bold')

    # Plot obstacles (as 3D spheres)
    obstacles_plot = ax.scatter(obstacles[:, 0], obstacles[:, 1], np.zeros(len(obstacles)), c='red', s=100, marker='o', alpha=0.7, label='Obstacles')

    # Plot agent goals (as stars)
    goals_x = [agent['goal'][0] for agent in agents]
    goals_y = [agent['goal'][1] for agent in agents]
    goals_z = [agent['goal'][2] if len(agent['goal']) > 2 else 2.0 for agent in agents]  # Default Z goal
    goals_plot = ax.scatter(goals_x, goals_y, goals_z, c='green', s=80, marker='*', alpha=0.8, label='Goals')

    # Initialize agent trajectory plots
    optimal_lines = []
    safe_lines = []
    optimal_markers = []
    safe_markers = []

    colors = plt.cm.tab20(np.linspace(0, 1, len(agents)))

    for i, agent in enumerate(agents):
        color = colors[i]

        # Optimal trajectory line
        opt_line, = ax.plot([], [], [], '--', color=color, linewidth=2, alpha=0.7, label=f'Agent {i} Optimal' if i < 3 else "")
        optimal_lines.append(opt_line)

        # Safe trajectory line
        safe_line, = ax.plot([], [], [], '-', color=color, linewidth=3, alpha=0.9, label=f'Agent {i} Safe' if i < 3 else "")
        safe_lines.append(safe_line)

        # Current position markers
        opt_marker = ax.scatter([], [], [], c=[color], s=100, marker='o', alpha=0.8, edgecolors='black', linewidth=2)
        safe_marker = ax.scatter([], [], [], c=[color], s=120, marker='o', alpha=1.0, edgecolors='black', linewidth=2)

        optimal_markers.append(opt_marker)
        safe_markers.append(safe_marker)

    # Add legend (only show first few agents to avoid clutter)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='red', markersize=10, linestyle='None', alpha=0.7, label='Obstacles'),
        plt.Line2D([0], [0], marker='*', color='green', markersize=10, linestyle='None', alpha=0.8, label='Goals')
    ]
    # Add agent trajectory examples
    for i in range(min(3, len(agents))):
        legend_elements.extend([
            plt.Line2D([0], [0], color=colors[i], linestyle='--', linewidth=2, alpha=0.7, label=f'Agent {i} Optimal'),
            plt.Line2D([0], [0], color=colors[i], linestyle='-', linewidth=3, alpha=0.9, label=f'Agent {i} Safe')
        ])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    def animate(frame):
        try:
            for i, agent in enumerate(agents):
                # Update optimal trajectory
                opt_positions = np.array([s[:3] for s in agent['optimal_trajectory']])
                if len(opt_positions) == 0 or opt_positions.ndim != 2 or opt_positions.shape[1] != 3:
                    continue

                if frame < len(opt_positions):
                    pos_slice = opt_positions[:frame+1]
                    optimal_lines[i].set_data(pos_slice[:, 0], pos_slice[:, 1])
                    optimal_lines[i].set_3d_properties(pos_slice[:, 2])
                    optimal_markers[i]._offsets3d = (opt_positions[frame, 0], opt_positions[frame, 1], opt_positions[frame, 2])
                else:
                    optimal_lines[i].set_data(opt_positions[:, 0], opt_positions[:, 1])
                    optimal_lines[i].set_3d_properties(opt_positions[:, 2])
                    optimal_markers[i]._offsets3d = (opt_positions[-1, 0], opt_positions[-1, 1], opt_positions[-1, 2])

                # Update safe trajectory
                safe_positions = np.array([s[:3] for s in agent['safe_trajectory']])
                if len(safe_positions) == 0 or safe_positions.ndim != 2 or safe_positions.shape[1] != 3:
                    continue

                if frame < len(safe_positions):
                    pos_slice = safe_positions[:frame+1]
                    safe_lines[i].set_data(pos_slice[:, 0], pos_slice[:, 1])
                    safe_lines[i].set_3d_properties(pos_slice[:, 2])
                    safe_markers[i]._offsets3d = (safe_positions[frame, 0], safe_positions[frame, 1], safe_positions[frame, 2])
                else:
                    safe_lines[i].set_data(safe_positions[:, 0], safe_positions[:, 1])
                    safe_lines[i].set_3d_properties(safe_positions[:, 2])
                    safe_markers[i]._offsets3d = (safe_positions[-1, 0], safe_positions[-1, 1], safe_positions[-1, 2])

            return optimal_lines + safe_lines + optimal_markers + safe_markers
        except Exception as e:
            print(f"Animation error at frame {frame}: {e}")
            return []    # Create animation
    frames = max(len(agent['optimal_trajectory']) for agent in agents)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=300, blit=False)

    # Save video
    try:
        # Try to save as video
        anim.save(filename, writer='ffmpeg', fps=2, dpi=80)
        print(f"✓ 3D multi-agent simulation video saved to '{filename}'")
    except Exception as e:
        print(f"⚠️ Video saving failed: {e}")
        # Create a static plot of the final state instead
        try:
            fig_static, ax_static = plt.subplots(figsize=(14, 10), subplot_kw={'projection': '3d'})
            ax_static.set_xlim(0, 12)
            ax_static.set_ylim(0, 12)
            ax_static.set_zlim(0, 8)
            ax_static.set_xlabel('X Position', fontsize=12)
            ax_static.set_ylabel('Y Position', fontsize=12)
            ax_static.set_zlabel('Z Position', fontsize=12)
            ax_static.set_title(f'3D Multi-Agent Quadrotor Safety Controller - Final State', fontsize=14, fontweight='bold')

            # Plot obstacles
            ax_static.scatter(obstacles[:, 0], obstacles[:, 1], np.zeros(len(obstacles)), c='red', s=100, marker='o', alpha=0.7, label='Obstacles')

            # Plot goals
            goals_x = [agent['goal'][0] for agent in agents]
            goals_y = [agent['goal'][1] for agent in agents]
            goals_z = [agent['goal'][2] if len(agent['goal']) > 2 else 2.0 for agent in agents]
            ax_static.scatter(goals_x, goals_y, goals_z, c='green', s=80, marker='*', alpha=0.8, label='Goals')

            # Plot final positions and trajectories
            colors = plt.cm.tab20(np.linspace(0, 1, len(agents)))
            for i, agent in enumerate(agents):
                color = colors[i]

                # Plot optimal trajectory
                opt_pos = np.array([s[:3] for s in agent['optimal_trajectory']])
                if len(opt_pos) > 0:
                    ax_static.plot(opt_pos[:, 0], opt_pos[:, 1], opt_pos[:, 2], '--', color=color, linewidth=2, alpha=0.7, label=f'Agent {i} Optimal' if i < 3 else "")

                # Plot safe trajectory
                safe_pos = np.array([s[:3] for s in agent['safe_trajectory']])
                if len(safe_pos) > 0:
                    ax_static.plot(safe_pos[:, 0], safe_pos[:, 1], safe_pos[:, 2], '-', color=color, linewidth=3, alpha=0.9, label=f'Agent {i} Safe' if i < 3 else "")

                # Plot final position
                if len(safe_pos) > 0:
                    ax_static.scatter(safe_pos[-1, 0], safe_pos[-1, 1], safe_pos[-1, 2], c=[color], s=120, marker='o', alpha=1.0, edgecolors='black', linewidth=2)

            ax_static.legend(loc='upper right', fontsize=10)
            static_filename = filename.replace('.mp4', '_static.png')
            fig_static.savefig(static_filename, dpi=150, bbox_inches='tight')
            print(f"✓ Static 3D plot saved to '{static_filename}'")
            plt.close(fig_static)
        except Exception as e2:
            print(f"⚠️ Could not create static plot either: {e2}")

    plt.close()


if __name__ == "__main__":
    # Run interactive training pipeline
    run_interactive_training()