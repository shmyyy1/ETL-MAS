# Multi-Agent Control & Safety

A comprehensive framework for training and deploying neural network-based safety controllers for multi-agent systems using Graph Neural Networks (GNNs) and optimal control theory.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This system implements an advanced multi-agent safety controller that combines:

- **Optimal Control**: LQR controllers for single integrator, double integrator, and quadrotor agents
- **Safety Prediction**: Graph Neural Networks (GNNs) for collision risk assessment
- **Neural Safety Filters**: Learned controllers that modify optimal actions to ensure safety
- **Multi-Agent Coordination**: Real-time safety-aware trajectory planning
- **3D Visualization**: Comprehensive plotting and animation of agent trajectories

The system successfully demonstrates **20 quadrotor agents reaching their goals** while maintaining safety constraints in obstacle-rich environments.

## Key Features

### Core Capabilities
- **Multi-Agent Types**: Single integrator, double integrator, and quadrotor agents
- **Optimal Control**: LQR-based controllers for different dynamics
- **Safety Learning**: GNN-based collision prediction with neural safety filters
- **Goal-Based Termination**: Simulations run until all agents reach their goals
- **Real-Time Coordination**: Multi-agent safety-aware decision making

### Advanced Features
- **ETL Pipeline**: MongoDB-based data processing pipeline
- **Training Infrastructure**: Automated model training with validation
- **3D Visualization**: Static and animated trajectory plots
- **Performance Metrics**: Comprehensive safety and goal-reaching statistics
- **Modular Architecture**: Extensible design for new agent types

### Visualization
- **3D Trajectory Plots**: Complete agent paths with obstacles and goals
- **Safety Analysis**: Collision risk heatmaps and statistics
- **Training Curves**: Loss convergence and performance metrics
- **Multi-Agent Scenarios**: 20+ agents in complex environments

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ETL Pipeline  │───▶│  GNN Training   │───▶│ Safety Training │
│   (MongoDB)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Storage   │    │ Safety Predictor│    │ Safety Filter   │
│                 │    │    (GNN)        │    │  (Neural Net)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Multi-Agent     │
                    │ Simulation      │
                    │ & Visualization │
                    └─────────────────┘
```

### Core Components

1. **ETL Pipeline** (`ETLPipeline` class)
   - Extracts raw simulation data
   - Transforms data into training format
   - Loads processed data to MongoDB

2. **GNN Safety Predictor** (`SafetyGNN` class)
   - Graph neural network for collision risk prediction
   - Multi-agent state encoding
   - Safety score generation

3. **Neural Safety Controller** (`SafetyController` class)
   - Modifies optimal actions for safety
   - State-dependent action correction
   - Safety constraint satisfaction

4. **Optimal Controllers** (`compute_lqr_control`, `compute_optimal_control`)
   - LQR controllers for different agent types
   - Goal-directed trajectory planning
   - Dynamics-aware control

## Quick Start

### Prerequisites
```bash
# Required packages
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pymongo
pip install numpy scipy matplotlib
pip install ffmpeg  # For video generation (optional)
```

### MongoDB Setup
```bash
# Install MongoDB
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community

# Or run manually
mongod --dbpath /usr/local/var/mongodb
```

### Running the System
```bash
# Clone/download the repository
cd /path/to/project

# Run the complete pipeline
python mongoGNN.py
```

The system will automatically:
1. Generate training data
2. Train GNN safety predictor
3. Train neural safety controller
4. Run multi-agent simulation
5. Generate visualizations

## Usage Guide

### Basic Usage
```python
from mongoGNN import run_interactive_training

# Run complete training pipeline
run_interactive_training()
```

### Custom Configuration
```python
# Modify agent types and parameters
AGENT_TYPES = {
    "single_integrator": {
        "state_dim": 4,
        "control_dim": 2,
        "dynamics": "kinematic",
        "safety_threshold": 0.5,
        "max_velocity": 1.0
    }
}

# Adjust simulation parameters
NUM_AGENTS = 20
NUM_OBSTACLES = 15
MAX_STEPS = 200
GOAL_TOLERANCE = 1.0
```

### Advanced Features

#### Custom Agent Types
```python
# Add new agent type
AGENT_TYPES["custom_agent"] = {
    "state_dim": 6,
    "control_dim": 3,
    "dynamics": "custom",
    "safety_threshold": 0.8,
    "max_velocity": 2.5
}

# Implement dynamics
def simulate_dynamics(agent_type, state, action, dt):
    if agent_type == "custom_agent":
        # Custom dynamics implementation
        pass
```

#### Safety Controller Tuning
```python
# Adjust training parameters
safety_controller = SafetyController(
    state_dim=config["state_dim"],
    control_dim=config["control_dim"],
    hidden_dim=128  # Increase for complex scenarios
)

optimizer = torch.optim.Adam(safety_controller.parameters(), lr=0.001)
```

## Results & Performance

### Goal Reaching Success
- ** 20/20 agents** reached goals within 1.0m tolerance
- **58 simulation steps** to complete multi-agent coordination
- **Average final distance**: 0.439m from goals

### Safety Performance
- **Zero collisions** in obstacle-rich environments
- **GNN accuracy**: Converged to low loss (< 0.13)
- **Safety controller**: Effective action modification

### Training Metrics
```
GNN Training: Loss converged to ~0.125
Safety Controller: Loss converged to ~0.0003
Simulation: 100% goal success rate
```

## 📁 File Structure

```
├── mongoGNN.py                 # Main system implementation
├── mongoGNN_backup.py          # Backup version
├── test_interactive.py         # Testing utilities
├── README.md                   # This file
├── data/                       # Processed datasets
│   ├── quadrotor_training_data.csv
│   └── quadrotor_dataset.pt
├── *.png                       # Generated visualizations
├── *_history.json              # Training logs
└── *.pth                       # Trained models
```

### Key Files Description

- **`mongoGNN.py`**: Complete implementation with all classes and functions
- **`data/`**: Training datasets in CSV and PyTorch format
- **`*.png`**: Visualization outputs (trajectories, training curves)
- **`*.pth`**: Saved neural network models
- **`*_history.json`**: Training metrics and convergence data

## 🔧 Technical Details

### Agent Dynamics

#### Single Integrator
- **State**: [x, y, vx, vy] (4D)
- **Control**: [vx_cmd, vy_cmd] (2D)
- **Dynamics**: Direct velocity control

#### Double Integrator
- **State**: [x, y, vx, vy, ax, ay] (6D)
- **Control**: [ax_cmd, ay_cmd] (2D)
- **Dynamics**: Acceleration control with LQR

#### Quadrotor
- **State**: [x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r] (12D)
- **Control**: [ax, ay, az, dummy] (4D)
- **Dynamics**: Simplified kinematic model

### Control Architecture

1. **Optimal Control Layer**
   - LQR controllers for trajectory planning
   - Goal-directed motion primitives

2. **Safety Prediction Layer**
   - GNN-based multi-agent state processing
   - Collision risk assessment

3. **Safety Control Layer**
   - Neural network action modification
   - Safety constraint enforcement

### Training Pipeline

1. **Data Generation**: Monte Carlo simulations with various scenarios
2. **GNN Training**: Supervised learning on safety labels
3. **Safety Controller Training**: Reinforcement learning with safety constraints
4. **Validation**: Multi-agent simulation with performance metrics

## Visualization Examples

The system generates several types of visualizations:

### 3D Trajectory Plot
- Shows complete agent trajectories
- Displays obstacles (red) and goals (green)
- Optimal vs safe trajectory comparison

### Training Curves
- GNN loss convergence
- Safety controller performance
- Validation metrics over epochs

### Safety Analysis
- Collision risk distributions
- Agent proximity statistics
- Safety margin analysis

### Development Setup
```bash
# Fork the repository
# Clone your fork
git clone https://github.com/your-username/multi-agent-safety-controller.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest
```
