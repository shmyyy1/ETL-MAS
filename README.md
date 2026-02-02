# Multi-Agent Control & Safety

This system implements an advanced multi-agent safety controller that combines:

- **Optimal Control**: LQR controllers for single integrator, double integrator, and quadrotor agents
- **Safety Prediction**: Graph Neural Networks (GNNs) for collision risk assessment
- **Neural Safety Filters**: Learned controllers that modify optimal actions to ensure safety
- **Multi-Agent Coordination**: Real-time safety-aware trajectory planning
- **3D Visualization**: Comprehensive plotting and animation of agent trajectories

The system successfully demonstrates **20 quadrotor agents reaching their goals** while maintaining safety constraints in obstacle-rich environments.
