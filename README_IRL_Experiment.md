# Iterative Feedback Experiment with IRL

This system implements an iterative feedback experiment to study whether humans provide teaching-optimal or reward-maximizing trajectories when teaching a robot to drive. The experiment alternates between human demonstrations and agent demonstrations, using Inverse Reinforcement Learning (IRL) to learn from human trajectories.

## Overview

The experiment follows this iterative process:
1. **Human Demonstration**: Human drives and demonstrates good driving behavior
2. **IRL Learning**: System learns reward function from human trajectories
3. **Agent Demonstration**: Robot/agent demonstrates what it learned
4. **Repeat**: Process continues for multiple iterations

## Files Structure

```
trajectory_preferences/
├── naive_policies.py           # Naive driving policies for agents that don't know how to drive
├── irl_framework.py           # IRL learning framework and experiment management
├── iterative_experiment.py    # Main experiment runner
├── test_naive_policies.py     # Test script for naive policies
├── README_IRL_Experiment.md   # This file
├── map_maker.py              # Creates preset maps (existing)
├── trajectory.py             # Original trajectory logging (existing)
└── preset_maps/              # Directory for saved maps
    └── my_preset_map.json    # Preset map file
```

## Prerequisites

1. **MetaDrive Environment**: Ensure MetaDrive is properly installed and working
2. **Preset Map**: Run `map_maker.py` first to create a preset map file
3. **Python Dependencies**: numpy, json, csv, datetime, dataclasses, pickle

## Quick Start

### 1. Create a Preset Map (if not done already)
```bash
cd trajectory_preferences
python map_maker.py
```

### 2. Test Naive Policies (Optional)
```bash
python test_naive_policies.py
```
This lets you see how different naive policies behave before running the full experiment.

### 3. Run the Full Iterative Experiment
```bash
python iterative_experiment.py
```

## Naive Policy Types

The system includes several types of "naive" agents that don't know how to drive well initially:

### 1. RandomPolicy
- **Behavior**: Completely random steering and throttle/brake actions
- **Use Case**: Represents truly naive behavior, but might be too chaotic
- **Visual Mark**: Red

### 2. NoisyExpertPolicy  
- **Behavior**: Uses the expert policy but adds significant noise and random corruption
- **Use Case**: More realistic "bad driving" that still has some structure
- **Visual Mark**: Orange
- **Parameters**: 
  - `steering_noise_std = 0.3`
  - `throttle_noise_std = 0.2` 
  - `action_corruption_prob = 0.1`

### 3. SimpleForwardPolicy
- **Behavior**: Basic forward driving with poor control and random steering
- **Use Case**: Represents a very basic driving attempt
- **Visual Mark**: Green
- **Parameters**: `target_speed = 0.5`

### 4. ProgressivePolicy
- **Behavior**: Can be updated with learned parameters from IRL
- **Use Case**: The main learning agent that improves over iterations
- **Visual Mark**: Blue
- **Features**: Interpolates between fallback policy and expert based on learning progress

## IRL Framework Components

### TrajectoryManager
- Records and manages trajectory data from both human and agent demonstrations
- Saves trajectories in both CSV (compatible with existing system) and pickle formats
- Tracks states, actions, positions, and metadata

### SimpleIRLLearner
- Basic IRL implementation for demonstration purposes
- Can be replaced with more sophisticated IRL algorithms (MaxEnt IRL, GAIL, etc.)
- Updates reward function based on human demonstrations
- Provides policy weights for agent updates

### IterativeFeedbackExperiment
- Main controller for the experiment process
- Manages phase transitions (human → IRL update → agent → repeat)
- Handles trajectory recording and experiment state saving
- Provides experiment summaries and logging

## Experiment Workflow

### Phase 1: Human Demonstration
- Environment created with `manual_control=True`
- Human drives using WASD keys
- System records:
  - Vehicle states (lidar observations)
  - Actions taken (steering, throttle/brake)
  - Positions (x, y coordinates)
- Trajectory saved when episode ends

### Phase 2: IRL Learning Update
- Collects all human trajectories from previous demonstrations
- Updates reward function using simple gradient ascent on human-visited states
- Generates new policy weights for the agent
- Logs update results and statistics

### Phase 3: Agent Demonstration
- Environment created with `manual_control=False`
- Agent uses selected naive policy (updated with learned weights if applicable)
- System records agent's trajectory for comparison
- Visual feedback shows agent's driving behavior

## Configuration Options

### Experiment Parameters
- `max_iterations`: Number of human-agent cycles (default: 5)
- `steps_per_episode`: Maximum steps per demonstration (default: 1000)
- `record_every_n_steps`: Frequency of trajectory recording (default: 10)

### Policy Selection
- Choose from 4 naive policy types when starting experiment
- Each policy has different characteristics and learning capabilities

### Environment Settings
- Uses preset map for consistency across iterations
- Configurable traffic density, rendering options
- Top-down view for clear trajectory visualization

## Output and Data

### Experiment Directory Structure
```
irl_experiments/
└── [experiment_name]/
    ├── experiment_state.json           # Experiment metadata and progress
    ├── human_episode_1_[timestamp].csv # Human trajectory (CSV format)
    ├── human_episode_1_[timestamp].pkl # Human trajectory (full data)
    ├── agent_episode_2_[timestamp].csv # Agent trajectory (CSV format)
    ├── agent_episode_2_[timestamp].pkl # Agent trajectory (full data)
    └── ...
```

### Data Formats

**CSV Format** (compatible with existing system):
```csv
step,pos_x,pos_y,action_0,action_1
0,10.5,20.3,0.1,-0.2
1,10.6,20.4,0.0,0.1
...
```

**Pickle Format** (full trajectory data):
- Complete state observations
- Action sequences
- Position trajectories
- Episode metadata
- Timestamps and agent type

## Extending the System

### Adding New IRL Algorithms
Replace `SimpleIRLLearner` with more sophisticated implementations:
- Maximum Entropy IRL
- Generative Adversarial Imitation Learning (GAIL)
- ValueDice
- Other state-of-the-art IRL methods

### Adding New Naive Policies
Extend `naive_policies.py` with new policy types:
```python
class MyCustomPolicy(BasePolicy):
    def act(self, *args, **kwargs):
        # Your policy implementation
        return action
```

### Customizing Experiment Flow
Modify `IterativeExperimentRunner` to:
- Change phase ordering
- Add additional data collection
- Implement different learning schedules
- Add custom evaluation metrics

## Troubleshooting

### Common Issues

1. **"Preset map file not found"**
   - Run `map_maker.py` first to create the map file
   - Check that the path in the script matches your file location

2. **"Expert failed, using random action"**
   - This is normal for the NoisyExpertPolicy when expert can't process observations
   - The policy automatically falls back to random actions

3. **Environment rendering issues**
   - Ensure MetaDrive is properly installed with rendering support
   - Check that your system supports OpenGL rendering

4. **Trajectory recording problems**
   - Verify that the experiment directory has write permissions
   - Check disk space for trajectory file storage

### Performance Tips

1. **Reduce rendering load**: Lower `film_size` in render calls
2. **Faster recording**: Increase `record_every_n_steps` to record less frequently
3. **Shorter episodes**: Reduce `steps_per_episode` for quicker iterations

## Research Applications

This system is designed to investigate:

1. **Teaching vs. Reward-Maximizing Trajectories**: Do humans provide trajectories that are optimal for teaching (easier to learn from) or optimal for reward maximization?

2. **Iterative Learning Dynamics**: How does the agent's behavior change over multiple iterations of human feedback?

3. **Human Adaptation**: Do humans adapt their teaching strategy based on the agent's demonstrated learning?

4. **Policy Comparison**: How do different naive starting policies affect the learning trajectory?

## Future Enhancements

- Integration with more sophisticated IRL algorithms
- Real-time trajectory comparison visualization
- Automated experiment analysis and reporting
- Support for multi-agent scenarios
- Integration with human preference learning methods
- Advanced reward function visualization
