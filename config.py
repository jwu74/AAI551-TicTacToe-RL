"""Configuration for RL agents."""

class Config:
    """Config settings for training."""

    # Environment
    ENV_NAME = "tictactoe_v3"
    BOARD_SIZE = 3
    STATE_DIM = 9
    ACTION_DIM = 9

    # Training
    TRAINING_EPISODES = {
        'q_learning': 100000,
        'sarsa': 100000,
        'dqn': 100000,
        'reinforce': 100000,
        'mcts': 0
    }

    EVAL_GAMES = 10000

    # Learning rates
    LEARNING_RATE_TABULAR = 0.2
    LEARNING_RATE_NEURAL = 0.001

    # Common params
    GAMMA = 0.99
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995

    # Agent configs
    Q_LEARNING_CONFIG = {
        'learning_rate': LEARNING_RATE_TABULAR,
        'gamma': GAMMA,
        'epsilon': EPSILON_START,
        'epsilon_decay': EPSILON_DECAY
    }

    SARSA_CONFIG = {
        'learning_rate': LEARNING_RATE_TABULAR,
        'gamma': GAMMA,
        'epsilon': EPSILON_START,
        'epsilon_decay': EPSILON_DECAY
    }

    DQN_CONFIG = {
        'learning_rate': LEARNING_RATE_NEURAL,
        'gamma': GAMMA,
        'epsilon': EPSILON_START,
        'epsilon_min': EPSILON_END,
        'epsilon_decay': EPSILON_DECAY,
        'batch_size': 64,
        'buffer_size': 10000,
        'target_update': 100,
    }

    REINFORCE_CONFIG = {
        'learning_rate': LEARNING_RATE_NEURAL,
        'gamma': GAMMA,
        'epsilon': 0.1,
    }

    MCTS_CONFIG = {
        'n_simulations': 1000,
        'exploration_constant': 1.414,
    }

    # Rewards
    REWARDS = {
        'win': 1.0,
        'loss': -1.0,
        'draw': 0.5,
        'invalid_move': -10.0,
        'valid_move': 0.0
    }

    # File paths
    MODEL_DIR = 'models/'
    RESULTS_DIR = 'results/'
    LOG_DIR = 'logs/'

    MODEL_PATHS = {
        'q_learning': MODEL_DIR + 'q_learning_qtable.pkl',
        'sarsa': MODEL_DIR + 'sarsa_qtable.pkl',
        'dqn': MODEL_DIR + 'dqn_model.pth',
        'reinforce': MODEL_DIR + 'reinforce_model.pth',
        'mcts': None
    }

    # Logging
    LOG_INTERVAL = 1000
    SAVE_INTERVAL = 5000
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    FIGURE_SIZE = (12, 8)

    RANDOM_SEED = 42

    @classmethod
    def get_agent_config(cls, agent_name):
        config_map = {
            'q_learning': cls.Q_LEARNING_CONFIG,
            'sarsa': cls.SARSA_CONFIG,
            'dqn': cls.DQN_CONFIG,
            'reinforce': cls.REINFORCE_CONFIG,
            'mcts': cls.MCTS_CONFIG
        }

        if agent_name not in config_map:
            raise ValueError(f"Unknown agent name: {agent_name}. "
                           f"Available agents: {list(config_map.keys())}")

        return config_map[agent_name].copy()

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("TIC-TAC-TOE RL PROJECT CONFIGURATION")
        print("=" * 60)
        print(f"\nEnvironment: {cls.ENV_NAME}")
        print(f"State Dimension: {cls.STATE_DIM}")
        print(f"Action Dimension: {cls.ACTION_DIM}")
        print(f"\nTraining Episodes: {cls.TRAINING_EPISODES}")
        print(f"Evaluation Games: {cls.EVAL_GAMES}")
        print(f"\nGamma: {cls.GAMMA}")
        print(f"Epsilon Range: {cls.EPSILON_END} - {cls.EPSILON_START}")
        print("=" * 60)


