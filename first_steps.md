# RobertaIA - First Steps

Welcome to the RobertaIA project! This document serves as a guide to understand the project structure, how to train, test, and debug the reinforcement learning models.

## 📁 Project Structure (Main Files)

The following lists exactly the files that form the core application, excluding temporary logs or untracked directories:

- **`requirements.txt`**: List of Python dependencies required to run the project. Install them using `pip install -r requirements.txt`.
- **`src/robertaEnv.py`**: Contains `RobertaEnv-v0`, a custom Gymnasium environment. This simulates the 1D balancer problem for an arm actuated by a motor, handling step physics, reward calculation, and Pygame rendering.
- **`src/robertaPolicy.py`**: Defines the custom PyTorch neural network `RobertaPolicy` (acting as a feature extractor) that processes the observation space for the SAC reinforcement learning algorithm.
- **`src/model_SAC_functions.py`**: Contains key helper functions such as setting random seeds, `linear_schedule` for learning rate decay, and `create_sac` which instantiates the Stable-Baselines3 Soft Actor-Critic algorithm with custom architectural arguments.
- **`src/training.py`**: The main entry point script to train the neural network. It provisions the vectorised environment, initiates callbacks, and executes the SAC training loop.
- **`src/debuging.py`**: Script used for testing (evaluating) a previously trained model, visualizing the test via Pygame, and plotting the episode tracking performance.

*(Note: Peripheral folders like `simulacao/` contain Matlab/Simulink logic used for mathematical modeling, `relatorios/` holds LaTeX reportss).*

---

## 🚀 How to Train the Neural Network

To start training the Soft Actor-Critic (SAC) model, you use the `src/training.py` script. The script prepares the environment, initializes the algorithm, and starts the learning loop using the `stable-baselines3` library.

### Basic Command:
```bash
python src/training.py --timesteps 10000 
```

### Adjustable Parameters:
- `--timesteps`: **(Required)** The total number of steps the environment will run to train the agent.
- `--lr`: *(Optional)* Initial learning rate for the neural network optimizer. Uses a linear decaying schedule over time. Default: `0.0003` (`3e-4`).
- `--seed`: *(Optional)* The random seed to guarantee reproducibility across the Python environment and PyTorch initializations. Default: `42`.

---

## 🧪 How to Test the Neural Network

After training, test your agent to see how well it balances and track its internal angle changes per timestep. This is performed using the `src/debuging.py` script.

### Basic Command:
```bash
python src/debuging.py --model logs/YOUR_MODEL_DIRECTORY/SAC_Roberta_lr0.{lr}_seed{seed}.zip
```

### Adjustable Parameters:
- `--model`: **(Required)** The exact path to the trained model's `.zip` file.
- `--episodes`: *(Optional)* The number of test rounds (episodes) you would like it to play and record. Default: `10`.

During execution, the script will:
1. Render the visual arm angle environment on screen.
2. Measure the angle (`phi`) and setpoint data at each simulation step.
3. Automatically generate and save graphical plots (Angle vs Steps) inside the folder `images_test/`, located in the same directory as your trained model.

---

## 📂 Where Logs are Saved

During training, all metadata, model states, and backups are written and organized inside the `logs/` directory.

Each training run generates a unique uniquely named sub-folder:
`logs/SAC_Roberta_lr<LR>_seed<SEED>_episodes<TIMESTEPS>_<TIMESTAMP>/`

Inside this specific directory, you will likely find:
- `SAC_Roberta_lr<...>.zip`: The final saved model upon execution end.
- `info.json`: A snapshot of environment parameters and physical factors active during the run.
- `/SAC_1/` / `/eval/`: Underlying folders containing the TensorBoard logging events and intermediate evaluation models (e.g. `best_model.zip`).

---

## 📈 Debugging with TensorBoard

TensorBoard gives you a powerful visual overview of the SAC learning process, including Episode Rewards, Lengths, value losses, and entropy metrics. 

### How to open it:
1. Open a new terminal.
2. Ensure you have your project's virtual environment activated (`source .venv/bin/activate`).
3. Point TensorBoard to the root logs directory by running:

```bash
tensorboard --logdir logs/
```

4. Open your web browser and navigate to: `http://localhost:6006/`
