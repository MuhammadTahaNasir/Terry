# 🤖 Nego Bots Documentation

<div align="center">
  <strong>Comprehensive guide to the Reinforcement Learning Architecture and Custom Gymnasium Environment</strong>
</div>

---

## 1. Core Architecture

The Nego Bots engine is divided into three critical layers:
1.  **Presentation Layer (Flask / HTML5)**: Handles the HTTP communication, state bridging, and rendering the Neobrutalist UI.
2.  **Reinforcement Learning Layer (Stable-Baselines3)**: Contains the Proximal Policy Optimization (PPO) agents for both Buyer and Seller.
3.  **Environment Layer (Gymnasium)**: The custom `MultiTurnNegotiationEnv` where reward structures and state spaces are calculated.

---

## 2. The Gymnasium Environment (`negotiation_env.py`)

Unlike standard Gym environments, negotiation is a multi-turn, multi-agent process with **hidden information**.

### State Space (Observation)
The environment provides the following 4-dimensional observation vector to the agents:
*   `budget` (Normalized): The buyer's maximum budget (hidden from the seller).
*   `floor` (Normalized): The seller's minimum acceptable price (hidden from the buyer).
*   `last_offer` (Normalized): The most recent offer made on the table.
*   `turn_count` (Integer): How many turns have elapsed (max 10).

### Action Space
The agents output a continuous value `[-1.0, 1.0]` which maps to a percentage of the remaining negotiation range. 
*   **Buyer**: Suggests a price between `last_offer` and `budget`.
*   **Seller**: Suggests a price between `floor` and `last_offer`.

### Reward Structure
The reward function heavily penalizes walking away (greed) while encouraging surplus capture:
*   `Deal Reached`: Large positive reward proportional to the surplus captured.
*   `Walk Away`: Large negative penalty (`-10.0`) for failing to find the Nash Equilibrium.
*   `Time Penalty`: Small negative penalty (`-0.1`) per turn to encourage rapid convergence.

---

## 3. Training the Agents (`train.py`)

The agents are trained using **PPO (Proximal Policy Optimization)** via Stable-Baselines3.

To retrain the agents with new behaviors (e.g., making the Buyer more aggressive):
```bash
# In the training directory
python train.py --epochs 50000 --learning_rate 0.0003
```
This will overwrite `models/trained_buyer.zip` and `models/trained_seller.zip`.

---

## 4. Frontend Integration (`app.py`)

The Flask backend loads the zipped models into memory on boot. When the UI sends a `POST /negotiate` request, the server spins up a sandbox environment, drops both trained agents into it, and extracts the full action trajectory (the chat log and offer history).

*Designed for the AI-Negotiation-Bot Project by Muhammad Taha Nasir.*
