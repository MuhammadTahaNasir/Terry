# MultiAgent-RL Negotiation Simulator

An AI-driven negotiation simulator built using Reinforcement Learning (Proximal Policy Optimization via `stable-baselines3`) and a Flask backend.

## Overview
This project simulates negotiations between buyers and sellers. It supports two modes:
1. **AI vs AI:** Two trained reinforcement learning models (a buyer and a seller) negotiate against each other based on hidden budgets and minimum floors.
2. **Human vs AI:** A human user takes the role of the buyer and negotiates against the trained AI seller, with real-time UI feedback, pressure gauges, and dynamic market appraisals.

## Features
- Custom Gymnasium environment for negotiations.
- PPO models trained for Buyer and Seller roles.
- Interactive Web UI with Neobrutalist design.
- Live chat, concession charts, and typing indicators.

## Previews
![Simulator UI](SS/1.png)
![Simulator Charts](SS/10.png)

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd negotiation_project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app/app.py
   ```
   Access the app at `http://127.0.0.1:5000`

## Project Structure
- `app/` - Flask web application and HTML templates.
- `env/` - Custom Gymnasium environment (`negotiation_env.py`).
- `models/` - Saved trained PPO models (`trained_buyer.zip`, `trained_seller.zip`).
- `training/` - Scripts used to train the RL agents.
- `analysis/` & `graphs/` - Analytics and visualizations of model performance.
