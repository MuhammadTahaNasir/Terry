import os
import warnings

# mute logs (must be done before imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")

import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.negotiation_env import NegotiationEnv

def main():
    print("Training start...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)

    b_env = NegotiationEnv(role='buyer')
    s_env = NegotiationEnv(role='seller')

    b_model = PPO("MlpPolicy", b_env, verbose=0)
    s_model = PPO("MlpPolicy", s_env, verbose=0)

    # config
    iters = 20 # more iterations for more frequent updates
    steps = 30000 # fewer steps per turn to make each batch "faster"
    rates = []

    for i in range(iters):
        print(f"\n--- Iteration {i+1}/{iters} ---")
        
        # Train Buyer
        print("Buyer is learning...")
        b_env.opponent_model = s_model
        b_model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=True)
        
        # Train Seller
        print("Seller is learning...")
        s_env.opponent_model = b_model
        s_model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=True)
        
        # Eval
        test_env = NegotiationEnv(role='buyer', opponent_model=s_model)
        deals = 0
        for _ in range(100):
            obs, info = test_env.reset()
            done = False
            while not done:
                action, _ = b_model.predict(obs, deterministic=False)
                obs, _, done, _, info = test_env.step(action)
            if info['deal_reached']:
                deals += 1
        
        rates.append(deals / 100.0)
        print(f"Current Agreement Rate: {deals}%")

    print("Saving models...")
    b_model.save("models/trained_buyer.zip")
    s_model.save("models/trained_seller.zip")

    plt.figure()
    plt.plot(range(1, iters + 1), rates, marker='o', color='green')
    plt.title("Training Progress (Deal Rate)")
    plt.xlabel("Iteration")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.savefig("graphs/1_deal_rate.png")
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
