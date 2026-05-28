import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

# custom path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.negotiation_env import NegotiationEnv

def main():
    print("Running analysis (1000 eps)...")
    os.makedirs('graphs', exist_ok=True)
    
    try:
        b_model = PPO.load("models/trained_buyer.zip")
        s_model = PPO.load("models/trained_seller.zip")
    except FileNotFoundError:
        print("Error: Models missing. Run training first.")
        return

    env = NegotiationEnv(role='buyer', opponent_model=s_model)
    
    deals = 0
    prices = []
    anchors = []
    splits = [] 
    b_hists = []
    s_hists = []
    
    n = 1000
    for i in range(n):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = b_model.predict(obs, deterministic=False)
            obs, _, done, _, info = env.step(action)
            
        if info['deal_reached']:
            deals += 1
            prices.append(info['deal_price'])
            
            b_log = info['buyer_history']
            anchors.append(b_log[0] if b_log else info['deal_price'])
                
            z_max = info['buyer_budget']
            z_min = info['seller_floor']
            if z_max >= z_min:
                # relative location in ZOPA
                split = (info['deal_price'] - z_min) / (z_max - z_min)
                splits.append(split)
            
            b_hists.append(info['buyer_history'])
            s_hists.append(info['seller_history'])

    rate = deals / n
    avg_price = np.mean(prices) if prices else 0
    corr = np.corrcoef(anchors, prices)[0, 1] if len(anchors) > 1 else 0
    
    print(f"\nResults:\nDeal Rate: {rate*100:.1f}%\nAvg Price: ${avg_price:.2f}\nAnchoring: {corr:.3f}")
    
    # Graphs
    plt.figure()
    max_b = max([len(h) for h in b_hists]) if b_hists else 0
    max_s = max([len(h) for h in s_hists]) if s_hists else 0
    
    b_avg = [np.mean([h[r] for h in b_hists if len(h) > r]) for r in range(max_b)]
    s_avg = [np.mean([h[r] for h in s_hists if len(h) > r]) for r in range(max_s)]
        
    plt.plot(range(1, len(b_avg)+1), b_avg, label='Buyer', marker='o', color='blue')
    plt.plot(range(1, len(s_avg)+1), s_avg, label='Seller', marker='s', color='orange')
    plt.title("Avg Concession Curves")
    plt.xlabel("Round")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/2_concession_curves.png")
    plt.close()
    
    plt.figure()
    if splits:
        sns.histplot(splits, bins=20, kde=True, color='green')
        plt.title("Price Distribution in ZOPA")
        plt.xlabel("Capture (0=Floor, 1=Budget)")
        plt.ylabel("Freq")
        plt.savefig("graphs/3_final_price_heatmap.png")
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
