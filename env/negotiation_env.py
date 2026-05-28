import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class NegotiationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, role='buyer', opponent_model=None):
        super().__init__()
        self.role = role
        self.opponent_model = opponent_model
        
        # [choice, price] -> choice: 0-0.33 propose, 0.33-0.66 accept, 0.66-1 walkaway
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # [round, last_offer, private_price]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.max_rounds = 10
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # randomized scale for training generalization
        # handles anything from $10 to $50,000+
        base = np.random.choice([100, 500, 1000, 5000, 10000])
        self.buyer_budget = np.random.uniform(base * 0.8, base * 1.5)
        self.seller_floor = np.random.uniform(base * 0.6, base * 1.2)
        
        self.current_round = 1
        self.last_offer_val = 0.0
        self.deal_reached = False
        self.deal_price = 0.0
        
        self.buyer_history = []
        self.seller_history = []
        self.chat_log = []
        
        if options:
            if 'buyer_budget' in options:
                self.buyer_budget = float(options['buyer_budget'])
            if 'seller_floor' in options:
                self.seller_floor = float(options['seller_floor'])
                
        return self._get_obs(self.role), self._get_info()
        
    def _get_obs(self, role):
        obs = np.zeros(3, dtype=np.float32)
        obs[0] = self.current_round / 10.0
        
        # relative scale: how far is the offer from my limit?
        limit = self.buyer_budget if role == 'buyer' else self.seller_floor
        if self.last_offer_val == 0:
            obs[1] = 0.0
        else:
            # normalized ratio (0 to 2)
            obs[1] = np.clip(self.last_offer_val / limit, 0.0, 2.0)
            
        # log-scale indicator for absolute magnitude handled by the model
        obs[2] = np.clip(np.log10(limit + 1) / 5.0, 0.0, 1.0) 
        return obs
        
    def _get_info(self):
        return {
            'deal_reached': self.deal_reached,
            'deal_price': self.deal_price,
            'buyer_budget': self.buyer_budget,
            'seller_floor': self.seller_floor,
            'rounds': self.current_round,
            'buyer_history': self.buyer_history.copy(),
            'seller_history': self.seller_history.copy(),
            'chat_log': self.chat_log.copy()
        }
        
    def _process_turn(self, agent_type, action):
        phrases = {
            'buyer': [
                "could u do ${price}?", "my best offer is ${price}", 
                "how about ${price}?", "i can do ${price}",
                "would u take ${price}?", "honestly ${price} is my max..."
            ],
            'seller': [
                "i can let it go for ${price}.", "lowest i'll go is ${price}", 
                "how does ${price} sound?", "i counter with ${price}.",
                "nah i need at least ${price}.", "${price} and we have a deal."
            ],
            'accept': ["alright, ${price} works.", "ok fine. deal.", "deal. ${price} is good.", "yeah i can do that."],
            'walkaway': ["no deal, we're too far apart.", "i'm good tbh. no deal.", "out.", "pass."]
        }
        
        choice_val = action[0]
        concession_ratio = (self.current_round - 1) / 9.0  
        if 1 < self.current_round < self.max_rounds:
            concession_ratio += random.uniform(-0.08, 0.08)
            concession_ratio = max(0.0, min(1.0, concession_ratio))
            
        if getattr(self, 'is_human_turn', False) and agent_type == getattr(self, 'human_role', None):
            price_val = getattr(self, 'human_exact_price', 0.0)
            self.is_human_turn = False
        else:
            if agent_type == 'buyer':
                # relative ranges instead of hardcoded numbers
                s_min, s_max = self.buyer_budget * 0.4, self.buyer_budget * 0.6
                min_p = s_min + (concession_ratio * (self.buyer_budget - s_min))
                max_p = s_max + (concession_ratio * (self.buyer_budget * 0.1))
                price_val = min_p + (action[1] * (max(max_p, min_p + (self.buyer_budget * 0.05)) - min_p))
                
                # monotonicity rules
                last_b = self.buyer_history[-1] if self.buyer_history else 0
                price_val = max(price_val, last_b)
                
                if self.seller_history:
                    price_val = min(price_val, self.seller_history[-1])

                # Final Round Pragmatism: In round 10, if it's below budget, TAKE IT.
                if self.current_round == self.max_rounds and self.last_offer_val > 0 and self.last_offer_val <= self.buyer_budget:
                    choice_val = 0.5 # Force Accept
                elif self.last_offer_val > 0 and self.last_offer_val <= self.buyer_budget:
                    # Normal auto-accept if offer is better than our target
                    if self.last_offer_val <= price_val: choice_val = 0.5
            else:
                s_max, s_min = self.seller_floor * 1.8, self.seller_floor * 1.4
                
                # Adaptive Concession: If buyer is moving, we move slightly more
                adaptive_bonus = 0.0
                if self.buyer_history and self.last_offer_val > 0:
                    # Calculate how much of the gap the buyer has closed
                    gap = s_max - self.seller_floor
                    progress = (self.last_offer_val - (self.buyer_history[0] if self.buyer_history else 0)) / (self.seller_floor + 1)
                    adaptive_bonus = np.clip(progress * 0.2, 0.0, 0.1) # Max 10% extra concession

                effective_concession = np.clip(concession_ratio + adaptive_bonus, 0.0, 1.0)
                
                max_p = s_max - (effective_concession * (s_max - self.seller_floor))
                min_p = s_min - (effective_concession * (s_min * 0.1))
                price_val = min_p + (action[1] * (max(max_p, min_p + (self.seller_floor * 0.05)) - min_p))
                
                last_s = self.seller_history[-1] if self.seller_history else price_val * 2
                price_val = min(price_val, last_s)
                
                if self.buyer_history:
                    price_val = max(price_val, self.buyer_history[-1])

                if self.buyer_history:
                    price_val = max(price_val, self.buyer_history[-1])

                # Final Round Pragmatism: In round 10, if it's above floor, TAKE IT.
                if self.current_round == self.max_rounds and self.last_offer_val >= self.seller_floor:
                    choice_val = 0.5 # Force Accept
                elif self.last_offer_val > 0 and self.last_offer_val >= self.seller_floor:
                    # Normal auto-accept if offer is better than our target
                    if self.last_offer_val >= price_val: choice_val = 0.5

        # Hard Constraints: Never propose a deal that violates your own secret limit
        if agent_type == 'buyer':
            price_val = min(price_val, self.buyer_budget)
        else:
            price_val = max(price_val, self.seller_floor)
                
        # price must be > 0
        price_val = max(1.0, round(price_val))
        
        # Insult Rule: walkaway if offer is >70% divergent from limit
        limit = self.buyer_budget if agent_type == 'buyer' else self.seller_floor
        if self.last_offer_val > 0:
            divergence = abs(self.last_offer_val - limit) / limit
            if divergence > 0.7:
                self.deal_reached = False
                self.chat_log.append(f"{agent_type.capitalize()}: That's an insulting offer. I'm out.")
                return True

        if choice_val > 0.66: # Walkaway
            # Intercession Logic: If the other party's last offer was fair, don't let them walk!
            if agent_type == 'buyer' and self.buyer_history and self.buyer_history[-1] >= self.seller_floor:
                # User (Buyer) walks away, AI Seller chases them by accepting their last offer
                self.deal_reached = True
                self.deal_price = self.buyer_history[-1]
                self.chat_log.append(f"Seller: Wait! Before you go, I've reconsidered. I'll take your offer of ${round(self.deal_price)}.")
                return True
            if agent_type == 'seller' and self.seller_history and self.seller_history[-1] <= self.buyer_budget:
                # AI Seller walks away, Buyer (Human) chases by taking their last offer (Optional, but kept for symmetry)
                self.deal_reached = True
                self.deal_price = self.seller_history[-1]
                self.chat_log.append(f"Buyer: Wait! I'll take that offer. ${round(self.deal_price)} works.")
                return True

            self.deal_reached = False
            self.chat_log.append(f"{agent_type.capitalize()}: {random.choice(phrases['walkaway'])}")
            return True
            
        elif choice_val > 0.33: # Accept Attempt
            if self.current_round == 1 or self.last_offer_val <= 0:
                self.chat_log.append(f"System: No offer available to accept yet.")
                return True # End turn with no-op

            # Strict Rationality Guard: Never accept a lossy deal
            if (agent_type == 'buyer' and self.last_offer_val > self.buyer_budget) or \
               (agent_type == 'seller' and self.last_offer_val < self.seller_floor):
                self.chat_log.append(f"System: Invalid acceptance. Offer ${round(self.last_offer_val)} exceeds limits.")
                return True # End turn as failure
            else:
                self.deal_reached = True
                self.deal_price = self.last_offer_val
                msg = random.choice(phrases['accept']).format(price=self.deal_price)
                self.chat_log.append(f"{agent_type.capitalize()}: {msg}")
                (self.buyer_history if agent_type == 'buyer' else self.seller_history).append(self.deal_price)
                return True
                
        self.last_offer_val = price_val
        msg = random.choice(phrases[agent_type]).format(price=price_val)
        self.chat_log.append(f"{agent_type.capitalize()}: {msg}")
        (self.buyer_history if agent_type == 'buyer' else self.seller_history).append(price_val)
        return False
            
    def _get_reward(self, role):
        # purely relative reward based on surplus capture
        penalty = (self.current_round / self.max_rounds) * 0.1
        if not self.deal_reached:
            return -0.5 - penalty
            
        if role == 'buyer':
            res = (self.buyer_budget - self.deal_price) / self.buyer_budget
        else:
            res = (self.deal_price - self.seller_floor) / self.seller_floor
            
        return res - penalty

    def step(self, action):
        if self._process_turn(self.role, action):
            return self._get_obs(self.role), self._get_reward(self.role), True, False, self._get_info()
            
        opp = 'seller' if self.role == 'buyer' else 'buyer'
        opp_obs = self._get_obs(opp)
        
        opp_action = self.action_space.sample() if self.opponent_model is None else self.opponent_model.predict(opp_obs, deterministic=False)[0]
            
        if self._process_turn(opp, opp_action):
            return self._get_obs(self.role), self._get_reward(self.role), True, False, self._get_info()
            
        self.current_round += 1
        if self.current_round >= self.max_rounds:
            self.deal_reached = False
            self.chat_log.append("Max rounds reached. No deal.")
            return self._get_obs(self.role), self._get_reward(self.role), True, False, self._get_info()
            
        return self._get_obs(self.role), 0.0, False, False, self._get_info()
