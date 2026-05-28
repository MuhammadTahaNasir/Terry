import os
import sys
import warnings

# mute logs (must be done before other imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")

import numpy as np
from flask import Flask, render_template, request, jsonify, make_response
from stable_baselines3 import PPO

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.negotiation_env import NegotiationEnv

app = Flask(__name__)

buyer_model = None
seller_model = None
interactive_env = None

def load_models():
    global buyer_model, seller_model
    try:
        m_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        buyer_model = PPO.load(os.path.join(m_dir, "trained_buyer.zip"))
        seller_model = PPO.load(os.path.join(m_dir, "trained_seller.zip"))
        return True
    except Exception as e:
        print(f"faulty load: {e}")
        return False

@app.route('/')
def home():
    res = make_response(render_template('landing.html'))
    res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return res

@app.route('/simulator')
def simulator():
    res = make_response(render_template('index.html'))
    res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return res

@app.route('/negotiate', methods=['POST'])
def negotiate():
    if not buyer_model or not seller_model:
        if not load_models():
            return jsonify({'error': 'models missing'})

    data = request.json
    b_budget = float(data.get('buyer_budget', 120))
    s_floor = float(data.get('seller_floor', 80))

    env = NegotiationEnv(role='buyer', opponent_model=seller_model)
    obs, info = env.reset(options={'buyer_budget': b_budget, 'seller_floor': s_floor})
    
    done = False
    while not done:
        action, _ = buyer_model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        
    return jsonify({
        'deal_reached': bool(info['deal_reached']),
        'deal_price': round(float(info['deal_price']), 2),
        'buyer_budget': round(float(info['buyer_budget']), 2),
        'seller_floor': round(float(info['seller_floor']), 2),
        'rounds': int(info['rounds']),
        'log': info['chat_log'],
        'buyer_history': [round(float(x), 2) for x in info.get('buyer_history', [])],
        'seller_history': [round(float(x), 2) for x in info.get('seller_history', [])]
    })

@app.route('/human_start', methods=['POST'])
def human_start():
    global interactive_env
    if not seller_model:
        if not load_models():
            return jsonify({'error': 'models missing'})

    data = request.json
    env_options = {
        'buyer_budget': float(data.get('buyer_budget', 120)),
        'seller_floor': float(data.get('seller_floor', 80))
    }

    interactive_env = NegotiationEnv(role='buyer', opponent_model=seller_model)
    _, info = interactive_env.reset(options=env_options)
    
    return jsonify({
        'log': info['chat_log'],
        'buyer_history': [],
        'seller_history': [],
    })

@app.route('/human_step', methods=['POST'])
def human_step():
    global interactive_env
    if not interactive_env:
        return jsonify({'error': 'session inactive'})
        
    data = request.json
    action_type = data.get('action') 
    price = float(data.get('price', 0.0))
    
    mapping = {'propose': 0.0, 'accept': 0.5, 'walkaway': 1.0}
    choice_val = mapping.get(action_type, 1.0)
    action = np.array([choice_val, 0.5], dtype=np.float32) # agent handles price via direct override
    
    # manual override for human input
    interactive_env.is_human_turn = True
    interactive_env.human_role = 'buyer'
    interactive_env.human_exact_price = float(price)
    
    obs, _, done, _, info = interactive_env.step(action)
    
    return jsonify({
        'deal_reached': bool(info['deal_reached']),
        'deal_price': round(float(info['deal_price']), 2),
        'seller_floor': round(float(interactive_env.seller_floor), 2),
        'rounds': int(info['rounds']),
        'done': bool(done),
        'log': info['chat_log'],
        'buyer_history': [round(float(x), 2) for x in info.get('buyer_history', [])],
        'seller_history': [round(float(x), 2) for x in info.get('seller_history', [])]
    })

if __name__ == '__main__':
    load_models()
    app.run(debug=False, port=5000)
