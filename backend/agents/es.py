import numpy as np
import copy
import pickle

class ESAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        ES-Optimized Agent:
        - Larger network (128 vs 32) for complex control
        - Observation normalization
        - Action smoothing for stability
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Larger 3-layer MLP with biases
        self.layer_shapes = [
            (input_dim, hidden_dim),  # W1
            (hidden_dim,),            # b1  
            (hidden_dim, hidden_dim), # W2 (extra layer)
            (hidden_dim,),            # b2
            (hidden_dim, output_dim), # W3
            (output_dim,)             # b3
        ]
        
        self.param_count = 0
        self.weights = []
        for shape in self.layer_shapes:
            size = np.prod(shape)
            self.param_count += size
            self.weights.append(np.random.randn(*shape) * 0.1)
        
        # Action smoothing (low-pass filter)
        self.prev_action = np.zeros(output_dim)
        self.smoothing_factor = 0.7
        
        # Observation normalization (running stats)
        self.obs_mean = np.zeros(input_dim)
        self.obs_std = np.ones(input_dim)
        self.obs_count = 0
    
    def reset(self):
        """Reset internal state (stats and smoothing)"""
        self.obs_mean = np.zeros(self.input_dim)
        self.obs_std = np.ones(self.input_dim)
        self.obs_count = 0
        self.prev_action = np.zeros(self.output_dim)

    def get_flat_weights(self):
        flat = np.concatenate([w.flatten() for w in self.weights])
        return flat

    def set_flat_weights(self, flat_weights):
        idx = 0
        new_weights = []
        for shape in self.layer_shapes:
            size = np.prod(shape)
            w = flat_weights[idx:idx+size].reshape(shape)
            new_weights.append(w)
            idx += size
        self.weights = new_weights
    
    def normalize_obs(self, obs):
        """Running normalization for stable ES gradient"""
        self.obs_count += 1
        alpha = 1.0 / min(self.obs_count, 1000)
        self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs
        self.obs_std = (1 - alpha) * self.obs_std + alpha * np.abs(obs - self.obs_mean)
        self.obs_std = np.maximum(self.obs_std, 0.01)
        return (obs - self.obs_mean) / self.obs_std

    def predict(self, observation):
        """Forward pass with normalization and smoothing"""
        # Normalize observation
        x = self.normalize_obs(observation)
        
        # 3-layer forward pass
        W1, b1 = self.weights[0], self.weights[1]
        h1 = np.tanh(x @ W1 + b1)
        
        W2, b2 = self.weights[2], self.weights[3]
        h2 = np.tanh(h1 @ W2 + b2)
        
        W3, b3 = self.weights[4], self.weights[5]
        out = np.tanh(h2 @ W3 + b3)
        
        # Map to action space
        steering = out[0]
        throttle = (out[1] + 1.0) / 2.0
        
        # Add throttle floor (prevent standing still)
        throttle = max(throttle, 0.3)
        
        # NO ACTION SMOOTHING - allow precise steering for centerline following
        action = np.array([steering, throttle])
        
        return action, None

    def save(self, path):
        # Save weights and normalization stats
        data = {
            'weights': self.get_flat_weights(),
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'obs_count': self.obs_count
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    
    def get_perturbed_agent(self, sigma=0.1):
        """Returns a new ESAgent with perturbed weights (for rollout)"""
        # 1. Generate Noise
        noise = []
        for w in self.weights:
            noise.append(np.random.randn(*w.shape))
            
        # 2. Create Clone
        agent = copy.deepcopy(self)
        
        # 3. Apply Noise: w_new = w + sigma * noise
        for i in range(len(agent.weights)):
            agent.weights[i] += sigma * noise[i]
            
        return noise, agent

    def update(self, results, alpha=0.01, sigma=0.1):
        """
        Canonical ES Update:
        w_new = w + alpha * (1 / (sigma * N)) * sum(F_i * epsilon_i)
        results: list of (noise, reward)
        """
        # Rank Normalization (optional but recommended for robustness)
        rewards = np.array([r for _, r in results])
        # Centering ranks to [-0.5, 0.5] usually works well
        # But for simple implementation, we stick to raw rewards or simple centering
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        # Aggregate Gradient
        grad = [np.zeros_like(w) for w in self.weights]
        
        for i, (noise, _) in enumerate(results):
            r = rewards[i]
            for layer_idx in range(len(grad)):
                grad[layer_idx] += r * noise[layer_idx]
        
        # Apply Update
        for i in range(len(self.weights)):
            self.weights[i] += alpha / (sigma * len(results)) * grad[i]
            
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # New format with normalization stats
            self.set_flat_weights(data['weights'])
            self.obs_mean = data.get('obs_mean', self.obs_mean)
            self.obs_std = data.get('obs_std', self.obs_std)
            self.obs_count = data.get('obs_count', 0)
        else:
            # Old format (just weights)
            self.set_flat_weights(data)
