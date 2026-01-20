# Policy Under Pressure

**Interactive experiments exploring how intelligent agents behave when rewards, sensors, and physics turn hostile**

🔗 **[Live Demo](https://your-deployment-url.vercel.app)** | 📊 **Research Project** | 🎓 **September 2025 - January 2026**

![Policy Under Pressure](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🎯 Overview

This project systematically compares **Reinforcement Learning (RL)** and **Evolution Strategies (ES)** through adversarial testing in a 2D racing environment. We expose both methods to hostile conditions—reward misalignment, sensor corruption, and physics manipulation—to reveal fundamental differences in their learned behaviors.

### Key Findings

- ✅ **ES wins on robustness**: Superior performance under all adversarial conditions
- ⚠️ **RL exhibits policy collapse**: Learned to minimize speed as a survival strategy
- 📊 **Reward misalignment matters**: Small changes in reward functions drastically alter behavior
- 🔬 **Gradient-based vs Population-based**: Fundamental differences in optimization landscapes

---

## 📂 Project Structure

```
d:/agent compare project/
├── frontend/                    # React + Vite web application
│   ├── src/
│   │   ├── pages/              # 4 segment pages + homepage
│   │   ├── components/         # Reusable UI components
│   │   └── ...
│   └── public/                 # Pre-generated JSON experiment results
│       ├── experiment_results.json
│       ├── segment2_results_v2.json
│       ├── gradient_results_v2.json
│       └── segment4_*.json (4 files)
│
├── backend/                    # Python physics/reward engine
│   ├── physics/               # Car dynamics, friction, noise
│   ├── rewards/               # Reward function definitions
│   └── experiments/           # Wrappers for adversarial testing
│
├── models/                    # Trained agent checkpoints (.pkl, .zip)
├── train_*.py                 # Training scripts for all agents
├── run_*.py                   # Experiment execution scripts
└── experiments.md             # Detailed methodology notes

```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.9+ (for training/experiments - optional)

### Running Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/policy-under-pressure.git
cd policy-under-pressure

# Install frontend dependencies
cd frontend
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:8080`

### Building for Production

```bash
cd frontend
npm run build
# Output in frontend/dist/
```

---

## 🌐 Deployment

### Deploy to Vercel (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Deploy**:
   ```bash
   vercel
   ```

3. **Follow prompts** - Vercel auto-detects the configuration from `vercel.json`

### Manual Deployment

Deploy the `frontend/dist/` folder to any static hosting:
- **Netlify**: Drag & drop `dist` folder
- **GitHub Pages**: Push `dist` to `gh-pages` branch
- **Cloudflare Pages**: Connect repo and set build command

---

## 📊 Experiments Overview

### Segment 1: Fragility
**Baseline comparison** - Both agents tested on standard track (Figure-8)
- RL achieves higher speed but exhibits brittleness
- ES shows more conservative, stable behavior

### Segment 2: Robustness
**Adversarial physics** - Friction reduction and sensor noise injection
- **Friction Ladder**: RL collapses to near-zero speed, ES maintains function
- **Noise Injection**: ES degrades gracefully, RL already at floor performance

### Segment 3: Gradient Analysis  
**Statistical robustness** - Aggregated metrics across severity levels
- RL stuck in local minimum (policy collapse during training)
- ES benefits from population-based exploration
- Reveals fundamental differences in optimization landscapes

### Segment 4: Reward Manipulation
**Controlled experiments** - Systematic reward engineering
1. **Exploit Behavior**: Both agents show preference for easier tracks
2. **Sensitivity Analysis**: ±10%, ±20% reward shifts expose brittleness
3. **Misalignment Test**: RL maintains quality despite wrong rewards (learned representations)
4. **Parallel Scaling**: ES scales linearly, RL plateaus

---

## 🔬 Technical Details

### Agents

**Reinforcement Learning (RL - PPO)**
- **Algorithm**: Proximal Policy Optimization
- **Framework**: Stable-Baselines3
- **Architecture**: 2-layer MLP (64 units)
- **Training**: 500K timesteps, gradient-based optimization
- **Observation**: 8D state vector (position, velocity, orientation, track)
- **Action**: 2D continuous (steering, throttle)

**Evolution Strategies (ES)**
- **Algorithm**: OpenAI ES (CMA-ES variant)
- **Framework**: Custom implementation
- **Architecture**: 2-layer MLP (64 units, matching RL)
- **Training**: 300 generations, population size 50
- **Fitness**: Episodic reward (no gradients)

### Environment

- **Track**: Figure-8 layout (2 loops, sharp turns)
- **Physics**: 2D car dynamics with friction, momentum
- **Adversarial Modes**:
  - Friction reduction (μ: 1.0 → 0.2)
  - Sensor noise (Gaussian, σ: 0.0 → 1.0)
  - Reward delay (0-20 timesteps)
  - Action masking (random dropout)

### Data Pipeline

All experiments were run **once** and results saved as JSON:
1. **Training**: `train_*.py` scripts → save models to `/models/`
2. **Experiments**: `run_*.py` scripts → load models, run tests, save JSON
3. **Frontend**: React app fetches pre-generated JSON, no server needed

---

## 📈 Key Metrics

- **Speed**: Mean velocity (m/s) during episode
- **Lateral Error**: RMS distance from track centerline
- **Survival Rate**: % of episodes completing without crash
- **Steering Variance**: Control activity indicator

---

## 🛠️ Tech Stack

### Frontend
- **React** 18.3 + **TypeScript**
- **Vite** 5.4 (build tool)
- **TailwindCSS** 3.4 (styling)
- **Recharts** 2.15 (data visualization)
- **Framer Motion** (animations)
- **Lucide React** (icons)

### Backend (Training Only)
- **Python** 3.9+
- **PyTorch** 2.0+
- **Stable-Baselines3** (RL)
- **NumPy**, **Matplotlib** (data/viz)

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{policyunderpressure2026,
  title={Policy Under Pressure: Adversarial Testing of RL vs ES},
  author={Keshav Majithia},
  year={2026},
  howpublished={\url{https://your-deployment-url.vercel.app}}
}
```

---

## 🤝 Contributing

This is a completed research project, but feedback and discussions are welcome!

1. **Issues**: Report bugs or suggest improvements
2. **Discussions**: Technical questions about methodology
3. **Forks**: Feel free to extend with your own experiments

---

## 📄 License

MIT License - see `LICENSE` file for details

---

## 👤 Author

**Keshav Majithia**
- 🔗 [LinkedIn](https://linkedin.com/in/keshav-m-9a2701252)
- 🐦 [Twitter](https://twitter.com/keshav_m__)
- 📧 [Email](mailto:keshavmajithia13@gmail.com)

---

## 🙏 Acknowledgments

- **OpenAI** - Evolution Strategies inspiration
- **Stable-Baselines3** - RL implementation
- **Recharts** - Beautiful charts
- **Vercel** - Hosting platform

---

**Built with ❤️ over 5 months** | September 2025 - January 2026

⭐ **Star this repo** if you found it interesting!
