# Generative Artificial Intelligence: Generative Image Transformations

This project explores generative artificial intelligence techniques focusing on Temporal Domain Generalization with Drift-Aware Dynamic Neural Networks (DRAIN).
The goal is to understand and visualize how generative models progressively modify or synthesize image content through structured flows.

🧪 Key Objectives:
- Implement and test DRAIN-based image transformation pipelines
- Provide simple, workshop-level demo code for experimentation and learning
- Visualize the sourece, intermediate and target domain

📦 Technologies:
- PyTorch
- DRAIN methods
- VAE methods

This repository serves as a sandbox for AI Lab’s rapid prototyping and educational demos.

## Setup
1. Create a Conda environment
```bash
conda create -n generative_demo python=3.10
```

2. Activate the environment
conda activate generative_demo
```bash
pip install -r requirements.txt
```

## Usage
1. Train the VAE mode
```bash
python train_vae.py --dataset Moons
```
2. Train the classifier
```bash
python train_classifier.py --dataset Moons
```
3. Visualize the results
```bash
python vvs_moons.py
```
4. Check generated visualizations
```bash
ls ./visual
```