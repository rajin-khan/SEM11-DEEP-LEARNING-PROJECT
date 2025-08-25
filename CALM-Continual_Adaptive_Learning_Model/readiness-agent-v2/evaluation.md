# CALM Framework Evaluation Guide: Complete Command Reference

This guide provides all commands to evaluate your CALM (Continual Adaptive Learning Model) framework, organized by execution time and research priority.

## 📋 Prerequisites

### 1. Setup Environment
```bash
# Navigate to project directory
cd CALM-Continual_Adaptive_Learning_Model/readiness-agent

# Install dependencies
pip install torch torchvision transformers scikit-learn matplotlib numpy tqdm
```

### 2. Verify Files
Ensure these files exist in your directory:
- `memory.py` (Enhanced version)
- `calibration.py` (Enhanced version)  
- `agent.py` (Enhanced version)
- `evaluate_agent_improved.py` (Complete version)

---

## ⚡ Quick Start Commands (1-5 minutes each)

Perfect for initial testing and debugging.

### Test Run with Debug Output
```bash
python evaluate_agent_improved.py --debug
```
**Time:** ~2-3 minutes  
**Purpose:** Verify everything works, see detailed output  
**Dataset:** CIFAR-10 (10 classes, small)  
**Configuration:** 10-shot, temperature calibration, adaptive thresholding  

### Minimal Configuration Test
```bash
python evaluate_agent_improved.py --k_shot 1 --dataset fashionmnist
```
**Time:** ~1-2 minutes  
**Purpose:** Fastest possible test  
**Why fast:** 1-shot memory + FashionMNIST (simpler dataset)  

### No Calibration Baseline
```bash
python evaluate_agent_improved.py --calibration_method none --k_shot 5
```
**Time:** ~2 minutes  
**Purpose:** Quick baseline without calibration overhead  

---

## 🏃 Fast Experiments (5-10 minutes each)

Good for parameter exploration and initial results.

### Different Memory Sizes (CIFAR-10)
```bash
# 1-shot (fastest)
python evaluate_agent_improved.py --k_shot 1

# 5-shot (balanced)
python evaluate_agent_improved.py --k_shot 5

# 10-shot (default)
python evaluate_agent_improved.py --k_shot 10
```
**Time:** ~3-5 minutes each  
**Purpose:** Understand memory size vs performance trade-off  

### Calibration Method Comparison (5-shot, CIFAR-10)
```bash
# Temperature scaling (fastest calibration)
python evaluate_agent_improved.py --k_shot 5 --calibration_method temperature

# No calibration (baseline)
python evaluate_agent_improved.py --k_shot 5 --calibration_method none

# Isotonic regression
python evaluate_agent_improved.py --k_shot 5 --calibration_method isotonic

# Platt scaling
python evaluate_agent_improved.py --k_shot 5 --calibration_method platt
```
**Time:** ~4-7 minutes each  
**Purpose:** Compare calibration methods  

### Safety vs Efficiency Trade-offs
```bash
# High safety (conservative)
python evaluate_agent_improved.py --target_deployment 0.5 --k_shot 5

# Balanced
python evaluate_agent_improved.py --target_deployment 0.75 --k_shot 5

# High efficiency (aggressive)
python evaluate_agent_improved.py --target_deployment 0.9 --k_shot 5
```
**Time:** ~4-6 minutes each  
**Purpose:** Explore safety-efficiency trade-off  

---

## 🚶 Medium Experiments (10-20 minutes each)

Core experimental comparisons for research.

### Cross-Dataset Generalization (5-shot, Temperature)
```bash
# CIFAR-10 (10 classes, 32x32, natural images)
python evaluate_agent_improved.py --dataset cifar10 --k_shot 5 --calibration_method temperature

# FashionMNIST (10 classes, 28x28, fashion items)
python evaluate_agent_improved.py --dataset fashionmnist --k_shot 5 --calibration_method temperature

# STL-10 (10 classes, 96x96, high-resolution)
python evaluate_agent_improved.py --dataset stl10 --k_shot 5 --calibration_method temperature
```
**Time:** ~5-15 minutes each  
**Purpose:** Test generalization across different visual domains  
**Note:** STL-10 takes longer due to higher resolution  

### Comprehensive Calibration Study (CIFAR-10, 10-shot)
```bash
python evaluate_agent_improved.py --dataset cifar10 --k_shot 10 --calibration_method temperature --debug
python evaluate_agent_improved.py --dataset cifar10 --k_shot 10 --calibration_method isotonic --debug
python evaluate_agent_improved.py --dataset cifar10 --k_shot 10 --calibration_method platt --debug
python evaluate_agent_improved.py --dataset cifar10 --k_shot 10 --calibration_method none --debug
```
**Time:** ~8-12 minutes each  
**Purpose:** Detailed calibration comparison with full debugging output  

### High-Memory Experiments
```bash
# 20-shot (high memory)
python evaluate_agent_improved.py --k_shot 20 --dataset cifar10

# 20-shot with debug for detailed analysis
python evaluate_agent_improved.py --k_shot 20 --dataset cifar10 --debug
```
**Time:** ~10-15 minutes each  
**Purpose:** Test performance with larger memory  

---

## 🏋️ Heavy Experiments (20-60 minutes each)

Comprehensive evaluations for publication-quality results.

### CIFAR-100 Experiments (Most Challenging)
```bash
# CIFAR-100 baseline (100 classes!)
python evaluate_agent_improved.py --dataset cifar100 --k_shot 5 --calibration_method temperature

# CIFAR-100 with minimal memory (hardest scenario)
python evaluate_agent_improved.py --dataset cifar100 --k_shot 1 --calibration_method temperature

# CIFAR-100 with high memory
python evaluate_agent_improved.py --dataset cifar100 --k_shot 10 --calibration_method temperature

# CIFAR-100 full analysis
python evaluate_agent_improved.py --dataset cifar100 --k_shot 5 --calibration_method temperature --debug
```
**Time:** ~20-45 minutes each  
**Purpose:** Most challenging dataset (100 classes vs 10)  
**Why slow:** Much more complex classification problem  

### STL-10 High-Resolution Analysis
```bash
# STL-10 comprehensive study
python evaluate_agent_improved.py --dataset stl10 --k_shot 5 --calibration_method temperature --debug
python evaluate_agent_improved.py --dataset stl10 --k_shot 10 --calibration_method temperature --debug
python evaluate_agent_improved.py --dataset stl10 --k_shot 20 --calibration_method temperature --debug
```
**Time:** ~15-30 minutes each  
**Purpose:** High-resolution image analysis (96x96 vs 32x32)  
**Why slow:** Larger images = more computation  

### Publication-Quality Configurations
```bash
# Best performance setup
python evaluate_agent_improved.py --dataset cifar10 --k_shot 20 --calibration_method temperature --target_deployment 0.75 --debug

# Safety-critical setup
python evaluate_agent_improved.py --dataset cifar10 --k_shot 10 --calibration_method isotonic --target_deployment 0.6 --debug

# Efficiency-focused setup
python evaluate_agent_improved.py --dataset cifar10 --k_shot 5 --calibration_method temperature --target_deployment 0.85 --debug
```
**Time:** ~15-25 minutes each  
**Purpose:** Generate publication-ready results  

---

## 🌟 Complete Research Protocol

### Phase 1: Quick Validation (30-45 minutes total)
Run these first to ensure everything works:

```bash
# 1. Basic functionality test
python evaluate_agent_improved.py --debug

# 2. Minimal test
python evaluate_agent_improved.py --k_shot 1 --dataset fashionmnist

# 3. No calibration baseline
python evaluate_agent_improved.py --calibration_method none --k_shot 5

# 4. Different memory sizes
python evaluate_agent_improved.py --k_shot 1
python evaluate_agent_improved.py --k_shot 5
python evaluate_agent_improved.py --k_shot 10
```

### Phase 2: Core Comparisons (2-3 hours total)
Main experimental results for your research:

```bash
# Memory size analysis (CIFAR-10)
python evaluate_agent_improved.py --k_shot 1 --dataset cifar10
python evaluate_agent_improved.py --k_shot 5 --dataset cifar10  
python evaluate_agent_improved.py --k_shot 10 --dataset cifar10
python evaluate_agent_improved.py --k_shot 20 --dataset cifar10

# Calibration method comparison (5-shot)
python evaluate_agent_improved.py --k_shot 5 --calibration_method temperature
python evaluate_agent_improved.py --k_shot 5 --calibration_method isotonic
python evaluate_agent_improved.py --k_shot 5 --calibration_method platt
python evaluate_agent_improved.py --k_shot 5 --calibration_method none

# Cross-dataset validation
python evaluate_agent_improved.py --dataset cifar10 --k_shot 5
python evaluate_agent_improved.py --dataset fashionmnist --k_shot 5
python evaluate_agent_improved.py --dataset stl10 --k_shot 5
```

### Phase 3: Challenging Scenarios (3-4 hours total)
For comprehensive evaluation:

```bash
# CIFAR-100 (most challenging)
python evaluate_agent_improved.py --dataset cifar100 --k_shot 1
python evaluate_agent_improved.py --dataset cifar100 --k_shot 5
python evaluate_agent_improved.py --dataset cifar100 --k_shot 10

# High-resolution analysis (STL-10)
python evaluate_agent_improved.py --dataset stl10 --k_shot 5 --debug
python evaluate_agent_improved.py --dataset stl10 --k_shot 10 --debug

# Safety-efficiency trade-offs
python evaluate_agent_improved.py --target_deployment 0.5 --k_shot 5
python evaluate_agent_improved.py --target_deployment 0.75 --k_shot 5
python evaluate_agent_improved.py --target_deployment 0.9 --k_shot 5
```

### Phase 4: Publication Results (2-3 hours total)
Final high-quality results:

```bash
# Best configurations with full analysis
python evaluate_agent_improved.py --dataset cifar10 --k_shot 10 --calibration_method temperature --debug
python evaluate_agent_improved.py --dataset cifar100 --k_shot 5 --calibration_method isotonic --debug
python evaluate_agent_improved.py --dataset stl10 --k_shot 10 --calibration_method temperature --debug

# Safety-critical scenarios
python evaluate_agent_improved.py --dataset cifar10 --k_shot 15 --calibration_method isotonic --target_deployment 0.6 --debug
```

---

## 📊 Expected Results Interpretation

### Key Metrics to Track:
- **Base Model Accuracy**: Core few-shot learning performance
- **False Positive Rate (FPR)**: Safety metric (target < 10%)
- **Deployment Decision Accuracy**: Agent effectiveness (target > 85%)
- **Deployment Rate**: Efficiency metric (balance with safety)

### Performance Expectations:
- **CIFAR-10**: 75-90% base accuracy
- **CIFAR-100**: 45-70% base accuracy (much harder)
- **STL-10**: 80-95% base accuracy
- **FashionMNIST**: 85-95% base accuracy

### Time Estimates by Configuration:
- **1-shot, small dataset**: 1-2 minutes
- **5-shot, CIFAR-10**: 3-5 minutes
- **10-shot, CIFAR-10**: 5-8 minutes
- **5-shot, CIFAR-100**: 15-30 minutes
- **10-shot, STL-10**: 10-20 minutes
- **20-shot, any dataset**: +50% time
- **Debug mode**: +25% time

---

## 🚨 Troubleshooting Commands

### If Out of Memory:
```bash
# Reduce memory usage
python evaluate_agent_improved.py --k_shot 1
python evaluate_agent_improved.py --k_shot 5 --dataset fashionmnist
```

### If Crashes:
```bash
# Test minimal configuration
python evaluate_agent_improved.py --dataset fashionmnist --k_shot 1 --calibration_method none
```

### If Slow Performance:
```bash
# Use CPU instead of GPU (add this if needed)
export CUDA_VISIBLE_DEVICES=""
python evaluate_agent_improved.py --k_shot 5
```

---

## 🎯 Recommended Daily Schedule

### Day 1: Setup & Validation (1-2 hours)
- Run Phase 1 commands
- Verify all functionality works
- Test different memory sizes

### Day 2: Core Experiments (3-4 hours)
- Run Phase 2 commands
- Compare calibration methods
- Test cross-dataset performance

### Day 3: Challenging Scenarios (4-5 hours)
- Run Phase 3 commands
- Focus on CIFAR-100 experiments
- Analyze safety-efficiency trade-offs

### Day 4: Publication Results (3-4 hours)
- Run Phase 4 commands
- Generate final publication-quality results
- Create results tables and graphs

**Total Time Investment: ~12-15 hours of compute time**

Start with the quick commands to verify everything works, then progressively move to more comprehensive experiments. The time estimates include both computation and result analysis time.