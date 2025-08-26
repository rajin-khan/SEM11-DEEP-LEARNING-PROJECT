import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

# Create output directory
output_dir = Path("calm_paper_figures")
output_dir.mkdir(exist_ok=True)

def save_figure(fig, filename, dpi=300):
    """Save figure in high quality"""
    filepath = output_dir / f"{filename}.png"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved: {filepath}")

# Data from the paper tables
def create_few_shot_comparison():
    """Table I: Few-Shot Learning Performance Comparison"""
    methods = ['Prototypical\nNetworks', 'CLIP\n(Zero-Shot)', 'Tip-Adapter', 'CALM\n(Ours)']
    accuracies = [77.5, 85.4, 89.8, 96.88]
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Few-Shot Learning Performance Comparison\n(5-way 5-shot on ImageNet)')
    ax.set_ylim(70, 100)
    ax.grid(True, alpha=0.3)
    
    # Highlight our method
    bars[-1].set_edgecolor('red')
    bars[-1].set_linewidth(3)
    
    plt.tight_layout()
    save_figure(fig, "few_shot_comparison")
    plt.close()

def create_continual_learning_comparison():
    """Table II: Continual Learning Performance"""
    methods = ['Fine-tuning', 'iCaRL', 'DualPrompt', 'CALM']
    avg_acc = [25, 72, 88, 59.61]
    bwt = [-65, -11, -1.2, -5.52]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average Accuracy
    colors1 = ['red', 'orange', 'lightgreen', 'gold']
    bars1 = ax1.bar(methods, avg_acc, color=colors1, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Final Average Accuracy (%)')
    ax1.set_title('Continual Learning: Final Accuracy')
    ax1.set_ylim(0, 100)
    
    for bar, acc in zip(bars1, avg_acc):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # Backward Transfer (closer to 0 is better)
    colors2 = ['red', 'orange', 'lightgreen', 'gold']
    bars2 = ax2.bar(methods, bwt, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Backward Transfer (%)')
    ax2.set_title('Continual Learning: Forgetting Resistance\n(Higher = Less Forgetting)')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for bar, b in zip(bars2, bwt):
        height = bar.get_height()
        offset = -2 if height < 0 else 1
        ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{b}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, "continual_learning_comparison")
    plt.close()

def create_kshot_analysis():
    """Table V: Performance vs. Episodic Memory Size"""
    datasets = ['CIFAR-10', 'CIFAR-100', 'STL-10', 'FashionMNIST']
    k_values = [1, 5, 10, 20]
    
    # Base Model Accuracy data
    accuracy_data = {
        'CIFAR-10': [59.49, 80.24, 84.74, 88.62],
        'CIFAR-100': [29.64, 50.39, 57.01, 61.81],
        'STL-10': [67.65, 94.75, 95.67, 97.27],
        'FashionMNIST': [53.01, 70.59, 73.79, 75.20]
    }
    
    # Deployed Predictions Accuracy data
    deployed_accuracy_data = {
        'CIFAR-10': [76.40, 91.22, 94.08, 95.07],
        'CIFAR-100': [62.21, 74.55, 78.12, 78.91],
        'STL-10': [80.14, 98.70, 99.43, 99.71],
        'FashionMNIST': [70.04, 84.10, 84.77, 84.01]
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Base Model Accuracy
    for dataset in datasets:
        ax1.plot(k_values, accuracy_data[dataset], marker='o', linewidth=2, 
                label=dataset, markersize=8)
    
    ax1.set_xlabel('k-shot (Memory Size)')
    ax1.set_ylabel('Base Model Accuracy (%)')
    ax1.set_title('Impact of Memory Size on Base Model Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Deployed Predictions Accuracy
    for dataset in datasets:
        ax2.plot(k_values, deployed_accuracy_data[dataset], marker='s', linewidth=2, 
                label=dataset, markersize=8)
    
    ax2.set_xlabel('k-shot (Memory Size)')
    ax2.set_ylabel('Deployed Predictions Accuracy (%)')
    ax2.set_title('Impact of Memory Size on Agent-Filtered Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    save_figure(fig, "kshot_analysis")
    plt.close()

def create_calibration_comparison():
    """Table VI: Comparison of Calibration Methods"""
    datasets = ['CIFAR-10', 'FashionMNIST', 'STL-10']
    methods = ['Temperature', 'Isotonic', 'Platt', 'None']
    
    # False Positive Rate data (lower is better)
    fpr_data = {
        'CIFAR-10': [25.02, 27.20, 28.76, 26.70],
        'FashionMNIST': [30.53, 26.91, 30.47, 30.47],
        'STL-10': [10.73, 16.39, 13.45, 13.45]
    }
    
    # True Negative Rate data (higher is better)
    tnr_data = {
        'CIFAR-10': [74.98, 76.51, 78.24, 73.24],
        'FashionMNIST': [69.47, 67.29, 69.53, 69.53],
        'STL-10': [81.27, 83.61, 80.95, 80.95]
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # FPR Comparison (lower is better)
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, method in enumerate(methods):
        fpr_values = [fpr_data[dataset][i] for dataset in datasets]
        ax1.bar(x + i*width - 1.5*width, fpr_values, width, label=method, alpha=0.8)
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('False Positive Rate (%)')
    ax1.set_title('Agent Safety: False Positive Rate by Calibration Method\n(Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # TNR Comparison (higher is better)
    for i, method in enumerate(methods):
        tnr_values = [tnr_data[dataset][i] for dataset in datasets]
        ax2.bar(x + i*width - 1.5*width, tnr_values, width, label=method, alpha=0.8)
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('True Negative Rate (%)')
    ax2.set_title('Agent Effectiveness: True Negative Rate by Calibration Method\n(Higher is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "calibration_comparison")
    plt.close()

def create_continual_learning_detailed():
    """Table VII: Cross-Dataset Continual Learning Analysis"""
    datasets = ['CIFAR-10', 'CIFAR-100', 'STL-10', 'FashionMNIST']
    
    # Average Accuracy data
    zero_shot = [88.80, 61.71, 97.36, 61.49]
    one_shot = [71.14, 40.86, 88.72, 61.42]
    five_shot = [90.78, 63.31, 98.21, 79.97]
    
    # BWT data
    bwt_one_shot = [-24.25, -11.99, -6.06, -18.01]
    bwt_five_shot = [-11.40, -11.74, -2.33, -14.19]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average Accuracy Comparison
    x = np.arange(len(datasets))
    width = 0.25
    
    ax1.bar(x - width, zero_shot, width, label='Zero-Shot Baseline', alpha=0.8, color='lightblue')
    ax1.bar(x, one_shot, width, label='1-Shot CALM', alpha=0.8, color='orange')
    ax1.bar(x + width, five_shot, width, label='5-Shot CALM', alpha=0.8, color='green')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Average Accuracy (%)')
    ax1.set_title('Continual Learning: Average Accuracy Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # BWT Comparison
    ax2.bar(x - width/2, bwt_one_shot, width, label='1-Shot CALM', alpha=0.8, color='orange')
    ax2.bar(x + width/2, bwt_five_shot, width, label='5-Shot CALM', alpha=0.8, color='green')
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Backward Transfer (%)')
    ax2.set_title('Continual Learning: Forgetting Resistance\n(Closer to 0 = Less Forgetting)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig, "continual_learning_detailed")
    plt.close()

def create_safety_autonomy_tradeoff():
    """Simulated Safety-Autonomy Trade-off Curve"""
    # Simulate confidence thresholds and corresponding metrics
    thresholds = np.linspace(0.1, 0.9, 20)
    
    # Simulate realistic curves based on paper metrics
    deployment_rates = 100 * (1 - thresholds)**1.5  # Decreasing with threshold
    fpr_rates = 50 * thresholds**0.5 * (1 - thresholds)  # Bell-shaped, peaks around 0.5
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot deployment rate
    color1 = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Deployment Rate (%)', color=color1, fontsize=12)
    line1 = ax1.plot(thresholds, deployment_rates, color=color1, linewidth=3, 
                     label='Deployment Rate (Autonomy)', marker='o', markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 110)
    
    # Plot FPR on secondary axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('False Positive Rate (%)', color=color2, fontsize=12)
    line2 = ax2.plot(thresholds, fpr_rates, color=color2, linewidth=3, 
                     label='False Positive Rate (Safety Risk)', marker='s', markersize=6)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 25)
    
    # Add optimal point annotation with better positioning
    optimal_idx = np.argmin(fpr_rates + (100 - deployment_rates) * 0.3)
    optimal_x = thresholds[optimal_idx]
    optimal_y = deployment_rates[optimal_idx]
    
    # Position annotation to avoid overlap
    if optimal_x < 0.5:
        text_x = optimal_x + 0.15
        text_y = optimal_y - 15
    else:
        text_x = optimal_x - 0.15
        text_y = optimal_y + 15
    
    ax1.annotate(f'Optimal Balance\n(δ = {optimal_x:.2f})', 
                xy=(optimal_x, optimal_y),
                xytext=(text_x, text_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Set title with proper spacing
    fig.suptitle('Readiness Agent: Safety-Autonomy Trade-off', fontsize=14, y=0.95)
    
    # Create combined legend with better positioning
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.98, 0.85))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    save_figure(fig, "safety_autonomy_tradeoff")
    plt.close()

def create_error_rate_comparison():
    """Error Rate Improvement Visualization"""
    scenarios = ['Base Model\n(Naive Deployment)', 'CALM Framework\n(Agent Filtered)']
    error_rates = [3.12, 0.75]
    accuracies = [96.88, 99.25]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Error Rate Comparison
    colors = ['red', 'green']
    bars1 = ax1.bar(scenarios, error_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, rate in zip(bars1, error_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_ylabel('Error Rate (%)')
    ax1.set_title('Error Rate Comparison\n(Lower is Better)')
    ax1.set_ylim(0, 4)
    ax1.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = ((error_rates[0] - error_rates[1]) / error_rates[0]) * 100
    ax1.annotate(f'{improvement:.0f}% Reduction', 
                xy=(0.5, max(error_rates) + 0.3), ha='center',
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Accuracy Comparison
    bars2 = ax2.bar(scenarios, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Comparison\n(Higher is Better)')
    ax2.set_ylim(95, 100)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "error_rate_comparison")
    plt.close()

def create_imagenet_metrics_summary():
    """ImageNet Results Summary Dashboard - Focused on Metrics Only"""
    fig = plt.figure(figsize=(16, 10))
    
    # Create a 2x2 grid with better spacing
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, 
                         top=0.88, bottom=0.12, left=0.08, right=0.92)
    
    # 1. Base Performance Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Base\nAccuracy', 'Deployed\nAccuracy', 'Deployment\nRate']
    values = [96.88, 99.25, 84.86]
    colors = ['lightblue', 'green', 'orange']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Core Performance Metrics', fontsize=14, pad=15)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)
    
    # 2. Agent Decision Quality
    ax2 = fig.add_subplot(gs[0, 1])
    decisions = ['False\nPositive\nRate', 'True\nNegative\nRate', 'Decision\nAccuracy']
    decision_values = [12.01, 87.99, 86.72]
    decision_colors = ['red', 'green', 'blue']
    
    bars2 = ax2.bar(decisions, decision_values, color=decision_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    for bar, val in zip(bars2, decision_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Agent Decision Quality', fontsize=14, pad=15)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. Error Reduction Impact with Improvement Arrow
    ax3 = fig.add_subplot(gs[1, 0])
    error_scenarios = ['Before\nAgent', 'After\nAgent']
    error_values = [3.12, 0.75]
    error_colors = ['red', 'green']
    
    bars3 = ax3.bar(error_scenarios, error_values, color=error_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    for bar, val in zip(bars3, error_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add improvement annotation
    ax3.annotate('76% Reduction', xy=(0.5, 2), ha='center', va='center',
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    ax3.set_ylabel('Error Rate (%)', fontsize=12)
    ax3.set_title('Error Rate Reduction Impact', fontsize=14, pad=15)
    ax3.set_ylim(0, 4)
    ax3.grid(True, alpha=0.3)
    
    # 4. Comprehensive Results Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create a comprehensive summary table
    summary_data = [
        ['Metric', 'Value', 'Achievement'],
        ['Few-Shot Accuracy', '96.88%', 'State-of-the-art'],
        ['Agent-Filtered Accuracy', '99.25%', 'Near-perfect'],
        ['Error Rate Reduction', '76%', 'Exceptional'],
        ['Continual Learning BWT', '-5.52%', 'Strong retention'],
        ['Deployment Rate', '84.86%', 'High autonomy'],
        ['False Positive Rate', '12.01%', 'Good safety'],
        ['True Negative Rate', '87.99%', 'Reliable filtering']
    ]
    
    # Create and style table
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style the table headers
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the data rows with alternating colors
    for i in range(1, len(summary_data)):
        color = '#E3F2FD' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(summary_data[0])):
            table[(i, j)].set_facecolor(color)
            if j == 1:  # Highlight values
                table[(i, j)].set_text_props(weight='bold')
    
    ax4.set_title('CALM Framework: Complete Results Summary', fontsize=14, pad=20)
    
    # Main title
    fig.suptitle('CALM Framework: ImageNet Evaluation Results', 
                fontsize=18, fontweight='bold', y=0.95)
    
    save_figure(fig, "imagenet_metrics_summary")
    plt.close()

def create_calm_architecture_diagram():
    """CALM Framework Architecture Flow - Optimized Layout"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Define main pipeline with tighter spacing
    step_width = 0.09
    step_height = 0.12
    y_main = 0.65
    x_start = 0.02
    x_gap = 0.01
    
    main_steps = [
        {"name": "Input\nImage", "color": "#FFE0B2"},
        {"name": "Frozen VLM\n(CLIP)", "color": "#E1F5FE"},
        {"name": "Feature\nEmbedding", "color": "#F3E5F5"},
        {"name": "Episodic\nMemory", "color": "#E8F5E8"},
        {"name": "Prototypical\nClassifier", "color": "#FFF3E0"},
        {"name": "Confidence\nCalibration", "color": "#FCE4EC"},
        {"name": "Readiness\nAgent", "color": "#F1F8E9"}
    ]
    
    step_positions = []
    
    # Draw main pipeline
    for i, step in enumerate(main_steps):
        x_pos = x_start + i * (step_width + x_gap)
        pos = (x_pos, y_main)
        step_positions.append(pos)
        
        # Draw rectangle
        rect = plt.Rectangle(pos, step_width, step_height, 
                           facecolor=step["color"], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        # Add text
        text_x = x_pos + step_width/2
        text_y = y_main + step_height/2
        ax.text(text_x, text_y, step["name"], ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Add arrow to next step
        if i < len(main_steps) - 1:
            arrow_start_x = x_pos + step_width + 0.002
            arrow_end_x = x_start + (i+1) * (step_width + x_gap) - 0.002
            arrow_y = y_main + step_height/2
            
            ax.annotate('', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    
    # Agent decision branches - more compact
    agent_x = step_positions[-1][0] + step_width/2
    agent_y = step_positions[-1][1]
    
    # Deploy branch (upper right)
    deploy_x = 0.78
    deploy_y = 0.4
    deploy_w = 0.18
    deploy_h = 0.1
    
    deploy_rect = plt.Rectangle((deploy_x, deploy_y), deploy_w, deploy_h, 
                              facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(deploy_rect)
    ax.text(deploy_x + deploy_w/2, deploy_y + deploy_h/2, 
           'DEPLOY Prediction\n(Confidence ≥ δ)', ha='center', va='center', 
           fontsize=10, fontweight='bold', color='darkgreen')
    
    # Request feedback branch (lower right)
    request_x = 0.78
    request_y = 0.2
    request_w = 0.18
    request_h = 0.1
    
    request_rect = plt.Rectangle((request_x, request_y), request_w, request_h, 
                               facecolor='#FFCDD2', edgecolor='red', linewidth=2)
    ax.add_patch(request_rect)
    ax.text(request_x + request_w/2, request_y + request_h/2, 
           'REQUEST Feedback\n(Confidence < δ)', ha='center', va='center', 
           fontsize=10, fontweight='bold', color='darkred')
    
    # Clean arrows from agent
    ax.annotate('', xy=(deploy_x, deploy_y + deploy_h/2), 
               xytext=(agent_x, agent_y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
    
    ax.annotate('', xy=(request_x, request_y + request_h/2), 
               xytext=(agent_x, agent_y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))
    
    # Human expert box - positioned to avoid overlap
    human_x = 0.55
    human_y = 0.05
    human_w = 0.15
    human_h = 0.08
    
    human_rect = plt.Rectangle((human_x, human_y), human_w, human_h, 
                              facecolor='#FFF9C4', edgecolor='orange', linewidth=2)
    ax.add_patch(human_rect)
    ax.text(human_x + human_w/2, human_y + human_h/2, 
           'Human Expert\nCorrection', ha='center', va='center', 
           fontsize=9, fontweight='bold', color='darkorange')
    
    # Arrow from request to human
    ax.annotate('', xy=(human_x + human_w/2, human_y + human_h), 
               xytext=(request_x + request_w/4, request_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    # Memory update arrow - cleaner path
    memory_x = step_positions[3][0] + step_width/2
    memory_y = step_positions[3][1]
    human_center_x = human_x + human_w/2
    human_center_y = human_y + human_h
    
    ax.annotate('', xy=(memory_x, memory_y), xytext=(human_center_x, human_center_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple', 
                             connectionstyle="arc3,rad=0.2"))
    
    # Add labels for decision paths
    ax.text(agent_x + 0.05, (agent_y + deploy_y + deploy_h/2)/2, 
           'High\nConf.', fontsize=9, color='green', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.1", facecolor="lightgreen", alpha=0.7))
    
    ax.text(agent_x + 0.05, (agent_y + request_y + request_h/2)/2, 
           'Low\nConf.', fontsize=9, color='red', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.1", facecolor="lightcoral", alpha=0.7))
    
    ax.text(0.35, 0.35, 'Memory\nUpdate', fontsize=9, color='purple', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.1", facecolor="plum", alpha=0.7))
    
    # Title and key features
    ax.text(0.5, 0.92, 'CALM Framework: System Architecture', ha='center', va='center', 
           fontsize=16, fontweight='bold')
    
    # Key features in a compact box
    features_text = 'Key Features: Non-parametric • Frozen Backbone • Self-Aware Agent • Human-in-Loop'
    ax.text(0.5, 0.02, features_text, ha='center', va='center', fontsize=10, style='italic',
           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.6))
    
    save_figure(fig, "calm_architecture_diagram")
    plt.close()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
from matplotlib.lines import Line2D
import numpy as np

INK = "#1E293B"
SLATE = "#94A3B8"
GREEN = "#2E7D32"
AMBER = "#F9A825"
PURPLE = "#7E57C2"
BG = "#FFFFFF"

FILL_INPUT = "#E3F2FD"
FILL_CLIP = "#E0F7FA"
FILL_EMB = "#EDE7F6"
FILL_MEM = "#F3E5F5"
FILL_PROTO = "#FFF3E0"
FILL_AGENT = "#E8F5E9"

def rounded(ax, x, y, w, h, text, fc, ec=INK, lw=1.8, fontsize=11, weight='bold', r=0.03):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={r}",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=INK)
    return box

def pill(ax, x, y, w, h, text, fc, ec, lw=1.8, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=INK)
    return box

def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.5, style='-|>', ls='solid', alpha=1.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, linestyle=ls, alpha=alpha))

def small_axes(ax, x, y, w, h):
    return ax.inset_axes([x, y, w, h])

def draw_prototype_inset(a):
    a.set_facecolor("#FFFFFF")
    a.set_xticks([]); a.set_yticks([])
    # random cluster
    np.random.seed(2)
    pts = np.r_[np.random.randn(20,2)*0.12 + [0.3,0.5],
                np.random.randn(20,2)*0.10 + [0.7,0.6],
                np.random.randn(20,2)*0.09  + [0.55,0.25]]
    a.scatter(pts[:,0], pts[:,1], s=10, c="#CFD8DC", edgecolor="none", alpha=0.8)
    # centroid (Cat)
    centroid = np.array([0.72, 0.62])
    a.scatter([centroid[0]],[centroid[1]], s=120, c=PURPLE, edgecolor=INK, linewidth=1.2, zorder=3)
    a.text(centroid[0]+0.02, centroid[1]+0.02, "Prototype: Cat", fontsize=8, color=INK)
    # query
    q = np.array([0.56, 0.58])
    a.scatter([q[0]],[q[1]], s=80, facecolors="none", edgecolors=INK, linewidths=1.4)
    # cosine arrow (bold)
    a.annotate('', xy=(centroid[0], centroid[1]), xytext=(q[0], q[1]),
               arrowprops=dict(arrowstyle='-|>', lw=2.5, color=PURPLE))
    a.set_title("Prototype match (cosine)", fontsize=9, pad=2)

def draw_calibration_inset(a):
    a.set_facecolor("#FFFFFF")
    a.set_xlim(0,1); a.set_ylim(0,1)
    a.set_xticks([]); a.set_yticks([])
    # y=x
    a.plot([0,1],[0,1], color=SLATE, lw=1.2)
    x = np.linspace(0.02, 0.98, 40)
    # before calibration (overconfident)
    y1 = x**0.8 - 0.03
    y1 = np.clip(y1, 0, 1)
    a.plot(x, y1, color=AMBER, lw=2.0, linestyle='--', label='uncalibrated')
    # after calibration (closer to diagonal)
    y2 = x**0.93 - 0.01
    y2 = np.clip(y2, 0, 1)
    a.plot(x, y2, color=INK, lw=2.5, label='T-scaled')
    a.text(0.05, 0.90, "Calibrated (T-scaled)", fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=7, loc='lower right')

def draw_bwt_inset(a):
    a.set_facecolor("#FFFFFF")
    methods = ["Fine-tuning", "iCaRL", "CALM"]
    vals = [-65.0, -11.0, -5.52]
    colors = [SLATE, SLATE, PURPLE]
    x = np.arange(len(vals))
    a.bar(x, vals, color=colors, edgecolor=INK, linewidth=1.0)
    a.set_xticks(x, methods, rotation=20, fontsize=8)
    a.axhline(0, color=INK, lw=1)
    a.set_ylim(-70, 5)
    for xi, v in zip(x, vals):
        a.text(xi, v-3 if v<0 else v+1, f"{v:.2f}", ha='center', va='top' if v<0 else 'bottom', fontsize=8)
    a.set_title("Forgetting (BWT, ↓ worse)", fontsize=9, pad=2)
    a.set_yticks([])

def create_calm_fig1(save_prefix="calm_fig1"):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(BG)

    # Layout params (normalized)
    y_center = 0.55
    block_w, block_h = 0.15, 0.14
    xs = [0.08, 0.25, 0.42, 0.59, 0.76, 0.89]  # centers
    def place(cx): return cx - block_w/2, y_center - block_h/2

    # Blocks
    b1 = rounded(ax, *place(xs[0]), block_w, block_h, "Input Image(s)", FILL_INPUT)
    b2 = rounded(ax, *place(xs[1]), block_w, block_h, "Frozen VLM\n(CLIP)\nFrozen θ", FILL_CLIP)
    b3 = rounded(ax, *place(xs[2]), block_w, block_h, "Embedding\n(d-dim)", FILL_EMB)
    b4 = rounded(ax, *place(xs[3]), block_w, block_h, "Episodic Memory\n(support)", FILL_MEM)
    b5 = rounded(ax, *place(xs[4]), block_w, block_h, "Prototypical\nClassifier", FILL_PROTO)
    b6 = rounded(ax, *place(xs[5]), block_w, block_h, "Readiness Agent\nq = max softmax(cos/T)", FILL_AGENT)

    # Forward arrows
    for i in range(len(xs)-1):
        x0 = xs[i] + block_w/2 - block_w/2
        x1 = xs[i+1] - block_w/2 + block_w/2
        arrow(ax, xs[i]+block_w/2*0.65, y_center, xs[i+1]-block_w/2*0.65, y_center, color=INK, lw=3)

    # Cosine rays (from Embedding to Memory)
    mem_x, mem_y = xs[3], y_center
    emb_x = xs[2] + block_w/2*0.75
    for dy, alpha in [(0.06, 0.8), (-0.05, 0.7), (0.0, 1.0)]:
        color = PURPLE if abs(dy) < 0.01 else SLATE
        lw = 2.8 if color == PURPLE else 1.5
        ls = 'solid' if color == PURPLE else 'dashed'
        arrow(ax, xs[2]+block_w*0.35, y_center, xs[3]-block_w*0.35, y_center+dy, color=color, lw=lw, ls=ls)

    # Readiness outputs (deploy & feedback)
    deploy = pill(ax, 0.84, 0.70, 0.12, 0.06, "Deploy (q ≥ δ) ✓", "#C8E6C9", GREEN, lw=2.0)
    feedback = pill(ax, 0.84, 0.33, 0.17, 0.06, "Request Feedback (q < δ) ?", "#FFE082", AMBER, lw=2.0)

    # Arrows from agent to outputs
    ax.annotate('', xy=(0.84, 0.70+0.03), xytext=(xs[5]+block_w/2, y_center+block_h*0.25),
                arrowprops=dict(arrowstyle='-|>', lw=3, color=GREEN))
    ax.annotate('', xy=(0.84, 0.33+0.03), xytext=(xs[5]+block_w/2, y_center-block_h*0.25),
                arrowprops=dict(arrowstyle='-|>', lw=3, color=AMBER))

    # Human icon (simple pill) & feedback loop
    human = pill(ax, 0.73, 0.18, 0.14, 0.06, "Human (labeling)", "#FFF9C4", AMBER, lw=2.0)
    # Arrow from feedback to human
    arrow(ax, 0.84+0.02, 0.33, 0.80, 0.24, color=AMBER, lw=2.5, ls='--')

    # Curved return to memory (purple)
    start = (0.80, 0.24)
    end = (xs[3]-block_w/2, y_center-0.02)
    con = ConnectionPatch(start, end, "data", "data",
                          arrowstyle='-|>', lw=2.5, linestyle='--', color=PURPLE, connectionstyle="arc3,rad=0.3")
    ax.add_artist(con)
    ax.text(0.70, 0.28, "Human correction → Memory update (Δcₖ)", fontsize=10, color=PURPLE)

    # Micro-insets
    # A: Prototype view (above classifier)
    aA = small_axes(ax, 0.71, 0.78, 0.20, 0.17)
    draw_prototype_inset(aA)
    # B: Calibration (above readiness agent)
    aB = small_axes(ax, 0.86, 0.78, 0.12, 0.17)
    draw_calibration_inset(aB)
    # C: BWT bars (above memory)
    aC = small_axes(ax, 0.58, 0.78, 0.20, 0.17)
    draw_bwt_inset(aC)

    # Title (optional; omit if your IEEE template adds fig caption)
    ax.text(0.02, 0.95, "CALM: Pipeline + Micro-Insets", fontsize=16, fontweight='bold', color=INK, ha='left')

    # Save
    fig.savefig(f"{save_prefix}.svg", bbox_inches='tight')
    fig.savefig(f"{save_prefix}.png", bbox_inches='tight')
    plt.close(fig)

# Example:
# create_calm_fig1("calm_fig1")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import numpy as np

# Colors
INK = "#1E293B"
SLATE = "#94A3B8"
GREEN = "#2E7D32"
AMBER = "#F9A825"
PURPLE = "#7E57C2"
BG = "#FFFFFF"

FILL_INPUT = "#E3F2FD"
FILL_CLIP = "#E0F7FA"
FILL_EMB = "#EDE7F6"
FILL_MEM = "#F3E5F5"
FILL_AGENT = "#E8F5E9"

def rounded(ax, x, y, w, h, text, fc, ec=INK, lw=1.8, fontsize=11, weight='bold', r=0.03):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={r}",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=INK)
    return box

def pill(ax, x, y, w, h, text, fc, ec, lw=1.8, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=INK)
    return box

def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.8, ls='solid', style='-|>', alpha=1.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, linestyle=ls, alpha=alpha))

def band(ax, x, y, w, h, r=0.02, fc="#F8FAFC"):
    panel = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                           facecolor=fc, edgecolor=INK, linewidth=0.8, alpha=0.35)
    ax.add_patch(panel)
    return panel

def create_calm_ecosystem_fig(save_prefix="calm_ecosystem"):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(BG)

    # --- Bands
    band(ax, 0.035, 0.70, 0.93, 0.22)  # Top: Perception
    band(ax, 0.035, 0.44, 0.93, 0.24)  # Middle: Memory
    band(ax, 0.035, 0.14, 0.93, 0.26)  # Bottom: Decision

    # --- Top band: inputs -> CLIP -> embedding
    # Input thumbnails (as pills)
    in1 = pill(ax, 0.06, 0.77, 0.07, 0.07, "cat", "#FFFFFF", INK, lw=1.2, fontsize=9)
    in2 = pill(ax, 0.06, 0.70, 0.07, 0.07, "shoe", "#FFFFFF", INK, lw=1.2, fontsize=9)
    in3 = pill(ax, 0.06, 0.84, 0.07, 0.07, "plane", "#FFFFFF", INK, lw=1.2, fontsize=9)

    clip = rounded(ax, 0.20, 0.73, 0.18, 0.12, "Frozen VLM\n(CLIP)\nFrozen θ", FILL_CLIP)
    arrow(ax, 0.13, 0.80, 0.20, 0.79)  # inputs -> CLIP (top)
    arrow(ax, 0.13, 0.74, 0.20, 0.79)  # mid
    arrow(ax, 0.13, 0.88, 0.20, 0.79)  # high

    emb = rounded(ax, 0.44, 0.75, 0.20, 0.10, "Embedding (d-dim)", FILL_EMB)
    arrow(ax, 0.38, 0.79, 0.44, 0.79)

    # Embedding glyph (little ticks bar)
    ticks_y = 0.79
    for i in range(12):
        ax.plot([0.47 + i*0.01, 0.47 + i*0.01], [ticks_y-0.01, ticks_y+0.01], color=INK, lw=1)

    # --- Middle band: Episodic Memory field + prototypes + cosine rays
    mem_panel = rounded(ax, 0.12, 0.47, 0.56, 0.18, "Episodic Memory (support)", FILL_MEM, lw=1.4, fontsize=11)
    # Point cloud
    np.random.seed(3)
    pts = np.r_[np.random.randn(40,2)*0.015 + [0.26,0.56],
                np.random.randn(40,2)*0.02  + [0.36,0.54],
                np.random.randn(40,2)*0.018 + [0.54,0.58]]
    ax.scatter(pts[:,0], pts[:,1], s=10, c="#CFD8DC", edgecolor="none", alpha=0.9)

    # Prototypes
    proto_cat  = (0.56, 0.585); ax.scatter(*proto_cat, s=200, c=PURPLE, edgecolor=INK, linewidth=1.2, zorder=3)
    ax.text(proto_cat[0]+0.015, proto_cat[1]+0.012, "Prototype: Cat", fontsize=10, color=INK)
    proto_dog  = (0.36, 0.545); ax.scatter(*proto_dog, s=160, c="#B39DDB", edgecolor=INK, linewidth=1.0, zorder=3)
    ax.text(proto_dog[0]-0.08, proto_dog[1]-0.035, "Prototype: Dog", fontsize=9, color=INK)
    proto_car  = (0.26, 0.565); ax.scatter(*proto_car, s=160, c="#B39DDB", edgecolor=INK, linewidth=1.0, zorder=3)
    ax.text(proto_car[0]-0.07, proto_car[1]+0.022, "Prototype: Car", fontsize=9, color=INK)

    # Cosine rays from embedding (bottom of emb block) to prototypes
    emb_emit = (0.54, 0.75)
    # winning (to Cat)
    arrow(ax, emb_emit[0], emb_emit[1]-0.01, proto_cat[0], proto_cat[1]+0.01, color=PURPLE, lw=3.0)
    ax.text((emb_emit[0]+proto_cat[0])/2+0.005, (emb_emit[1]+proto_cat[1])/2+0.01, "cosine ↑", fontsize=9, color=PURPLE)
    # non-winning
    arrow(ax, emb_emit[0], emb_emit[1]-0.01, proto_dog[0]+0.005, proto_dog[1]+0.005, color=SLATE, lw=1.6, ls='--')
    arrow(ax, emb_emit[0], emb_emit[1]-0.01, proto_car[0]+0.005, proto_car[1]+0.005, color=SLATE, lw=1.6, ls='--')

    # --- Bottom band: Readiness gate + outputs + human loop
    gate = rounded(ax, 0.24, 0.19, 0.22, 0.12, "Readiness Agent\nq = max softmax(cos/T)", FILL_AGENT)
    # pipe from memory down to gate
    arrow(ax, 0.40, 0.47, 0.35, 0.31)  # memory center -> gate

    deploy = pill(ax, 0.58, 0.28, 0.18, 0.08, "Deploy (q ≥ δ) ✓", "#C8E6C9", GREEN, lw=2.0)
    feedback = pill(ax, 0.58, 0.18, 0.22, 0.08, "Request Feedback (q < δ) ?", "#FFE082", AMBER, lw=2.0)

    arrow(ax, 0.46, 0.25, 0.58, 0.32, color=GREEN, lw=3.2)   # gate -> deploy
    arrow(ax, 0.46, 0.23, 0.58, 0.22, color=AMBER, lw=3.2)   # gate -> feedback

    human = pill(ax, 0.83, 0.16, 0.15, 0.07, "Human (labeling)", "#FFF9C4", AMBER, lw=2.0)
    # feedback -> human
    arrow(ax, 0.80, 0.22, 0.83, 0.19, color=AMBER, lw=2.6, ls='--')

    # human -> memory (curved, purple)
    con = ConnectionPatch((0.905, 0.195), (0.56, 0.50), "data", "data",
                          arrowstyle='-|>', lw=2.8, linestyle='--', color=PURPLE, connectionstyle="arc3,rad=0.35")
    ax.add_artist(con)
    ax.text(0.69, 0.37, "Human correction → Memory update (Δcₖ)", fontsize=10, color=PURPLE)

    # Optional title (omit if using IEEE caption)
    ax.text(0.04, 0.93, "CALM Ecosystem: Perception • Memory • Readiness", fontsize=16, fontweight='bold', color=INK)

    fig.savefig(f"{save_prefix}.svg", bbox_inches='tight')
    fig.savefig(f"{save_prefix}.png", bbox_inches='tight')
    plt.close(fig)

# Example:
# create_calm_ecosystem_fig("calm_ecosystem")


def create_all_visualizations():
    """Generate all visualizations for the CALM paper"""
    print("Generating CALM paper visualizations...")
    print("=" * 50)
    
    create_few_shot_comparison()
    create_continual_learning_comparison()
    create_kshot_analysis()
    create_calibration_comparison()
    create_continual_learning_detailed()
    create_safety_autonomy_tradeoff()
    create_error_rate_comparison()
    create_imagenet_metrics_summary()
    create_calm_architecture_diagram()
    create_calm_fig1("calm_fig1")
    create_calm_ecosystem_fig("calm_ecosystem")
    
    print("=" * 50)
    print(f"All visualizations saved to: {output_dir.absolute()}")
    print("Generated figures:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    create_all_visualizations()