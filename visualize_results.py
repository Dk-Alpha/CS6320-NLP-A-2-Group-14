
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# LEARNING CURVES - Training Loss and Validation Accuracy

def plot_learning_curves(stats_file, save_name=None):
    """
    Plot training loss and validation accuracy over epochs
    This is for the BONUS section in the report (1pt)
    """
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Training Loss
    ax1.plot(stats['epochs'], stats['train_loss'], 'b-o', linewidth=2, markersize=6, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{stats["model_type"]} Training Loss\n(Hidden Dim: {stats["hidden_dim"]})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Training vs Validation Accuracy
    ax2.plot(stats['epochs'], stats['train_accuracy'], 'g-o', linewidth=2, markersize=6, label='Training Accuracy')
    ax2.plot(stats['epochs'], stats['val_accuracy'], 'r-s', linewidth=2, markersize=6, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title(f'{stats["model_type"]} Accuracy\n(Hidden Dim: {stats["hidden_dim"]})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Saved learning curves to {save_name}")
    
    plt.show()
    
    # Print final statistics
    print(f"\n {stats['model_type']} Final Results:")
    print(f"   Final Training Loss: {stats['train_loss'][-1]:.4f}")
    print(f"   Final Training Accuracy: {stats['train_accuracy'][-1]:.4f}")
    print(f"   Final Validation Accuracy: {stats['val_accuracy'][-1]:.4f}")
    print(f"   Best Validation Accuracy: {max(stats['val_accuracy']):.4f}")


# COMPARE FFNN vs RNN


def compare_models(ffnn_stats_file, rnn_stats_file, save_name=None):
    """
    Compare FFNN and RNN performance side by side
    """
    with open(ffnn_stats_file, 'rb') as f:
        ffnn_stats = pickle.load(f)
    with open(rnn_stats_file, 'rb') as f:
        rnn_stats = pickle.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Training Loss Comparison
    ax1 = axes[0, 0]
    ax1.plot(ffnn_stats['epochs'], ffnn_stats['train_loss'], 'b-o', linewidth=2, label='FFNN', markersize=5)
    ax1.plot(rnn_stats['epochs'], rnn_stats['train_loss'], 'r-s', linewidth=2, label='RNN', markersize=5)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy Comparison
    ax2 = axes[0, 1]
    ax2.plot(ffnn_stats['epochs'], ffnn_stats['val_accuracy'], 'b-o', linewidth=2, label='FFNN', markersize=5)
    ax2.plot(rnn_stats['epochs'], rnn_stats['val_accuracy'], 'r-s', linewidth=2, label='RNN', markersize=5)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Training Accuracy Comparison
    ax3 = axes[1, 0]
    ax3.plot(ffnn_stats['epochs'], ffnn_stats['train_accuracy'], 'b-o', linewidth=2, label='FFNN', markersize=5)
    ax3.plot(rnn_stats['epochs'], rnn_stats['train_accuracy'], 'r-s', linewidth=2, label='RNN', markersize=5)
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Training Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Training Accuracy Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Bar Chart of Final Performance
    ax4 = axes[1, 1]
    models = ['FFNN', 'RNN']
    train_acc = [ffnn_stats['train_accuracy'][-1], rnn_stats['train_accuracy'][-1]]
    val_acc = [ffnn_stats['val_accuracy'][-1], rnn_stats['val_accuracy'][-1]]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, train_acc, width, label='Training Accuracy', color='#3498db')
    bars2 = ax4.bar(x + width/2, val_acc, width, label='Validation Accuracy', color='#e74c3c')
    
    ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax4.set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend(fontsize=10)
    ax4.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f" Saved comparison plots to {save_name}")
    
    plt.show()
    
    # Print comparison table
    print("\nModel Comparison Summary:")
    print("=" * 70)
    print(f"{'Metric':<30} {'FFNN':<20} {'RNN':<20}")
    print("=" * 70)
    print(f"{'Hidden Dimension':<30} {ffnn_stats['hidden_dim']:<20} {rnn_stats['hidden_dim']:<20}")
    print(f"{'Epochs Trained':<30} {len(ffnn_stats['epochs']):<20} {len(rnn_stats['epochs']):<20}")
    print(f"{'Final Training Loss':<30} {ffnn_stats['train_loss'][-1]:<20.4f} {rnn_stats['train_loss'][-1]:<20.4f}")
    print(f"{'Final Training Accuracy':<30} {ffnn_stats['train_accuracy'][-1]:<20.4f} {rnn_stats['train_accuracy'][-1]:<20.4f}")
    print(f"{'Final Validation Accuracy':<30} {ffnn_stats['val_accuracy'][-1]:<20.4f} {rnn_stats['val_accuracy'][-1]:<20.4f}")
    print(f"{'Best Validation Accuracy':<30} {max(ffnn_stats['val_accuracy']):<20.4f} {max(rnn_stats['val_accuracy']):<20.4f}")
    print("=" * 70)


# PERFORMANCE ACROSS DIFFERENT HIDDEN DIMENSIONS


def compare_hyperparameters(stats_files_dict, save_name=None):
    """
    Compare models with different hidden dimensions
    stats_files_dict format: {'FFNN_h10': 'file1.pkl', 'FFNN_h50': 'file2.pkl', ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats_files_dict)))
    
    for (name, file), color in zip(stats_files_dict.items(), colors):
        with open(file, 'rb') as f:
            stats = pickle.load(f)
        
        # Plot validation accuracy
        axes[0].plot(stats['epochs'], stats['val_accuracy'], '-o', 
                    linewidth=2, label=name, color=color, markersize=5)
        
        # Plot training loss
        axes[1].plot(stats['epochs'], stats['train_loss'], '-o', 
                    linewidth=2, label=name, color=color, markersize=5)
    
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Validation Accuracy - Different Hyperparameters', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Training Loss - Different Hyperparameters', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f" Saved hyperparameter comparison to {save_name}")
    
    plt.show()


# CREATE CONFUSION MATRIX

def plot_confusion_matrix(confusion_matrix, model_name='Model', save_name=None):
    """
    Plot confusion matrix
    confusion_matrix should be a 5x5 numpy array
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize to show percentages
    cm_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
                yticklabels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
                cbar_kws={'label': 'Percentage (%)'},
                square=True)
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}\n(Values show percentage of predictions)', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f" Saved confusion matrix to {save_name}")
    
    plt.show()


# DATA STATISTICS VISUALIZATION


def plot_data_statistics(train_file, val_file, test_file=None, save_name=None):
    """
    Visualize dataset statistics
    """
    # Load data
    with open(train_file) as f:
        train_data = json.load(f)
    with open(val_file) as f:
        val_data = json.load(f)
    
    # Count distribution by stars
    train_dist = {i: 0 for i in range(1, 6)}
    val_dist = {i: 0 for i in range(1, 6)}
    
    for item in train_data:
        train_dist[item['stars']] += 1
    for item in val_data:
        val_dist[item['stars']] += 1
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Count by dataset
    datasets = ['Training', 'Validation']
    counts = [len(train_data), len(val_data)]
    bars = axes[0].bar(datasets, counts, color=['#3498db', '#e74c3c'], width=0.6)
    axes[0].set_ylabel('Number of Examples', fontsize=11, fontweight='bold')
    axes[0].set_title('Dataset Size', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Distribution by stars
    stars = list(range(1, 6))
    train_counts = [train_dist[i] for i in stars]
    val_counts = [val_dist[i] for i in stars]
    
    x = np.arange(len(stars))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, train_counts, width, label='Training', color='#3498db')
    bars2 = axes[1].bar(x + width/2, val_counts, width, label='Validation', color='#e74c3c')
    
    axes[1].set_xlabel('Star Rating', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution by Star Rating', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['1★', '2★', '3★', '4★', '5★'])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved data statistics to {save_name}")
    
    plt.show()
    
    # Print statistics table
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"Training examples: {len(train_data):,}")
    print(f"Validation examples: {len(val_data):,}")
    print(f"\nDistribution by stars:")
    print(f"{'Star Rating':<15} {'Training':<15} {'Validation':<15}")
    print("-" * 50)
    for i in range(1, 6):
        print(f"{i}★ {'':<12} {train_dist[i]:<15,} {val_dist[i]:<15,}")
    print("=" * 50)


# GENERATE ALL PLOTS AT ONCE

def generate_all_plots(ffnn_stats='ffnn_stats_h50_e10.pkl', 
                       rnn_stats='rnn_stats_h64_e6.pkl',
                       train_file='training.json',
                       val_file='validation.json'):
    """
    Generate all plots needed for the report
    """
    print("Generating all visualizations for report...")
    print("=" * 60)
    
    if not Path(ffnn_stats).exists():
        print(f" File not found: {ffnn_stats}")
        print("   Please train FFNN first using ffnn_with_logging.py")
        return
    
    if not Path(rnn_stats).exists():
        print(f"File not found: {rnn_stats}")
        print("   Please train RNN first using rnn_with_logging.py")
        return
    
    # 1. Data statistics
    print("\nGenerating data statistics...")
    plot_data_statistics(train_file, val_file, save_name='data_statistics.png')
    
    # 2. FFNN learning curves
    print("\nGenerating FFNN learning curves...")
    plot_learning_curves(ffnn_stats, save_name='ffnn_learning_curves.png')
    
    # 3. RNN learning curves
    print("\nGenerating RNN learning curves...")
    plot_learning_curves(rnn_stats, save_name='rnn_learning_curves.png')
    
    # 4. Model comparison
    print("\nGenerating model comparison...")
    compare_models(ffnn_stats, rnn_stats, save_name='model_comparison.png')
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("  - data_statistics.png")
    print("  - ffnn_learning_curves.png")
    print("  - rnn_learning_curves.png")
    print("  - model_comparison.png")
    
if __name__ == "__main__":
    print("\n generating all plots...")
    generate_all_plots()