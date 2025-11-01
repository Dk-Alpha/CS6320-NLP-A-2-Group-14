"""
Error Analysis Script for Report
Generates confusion matrix and finds error examples for analysis
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# MODEL DEFINITIONS (same as in training scripts)
# ============================================

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()

    def forward(self, input_vector):
        hidden_layer = self.activation(self.W1(input_vector))
        output_layer = self.W2(hidden_layer)
        predicted_vector = self.softmax(output_layer)
        return predicted_vector


class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, inputs):
        output, hidden = self.rnn(inputs)
        output = self.W(output)
        output = torch.sum(output, dim=0)
        predicted_vector = self.softmax(output)
        return predicted_vector

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_data(data_file):
    """Load data from JSON file"""
    with open(data_file) as f:
        data = json.load(f)
    return [(item["text"].split(), int(item["stars"]-1), item["text"]) for item in data]


def make_vocab(train_data):
    """Create vocabulary from training data"""
    vocab = set()
    for document, _, _ in train_data:
        for word in document:
            vocab.add(word)
    return vocab


def make_indices(vocab):
    """Create word-to-index mappings"""
    vocab_list = sorted(vocab)
    vocab_list.append('<UNK>')
    word2index = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
    vocab.add('<UNK>')
    return vocab, word2index


def convert_to_vector(document, word2index):
    """Convert document to bag-of-words vector"""
    vector = torch.zeros(len(word2index))
    for word in document:
        index = word2index.get(word, word2index['<UNK>'])
        vector[index] += 1
    return vector

# ============================================
# CONFUSION MATRIX GENERATION
# ============================================

def generate_confusion_matrix_ffnn(model_path, test_file, train_file, save_name=None):
    """Generate confusion matrix for FFNN model"""
    
    # Load model
    print("Loading FFNN model...")
    train_data = load_data(train_file)
    vocab = make_vocab(train_data)
    vocab, word2index = make_indices(vocab)
    
    model = FFNN(input_dim=len(vocab), h=50)  # Adjust hidden_dim if needed
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_data = load_data(test_file)
    
    # Generate predictions
    confusion_matrix = np.zeros((5, 5), dtype=int)
    errors = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for document, true_label, text in test_data:
            input_vector = convert_to_vector(document, word2index)
            output = model(input_vector)
            predicted_label = torch.argmax(output).item()
            
            confusion_matrix[true_label][predicted_label] += 1
            
            # Collect errors
            if predicted_label != true_label:
                errors.append({
                    'text': text,
                    'true_label': true_label + 1,  # Convert back to 1-5
                    'predicted_label': predicted_label + 1,
                    'confidence': torch.exp(output[predicted_label]).item()
                })
    
    # Plot confusion matrix
    plot_confusion_matrix_heatmap(confusion_matrix, 'FFNN', save_name)
    
    # Print statistics
    print_confusion_stats(confusion_matrix)
    
    return confusion_matrix, errors


def generate_confusion_matrix_rnn(model_path, test_file, word_embedding_path='word_embedding.pkl', save_name=None):
    """Generate confusion matrix for RNN model"""
    
    # Load model
    print("Loading RNN model...")
    model = RNN(input_dim=50, h=64)  # Adjust hidden_dim if needed
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load word embeddings
    print("Loading word embeddings...")
    word_embedding = pickle.load(open(word_embedding_path, 'rb'))
    
    # Load test data
    print("Loading test data...")
    test_data = load_data(test_file)
    
    # Generate predictions
    confusion_matrix = np.zeros((5, 5), dtype=int)
    errors = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for document, true_label, text in test_data:
            # Process text
            text_processed = " ".join(document)
            text_processed = text_processed.translate(text_processed.maketrans("", "", string.punctuation)).split()
            
            # Get embeddings
            vectors = [word_embedding[w.lower()] if w.lower() in word_embedding.keys() 
                      else word_embedding['unk'] for w in text_processed]
            
            if len(vectors) == 0:
                continue
            
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output).item()
            
            confusion_matrix[true_label][predicted_label] += 1
            
            # Collect errors
            if predicted_label != true_label:
                errors.append({
                    'text': text,
                    'true_label': true_label + 1,
                    'predicted_label': predicted_label + 1,
                    'confidence': torch.exp(output[0][predicted_label]).item()
                })
    
    # Plot confusion matrix
    plot_confusion_matrix_heatmap(confusion_matrix, 'RNN', save_name)
    
    # Print statistics
    print_confusion_stats(confusion_matrix)
    
    return confusion_matrix, errors

# ============================================
# VISUALIZATION
# ============================================

def plot_confusion_matrix_heatmap(confusion_matrix, model_name, save_name=None):
    """Plot confusion matrix as heatmap"""
    
    plt.figure(figsize=(10, 8))
    
    # Normalize to percentages
    cm_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(confusion_matrix, dtype=object)
    for i in range(5):
        for j in range(5):
            annotations[i, j] = f'{confusion_matrix[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Plot
    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                xticklabels=['1★', '2★', '3★', '4★', '5★'],
                yticklabels=['1★', '2★', '3★', '4★', '5★'],
                cbar_kws={'label': 'Percentage (%)'},
                square=True, linewidths=0.5, linecolor='gray')
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}\n(Count and Percentage)', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to {save_name}")
    
    plt.show()


def print_confusion_stats(confusion_matrix):
    """Print accuracy statistics from confusion matrix"""
    
    print("\n" + "=" * 60)
    print("📊 Confusion Matrix Statistics")
    print("=" * 60)
    
    total = confusion_matrix.sum()
    correct = np.trace(confusion_matrix)
    accuracy = correct / total
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    print("\nPer-Class Statistics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for i in range(5):
        # Precision: TP / (TP + FP)
        precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum() if confusion_matrix[:, i].sum() > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = confusion_matrix[i, i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0
        
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{i+1}★ {'':<7} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print("=" * 60)

# ============================================
# ERROR ANALYSIS
# ============================================

def analyze_errors(errors, num_examples=10):
    """Analyze and print error examples"""
    
    print("\n" + "=" * 80)
    print("🔍 ERROR ANALYSIS - Examples of Misclassifications")
    print("=" * 80)
    
    # Sort by confidence (most confident mistakes are interesting)
    errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)
    
    # Group errors by type
    error_types = defaultdict(list)
    for error in errors:
        key = f"{error['true_label']}→{error['predicted_label']}"
        error_types[key].append(error)
    
    print(f"\nTotal errors: {len(errors)}")
    print(f"Error types found: {len(error_types)}")
    
    print("\nMost common error types:")
    sorted_types = sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (error_type, examples) in enumerate(sorted_types[:5], 1):
        print(f"{i}. {error_type}: {len(examples)} instances")
    
    print(f"\n{'-'*80}")
    print(f"Top {num_examples} Most Confident Errors:")
    print(f"{'-'*80}\n")
    
    for i, error in enumerate(errors_sorted[:num_examples], 1):
        print(f"Error #{i}:")
        print(f"  Text: {error['text'][:200]}...")  # First 200 chars
        print(f"  True Label: {error['true_label']}★")
        print(f"  Predicted: {error['predicted_label']}★")
        print(f"  Confidence: {error['confidence']:.4f}")
        print(f"  Error Type: {error['true_label']}★ → {error['predicted_label']}★")
        print()
    
    print("=" * 80)
    
    return error_types


def suggest_improvements(errors, confusion_matrix):
    """Suggest improvements based on error analysis"""
    
    print("\n" + "=" * 80)
    print("💡 SUGGESTED IMPROVEMENTS")
    print("=" * 80)
    
    # Analyze confusion patterns
    print("\n1. Confusion Patterns:")
    for i in range(5):
        for j in range(5):
            if i != j and confusion_matrix[i, j] > 20:
                print(f"   - High confusion between {i+1}★ and {j+1}★ ({confusion_matrix[i, j]} cases)")
                print(f"     → Consider: Better features to distinguish these classes")
    
    # Analyze error distribution
    print("\n2. Class-Specific Issues:")
    for i in range(5):
        recall = confusion_matrix[i, i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0
        if recall < 0.5:
            print(f"   - {i+1}★ has low recall ({recall:.2f})")
            print(f"     → Consider: More training data or class weighting")
    
    print("\n3. General Recommendations:")
    print("   - Try different hidden dimensions: [32, 64, 128, 256]")
    print("   - Experiment with learning rates: [0.001, 0.01, 0.1]")
    print("   - For RNN: Try LSTM or GRU instead of vanilla RNN")
    print("   - Add dropout for regularization")
    print("   - Use pre-trained embeddings (GloVe, Word2Vec)")
    print("   - Ensemble FFNN and RNN predictions")
    
    print("=" * 80)

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run error analysis"""
    
    print("=" * 80)
    print("  ERROR ANALYSIS TOOL")
    print("=" * 80)
    
    # Example usage - adjust paths as needed
    
    # For FFNN
    print("\n1️⃣ Analyzing FFNN...")
    try:
        confusion_ffnn, errors_ffnn = generate_confusion_matrix_ffnn(
            model_path='ffnn_model_h50_e10.pt',
            test_file='validation.json',  # Use validation or test file
            train_file='training.json',
            save_name='ffnn_confusion_matrix.png'
        )
        error_types_ffnn = analyze_errors(errors_ffnn, num_examples=10)
        suggest_improvements(errors_ffnn, confusion_ffnn)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure you've trained the model and files exist")
    
    print("\n" + "="*80 + "\n")
    
    # For RNN
    print("2️⃣ Analyzing RNN...")
    try:
        confusion_rnn, errors_rnn = generate_confusion_matrix_rnn(
            model_path='rnn_model_h64_e10.pt',
            test_file='validation.json',  # Use validation or test file
            word_embedding_path='word_embedding.pkl',
            save_name='rnn_confusion_matrix.png'
        )
        error_types_rnn = analyze_errors(errors_rnn, num_examples=10)
        suggest_improvements(errors_rnn, confusion_rnn)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure you've trained the model and files exist")
    
    print("\n✅ Error analysis complete!")
    print("\n📝 Use these insights and images in your report's Analysis section")


if __name__ == "__main__":
    main()

# ============================================
# ALTERNATIVE: Quick confusion matrix only
# ============================================

"""
# If you just want confusion matrix without full analysis:

from error_analysis import generate_confusion_matrix_ffnn, generate_confusion_matrix_rnn

# FFNN
confusion_ffnn, _ = generate_confusion_matrix_ffnn(
    'ffnn_model_h50_e10.pt', 
    'validation.json', 
    'training.json',
    save_name='ffnn_cm.png'
)

# RNN  
confusion_rnn, _ = generate_confusion_matrix_rnn(
    'rnn_model_h64_e10.pt',
    'validation.json',
    save_name='rnn_cm.png'
)
"""