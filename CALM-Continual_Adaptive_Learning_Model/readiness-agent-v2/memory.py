# File: memory.py (ENHANCED VERSION)
# Enhanced episodic memory with forgetting mechanisms, quality control, and continual learning support

import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings

class EpisodicMemory:
    """
    Enhanced episodic memory for continual learning with intelligent forgetting,
    quality control, and dynamic memory management.
    
    This memory supports:
    - Prototype-based few-shot learning
    - Intelligent forgetting to prevent memory overflow
    - Quality-based example selection
    - Memory statistics and analysis
    - Support for continual learning scenarios
    """
    
    def __init__(self, max_examples_per_class=100, quality_threshold=0.1, 
                 enable_forgetting=True, forgetting_strategy="oldest"):
        """
        Initialize the enhanced episodic memory.
        
        Args:
            max_examples_per_class (int): Maximum number of examples to store per class
            quality_threshold (float): Minimum quality score for adding examples
            enable_forgetting (bool): Whether to enable intelligent forgetting
            forgetting_strategy (str): Strategy for forgetting ("oldest", "lowest_quality", "random")
        """
        # Core storage: class_id -> list of (embedding, metadata) tuples
        self.storage = defaultdict(list)
        
        # Memory management parameters
        self.max_examples_per_class = max_examples_per_class
        self.quality_threshold = quality_threshold
        self.enable_forgetting = enable_forgetting
        self.forgetting_strategy = forgetting_strategy
        
        # Statistics tracking
        self.total_added = 0
        self.total_rejected = 0
        self.total_forgotten = 0
        self.class_statistics = defaultdict(lambda: {
            'added': 0, 'rejected': 0, 'forgotten': 0, 'avg_quality': 0.0
        })
        
        print(f"Enhanced Episodic Memory initialized:")
        print(f"  Max examples per class: {max_examples_per_class}")
        print(f"  Quality threshold: {quality_threshold}")
        print(f"  Forgetting enabled: {enable_forgetting}")
        print(f"  Forgetting strategy: {forgetting_strategy}")

    def add_example(self, embedding: torch.Tensor, label: int, 
                   quality_score: Optional[float] = None, 
                   metadata: Optional[Dict] = None):
        """
        Add a single example to memory with quality control and forgetting.
        
        Args:
            embedding (torch.Tensor): The embedding vector of the example
            label (int): The integer label of the example
            quality_score (float, optional): Quality score of the example (0-1)
            metadata (dict, optional): Additional metadata (timestamp, source, etc.)
        """
        # Ensure embedding is on CPU and detached
        embedding = embedding.detach().cpu()
        
        # Calculate quality score if not provided
        if quality_score is None:
            quality_score = self._calculate_quality_score(embedding, label)
        
        # Quality filtering
        if quality_score < self.quality_threshold:
            self.total_rejected += 1
            self.class_statistics[label]['rejected'] += 1
            if len(self.storage[label]) < 5:  # Allow some low-quality examples for very new classes
                warnings.warn(f"Low quality example for class {label} (score: {quality_score:.3f}) - adding anyway as class is new")
            else:
                return False  # Rejected
        
        # Create metadata
        example_metadata = {
            'quality_score': quality_score,
            'timestamp': self.total_added,
            'label': label
        }
        if metadata:
            example_metadata.update(metadata)
        
        # Add the example
        self.storage[label].append((embedding, example_metadata))
        self.total_added += 1
        self.class_statistics[label]['added'] += 1
        
        # Update class quality statistics
        class_qualities = [meta['quality_score'] for _, meta in self.storage[label]]
        self.class_statistics[label]['avg_quality'] = np.mean(class_qualities)
        
        # Apply forgetting if necessary
        if self.enable_forgetting and len(self.storage[label]) > self.max_examples_per_class:
            self._forget_examples(label)
        
        return True  # Successfully added

    def _calculate_quality_score(self, embedding: torch.Tensor, label: int) -> float:
        """
        Calculate a quality score for a new example based on its relationship to existing examples.
        
        Args:
            embedding: The new example's embedding
            label: The example's label
            
        Returns:
            float: Quality score between 0 and 1
        """
        if label not in self.storage or len(self.storage[label]) == 0:
            return 1.0  # First example for this class is always high quality
        
        # Get existing embeddings for this class
        existing_embeddings = torch.stack([emb for emb, _ in self.storage[label]])
        
        # Calculate similarities to existing examples
        similarities = F.cosine_similarity(embedding.unsqueeze(0), existing_embeddings, dim=1)
        
        # Quality is based on informativeness (not too similar to existing examples)
        max_similarity = similarities.max().item()
        mean_similarity = similarities.mean().item()
        
        # High quality if it's somewhat different from existing examples
        # but not completely different (which might indicate noise)
        diversity_score = 1.0 - max_similarity  # Reward diversity
        consistency_score = min(mean_similarity + 0.3, 1.0)  # Reward some consistency
        
        # Combine scores (favor diverse but not outlier examples)
        quality_score = 0.7 * diversity_score + 0.3 * consistency_score
        
        return max(0.0, min(1.0, quality_score))

    def _forget_examples(self, label: int):
        """
        Remove examples from memory based on the forgetting strategy.
        
        Args:
            label: Class label to apply forgetting to
        """
        examples = self.storage[label]
        target_size = max(1, int(0.8 * self.max_examples_per_class))  # Keep 80% of max
        num_to_forget = len(examples) - target_size
        
        if num_to_forget <= 0:
            return
        
        if self.forgetting_strategy == "oldest":
            # Remove oldest examples (by timestamp)
            examples.sort(key=lambda x: x[1]['timestamp'])
            forgotten = examples[:num_to_forget]
            self.storage[label] = examples[num_to_forget:]
            
        elif self.forgetting_strategy == "lowest_quality":
            # Remove lowest quality examples
            examples.sort(key=lambda x: x[1]['quality_score'])
            forgotten = examples[:num_to_forget]
            self.storage[label] = examples[num_to_forget:]
            
        elif self.forgetting_strategy == "random":
            # Random forgetting
            indices = np.random.choice(len(examples), num_to_forget, replace=False)
            forgotten = [examples[i] for i in indices]
            self.storage[label] = [examples[i] for i in range(len(examples)) if i not in indices]
            
        else:
            raise ValueError(f"Unknown forgetting strategy: {self.forgetting_strategy}")
        
        # Update statistics
        self.total_forgotten += len(forgotten)
        self.class_statistics[label]['forgotten'] += len(forgotten)

    def get_prototypes(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Calculate prototypes for each class with quality weighting.
        
        Returns:
            Tuple containing:
            - prototypes (torch.Tensor): [num_classes, embedding_dim] prototype tensor
            - class_ids (list): List of class IDs corresponding to each prototype
        """
        class_ids = sorted(self.storage.keys())
        if not class_ids:
            return torch.empty(0), []
        
        all_prototypes = []
        
        for cid in class_ids:
            embeddings = []
            weights = []
            
            for embedding, metadata in self.storage[cid]:
                embeddings.append(embedding)
                # Use quality score as weight for averaging
                weights.append(metadata['quality_score'])
            
            embeddings_tensor = torch.stack(embeddings)
            weights_tensor = torch.tensor(weights, dtype=embeddings_tensor.dtype)
            
            # Normalize weights
            weights_tensor = weights_tensor / weights_tensor.sum()
            
            # Calculate quality-weighted prototype
            prototype = (embeddings_tensor * weights_tensor.unsqueeze(1)).sum(dim=0)
            all_prototypes.append(prototype)
        
        prototypes = torch.stack(all_prototypes)
        return prototypes, class_ids

    def get_simple_prototypes(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Get simple (unweighted) prototypes for backward compatibility.
        
        Returns:
            Tuple containing prototypes and class IDs
        """
        class_ids = sorted(self.storage.keys())
        if not class_ids:
            return torch.empty(0), []
        
        all_prototypes = [
            torch.stack([emb for emb, _ in self.storage[cid]]).mean(dim=0) 
            for cid in class_ids
        ]
        
        prototypes = torch.stack(all_prototypes)
        return prototypes, class_ids

    def get_class_examples(self, label: int, max_examples: Optional[int] = None) -> List[torch.Tensor]:
        """
        Get examples for a specific class.
        
        Args:
            label: Class label
            max_examples: Maximum number of examples to return
            
        Returns:
            List of embedding tensors
        """
        if label not in self.storage:
            return []
        
        examples = [emb for emb, _ in self.storage[label]]
        
        if max_examples and len(examples) > max_examples:
            # Return highest quality examples
            scored_examples = [(emb, meta['quality_score']) for emb, meta in self.storage[label]]
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            examples = [emb for emb, _ in scored_examples[:max_examples]]
        
        return examples

    def get_memory_statistics(self) -> Dict:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            'total_examples': len(self),
            'total_classes': len(self.storage),
            'total_added': self.total_added,
            'total_rejected': self.total_rejected,
            'total_forgotten': self.total_forgotten,
            'rejection_rate': self.total_rejected / max(1, self.total_added + self.total_rejected),
            'forgetting_rate': self.total_forgotten / max(1, self.total_added),
            'examples_per_class': {},
            'quality_per_class': {},
            'memory_efficiency': len(self) / max(1, self.total_added)
        }
        
        for class_id in self.storage:
            stats['examples_per_class'][class_id] = len(self.storage[class_id])
            stats['quality_per_class'][class_id] = self.class_statistics[class_id]['avg_quality']
        
        return stats

    def print_statistics(self):
        """Print detailed memory statistics."""
        stats = self.get_memory_statistics()
        
        print(f"\n=== Episodic Memory Statistics ===")
        print(f"Total examples stored: {stats['total_examples']}")
        print(f"Total classes: {stats['total_classes']}")
        print(f"Total examples processed: {stats['total_added']}")
        print(f"Examples rejected (low quality): {stats['total_rejected']} ({stats['rejection_rate']:.1%})")
        print(f"Examples forgotten: {stats['total_forgotten']} ({stats['forgetting_rate']:.1%})")
        print(f"Memory efficiency: {stats['memory_efficiency']:.1%}")
        
        print(f"\nPer-class breakdown:")
        for class_id in sorted(self.storage.keys()):
            count = stats['examples_per_class'][class_id]
            quality = stats['quality_per_class'][class_id]
            print(f"  Class {class_id}: {count} examples, avg quality: {quality:.3f}")

    def compact_memory(self, target_examples_per_class: Optional[int] = None):
        """
        Compact memory by keeping only the highest quality examples.
        
        Args:
            target_examples_per_class: Target number of examples per class
        """
        if target_examples_per_class is None:
            target_examples_per_class = max(1, self.max_examples_per_class // 2)
        
        print(f"Compacting memory to {target_examples_per_class} examples per class...")
        
        total_before = len(self)
        
        for label in self.storage:
            if len(self.storage[label]) > target_examples_per_class:
                # Sort by quality score (descending)
                examples = self.storage[label]
                examples.sort(key=lambda x: x[1]['quality_score'], reverse=True)
                
                # Keep only the highest quality examples
                forgotten_count = len(examples) - target_examples_per_class
                self.storage[label] = examples[:target_examples_per_class]
                
                # Update statistics
                self.total_forgotten += forgotten_count
                self.class_statistics[label]['forgotten'] += forgotten_count
        
        total_after = len(self)
        print(f"Memory compacted: {total_before} -> {total_after} examples")

    def clear_class(self, label: int):
        """Remove all examples for a specific class."""
        if label in self.storage:
            forgotten_count = len(self.storage[label])
            del self.storage[label]
            self.total_forgotten += forgotten_count
            self.class_statistics[label]['forgotten'] += forgotten_count

    def clear_all(self):
        """Clear all memory."""
        total_forgotten = len(self)
        self.storage.clear()
        self.total_forgotten += total_forgotten
        print(f"All memory cleared. {total_forgotten} examples forgotten.")

    def __len__(self):
        """Return total number of examples stored in memory."""
        return sum(len(examples) for examples in self.storage.values())

    def get_seen_classes(self) -> List[int]:
        """Return list of unique class labels seen so far."""
        return sorted(self.storage.keys())

    def merge_with(self, other_memory: 'EpisodicMemory', quality_threshold_boost: float = 0.0):
        """
        Merge another memory into this one.
        
        Args:
            other_memory: Another EpisodicMemory instance to merge
            quality_threshold_boost: Temporary boost to quality threshold for merged examples
        """
        print(f"Merging with another memory ({len(other_memory)} examples)...")
        
        original_threshold = self.quality_threshold
        self.quality_threshold = max(0.0, self.quality_threshold - quality_threshold_boost)
        
        merged_count = 0
        for label, examples in other_memory.storage.items():
            for embedding, metadata in examples:
                if self.add_example(embedding, label, 
                                  metadata.get('quality_score', 0.5), 
                                  metadata):
                    merged_count += 1
        
        self.quality_threshold = original_threshold
        print(f"Successfully merged {merged_count} examples.")