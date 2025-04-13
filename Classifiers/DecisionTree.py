import numpy as np
from collections import deque
from typing import Deque

class Condition:
    # Helper class store all split conditions informations
    def __init__(self, feature: int = 0, threshold: float = 0):
        self.feature = feature
        self.threshold = threshold

class Node:  
    # Helper class: Store decision nodes (feature, threshold, left/right children)
    def __init__(self, samples: np.ndarray, condition: Condition = None):
            self.samples = samples
            self.n_samples = len(samples)
            self.condition = condition if condition else Condition()
            self.class_majority = -1
            self.impurity = 1
            self.left = None
            self.right = None
             

class DecisionTree:  
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1):  
        # Initialize tree constraints (depth, min samples to split)  
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):  
        # build the tree by splitting nodes using Gini impurity
        self.n_classes = len(np.unique(y))
        self.root = Node(np.arange(len(X))) # We will store indices to keep track of sampels per node during training
        depth=0
        q: Deque[Node] = deque()
        q.append(self.root)
        
        #We stop the tree at max_depth
        while q:
            size = len(q)
            for _ in range(size):
                node = q.popleft()

                # Define it's majority class
                values, counts = np.unique(y[node.samples], return_counts=True)
                node.class_majority = values[np.argmax(counts)]

                # The node becomes a leaf if it has a perfect impurity
                node_impurity = self._gini(y[node.samples])
                node.impurity = node_impurity
                if node_impurity == 0:
                    continue

                # we don't split if we don't have enough samples
                if node.n_samples < self.min_samples_split:
                    continue
                
                condition = self._best_split(node_impurity, X[node.samples], y[node.samples])
                if condition == None:
                    continue

                if depth >= self.max_depth:
                    continue
                
                node.condition = condition

                mask = X[node.samples][:, node.condition.feature] <= node.condition.threshold

                node.left = Node(node.samples[mask])
                node.right = Node(node.samples[~mask])
                    
                q.append(node.left)
                q.append(node.right)
            depth+=1
        print(self)
        predictions = self.predict(X)
        accuracy = np.sum(predictions==y)/len(y)

        return accuracy


    def predict(self, X: np.ndarray):  
        # Traverse the tree to predict class labels for input X
        predictions = []

        for input in X:
            root = self.root
            while root.left:
                if input[root.condition.feature] <= root.condition.threshold:
                    root = root.left
                else:
                    root = root.right
            predictions.append(root.class_majority)
        
        return predictions

    def _gini(self, y: np.ndarray):  
        # Helper: Compute Gini impurity for a set of labels 
        _, counts = np.unique(y, return_counts=True)
        weighted_counts = counts/len(y)
        
        return 1-np.sum(np.square(weighted_counts))

    def _best_split(self, parent_impurity: float, X: np.ndarray, y: np.ndarray):  
        # Helper: Find the best feature/threshold to split the data
        n_features = X.shape[1]
        max_gain = -np.inf
        best_feature, best_threshold = -1, -1

        #iterate over each feature
        for n in range(n_features):
            # Create an array containing only the feature n from each sample
            feature_array = np.sort(X[:,n].squeeze())
            thresholds = (feature_array[:-1]+feature_array[1:])/2
            for threshold in thresholds:
                threshold_mask = X[:,n] <= threshold
                left_mask = y[threshold_mask]
                right_mask = y[~threshold_mask]

                # Check that we respect the minimum number of samples to create a leaf
                if len(right_mask)<self.min_samples_leaf or len(left_mask) < self.min_samples_leaf:
                    continue

                left_impurity = self._gini(left_mask)
                right_impurity = self._gini(right_mask)

                weighted_impurity = (len(left_mask)/len(y))*left_impurity + (len(right_mask)/len(y))*right_impurity

                gain = parent_impurity - weighted_impurity

                if gain > max_gain:
                    max_gain = gain
                    best_feature, best_threshold = n, threshold
        if best_feature == -1:
            return None
        else:
            return Condition(feature=best_feature, threshold=best_threshold)
        
    def __str__(self):
        if not self.root:
            return "Empty tree"
        
        lines = []
        stack = [(self.root, 0, True, True)]  # (node, depth, is_last, is_root)
        
        while stack:
            node, depth, is_last, is_root = stack.pop()
            
            # =====================================================================
            # 1. Build the prefix for this node (connectors + indentation)
            # =====================================================================
            prefix = ""
            if not is_root:
                # Indentation from previous levels
                prefix += "    " * (depth - 1)
                
                # Add connector (├── or └── )
                prefix += "└── " if is_last else "├── "
            
            # =====================================================================
            # 2. Build the node's description line
            # =====================================================================
            if node.left is None and node.right is None:
                # Leaf node
                line = f"{prefix}Leaf: class {node.class_majority} (samples={node.n_samples}, impurity={node.impurity:.3f})"
            else:
                # Decision node
                line = (f"{prefix}[Feature {node.condition.feature} <= {node.condition.threshold:.2f}] "
                        f"(samples={node.n_samples}, impurity={node.impurity:.3f})")
            
            lines.append(line)
            
            # =====================================================================
            # 3. Prepare children for processing (reverse order for stack LIFO)
            # =====================================================================
            children = []
            if node.right: children.append((node.right, False))  # Right child first
            if node.left: children.append((node.left, True))     # Left child last
            
            # Push children to stack with their metadata
            for idx, (child, is_last_child) in enumerate(children):
                stack.append((
                    child,             # Node to process
                    depth + 1,         # Next depth level
                    is_last_child,     # Is last child of its parent?
                    False              # No longer root
                ))
        
        return "\n".join(lines)
