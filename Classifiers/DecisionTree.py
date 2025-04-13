import numpy as np
from collections import deque
from typing import Deque

class Condition:
    # Helper class store all split conditions informations
    def __init__(self, feature: int = 0, threshold: float = 0):
        self.feature = feature
        self.threshold = threshold

class TrainNode:  
    # Helper class: Store training and conditions informations
    def __init__(self, samples: np.ndarray, condition: Condition = None):
            self.samples = samples
            self.n_samples = len(samples)
            self.condition = condition if condition else Condition()
            self.class_majority = -1
            self.impurity = 1
            self.left = None
            self.right = None

class TreeNode:
    """Lightweight tree node for prediction (no training data)."""
    def __init__(self, condition: Condition = None, class_label: int = -1):
        self.condition = condition
        self.class_label = class_label  # Only populated for leaf nodes
        self.left = None  # TreeNode
        self.right = None  # TreeNode

    def is_leaf(self) -> bool:
        return self.class_label != -1          

class DecisionTree:  
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1, criterion: str = "gini"):  
        # Initialize tree constraints (depth, min samples to split)  
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes = 0

        criterions = {
            "gini": self._gini,
            "entropy": self._entropy
        }
        self.criterion = criterions[criterion]

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:  
        # build the tree by splitting nodes using Gini impurity
        self.n_classes = len(np.unique(y))
        self.root = TrainNode(np.arange(len(X))) # We will store indices to keep track of sampels per node during training
        depth=0
        q: Deque[TrainNode] = deque()
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
                node_impurity = self.criterion(y[node.samples])
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

                node.left = TrainNode(node.samples[mask])
                node.right = TrainNode(node.samples[~mask])
                    
                q.append(node.left)
                q.append(node.right)
            depth+=1
        
        # Build the final tree
        self.root = self._build_prediction_tree(self.root)

        predictions = self.predict(X)
        accuracy = np.sum(predictions==y)/len(y)

        return accuracy


    def predict(self, X: np.ndarray) -> list:  
        # Traverse the tree to predict class labels for input X
        predictions = []

        for input in X:
            root = self.root
            while root.left:
                if input[root.condition.feature] <= root.condition.threshold:
                    root = root.left
                else:
                    root = root.right
            predictions.append(root.class_label)
        
        return predictions

    def _gini(self, y: np.ndarray) -> float:  
        # Helper: Compute Gini impurity for a set of labels 
        _, counts = np.unique(y, return_counts=True)
        proportions = counts/len(y)
        gini = 1-np.sum(np.square(proportions))

        return gini
    
    def _entropy(self, y: np.ndarray) -> float:
        # Helper: Compute entropy for a set of labels
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        entropies = proportions*np.log2(proportions)
        entropy = -np.sum(entropies)

        return entropy

    def _best_split(self, parent_impurity: float, X: np.ndarray, y: np.ndarray) -> Condition:  
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

                left_impurity = self.criterion(left_mask)
                right_impurity = self.criterion(right_mask)

                weighted_impurity = (len(left_mask)/len(y))*left_impurity + (len(right_mask)/len(y))*right_impurity

                gain = parent_impurity - weighted_impurity

                if gain > max_gain:
                    max_gain = gain
                    best_feature, best_threshold = n, threshold
        if best_feature == -1:
            return None
        else:
            return Condition(feature=best_feature, threshold=best_threshold)
    
    def _build_prediction_tree(self, train_root: TrainNode) -> TreeNode:
        if not train_root:
            return None
            
        # Initialize prediction tree root
        pred_root = TreeNode()
        stack = [(train_root, pred_root)]

        while stack:
            train_node, pred_node = stack.pop()
            
            # Handle leaf nodes
            if not train_node.left and not train_node.right:
                pred_node.class_label = train_node.class_majority
                continue
                
            # Copy split condition
            pred_node.condition = train_node.condition
            
            # Process right child first (stack is LIFO)
            if train_node.right:
                pred_node.right = TreeNode()
                stack.append((train_node.right, pred_node.right))
            
            # Process left child
            if train_node.left:
                pred_node.left = TreeNode()
                stack.append((train_node.left, pred_node.left))

        return pred_root