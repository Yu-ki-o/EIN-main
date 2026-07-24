# Gaussian Shared–Exclusive Semantic-Tree Queries

The dual-query path is opt-in. Existing `learned`, `root`, and
`root_learned` query modes keep their original modules and forward path.

## Enable the new path

```yaml
semantic_tree_query_mode: gaussian_shared_exclusive

# Gaussian query posterior
semantic_tree_gaussian_query_sample: true
semantic_tree_gaussian_query_condition_shared_variance: true
semantic_tree_gaussian_query_initial_logvar: -4.0
semantic_tree_gaussian_query_min_logvar: -8.0
semantic_tree_gaussian_query_max_logvar: 4.0

# Entropy-to-reliability fusion
semantic_tree_gaussian_query_reliability_temperature: 1.0
semantic_tree_gaussian_query_detach_uncertainty: false

# "context" prevents the graph-conditioned query from bypassing attention.
semantic_tree_dual_query_output_mode: context

# Optional auxiliary objectives
lambda_semantic_tree_query_kl_aux: 0.001
lambda_semantic_tree_query_diversity_aux: 0.01
lambda_semantic_tree_query_classification_aux: 0.01
```

## Forward path

1. The shared query mean is a dataset-level learned parameter. Its base
   log-variance is learned globally, with an optional graph-conditioned
   variance offset.
2. The exclusive posterior is produced from the concatenation of the
   mean-pooled backbone node representation and the backbone root
   representation.
3. Shared and exclusive queries independently retrieve semantic-tree
   values through the same key/value projections and attention-layer
   parameters.
4. Their retrieved contexts are fused with weights obtained by applying
   a softmax to negative Gaussian entropy.
5. The diversity loss minimizes a cross-covariance proxy between the two
   retrieved graph representations. It does not force raw query samples
   to be statistically independent.

The model exposes the latest shared/exclusive means, log-variances,
attention maps, and fusion weights through the corresponding
`_last_semantic_tree_*` diagnostic attributes.
