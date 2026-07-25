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

# Cross-graph Gaussian objectives
semantic_tree_gaussian_query_shared_mi_temperature: 0.2
semantic_tree_gaussian_query_exclusive_mi_temperature: 0.2

# "context" prevents the graph-conditioned query from bypassing attention.
semantic_tree_dual_query_output_mode: context

# Optional auxiliary objectives
lambda_semantic_tree_query_kl_aux: 0.001
lambda_semantic_tree_query_shared_mi_aux: 0.01
lambda_semantic_tree_query_exclusive_mi_aux: 0.01
lambda_semantic_tree_query_diversity_aux: 0.01
lambda_semantic_tree_query_classification_aux: 0.01
```

## Forward path

1. For every graph, concatenate the mean-pooled backbone node
   representation with the backbone root representation.
2. Feed that graph summary to two independent encoders. The shared
   encoder predicts a graph-conditioned offset around a learned
   dataset-level Gaussian prior; the exclusive encoder predicts a fully
   graph-conditioned Gaussian. Consequently, every graph has its own
   shared and exclusive posterior.
3. Sample one shared and one exclusive query (during training when
   sampling is enabled). They independently retrieve semantic-tree
   values through the same key/value projections and attention-layer
   parameters.
4. Apply a class-conditional cross-graph objective to the Gaussian
   posteriors. The shared objective uses diagonal-Gaussian Wasserstein
   distance in a supervised contrastive loss, pulling same-label shared
   posteriors together while separating different labels. The exclusive
   objective minimizes Gaussian overlap between same-label graphs so
   that graph-specific information does not collapse into another
   class-level prototype.
5. Give the two retrieved graph representations independent classifiers.
   Their cross-entropy losses ensure that both the shared and exclusive
   paths remain predictive of the graph label.
6. Fuse the two retrieved contexts with reliability weights obtained by
   applying a softmax to negative Gaussian entropy.
7. Retain the within-graph diversity loss, which minimizes a
   cross-covariance proxy between shared and exclusive retrieved graph
   representations.

The cross-graph objectives are tractable distribution-distance proxies,
not a closed-form estimate of mutual information. Conditioning them on
the class label avoids the contradictory requirement that an exclusive
query be completely label-independent while also predicting that label.
At least two examples of the same class must occur in a mini-batch for
the corresponding cross-graph terms to be non-zero.

The model exposes the latest shared/exclusive means, log-variances,
attention maps, and fusion weights through the corresponding
`_last_semantic_tree_*` diagnostic attributes.
