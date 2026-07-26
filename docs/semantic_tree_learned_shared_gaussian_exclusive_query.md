# Learned-Shared and Gaussian-Exclusive Semantic-Tree Queries

This hybrid query path is opt-in:

```yaml
semantic_tree_query_mode: learned_shared_gaussian_exclusive
semantic_tree_num_queries: 1

semantic_tree_gaussian_query_sample: true
semantic_tree_gaussian_query_initial_logvar: -4.0
semantic_tree_gaussian_query_min_logvar: -8.0
semantic_tree_gaussian_query_max_logvar: 4.0
semantic_tree_gaussian_query_reliability_temperature: 1.0
semantic_tree_gaussian_query_detach_uncertainty: false
semantic_tree_dual_query_output_mode: context

lambda_semantic_tree_query_kl_aux: 0.001
lambda_semantic_tree_query_shared_mi_aux: 0.0
lambda_semantic_tree_query_exclusive_mi_aux: 0.01
lambda_semantic_tree_query_diversity_aux: 0.01
lambda_semantic_tree_query_classification_aux: 0.01
```

Existing `learned`, `root`, `root_learned`, and
`gaussian_shared_exclusive` modes retain their original modules and
forward paths.

## Forward path

1. The shared query is the deterministic dataset-level
   `learned_query`. It is expanded across a mini-batch but is not
   assigned a mean/variance posterior and is never sampled.
2. For every graph, concatenate the mean-pooled backbone node
   representation with the backbone root representation. An exclusive
   encoder predicts the mean and diagonal log-variance of the
   graph-private Gaussian query.
3. The shared query and a sampled private query independently retrieve
   values from the same semantic tree.
4. The shared branch has a fixed zero reliability logit. The private
   branch uses the same Gaussian differential-entropy uncertainty as the
   full dual-Gaussian mode:

   \[
   u_e=\frac{1}{2}\operatorname{Mean}
   \left(\log\sigma_e^2+\log(2\pi e)\right).
   \]

   The private entropy is centered at the entropy \(u_0\) implied by
   `semantic_tree_gaussian_query_initial_logvar`. The two branch weights
   are:

   \[
   (w_s,w_e)=\operatorname{softmax}
   \left(-[0,u_e-u_0]/T\right).
   \]

   Fusion therefore starts at \(w_s=w_e=0.5\). The shared branch has no
   learned uncertainty parameters or uncertainty gradients. Only a
   change in private log-variance moves its reliability relative to the
   fixed shared reference.
5. Only the exclusive Gaussian receives the query KL loss and the
   cross-graph exclusive-overlap loss. The shared-MI auxiliary weight is
   ignored in this mode because the learned shared query is already
   identical across graphs.
6. The retrieved shared/private graph representations can still receive
   the diversity loss and their independent auxiliary classification
   losses.

## Interpretation

This mode separates a stable dataset-level semantic question from
graph-specific evidence seeking. It is less dependent on the backbone
than making both queries graph-conditioned, while retaining a
distributional confidence estimate for the private query.

The private variance is still a learned variational uncertainty proxy,
not guaranteed calibration. Compare sampling on/off, detach on/off, and
the learned-only baseline, and report the fusion weights or calibration
diagnostics before making a strong uncertainty claim.
