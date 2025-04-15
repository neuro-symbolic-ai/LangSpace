from enum import Enum


class DisentanglementMetric(Enum):
    """
        Metrics used to evaluate disentangled representations in latent variable models.

        These metrics provide multiple perspectives on the quality of learned representations, by
        assessing how exclusively and robustly individual latent dimensions capture specific generative
        factors.

        Attributes:
            Z_DIFF:
                The "z-diff" metric assesses how sensitive each latent dimension is to changes in just one
                generative factor. It calculates the differences in latent activations when varying a single
                factor, thereby measuring the responsiveness of each dimension.
                Reference: Higgins, I., Matthey, L., Pal, A., et al. “beta-VAE: Learning Basic Visual Concepts with a
                Constrained Variational Framework.” ICLR, 2017.

            Z_MIN_VAR:
                The "z-min-var" metric quantifies the stability of latent dimensions by measuring the minimum
                variance observed across dimensions when a particular generative factor is fixed. Lower variance
                in the corresponding latent code indicates a consistent encoding of that factor.
                Reference: Kim, H., & Mnih, A. “Disentangling by Factorising.” In Proceedings of the 35th International
                Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2649–2658. PMLR.

            MIG:
                The Mutual Information Gap (MIG) evaluates disentanglement by computing the difference in mutual
                information between the top two latent variables linked to a specific ground-truth factor. A
                larger gap implies that the factor is predominantly captured by a single latent dimension.
                Reference: Chen, R. T. Q., Li, X., Grosse, R., & Duvenaud, D. “Isolating Sources of Disentanglement in
                Variational Autoencoders.” NeurIPS, 2018.

            DISENTANGLEMENT:
                This overall metric reflects the degree to which the learned representation is disentangled.
                It typically aggregates per-dimension or per-factor scores to provide a single measure of
                representation quality.
                Reference: Eastwood, C., & Williams, C. K. I. “A Framework for the Quantitative Evaluation of
                Disentangled Representations.” ICLR, 2018.

            COMPLETENESS:
                Completeness measures the concentration of information about each generative factor within a
                limited set of latent variables. High completeness indicates that each factor is encoded without
                excessive redundancy across different dimensions.
                Reference: Eastwood, C., & Williams, C. K. I. “A Framework for the Quantitative Evaluation of
                Disentangled Representations.” ICLR, 2018.

            INFORMATIVENESS:
                This metric gauges how well the latent representation retains useful information about the
                underlying generative factors. It is often estimated via the performance of predictors that recover
                these factors from the latent codes, thus reflecting the overall predictive power of the representation.
                Reference: Eastwood, C., & Williams, C. K. I. “A Framework for the Quantitative Evaluation of
                Disentangled Representations.” ICLR, 2018.
        """

    Z_DIFF = "z-diff"
    Z_MIN_VAR = "z-min-var"
    MIG = "MIG"
    # MODULARITY = "Modularity"
    # EXPLICITNESS = "Explicitness"
    DISENTANGLEMENT = "Disentanglement"
    COMPLETENESS = "Completeness"
    INFORMATIVENESS = "Informativeness"
