from enum import Enum


class InterpolationMetric(Enum):
    """
        Metrics used to evaluate interpolation in generative models and latent space representations.

        These metrics assess the quality and continuity of transitions between points in the latent space to ensure
        that interpolated samples are both realistic and smoothly connected.

        Attributes:
            QUALITY:
                This metric evaluates the overall perceptual fidelity and realism of interpolated outputs.
                It considers aspects such as artifact-free generation, text clarity, and consistency with the learned
                data distribution. High quality interpolations are expected to appear indistinguishable from real data.
                Research in generative adversarial networks and variational autoencoders has repeatedly emphasized
                the importance of qualitative assessments in generative processes.

            SMOOTHNESS:
                This metric assesses the continuity of transitions along the latent space trajectory.
                A smooth interpolation implies gradual and coherent changes between successive points, ensuring that
                there are no abrupt jumps or artifacts. Smoothness is a key indicator of a well-behaved latent space,
                where semantic features change consistently and predictably. Studies evaluating latent space geometry
                have highlighted that smooth transitions are indicative of robust and disentangled representations.

        """

    QUALITY = "Quality"
    SMOOTHNESS = "Smoothness"
