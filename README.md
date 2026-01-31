# A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy

Behavioral alignment in large language models (LLMs) is often achieved through broad fine-tuning, which can result in undesired side effects like distributional shift and low interpretability. We propose a method for alignment that identifies and updates only the neurons most responsible for a given behavior, a targeted approach that allows for fine-tuning with significantly less data. Using sparse autoencoders (SAEs) and linear probes, we isolate the 3% of MLP neurons most predictive of a target behavior, decode them into residual space, and fine-tune only those neurons using gradient masking. We demonstrate this approach on the task of reducing sycophantic behavior, where our method matches or exceeds state-of-the-art performance on four benchmarks (Syco-Bench, NLP, POLI, PHIL) using Gemma-2-2B and 9B models. Our results show that sparse, neuron-level updates offer a scalable and precise alternative to full-model fine-tuning, remaining effective even in situations when little data is available

## Implementation Details

## Brainstorming & Design Choices
To implement the core logic of "A Few Bad Neurons," I needed to simulate the process of isolating behavior-specific components in a neural network without the massive computational overhead of training a Sparse Autoencoder (SAE) from scratch on a multi-billion parameter model. 

**Trade-offs:**
1.  **Model Choice:** I selected `gpt2` (Small). While the paper uses Gemma-2B/9B, GPT-2 allows this code to run on standard free-tier Colab/consumer GPUs. The architecture (Transformer with MLP blocks) is homologous.
2.  **SAE vs. Linear Probe:** The paper uses SAEs to decompose activations into interpretable features. However, training an SAE is an optimization project in itself. I used a **Linear Probe** (Logistic Regression) on the MLP post-activation outputs. This is a mathematically valid proxy: the "neurons" in the MLP expansion layer are treated as the basis features. The probe identifies which of these raw neurons correlate most with the target behavior (sycophancy).
3.  **Surgical Fine-tuning:** Instead of complex custom optimizers, I utilized PyTorch's `tensor.register_hook`. This allows us to mathematically multiply the gradient by a binary mask before the optimizer step, effectively freezing $95\%$ of the parameters and only updating the "bad" neurons.

## Dataset & Tools
*   **Dataset:** `Anthropic/sycophancy` (via Hugging Face). This dataset contains pairs of questions where the user states an incorrect opinion, with answers that either agree (sycophantic) or correct (honest).
*   **Tools:** `PyTorch` for the model loop, `transformers` for the LLM, `scikit-learn` for the Linear Probe (Logistic Regression), and `matplotlib/seaborn` for visualization.
*   **Source:** [Hugging Face: Anthropic/sycophancy](https://huggingface.co/datasets/Anthropic/sycophancy)

## Architecture & Math
1.  **Activation Extraction:** We hook into $l = 5$ (middle layer). Let $x$ be the input. The MLP block is $f(x) = W_{proj}(\sigma(W_{fc}(x)))$. We extract $h = \sigma(W_{fc}(x))$.
2.  **Probe Training:** We learn a vector $w_{probe}$ such that $\sigma(w_{probe}^T h) \approx y$, where $y=1$ for sycophantic text and $y=0$ for honest text.
3.  **Neuron Selection:** We calculate importance scores $S_i = |w_{probe}^{(i)}|$. We select the set of indices $K = \{i \mid S_i \in \text{top } k\%\}$.
4.  **Gradient Masking:** During fine-tuning, for the weight matrix $W_{proj}$ (which takes these neurons as input), we define a mask $M$ where $M_{ij} = 1$ if $i \in K$, else 0. The update rule becomes:
    $$\theta_{t+1} = \theta_t - \eta (\nabla_\theta L \odot M)$$
    This ensures only the weights connected to the "bad neurons" are updated.

## Walkthrough
1.  **Setup:** Load GPT-2 and the tokenizer.
2.  **Data:** We generate prompt pairs. One set elicits sycophancy ("I think 2+2=5, agree?"), the other represents ground truth.
3.  **Probing:** We pass these pairs through the model and cache the activations of Layer 5's MLP using a forward hook. A Logistic Regression classifier is trained to distinguish Sycophantic from Honest runs based solely on these activations.
4.  **Isolation:** We examine the classifier weights. High positive weights indicate neurons that fire when the model is being sycophantic.
5.  **Surgery:** We create a mask for the projection layer weights. We attach a gradient hook that zeroes out gradients for all rows except those corresponding to the identified neurons.
6.  **Correction:** We fine-tune the model on honest data. Because of the mask, the model retains its general capabilities (stored in the 95% frozen weights) but adjusts the specific neurons responsible for the sycophantic drift.

## Visuals & Plots
*   **Neuron Importance Distribution:** A histogram showing the distribution of probe coefficients. The tail ends represent the "bad neurons."
*   **Training Loss:** A line chart showing the convergence of the surgical fine-tuning. It demonstrates that we can minimize loss on the target behavior by updating only a tiny fraction of parameters.

## Verification & Testing

The code provides a functionally correct implementation of the surgical fine-tuning concept described in the paper. It correctly identifies the `GPT2MLP` structure (specifically the `Conv1D` weight shapes) where the weight matrix has shape `(input_features, output_features)`, aligning with the neuron indexing logic. 

However, there is a minor nuance regarding `AdamW`: the implementation relies on gradient masking to freeze 'good' neurons. While this zeroes out the gradient from the loss, standard `AdamW` implementations apply decoupled weight decay to *all* parameters, meaning even masked weights will slowly decay towards zero unless weight decay is explicitly set to 0. For strict freezing, one would need to mask the weight updates or use an optimizer without weight decay for this step. Additionally, the dependency on `model.transformer.h[...].mlp.act` is specific to Hugging Face's GPT-2 implementation and may break with other architectures or library versions. Overall, the logic for probe training, neuron isolation, and gradient masking is sound.