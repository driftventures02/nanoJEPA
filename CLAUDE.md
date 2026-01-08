# NanoJEPA Developer Guide & Architecture Spec

## 1. Core Philosophy
NanoJEPA is a **Joint-Embedding Predictive Architecture** for text. 
- **GPT (Standard):** Predicts `Token[t+1]` given `Token[0...t]`.
- **NanoJEPA:** Predicts `Embedding[Block B]` given `Embedding[Block A]`.

It separates **Thinking** (Latent Prediction) from **Speaking** (Token Generation).

## 2. Architecture Components
The model is composed of three distinct neural networks ("organs"):

### A. The Encoder (The Eye)
- **Role:** Compresses raw text into a high-dimensional "Thought Vector".
- **Input:** Token Sequence (Context).
- **Output:** A single vector `z_context` (Shape: `[Batch, Embed_Dim]`).
- **Constraint:** It must learn to ignore syntax noise and capture semantic meaning.

### B. The Predictor (The Brain)
- **Role:** Performs reasoning in latent space.
- **Input:** `z_context`.
- **Output:** `z_hat_target` (Predicted representation of the *future*).
- **Key Logic:** This is where the "intelligence" lives. It learns the causal physics of concepts (e.g., "Question" concept -> "Answer" concept).

### C. The Generator (The Mouth)
- **Role:** Translates abstract vectors back into English.
- **Input:** `z_hat_target` (The predicted thought).
- **Mechanism:** A standard Transformer Decoder that uses Cross-Attention to attend to the thought vector.
- **Output:** Token Sequence (Target).

## 3. Training Dynamics (The Dual Loss)
We train with two losses simultaneously to prevent "Representation Collapse" (where the model cheats by predicting zero).

1.  **Latent Loss (JEPA Loss):**
    - `MSE(Predictor(Context), StopGrad(Encoder(Target)))`
    - Forces the Brain to predict the *true meaning* of the future.
    - **Crucial:** We stop gradients on the Target Encoder so the Encoder doesn't change to match the Brain (cheating).

2.  **Generative Loss (Reconstruction):**
    - `CrossEntropy(Generator(Predicted_Vector), Target_Tokens)`
    - Forces the Mouth to learn how to articulate the specific thoughts produced by the Brain.

## 4. Difference from NanoGPT
| Feature | NanoGPT | NanoJEPA |
| :--- | :--- | :--- |
| **Objective** | Next Token Prediction | Latent Future Prediction |
| **State** | Key/Value Cache (Tokens) | Abstract Vector (Thought) |
| **Global Coherence** | Low (can ramble) | High (plans response first) |
| **Training Speed** | Fast | Slow (needs stable latent space) |

## 5. Hackability
- **Change the Brain:** Swap the MLP Predictor for a Transformer Block to enable multi-step reasoning.
- **Change the Mouth:** Swap the Decoder for a Stable Diffusion model to make it a Text-to-Image JEPA.
