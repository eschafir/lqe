======================================================================
           WINOGROUND COMPOSITIONALITY BENCHMARK COMPARISON
======================================================================
Total Samples Evaluated: CLIP = 50 | VLM = 50

### Winoground Performance Comparison Table
| Method | Text Score (%) | Image Score (%) | Group Score (%) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Vanilla CLIP** | 28.00% | 6.00% | 4.00% | Direct cross-modal score |
| **M-LQE (Average)** | 16.00% | 2.00% | 2.00% | Mean component score |
| **M-LQE (Product)** | 16.00% | 2.00% | 2.00% | Product component score |
| **M-LQE (Hybrid)** | 18.00% | 4.00% | 4.00% | Global + 0.5 * component |
| **VLM (meta/llama-3.2-11b-vision-instruct)** | 52.00% | 28.00% | 12.00% | Cross-attention reasoning |
======================================================================

### Examples of CLIP Failures Solved by VLM on Winoground
Below are samples where Vanilla CLIP failed (Group Score = 0), but the VLM succeeded (Group Score = 1):

1. **Sample ID 0**:
   * **Caption 0**: 'an old person kisses a young person'
   * **Caption 1**: 'a young person kisses an old person'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

2. **Sample ID 5**:
   * **Caption 0**: 'a bird eats a snake'
   * **Caption 1**: 'a snake eats a bird'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 1
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

3. **Sample ID 19**:
   * **Caption 0**: 'there is more dirt than empty space in the jar'
   * **Caption 1**: 'there is more empty space than dirt in the jar'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

4. **Sample ID 30**:
   * **Caption 0**: 'a blue bird is next to a red berry'
   * **Caption 1**: 'a red bird is next to a blue berry'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

5. **Sample ID 49**:
   * **Caption 0**: 'it wears a hat but the person doesn't'
   * **Caption 1**: 'the person wears a hat but it doesn't'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1