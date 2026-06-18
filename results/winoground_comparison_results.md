======================================================================
           WINOGROUND COMPOSITIONALITY BENCHMARK COMPARISON
======================================================================
Total Samples Evaluated: CLIP = 100 | VLM = 100

### Winoground Performance Comparison Table
| Method | Text Score (%) | Image Score (%) | Group Score (%) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Vanilla CLIP** | 29.00% | 13.00% | 9.00% | Direct cross-modal score |
| **M-LQE (Average)** | 16.00% | 9.00% | 5.00% | Mean component score |
| **M-LQE (Product)** | 15.00% | 8.00% | 4.00% | Product component score |
| **M-LQE (Hybrid)** | 23.00% | 11.00% | 7.00% | Global + 0.5 * component |
| **VLM (meta/llama-3.2-11b-vision-instruct)** | 45.00% | 19.00% | 12.00% | Cross-attention reasoning |
======================================================================

### Examples of CLIP Failures Solved by VLM on Winoground
Below are samples where Vanilla CLIP failed (Group Score = 0), but the VLM succeeded (Group Score = 1):

1. **Sample ID 0**:
   * **Caption 0**: 'an old person kisses a young person'
   * **Caption 1**: 'a young person kisses an old person'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

2. **Sample ID 6**:
   * **Caption 0**: 'a plant was harmed by another organism, and that organism broke the plant into pieces'
   * **Caption 1**: 'another organism was harmed by a plant, and that plant broke the organism into pieces'
   * **CLIP Vanilla Score**: Text Correct = 1 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

3. **Sample ID 7**:
   * **Caption 0**: 'a bottle is in water'
   * **Caption 1**: 'water is in a bottle'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

4. **Sample ID 17**:
   * **Caption 0**: 'there are more ladybugs than flowers'
   * **Caption 1**: 'there are more flowers than ladybugs'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1

5. **Sample ID 25**:
   * **Caption 0**: 'the person with the white collared shirt waters the plant while the other holds it'
   * **Caption 1**: 'the person with the white collared shirt holds the plant while the other waters it'
   * **CLIP Vanilla Score**: Text Correct = 0 | Image Correct = 0
   * **VLM Correctness**: Text Correct = 1 | Image Correct = 1