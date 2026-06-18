============================================================
           ARO ATTRIBUTION BENCHMARK COMPARISON
============================================================
Total Samples Evaluated: CLIP = 200 | VLM = 200

### Performance Comparison Table
| Method | Accuracy | Accuracy (%) | Description |
| :--- | :---: | :---: | :--- |
| **Vanilla CLIP** | 0.5550 | 55.50% | Direct cross-modal sentence score |
| **M-LQE (Average)** | 0.4850 | 48.50% | Mean component score (templates) |
| **M-LQE (Product)** | 0.4800 | 48.00% | Product component score (templates) |
| **M-LQE (Hybrid)** | 0.5050 | 50.50% | Global score + 0.5 * component score |
| **M-LQE (Grounded Crop)** | 0.4150 | 41.50% | Target object bounding-box crop |
| **M-LQE (Grounded Fusion)** | 0.4650 | 46.50% | Global score + 0.5 * crop score |
| **VLM (meta/llama-3.2-11b-vision-instruct)** | 0.7550 | 75.50% | Cross-attention visual reasoning |
============================================================

### Examples of CLIP Attribute-Binding Failures Solved by VLM
Below are instances where Vanilla CLIP was fooled by swapped attributes, but the VLM reasoned correctly:

1. **Sample ID 20**:
   * **True Caption**: 'the unpeeled banana and the square plate'
   * **False Caption**: 'the square banana and the unpeeled plate'
   * **VLM Selected**: Option B (Raw Output: 'B.')

2. **Sample ID 26**:
   * **True Caption**: 'the black table and the white plate'
   * **False Caption**: 'the white table and the black plate'
   * **VLM Selected**: Option A (Raw Output: 'A')

3. **Sample ID 28**:
   * **True Caption**: 'the unpeeled banana and the square plate'
   * **False Caption**: 'the square banana and the unpeeled plate'
   * **VLM Selected**: Option B (Raw Output: 'B')

4. **Sample ID 30**:
   * **True Caption**: 'the attached banana and the light colored plate'
   * **False Caption**: 'the light colored banana and the attached plate'
   * **VLM Selected**: Option A (Raw Output: 'A')

5. **Sample ID 34**:
   * **True Caption**: 'the gray street and the orange wall'
   * **False Caption**: 'the orange street and the gray wall'
   * **VLM Selected**: Option B (Raw Output: 'B')