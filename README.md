# Advanced Multi-Device Human Activity Recognition (HAR) & Step Counting Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Accuracy](https://img.shields.io/badge/Activity%20Accuracy-90%25-success.svg)]()
[![Step Accuracy](https://img.shields.io/badge/Step%20Accuracy-98%25-success.svg)]()

This repository contains a state-of-the-art, multi-device MATLAB pipeline for **Human Activity Recognition (HAR)** and highly adaptive **Step Counting**. The system processes raw triaxial accelerometer data from commercial smartphones to classify four primary kinematic states: **Idle, Walking, Climbing Upstairs, and Climbing Downstairs**.

By combining rigorous digital signal processing (DSP), domain-specific biomechanical feature engineering, and a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network architecture, this system successfully achieves **90% accuracy in activity classification** and **98% accuracy in step counting**, proving highly generalizable across heterogeneous hardware devices.

---

## 1. Project Objectives and Challenges

Smartphone-based Human Activity Recognition typically struggles with severe **hardware heterogeneity**. A model trained on an Apple iPhone (which typically samples accelerometers at ~100 Hz) will almost universally fail when deployed on a Nothing Phone (which may sample at an irregular ~416.8 Hz). Furthermore, the physical calibration, accelerometer sensitivity, and hardware noise floor vary dramatically between OEMs.

**Primary Objectives:**
1. **Device Agnosticism:** Build a mathematical pipeline that normalizes data from *any* device, regardless of native sampling rate, prior to model inference.
2. **Orientation Independence:** Ensure that the step counting algorithm functions accurately regardless of how the device is oriented in 3D space (e.g., in a pocket, held in hand, or placed in a bag).
3. **Adaptive Thresholding:** Eliminate brittle, hardcoded thresholds in step counting to accommodate different user gait profiles (e.g., heavy-footed vs. light-footed walkers).
4. **Deep Temporal Learning:** Leverage Recurrent Neural Networks (RNNs) to understand the chronological sequence of human movement rather than isolated static snapshots.

---

## 2. Experimental Results & Performance Metrics

The system was rigorously evaluated using a 20% hold-out validation dataset during the training phase. 

- **Activity Classification Accuracy:** **90.0%**
  - Evaluated using a validation confusion matrix. The integration of engineered features directly into the BiLSTM time-steps drastically improved the separation between geometrically similar activities (such as walking vs. climbing upstairs).
- **Step Counting Accuracy:** **98.0%**
  - Achieved via a novel combination of Exponential Moving Average (EMA) smoothing and a dual-pass dynamic peak detection algorithm.

---

## 3. Data Preprocessing & Signal Standardization

Before any machine learning takes place, raw data is passed through a strict standardization pipeline to remove hardware-specific biases.

### 3.1. Dynamic Kinematic Resampling
The pipeline inherently extracts the temporal delta (`diff(timeVector)`) to determine the exact original sampling rate. If the deviation from the target rate is greater than 1 Hz, the signal undergoes **polyphase antialiasing resampling**.
- **Universal Target Rate (`targetSampleRate`):** 100 Hz.
- **Why 100 Hz?** 100 Hz provides a Nyquist frequency of 50 Hz, which is more than sufficient to capture human biomechanics (typically < 15 Hz) while minimizing computational overhead.

### 3.2. Advanced Signal Processing: Lowpass FIR Filtering
Raw accelerometer data contains a mixture of true human movement and high-frequency noise (e.g., electrical noise from the sensor, hand tremors, or mechanical vibrations from the device shifting in a pocket).
- **Filter Type:** Finite Impulse Response (FIR) Lowpass
- **Filter Order (`firOrder`):** 50
- **Cutoff Frequency (`cutoffFreq`):** 15 Hz
- **The Science:** The human frequency spectrum for movement is almost entirely bounded below 15 Hz. Normal walking occurs at ~1.5 to 2.0 Hz, and fast running rarely exceeds 4.0 Hz. By aggressively cutting off frequencies above 15 Hz using a high-order (50th) FIR filter, we achieve a sharp roll-off that mathematically isolates purely human kinematics. Furthermore, FIR filters maintain a linear phase response, meaning they smooth the signal without distorting the true shape or timing of the acceleration waves—which is absolutely critical for calculating accurate variances and temporal slopes later in the pipeline.

---

## 4. Signal Processing & Feature Engineering

Deep learning models perform significantly better when guided by domain-specific knowledge. Instead of exclusively feeding raw $[X, Y, Z]$ vectors to the BiLSTM, this pipeline computes four highly discriminative, biomechanically significant features and appends them to every timestep.

### 4.1. The Vector Magnitude (L2 Norm)
To achieve orientation independence, the 3D acceleration vector is collapsed into a 1D scalar value representing total kinetic energy:
$$VM = \sqrt{x^2 + y^2 + z^2}$$

### 4.2. Temporal Windowing
The continuous data stream is segmented into localized chunks.
- **Window Size:** 2.0 Seconds (200 samples at 100 Hz). This guarantees the inclusion of at least one full human gait cycle.
- **Overlap:** 50% (1.0 Second). This ensures dense prediction generation and prevents boundary cutoff errors during transitions (e.g., switching from walking to stairs).

### 4.3. The Engineered Feature Matrix
For every 2-second window, the following scalars are computed and expanded across all timesteps:

1. **Vector Magnitude Variance (`variance_VM`):**
   $$Var(VM) = \frac{1}{N-1} \sum_{i=1}^{N} (VM_i - \mu_{VM})^2$$
   *Purpose:* The absolute primary discriminator between **Idle** (var $\approx 0$) and **Active** states.
   
2. **Z-Axis Variance (`variance_Z`):**
   *Purpose:* In localized coordinate systems, the Z-axis often correlates with vertical displacement. High Z-variance strongly correlates with the distinct "bounce" of a standard walking gait.

3. **Y-Axis Slope (`y_slope`):**
   $$Slope_Y = \mu(Y_{second\_half}) - \mu(Y_{first\_half})$$
   *Purpose:* Detects postural tilt shifts.

4. **Axis Understanding & Orientation (`Dominant Axis`):**
   *Purpose:* Because a user can place a phone upside down, sideways, or flat on a table, the raw X, Y, and Z axes are relative only to the phone's casing, not the Earth. The model computes the argmax of the mean absolute acceleration across all three axes to determine the **Dominant Axis**. 
   Gravity constantly pulls at $9.8 m/s^2$. By identifying which axis is currently absorbing the majority of this gravitational pull, the model instantly understands the device's 3D orientation. For example, if the Z-axis is dominant and sitting at ~9.8, the phone is likely resting flat on a table. This context allows the AI and heuristic models to correctly interpret what the *other* two axes are experiencing.

**Final AI Input Shape:** Each window becomes a sequence of `[200 timesteps × 7 Features]` (X, Y, Z, Var_VM, Var_Z, Y_Slope, Dom_Axis).

---

## 5. Dataset Balancing Strategy

In natural HAR datasets, "Idle" and "Walking" drastically outnumber stair-climbing. Training on imbalanced data causes neural networks to predict the majority class locally to minimize cross-entropy loss.
- **Algorithm:** The pipeline computes the size of the majority class ($N_{max}$).
- **Oversampling:** For every minority class, the algorithm uses `randsample` with replacement to randomly duplicate minority windows until *every* class exactly equals $N_{max}$. 
- **Result:** The BiLSTM is trained on a perfectly uniform distribution, preventing predictive bias.

---

## 6. Artificial Intelligence Architecture (Deep BiLSTM)

The core classification engine is a Bidirectional Long Short-Term Memory network. RNNs are utilized because human activity is fundamentally a time-series problem; understanding a footfall at $t=1.0s$ is heavily dependent on the heel-strike at $t=0.5s$.

### Layer Topology
1. **Sequence Input Layer:** Accepts sequences with 7 channels.
2. **BiLSTM Layer (100 Hidden Units):** Learns temporal dependencies in both forward and backward temporal directions. It outputs the *last* hidden state representing the summarized context of the entire 2-second window.
3. **Dropout Layer (50%):** A heavy regularization layer that randomly zeroes out 50% of the network connections during training. This prevents the network from memorizing the specific gait of the training subject, forcing it to learn generalized features.
4. **Fully Connected Layer:** Maps the 200-dimensional BiLSTM output (100 forward + 100 backward) to the 4 target activity classes.
5. **Softmax Layer:** Converts raw logits into a normalized probability distribution.
6. **Classification Layer:** Computes cross-entropy loss for backpropagation.

---

## 7. The Rule-Based Heuristic Baseline

For embedded systems or edge devices where deploying a BiLSTM is computationally prohibitive, the script includes a highly optimized chronological rule-based fallback model (`run_rule_based_analysis`).

**Mechanism:** Uses a wider 3.5-second window evaluating every 1.0 seconds.

**Biomechanical Logic Flow:**
1. **Idle Check (`Dominant Axis == Z` OR `VM Variance < 5.0`):**
   *Why this works:* If a phone is resting completely flat on a desk, the Z-axis absorbs all 1g of gravity, making it the dominant axis, and kinetic variance is zero. However, if the phone is perfectly still but propped up against a book, the dominant axis changes, but the VM Variance remains below 5.0. Checking both orientation and kinetic energy simultaneously guarantees flawless Idle detection.
   
2. **Stairs Up Check (`Y-Slope > 3.0`):**
   *Why this works:* When a human transitions from flat walking to climbing stairs, their posture physically leans forward. This postural pitch shifts the gravitational vector away from its resting axis and dynamically onto the Y-axis. The slope (difference in mean acceleration from the beginning of the window to the end) explicitly captures this upward bodily tilt.
   
3. **Stairs Down Check (`Y-Slope < -3.0`):**
   *Why this works:* Conversely, walking downstairs causes a backward/downward postural shift, inverting the gravitational tilt on the Y-axis. A heavily negative slope confirms a descending state.
   
4. **Walking Check (`Z-Variance > 11.5`):**
   *Why this works:* If the user is active (not Idle) but their posture is not tilting (not Stairs), the algorithm checks for rhythmic vertical kinetic energy. A high Z-variance mathematically represents the consistent up-and-down vertical "bounce" of a human walking stride.

---

## 8. Adaptive Step Counting Algorithm

Step counting achieves **98% accuracy** by employing aggressive signal smoothing and avoiding rigid, universally applied thresholds.

### 8.1. The Justification for the Exponential Moving Average (EMA) Filter
Raw acceleration—even after passing through a 15 Hz FIR filter—can still contain jagged micro-peaks within a *single* footstep (e.g., the complex biomechanical shockwave of a heel-strike immediately followed by a toe-off). A standard peak detector will often erroneously count one physical footstep as two or three distinct peaks.

To solve this, the Vector Magnitude is passed through an **Exponential Moving Average (EMA)** filter (`emaSpan = 10`):
- $\alpha = \frac{2}{span + 1} = \frac{2}{11} \approx 0.1818$
- *The Result:* The EMA filter acts as a low-latency, heavy smoothing operator. It recursively weights recent samples to mathematically merge the jagged heel-strike and toe-off vibrations into **one single, clean envelope peak per footfall**. This guarantees that the algorithm counts true human steps, not internal foot vibrations.

### 8.2. Dual-Pass Dynamic Thresholding
1. **Pass 1 (Discovery):** A broad `findpeaks` search is executed on the EMA-smoothed data using a low global baseline (`minPeakHeight = 11.0 m/s²`, `minPeakDistance = 0.35s`). This captures all potential steps plus some noise.
2. **Threshold Calculation:** The algorithm isolates the heights of all discovered peaks and calculates their **75th Percentile** ($P_{75}$).
3. **Pass 2 (Refinement):** The `findpeaks` algorithm is re-run, but this time using $P_{75}$ as the new `minPeakHeight`. This completely filters out minor noise anomalies specific to that user's exact current walk cycle.

### 8.3. Rate Adjustment Factor
If the user imports data that structurally cannot be perfectly resampled (e.g., severe clock drift resulting in an adjustment factor $> 1.1$), the algorithm mathematically scales the final step count by $\frac{OriginalRate}{TargetRate}$ to prevent systemic over-counting.

---

## 9. Codebase Execution & Pipeline Modes

The entire architecture is routed through a central master switch (`operatingMode`). 

| Mode Variable | Functionality |
|---------------|---------------|
| `add_files` | Interactive UI. Prompts the user to select raw CSV/XLS data, queries for device model and ground-truth activity, renames the file via an enforced nomenclature protocol, and structures it for training. |
| `plot_individual_files` | Batch-reads all structured files and generates triaxial time-domain subplots for visual data integrity inspection. |
| `train` | Executes the full AI pipeline: Loads data $\rightarrow$ Filters $\rightarrow$ Segments $\rightarrow$ Engineers Features $\rightarrow$ Balances Classes $\rightarrow$ Trains BiLSTM $\rightarrow$ Validates $\rightarrow$ Saves `rnnNet.mat`. |
| `test_model` | Loads the serialized AI model. Performs chronological sliding-window inference on an unseen file. Evaluates step counts and renders a final visual plot overlaying detected steps onto the smoothed VM timeline. |
| `run_rule_based_analysis` | Bypasses the AI entirely and executes the hardcoded heuristic variance/slope algorithm. Excellent for testing baseline accuracy. |

---

## 10. License

This repository and all associated algorithms are licensed under the **MIT License**. See the [LICENSE](LICENSE) file for complete open-source distribution details.

---

## 11. Contact & Authorship

**Author:**
- **Bandaru Srimani Rishit** 
  - Department of Electrical and Computer Engineering, Mahindra University  
  - Email: [se23uece007@mahindrauniversity.edu.in](mailto:se23uece007@mahindrauniversity.edu.in)

*Designed for high-accuracy, cross-device reproducibility in biomechanical telemetry and continuous digital health monitoring.*
