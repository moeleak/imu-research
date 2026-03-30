#import "@preview/touying:0.6.1": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [A simple research about IMU],
    subtitle: [A Comparative Study with Multiple Neural Models on Handheld Inertial Data],
    author: [Li X.Z., Ma J.Z., Meng H.J., Cai Z.X],
    date: datetime.today(),
    institution: [Object-Oriented Programming Final Project],
  ),
)

#let result-figure(path, cap) = figure(
  image(path, width: 100%),
  caption: cap,
)

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em))

= Introduction

== Problem Statement

- Inertial Measurement Unit (IMU) based Pedestrian Dead Reckoning (PDR) is attractive because it is infrastructure-free and can run on commodity mobile devices.
- The main difficulty is that heading noise, uncertain step length, and cumulative integration drift quickly distort long trajectories.
- A fixed-step baseline is easy to deploy, but it cannot adapt to different walking speeds, phone poses, or motion transitions.
- The goal of this project is to compare multiple learning-based models under a unified IMU pipeline and inject stride-aware logic into the displacement prediction task.
- The central question is whether gait-scale structure helps the model learn more realistic human motion patterns from inertial signals.


== Contributions

- A unified IMU preprocessing and reconstruction pipeline is used for all methods.
- The prediction target is redesigned as scalar displacement with stride-aware temporal logic.
- Classical, convolutional, recurrent, and graph-based models are compared under the same evaluation setting.


= Methodology

== End-to-End Pipeline

- Step 1: load the handheld sequences from the Oxford Inertial Odometry Dataset (OxIOD) and downsample the raw streams by a factor of four.
- Step 2: construct an 11-dimensional feature vector from linear acceleration, angular velocity, roll and pitch trigonometric encoding, and acceleration magnitude.
- Step 3: train each model to regress scalar displacement instead of absolute position.
- Step 4: smooth the yaw angle with Savitzky-Golay filtering and remove static intervals with Zero-Velocity Update.
- Step 5: reconstruct the trajectory by integrating predicted displacement with heading and perform bias, scale, and drift alignment for fair comparison.

#align(center)[
  $
    x_t = [a_t, omega_t, sin(r_t), cos(r_t), sin(p_t), cos(p_t), ||a_t||_2]
  $
]


== Mathematical Formulation

#slide(composer: (1fr, 1fr))[
  #align(center)[
    $
      d_t &= ||p_(t+Delta) - p_t||_2 \
      theta_t &= psi_t + b + t delta
    $
  ]
][
  #align(center)[
    $
      Delta x_t &= hat(d)_t cos theta_t \
      Delta y_t &= hat(d)_t sin theta_t \
      hat(p)_t &= hat(p)_(t-1) + (Delta x_t, Delta y_t)
    $
  ]
]


== Dataset and Target Design

#slide(composer: (1fr, 1fr))[
  - Dataset: Oxford Inertial Odometry Dataset (OxIOD).
  - Focus: handheld sequences and qualitative analysis on `data5`.
  - Statistics are estimated on the training split and reused in testing.
][
  - Window models use 20-frame inputs.
  - Sequence models use 100-frame segments.
  - The target is future scalar displacement, not coordinates.
]


== Compared Models

#slide(composer: (1fr, 1fr))[
  - Baseline: Weinberg step detector with fixed length.
  - CNN-MLP: local window displacement regression.
  - AR-CNN: stride-aware autoregressive regression.
][
  - LSTM: frame-wise sequential displacement prediction.
  - GNN: Graph Attention Network v2 with temporal edges.
  - Transformer: explored as an auxiliary extension.
]


== Stride-Aware Learning Logic

#align(center)[
  $
    y_t = d_t
  $
]

- Scalar displacement is used as the supervision target.
- Temporal links inject stride-scale continuity into prediction.
- The inductive bias is motion-aware rather than coordinate-aware.


== Optimization Objective

#slide(composer: (1fr, 1fr))[
  #align(center)[
    $
      z_t &= I(v_t < tau) \
      tilde(d)_t &= (1 - z_t) hat(d)_t
    $
  ]
][
  #align(center)[
    $
      L &= 1/T sum_(t=1)^T rho(hat(d)_t - d_t) \
      L_s &= 1/T sum_(t=1)^T rho(tilde(d)_t - d_t)
    $
  ]
]


= Experiments

== Evaluation Protocol

- All methods are trained on the same feature family and evaluated with the same heading reconstruction procedure.
- Yaw is smoothed before integration to reduce phase noise from the raw inertial orientation signal.
- Predicted displacement is converted into trajectory increments and accumulated over time.
- A global search over heading bias, scale, and drift is used to compare trajectory shape under consistent alignment assumptions.
- Because the repository includes figures instead of a full numeric log table, the analysis below focuses on convergence behavior and qualitative trajectory fidelity.


== Training Loss Comparison

#slide(composer: (1fr, 1fr))[
  #result-figure(
    "assets/comparison/loss-train.png",
    [Train loss]
  )
][
  #result-figure(
    "assets/comparison/loss-val.png",
    [Validation loss]
  )
]

- The loss plots are included to verify that each model can fit the scalar displacement objective.
- In IMU-based trajectory reconstruction, low pointwise loss is necessary but not sufficient, because small displacement errors accumulate during integration.
- The final judgment therefore combines loss behavior with the reconstructed path quality shown in the following figures.


== Baseline and Compact Regressor

#slide(composer: (1fr, 1fr))[
  #result-figure(
    "assets/standalone/baseline.png",
    [Baseline]
  )
][
  #result-figure(
    "assets/standalone/mlp.png",
    [CNN-MLP]
  )
]

- The classical baseline captures the coarse turning tendency of the walker, but its fixed step length makes it sensitive to pace changes and device handling variation.
- The compact regressor replaces the constant step assumption with learned displacement estimation, which improves adaptivity to local inertial patterns.
- However, a pure local window model still has limited access to longer temporal dependencies, so drift can remain after long integration.


== Stride-Aware and Sequential Models

#slide(composer: (1fr, 1fr))[
  #result-figure(
    "assets/standalone/mlp_autoregressive.png",
    [AR-CNN]
  )
][
  #result-figure(
    "assets/standalone/lstm.png",
    [LSTM]
  )
]

- The stride-aware autoregressive variant produces smoother step-to-step evolution because displacement is propagated with explicit sequential logic.
- This design better reflects the fact that human walking is not a collection of isolated windows but a continuous process with stride-to-stride correlation.
- The Long Short-Term Memory model further strengthens temporal continuity by maintaining hidden states over longer sequences.
- In practice, both strategies are more suitable than the fixed-step baseline when motion speed and turning behavior vary across time.


== Graph-Based Motion Modeling

#slide(composer: (1fr, 1.05fr))[
  #result-figure(
    "assets/standalone/gnn.png",
    [GNN]
  )
][
  - Edges at `t+1`, `t+5`, and `t+10` encode local and stride-scale dependencies.
  - The graph structure is robust to handheld signal wobble.
]


== Comparative Findings

- The fixed-step baseline is simple and interpretable, but it is the least adaptive because step length is hand-crafted.
- The compact learned regressor improves local displacement estimation and provides a strong lightweight baseline.
- Adding stride-aware sequential logic produces more physically plausible trajectories because adjacent predictions become motion-consistent.
- Long Short-Term Memory and Graph Neural Network models are better suited to learning the regularity of human motion under varying inertial conditions.


= Conclusion

== Conclusion and Future Work

- This project reformulates IMU-based Pedestrian Dead Reckoning as scalar displacement learning with explicit stride-aware logic.
- The comparison shows a clear progression from fixed-step heuristics to temporal and graph-based models that better capture human motion regularity.
- The methodological lesson is that a good inductive bias matters: step-scale targets, temporal continuity, and motion-aware reconstruction are all crucial.
- Future work can extend this study with joint heading and displacement prediction, uncertainty-aware trajectory fusion, cross-device evaluation, and a full numeric benchmark table.
- The current report provides a clean foundation for presenting the project as an academic study rather than only a collection of scripts.


== Keywords

- Inertial Measurement Unit
- Pedestrian Dead Reckoning
- Oxford Inertial Odometry Dataset
- Multi-Layer Perceptron
- Long Short-Term Memory
- Graph Neural Network
- Zero-Velocity Update
