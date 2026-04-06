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
  image(path),
  caption: cap,
)

#let stacked-result-row(left-path, left-cap, right-path, right-cap, height: 5.8cm) = {
  v(0.1em)
  set text(size: 0.78em)
  grid(
    columns: (1fr, 1fr),
    gutter: 0.25em,
    row-gutter: 0.1em,
    align(center)[
      #figure(
        image(left-path, height: height),
        caption: left-cap,
      )
    ],
    align(center)[
      #figure(
        image(right-path, height: height),
        caption: right-cap,
      )
    ],
  )
}

#let stacked-wide-result-row(left-path, left-cap, right-path, right-cap, width: 100%) = {
  v(0.1em)
  grid(
    columns: (1fr, 1fr),
    gutter: 0.35em,
    row-gutter: 0.1em,
    align(center)[
      #text(size: 0.66em)[#left-cap]
    ],
    align(center)[
      #text(size: 0.66em)[#right-cap]
    ],
    align(center)[
      #image(left-path, width: width)
    ],
    align(center)[
      #image(right-path, width: width)
    ],
  )
}

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em))

= Introduction

== Problem Statement

- IMU-based PDR is infrastructure-free and phone-ready.
- Main errors come from heading noise, step uncertainty, and drift.
- Fixed-step methods break under speed, pose, and motion changes.
- We compare learned models and test whether stride-aware design improves realism.


== Contributions

- One IMU preprocessing and reconstruction pipeline for all methods.
- Scalar displacement target with stride-aware temporal logic.
- Comparison across classical, CNN, recurrent, and graph models.


= Methodology

== End-to-End Pipeline

- Load OxIOD handheld sequences and downsample by `4x`.
- Build 11-D features from acceleration, gyro, attitude, and norm.
- Train each model to predict scalar displacement.
- Smooth yaw with Savitzky-Golay (S-G) filtering.
- Reconstruct trajectories with aligned bias, scale, and drift.

#align(center)[
  $
    x_t = [a_t, omega_t, sin(r_t), cos(r_t), sin(p_t), cos(p_t), ||a_t||_2]
  $
]

#text(size: 0.9em)[
  $a_t$: linear acceleration, $omega_t$: angular velocity, $r_t/p_t$: roll and pitch.
]


== Mathematical Formulation

#slide(composer: (1fr, 1fr))[
  #align(center)[
    $
      d_t &= ||p_(t+Delta) - p_t||_2 \
      theta_t &= tilde(psi)_t + b + t delta
    $
  ]
  
  #text(size: 0.85em)[
    $d_t$: target displacement, $p_t$: ground truth position, $tilde(psi)_t$: S-G smoothed yaw.
  ]
][
  #align(center)[
    $
      Delta x_t &= hat(d)_t cos theta_t \
      Delta y_t &= hat(d)_t sin theta_t \
      hat(p)_t &= hat(p)_(t-1) + (Delta x_t, Delta y_t)
    $
  ]
  
  #text(size: 0.85em)[
    $hat(d)_t$: predicted displacement, $Delta x_t / Delta y_t$: increments, $hat(p)_t$: path.
  ]
]


== Dataset and Target Design

#slide(composer: (1fr, 1fr))[
  - Dataset: OxIOD handheld sequences.
  - Case study: `data5`.
  - Train-set statistics are reused at test time.
][
  - Window models: 20 frames.
  - Sequence models: 100 frames.
  - Target: future scalar displacement.
]


== Compared Models

#slide(composer: (1fr, 1fr))[
  - Baseline: Weinberg step detector with fixed length.
  - CNN-MLP: local window displacement regression.
  - AR-CNN: stride-aware autoregressive regression.
][
  - LSTM: frame-wise sequential displacement prediction.
  - GNN: Graph Attention Network v2 with temporal edges.
]


== Stride-Aware Learning Logic

#align(center)[
  $
    y_t = d_t
  $
]

#text(size: 0.9em)[
  $y_t$ is the scalar displacement label.
]

- Supervision target: scalar displacement.
- Temporal links enforce stride continuity.
- The inductive bias is motion-aware, not coordinate-aware.

== S-G Smoothing and Loss

#slide(composer: (1fr, 1fr))[
  #align(center)[
    $
      tilde(psi)_t &= sum_(k=-m)^m c_k psi_(t+k) \
      sum_(k=-m)^m c_k &= 1
    $
  ]
  
  #text(size: 0.85em)[
    $psi_t$: raw yaw, $c_k$: S-G coefficients, $m$: half-window size.
  ]
][
  #align(center)[
    $
      e_t &= hat(d)_t - d_t \
      L &= 1/T sum_(t=1)^T rho(e_t)
    $
  ]
  
  #text(size: 0.85em)[
    $e_t$: displacement residual, $rho$: robust penalty, $L$: training loss.
  ]
]


= Experiments

== Evaluation Protocol

- Same features and heading reconstruction for all methods.
- Apply S-G yaw smoothing before integration.
- Convert predicted displacement into trajectory increments.
- Compare paths after bias, scale, and drift alignment.


== Training Loss Comparison

#[
  #set text(size: 0.82em)
  #set par(leading: 0.92em)
  - Loss curves show whether each model fits the displacement target.
  - Low pointwise loss does not guarantee a good trajectory.
  - Final comparison uses reconstructed path quality.
]

#stacked-wide-result-row(
  "assets/comparison/loss-train.png",
  [Train loss],
  "assets/comparison/loss-val.png",
  [Validation loss],
)


== Baseline and Compact Regressor

#[
  #set text(size: 0.82em)
  #set par(leading: 0.92em)
  - The baseline captures coarse turns but is sensitive to pace and pose changes.
  - CNN-MLP learns local displacement and adapts better to inertial patterns.
  - Limited long-range context still leaves drift after long integration.
]

#stacked-result-row(
  "assets/standalone/baseline.png",
  [Baseline],
  "assets/standalone/mlp.png",
  [CNN-MLP],
  height: 9.2cm,
)


== Stride-Aware and Sequential Models

#[
  #set text(size: 0.82em)
  #set par(leading: 0.92em)
  - AR-CNN gives smoother step-to-step predictions.
  - LSTM strengthens longer-range temporal continuity.
  - Both handle speed and turning changes better than fixed-step logic.
]

#stacked-result-row(
  "assets/standalone/mlp_autoregressive.png",
  [AR-CNN],
  "assets/standalone/lstm.png",
  [LSTM],
  height: 9.2cm,
)

== Graph-Based Motion Modeling

#slide(composer: (1fr, 1fr))[
  #result-figure(
    "assets/standalone/gnn.png",
    [GNN]
  )
][
  - Edges at `t+1`, `t+5`, and `t+10` encode local and stride-scale dependencies.
  - The graph structure is more robust to handheld wobble.
]

== Comparative Findings

- Fixed-step baseline: simple, interpretable, least adaptive.
- CNN-MLP: a strong lightweight learned baseline.
- AR-CNN, LSTM, and GNN produce more motion-consistent trajectories.
- Temporal structure matters for IMU PDR.

= Conclusion

== Conclusion and Future Work

- We reformulate IMU-PDR as scalar displacement learning.
- Stride-aware temporal models outperform fixed-step heuristics.
- Useful inductive bias comes from step-scale targets and temporal continuity.
- Future work: joint heading prediction, uncertainty fusion, cross-device tests, and a full benchmark table.
