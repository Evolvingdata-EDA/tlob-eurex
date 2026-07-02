i cpo# Detecting Transient, Exploitable Predictability in Non-Stationary Time Series
### 2026 state-of-the-art survey, metrics framework, and deep-neural research roadmap

> Scope: detecting brief, economically exploitable windows of structure in non-stationary time series, with emphasis on financial markets, deep neural networks, online adaptation, regime discovery, and rigorous validation.
>
> Core change from the prior version: this document is organized as an implementation/research pipeline rather than a bibliography:
>
> **Detect → Forecast → Decide → Validate → Adapt**
>
> Key conceptual upgrade: do not only predict returns. Also predict the probability that the current window is *predictable enough to trade*:
>
> \[
> P(\text{predictability/exploitability} > \tau \mid X_t)
> \]
>
> In other words, build a **meta-predictability model** that decides when forecasting is worthwhile, then activate specialized forecasters and trading policies only in those windows.

---

## 0. Executive summary

Transient predictability is best treated as a **two-level problem**:

1. **Meta-prediction:** Is the current market state locally predictable and exploitable after costs?
2. **Conditional forecasting/trading:** If yes, which specialized model should forecast, size risk, and execute?

A robust system should therefore measure and optimize more than forecast error. It should measure:

- local forecastability,
- changepoint and drift quality,
- regime recurrence,
- uncertainty calibration,
- post-cost profitability,
- signal decay,
- robustness against false discovery,
- online adaptation speed.

A practical architecture is:

```text
Raw market data
   ↓
Cleaning, alignment, corporate actions, survivorship-safe universe
   ↓
Stationarity-aware transforms: returns, fractional differentiation, RevIN/Dish-TS/SAN
   ↓
Rolling distribution-shift and changepoint layer
   ↓
Regime embedding / retrieval layer
   ↓
Predictability classifier: P(edge exists | current state)
   ↓
Regime-conditioned neural forecaster: Transformer / TCN / SSM / MoE / foundation model
   ↓
Uncertainty and persistence heads
   ↓
Cost-aware trading policy and position sizing
   ↓
Purged/embargoed validation, PBO/DSR, live paper-trading monitor
   ↓
Online adaptation / continual learning / model retirement
```

---

# I. Problem formulation

## 1. What is a transient predictability pocket?

A **predictability pocket** is a contiguous interval in which a signal has materially stronger out-of-sample predictive and economic value than the surrounding baseline regime.

Operationally, a pocket should satisfy all of the following:

1. **Local statistical signal:** local out-of-sample IC, rank IC, log score, or R² exceeds a threshold.
2. **Economic signal:** post-cost Sharpe, expected utility, or alpha is positive after turnover, spread, impact, borrow, and latency.
3. **Persistence:** the edge survives long enough to be acted upon.
4. **Non-artifact:** it survives purged cross-validation, embargo, deflated Sharpe, PBO, randomized-label tests, and cost stress.
5. **Actionability:** the signal maps to feasible trades given capacity, liquidity, and execution constraints.

This framing avoids the common error of detecting *interesting structure* that is not economically exploitable.

## 2. Recommended target variables

Instead of training only on future returns, add meta-targets:

| Target | Meaning | Example label |
|---|---|---|
| Return forecast | Direction or magnitude | future return, residual return, quantile return |
| Predictability state | Whether forecasting is currently useful | 1 if local OOS IC > threshold |
| Exploitability state | Whether trading is currently useful | 1 if post-cost local Sharpe > threshold |
| Regime identity | Latent market state | cluster, HMM state, learned embedding ID |
| Edge persistence | Whether signal remains useful | 1 if next k windows remain profitable |
| Risk state | Whether risk is acceptable | volatility, drawdown, liquidity, adverse selection |

The most important addition is the **predictability/exploitability classifier**:

\[
Y^{edge}_{t}=1\{\text{local post-cost performance over }[t,t+h] > \tau\}
\]

This makes the system selective. It can refuse to trade when the world is locally noisy.

---

# II. Detect — local structure, changepoints, drift, regimes

## 3. Conceptual foundations

- **Farmer, Schmidt & Timmermann (2023), _Pockets of Predictability_.** Empirical anchor for local, time-varying predictability.
- **Cakici, Fieberg, Neumaier, Poddig & Zaremba (2025), _Pockets of Predictability: A Replication_.** Crucial caution: two-sided kernels and in-sample smoothing can create look-ahead artifacts.
- **Dahlhaus (1997/2012), locally stationary processes.** Theory for series that are locally approximable as stationary while global dynamics drift.
- **Lo (2004), Adaptive Markets Hypothesis.** Predictability waxes and wanes as agents adapt.
- **McLean & Pontiff (2016).** Alpha decays after discovery/publication through data mining, investor learning, and crowding.
- **Ang & Timmermann (2012).** Regime changes in financial markets.

## 4. Changepoint and drift detection methods

### Classical/statistical

- Bayesian Online Changepoint Detection (BOCPD).
- Robust and scalable BOCPD variants for heavy tails and outliers.
- Autoregressive online changepoint models with time-varying variance/correlation.
- Markov-switching and hidden Markov models.
- Frequency-domain changepoint detection under evolutionary spectra.
- Wasserstein clustering of rolling return distributions.

### Streaming-ML concept drift

- ADWIN adaptive windows.
- DDM, EDDM, Page-Hinkley, KSWIN.
- Error-rate drift monitors.
- Distributional drift monitors using MMD, KL/JS divergence, PSI, Wasserstein distance.
- Recurrent drift models that identify regime reappearances.

### Neural regime discovery

- Neural HMMs / switching state-space models.
- Variational recurrent state-space models.
- Contrastive regime encoders.
- Mixture-of-experts gates.
- Sparse regime mixtures, e.g. DeRegiME-style residual uncertainty regimes.

## 5. Detection metrics

| Objective | Metric | Notes |
|---|---|---|
| Detect change quickly | Detection delay | Use synthetic changepoints or annotated events |
| Avoid false alarms | False alarms per month/year | Important for turnover and model churn |
| Avoid missed changes | Recall of changepoints | Particularly important around crises |
| Regime quality | Regime purity / silhouette | Use distributional or economic separation |
| Distribution shift | Wasserstein, MMD, PSI, KL/JS | Compare current window with reference windows |
| Correlation shift | Frobenius norm of correlation difference | Useful cross-sectionally |
| Risk shift | vol/spread/liquidity state transitions | Must connect to execution quality |
| Recurrence | nearest-neighbor regime reappearance score | Useful for retrieval/meta-learning |

Recommended minimum detection dashboard:

```text
rolling vol shift
rolling correlation shift
MMD(current features, training features)
Wasserstein(current returns, reference returns)
BOCPD run-length posterior
drift detector alarm state
regime embedding nearest historical neighbors
```

---

# III. Measure — local forecastability and exploitability

## 6. Local predictability metrics

Use rolling or online estimates. Always compute out-of-sample.

| Metric | Use |
|---|---|
| Rolling OOS R² | Local explanatory/predictive power |
| Rolling IC / rank IC | Cross-sectional predictive quality |
| Directional accuracy | Useful but insufficient alone |
| Mutual information | Nonlinear dependence between features and future returns |
| Permutation entropy | Model-free complexity/predictability |
| Spectral entropy / ForeCA score | Forecastability via spectral concentration |
| Transfer entropy / Granger causality | Directional dependence, with caution |
| Calibration error | Whether probabilistic forecasts are reliable |
| CRPS / pinball loss | Probabilistic/quantile forecast quality |

## 7. Economic exploitability metrics

| Metric | Why it matters |
|---|---|
| Net Sharpe / Sortino | Core risk-adjusted performance |
| Deflated Sharpe | Adjusts for non-normality and selection bias |
| Calmar / max drawdown | Captures path risk |
| Turnover-adjusted alpha | Prevents unrealistic high-churn signals |
| Post-cost PnL | Must include spread, fees, slippage, impact |
| Capacity curve | Edge decay as capital increases |
| Half-life of alpha | How fast the pocket disappears |
| Hit rate conditional on signal strength | Tests monotonicity |
| Tail conditional loss | Avoids rare crash exposure |
| Execution shortfall | Separates forecasting skill from execution losses |

## 8. Meta-predictability labels

To train a model that predicts whether a pocket exists, create labels such as:

```text
edge_label_t = 1 if rolling_forward_IC(t, t+h) > threshold
edge_label_t = 1 if rolling_forward_post_cost_Sharpe(t, t+h) > threshold
edge_label_t = 1 if model_confidence_t is calibrated and post_cost_return_t > threshold
```

Important: these labels must be generated with purging and embargo because forward windows overlap.

---

# IV. Forecast — deep neural network approaches

## 9. Baseline neural forecasters

Before using complex regime-aware systems, benchmark against:

- LSTM / GRU.
- Temporal Convolutional Networks.
- N-BEATS / N-HiTS.
- DeepAR / DeepVAR.
- Temporal Fusion Transformer.
- PatchTST.
- iTransformer.
- TimesNet.
- TimeMixer.
- DLinear/NLinear as strong simple baselines.

Financial data often punishes complexity. A deep model is only useful if it beats simple baselines under realistic costs and validation.

## 10. Non-stationarity-aware neural methods

### Normalization and distribution shift

- **RevIN** normalizes each instance/window and restores scale after prediction.
- **Non-stationary Transformers** combine de-stationarization with modified attention.
- **Dish-TS** separates intra-window and inter-window distribution shift.
- **SAN** predicts temporal slice statistics.
- **Koopman Neural Operator Forecaster** uses operator-theoretic structure for temporal shifts.
- **TAFAS / TSF-TTA** adapts pre-trained forecasters at test time under shifting distributions.
- **PETSA-style parameter-efficient TTA** is useful when full online updating is too costly.

## 11. Regime-aware neural architectures

### A. Regime Mixture Transformer

```text
Input window
  ↓
Temporal encoder: Transformer/TCN/SSM
  ↓
Regime gate: sparse softmax / Dirichlet / stick-breaking
  ↓
Expert forecasters: momentum, reversion, volatility, liquidity, crisis
  ↓
Uncertainty heads: quantiles / Student-t / mixture density
  ↓
Cost-aware position sizing
```

Train with a combined objective:

\[
\mathcal{L}=\mathcal{L}_{forecast}+\lambda_1\mathcal{L}_{edge}+\lambda_2\mathcal{L}_{calibration}+\lambda_3\mathcal{L}_{turnover}
\]

### B. Changepoint + neural forecaster

```text
Rolling features
  ↓
BOCPD / neural CPD
  ↓
Reset or reweight hidden state
  ↓
LSTM/TCN/Transformer forecaster
  ↓
Trade only if edge classifier is active
```

Best for abrupt transitions.

### C. Retrieval-augmented regime forecaster

```text
Current window encoder
  ↓
Nearest historical regime windows
  ↓
Cross-attention over retrieved analogues
  ↓
Forecast + edge persistence estimate
```

Best when market patterns recur but with long gaps.

### D. Contrastive regime encoder

Train embeddings so that windows with similar future behavior are close:

- positive pairs: windows with similar subsequent return/risk/edge profiles,
- negative pairs: windows with different future behavior,
- optional supervised contrastive labels: volatility regime, liquidity regime, trend/reversion regime.

### E. Meta-learning / few-shot adaptation

Use when pockets are short:

- MAML/Reptile-style adaptation,
- hypernetworks generating forecaster parameters from recent context,
- adapters/LoRA for time-series foundation models,
- online fine-tuning with strong regularization,
- replay buffers to avoid catastrophic forgetting.

### F. State Space Models and Mamba-style models

State-space sequence models can be attractive because they scale well to long contexts. Use them for:

- long-context market histories,
- multivariate macro/asset panels,
- low-latency inference,
- regime embedding over long horizons.

But benchmark carefully against PatchTST/iTransformer/TCN and simple linear baselines.

## 12. Foundation models for time series

Time-series foundation models are now relevant as **feature extractors**, **zero-shot baselines**, and **few-shot adapters**, but they should not be assumed to work out-of-the-box for financial alpha.

Important models to include:

| Model | Core idea | Use in this project |
|---|---|---|
| TimesFM | Decoder-only pre-trained forecasting model | Zero-shot/few-shot baseline, feature extractor |
| Chronos | Tokenizes time-series values, trains LM-style models | Probabilistic forecasting baseline |
| Lag-Llama | Decoder-only probabilistic forecaster with lag covariates | Probabilistic baseline; uncertainty estimates |
| Moirai / Uni2TS | Universal masked-encoder forecasting transformer trained across many datasets | Strong general forecaster; transfer learning |
| MOMENT | Open general-purpose time-series foundation model | Embeddings, anomaly detection, classification, forecasting |
| Moirai-MoE | Sparse mixture-of-experts foundation model | Natural fit for regime specialization |
| Tiny Time Mixers / Granite TTM | Efficient foundation-style forecasters | Fast baselines and low-latency deployment |
| TimeGPT | Commercial foundation forecaster | Benchmark only if licensing/data constraints allow |

Recommended use: fine-tune or adapt them to predict **edge states**, not only future values.

## 13. Finance-specific neural architectures

### Deep Momentum Networks

- LSTM learns trend and position size directly against Sharpe-like objectives.
- Useful template for end-to-end trading but must be cost-aware.

### Momentum Transformer / X-Trend

- Attention-based strategy selection and interpretable trend/reversion pattern matching.
- X-Trend-like few-shot pattern recognition is highly relevant for recurring pockets.

### Deep Learning Statistical Arbitrage

- Construct residual portfolios from latent factor models.
- Use CNN/Transformer on residual time series.
- Optimize trading policy under constraints.
- Strong fit for cross-sectional transient mean-reversion.

### DeepLOB / order-book models

- CNN + Inception + LSTM for limit-order-book forecasting.
- Deep Order Flow Imbalance for multi-horizon microstructure alpha.
- Bayesian DeepLOB variants for uncertainty-aware position sizing.

For microstructure, add execution-specific targets:

- mid-price move,
- spread crossing probability,
- fill probability,
- adverse selection,
- queue position,
- cancel/replace timing,
- market impact.

---

# V. Decide — from forecast to trade

## 14. Trading policy design

A good forecast is not a trading strategy. Add a policy layer:

```text
forecast distribution
  ↓
expected utility after costs
  ↓
uncertainty gate
  ↓
position size
  ↓
risk constraints
  ↓
execution schedule
```

Decision rules:

- Trade only when expected return exceeds cost + risk buffer.
- Scale by calibrated confidence, not raw model probability.
- Penalize turnover directly in the loss.
- Cap exposure by regime-level drawdown risk.
- Deactivate strategies when edge persistence probability falls.

## 15. Policy metrics

| Metric | Purpose |
|---|---|
| Expected utility | Combines return and risk preference |
| Turnover penalty | Controls over-trading |
| Cost-adjusted signal-to-noise | Determines trade threshold |
| Probability of ruin / drawdown | Risk gate |
| Capacity-adjusted alpha | Realism |
| Execution shortfall | Implementation quality |
| Edge persistence probability | Whether to keep strategy active |

---

# VI. Validate — avoid look-ahead, false discovery, and overfitting

## 16. Validation protocol

Minimum protocol:

1. Use point-in-time data only.
2. Use walk-forward testing.
3. Use purged and embargoed cross-validation for overlapping labels.
4. Separate model selection, hyperparameter tuning, and final test periods.
5. Record all trials for deflated Sharpe and PBO.
6. Stress costs, slippage, spread, borrow, latency, and impact.
7. Run placebo tests: randomized labels, shifted labels, permuted features.
8. Test across assets, regions, and regimes.
9. Track live paper-trading degradation.

## 17. False-discovery metrics

| Metric | Use |
|---|---|
| Deflated Sharpe Ratio | Corrects Sharpe for multiple testing/non-normality |
| Probability of Backtest Overfitting | Estimates selection overfit probability |
| White Reality Check / SPA | Data-snooping robust test |
| Multiple-testing adjusted p-values | Especially for factor searches |
| Minimum backtest length | Avoids tiny-sample Sharpe illusions |
| Label-randomization performance | Should collapse to noise |
| Feature-permutation performance | Detects leakage or unstable dependence |

## 18. Leakage checklist

Common failure modes:

- two-sided rolling kernels,
- future constituents or survivorship bias,
- using final-day adjusted prices incorrectly,
- stale or revised macro data,
- using contemporaneous but unavailable signals,
- overlapping labels without purging,
- hyperparameter tuning on the test set,
- data vendor timestamp errors,
- ignoring delistings,
- cost assumptions fitted after seeing results.

---

# VII. Adapt — online learning and model lifecycle

## 19. Adaptation methods

| Method | When useful | Risk |
|---|---|---|
| Sliding-window retraining | slow drift | loses old regimes |
| Expanding window with decay | stable long memory | slow to adapt |
| Online gradient updates | fast drift | overreacts to noise |
| Test-time adaptation | source model under live shift | adaptation leakage if labels are mishandled |
| Adapters/LoRA | efficient foundation-model adaptation | underfits large drift |
| Replay buffers | recurring regimes | stale examples |
| Model ensembles | uncertainty and robustness | complexity and selection bias |
| Regime-conditioned experts | recurring market states | gate instability |

## 20. Adaptation metrics

| Metric | Meaning |
|---|---|
| Recovery time after drift | How fast performance returns |
| Adaptation gain | OOS improvement vs frozen model |
| Forgetting score | Loss on old regimes after adaptation |
| Regime reuse rate | How often old experts are reused |
| Model churn | Frequency of retraining/switching |
| Online calibration drift | Reliability over time |
| Live-vs-backtest degradation | Reality gap |

---

# VIII. Datasets and benchmarks

## 21. Financial datasets

| Dataset | Use |
|---|---|
| CRSP / Compustat | US equities, fundamentals, survivorship-safe research |
| TAQ | Intraday trades and quotes |
| LOBSTER | Limit order book research |
| FI-2010 | Standard DeepLOB-style benchmark |
| Nasdaq Nordic / Helsinki LOB datasets | Microstructure benchmarks |
| Optiver realized volatility competitions | Volatility forecasting |
| Jane Street competition data | Anonymized market prediction benchmark |
| Crypto order books | 24/7 high-frequency regime shifts |
| Futures data | Trend/momentum and macro regimes |
| Options data | volatility surface and risk-neutral signals |

## 22. General time-series datasets for pretesting

| Dataset | Use |
|---|---|
| ETT | long-horizon transformer benchmark |
| Electricity | multivariate demand forecasting |
| Exchange Rate | non-stationary financial-like benchmark |
| Traffic | large-panel forecasting |
| Weather | multivariate continuous forecasting |
| M4/M5 | hierarchical and retail forecasting |
| UCR/UEA | classification and representation testing |
| Monash Time Series Forecasting Repository | diverse forecasting tasks |
| LOTSA / Time Series Pile | foundation-model pretraining/evaluation |

Use general datasets to test architecture mechanics, but financial claims require financial validation.

---

# IX. Libraries and implementation stack

## 23. Forecasting and deep learning

- **PyTorch Forecasting** — high-level PyTorch/Lightning forecasting framework.
- **Nixtla NeuralForecast** — broad neural forecasting model collection.
- **GluonTS** — probabilistic forecasting and DeepAR-style models.
- **Darts** — unified forecasting interface.
- **sktime** — classical and ML time-series toolkit.
- **tsai** — deep learning for time-series classification/forecasting.
- **Uni2TS** — Moirai/Universal Time Series Transformer tooling.
- **Chronos Forecasting** — Amazon Chronos implementation.
- **Lag-Llama** — probabilistic foundation-model implementation.
- **MOMENT** — open time-series foundation model.

## 24. Online learning, drift, and quant research

- **River** — online ML and concept drift detection.
- **ruptures** — offline changepoint detection.
- **Kats** — time-series analysis and changepoints.
- **Qlib** — AI-oriented quantitative investment platform.
- **mlfinlab / mlfinpy-style tooling** — purged CV, fractional differentiation, labeling ideas.
- **FinRL** — reinforcement-learning trading experiments.
- **vectorbt / backtrader / zipline-like engines** — backtesting, with caution around realism.

---

# X. Proposed research program

## 25. Phase 1 — build the measurement layer

Deliverables:

- rolling local predictability dashboard,
- drift/changepoint dashboard,
- local post-cost performance dashboard,
- leakage-safe validation framework,
- baseline models: linear, tree, TCN, PatchTST/iTransformer, simple momentum/reversion.

Success criterion:

> The system can identify historical windows where predictability was present without using future information.

## 26. Phase 2 — train the edge classifier

Train:

\[
P(\text{edge exists} \mid X_t)
\]

Inputs:

- recent returns,
- volatility/liquidity/spread,
- cross-sectional dispersion,
- factor residuals,
- order-flow features,
- regime embeddings,
- drift metrics,
- uncertainty metrics.

Labels:

- forward local IC,
- forward post-cost Sharpe,
- forward drawdown-safe alpha,
- edge persistence over multiple horizons.

Success criterion:

> The classifier improves trade selectivity: fewer trades, higher net Sharpe, lower drawdown, lower PBO.

## 27. Phase 3 — add regime-conditioned neural forecasting

Test architectures:

1. TCN/Transformer + edge classifier.
2. Regime Mixture Transformer.
3. Retrieval-augmented regime forecaster.
4. MoE foundation-model adapter.
5. Changepoint + hidden-state reset forecaster.
6. Probabilistic DeRegiME-style residual uncertainty model.

Success criterion:

> Regime-conditioned models outperform frozen global models specifically during edge-positive windows.

## 28. Phase 4 — live paper-trading and online adaptation

Track:

- live calibration,
- live edge probability,
- live cost slippage,
- model drift,
- adaptation gain,
- signal decay,
- strategy crowding proxies.

Success criterion:

> Live paper performance degrades gracefully and adaptation improves recovery after drift without increasing false positives.

---

# XI. Suggested reading order

1. Farmer, Schmidt & Timmermann + Cakici replication — phenomenon and look-ahead trap.
2. López de Prado — fractional differentiation, meta-labeling, triple-barrier labeling, purged CV.
3. Bailey & López de Prado — deflated Sharpe and PBO.
4. Gama/Lu concept drift surveys — streaming framing.
5. RevIN / Non-stationary Transformers / Dish-TS / SAN — non-stationary neural forecasting.
6. Wood/Zohren/Roberts deep momentum, changepoint, X-Trend, DeRegiME — finance-specific DL + regimes.
7. Chronos, TimesFM, Lag-Llama, Moirai, MOMENT — foundation-model baselines and embedding sources.
8. DeepLOB / Deep Order Flow Imbalance / Briola microstructure guide — high-frequency implementation reality.
9. Guijarro-Ordonez, Pelger & Zanotti — deep statistical arbitrage pipeline.

---

# XII. Web-verified 2026 additions and source notes

The following items were checked/added during the 2026 update:

- **TAFAS / TSF-TTA**: test-time adaptation for non-stationary time-series forecasting, AAAI 2025, with code available.
- **DeRegiME**: 2026 arXiv paper on deep regime mixtures for probabilistic forecasting under distribution shift.
- **Deep Learning Statistical Arbitrage**: accepted in Management Science according to author research page; public code exists.
- **Chronos**: TMLR 2024; LM-style tokenized probabilistic forecasting.
- **TimesFM**: ICML 2024 decoder-only foundation model; Google Research code and checkpoints available.
- **Lag-Llama**: probabilistic decoder-only time-series foundation model.
- **Moirai / Uni2TS**: universal masked-encoder forecasting transformer trained on LOTSA.
- **MOMENT**: ICML 2024 open time-series foundation model.
- **Moirai-MoE**: sparse MoE time-series foundation model.
- **PyTorch Forecasting, NeuralForecast, River, Qlib**: active implementation tools relevant to this stack.

---

# XIII. Bottom line

The strongest research direction is not “use a bigger return forecaster.” It is:

> **learn when the market is locally predictable, identify which recurring regime it resembles, activate the appropriate specialized forecaster, and trade only when uncertainty, costs, and validation gates agree.**

That turns transient predictability from an after-the-fact observation into an online decision system.

