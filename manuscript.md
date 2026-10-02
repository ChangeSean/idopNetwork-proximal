# idopNetwork proximal causal inference for molecular exposures and restricted survival

## Abstract

Protein abundance reflects molecular regulation and systemic variation that can confound survival associations. We develop idopNetwork–proximal inference for total molecular exposure effects. Niche-ordered curves and patient deviations guide network-based proxy construction. Structural conditions connect these roles to proximal exclusions, and a linear outcome bridge preserves downstream contributions to the total effect. Joint information learns systemic readout coordinates; conditional information selects an identifying design. Independent discovery and estimation combine this construction with concentrated moment inference. Inverse censoring weights extend the bridge to restricted mean survival time (RMST) and survival probability. A ten-estimator study compares seven molecular systems. In the modular system, linear-effect RMSE is 0.049, compared with 0.195 for a global-window design and 0.184 for correlation-selected proxies. Joint readout reduces linear bias from 0.203 to 0.013 in a weak two-factor system. Separate validation of the complete 75:25 discovery/estimation workflow uses 200 fresh datasets per system; coverage is 0.950–0.995, with bounded-set frequency quantifying precision. The ovarian-cancer application supplies 24 completed estimates from 60 proteins. PTEN has an estimated RMST contrast of 1.02 months per standard deviation and a 95% set of [-2.07, 4.24]; SERPINE1 and CCNE1 have negative point directions with real-line sets. Two ovarian proteins have bounded RMST sets; all sets include zero. These analyses connect molecular intervention questions to clinical effect scales and identifying information.

**Keywords:** proximal causal inference; idopNetwork; molecular networks; latent confounding; restricted mean survival time; outcome bridge

## 1 Introduction

Protein abundance is shaped by local regulation and systemic variation across patients. In tumour cohorts, purity, immune infiltration, stromal content and proliferation can influence many proteins together and also affect survival. A protein–outcome association therefore combines exposure effects with variation attributable to an incompletely measured biological state. Ovarian profiling documents stromal and mesenchymal variation [1] and multiple transcriptional subtypes [2]; intratumoral T cells are associated with survival [3]. These findings motivate measuring the shared biological state alongside an individual protein. Estimating the effect of changing that protein requires information about the state and a clinical intervention target.

idopNetwork [4, 5] reconstructs a molecular network from cross-sectional observations by ordering patients along a niche index, fitting protein-specific population curves, and decomposing the curves into self and cross-protein contributions. Proximal causal inference [6, 7, 8] uses treatment-inducing proxies $Z$ and outcome-inducing proxies $W$ of an unmeasured confounder. An outcome bridge connects these measurements to the outcome and identifies an intervention contrast. The methodological task for a molecular panel is to connect network structure to exposure-specific proxy roles and then to identifying bridge information.

We develop idopNetwork–proximal to estimate total protein-exposure effects from these molecular measurements. Exposure neighbourhoods and separated network components define candidate proxy roles. Structural conditions establish their exclusions, and an exposure-specific bridge identifies the total effect. Joint information learns systemic readout coordinates; conditional information selects a design that supplies the dimensions needed for causal estimation.

The clinical target is a change in restricted mean survival time (RMST) or survival probability at a specified horizon. Inverse censoring weights preserve the complete-data conditional mean, extending the bridge to censored outcomes. Recent proximal methods for counterfactual survival curves and censored event regression provide related approaches [9, 10].

The framework builds on negative-control designs [11, 12], proximal regression and semiparametric identification [6, 7, 8]. Fortified and adaptive proximal approaches [13, 14] address collections containing invalid proxies. We combine molecular design with independent discovery/estimation inference, evaluate network construction and readout learning in simulations, and study 36-month survival contrasts in ovarian cancer.

## 2 Setting and estimand

For patient $k$, observe a protein panel $X_k\in\mathbb R^p$, covariates $C_k$, and outcome $Y_k$. One protein $A=X_a$ is the exposure. The total intervention contrast is

$$
\Delta_A(a,a')=E[Y^a-Y^{a'}]=\tau_A(a-a').
$$

The last equality defines the constant-effect mean model over the exposure contrast under study. For survival, observe $O=\min(T,K)$ and $D=I(T\le K)$, where $T$ is event time and $K$ censoring time. The clinical targets at horizon $\ell$ are

$$
M_\ell(a)=E[\min(T^a,\ell)],\qquad S_a(\ell)=P(T^a>\ell).
$$

We estimate $M_\ell(a)-M_\ell(a')$ and $S_a(\ell)-S_{a'}(\ell)$ with target-specific linear mean bridges. The TCGA horizon is 36 months; a one-standard-deviation exposure contrast is expressed in months of restricted survival and in percentage points of survival probability. Probability mean models apply over contrasts within their probability range.

After accounting for $C$, represent the molecular system by

$$
X=X\Theta+F\Lambda^\top+E,\qquad \operatorname{diag}(\Theta)=0,
$$

with regulatory coefficients $\Theta$, unmeasured systemic state $F\in\mathbb R^r$, and loadings $\Lambda$. The structural derivation uses an acyclic graph with independent structural errors given $(F,C)$. For a linear outcome,

$$
Y=\gamma A+\sum_{v\ne a}\pi_vX_v+\lambda^\top F+d^\top C+\epsilon_Y,
\qquad
\tau_A=\gamma+\sum_{v\ne a}\pi_v\omega_{Av},\quad
\omega_{Av}=\sum_{\mathcal P:A\leadsto v}\prod_{(i,j)\in\mathcal P}\theta_{ij}.
$$

The path contribution is zero for a non-descendant. Substitution gives the exposure-specific mean model of Section 4.2 when its mean-zero error conditions hold. Thus the bridge retains downstream contributions to the total effect. The observed panel determines the niche ordering, representation, support, roles and information ranks; clinical outcomes enter estimation after this design is constructed.

## 3 Method

Figure 1 connects molecular representation, exposure-specific proxy information and clinical effect estimation. The application and simulation use the same network construction and bridge functions.

### 3.1 Niche index, power curves and deviations

Levels are standardised per protein and shifted so that the minimum of each is 1. The niche index of patient $k$ is $s_k=\sum_j X_{kj}$; sorting patients by $s$ turns the cross-section into quasi-dynamic data $X_j(s)$, and each protein is fitted by a power curve $c_j(s)=a_j s^{b_j}$ by least squares (Figure 2). The fitted curves $C(s)$ describe the population trend along the index; the deviations $U=X-C(s)$ carry the patient-specific information. The proximal analysis uses the deviations to estimate the latent state and runs the estimator itself on the standardised levels. Retaining observed patient levels in the bridge preserves variation around the population curves.

### 3.2 Molecular factor representation

Let $U_c$ be the centred patient deviations. The spectrum dimension is $\hat r_{\mathrm{PCA}}=\arg\max_{k\le8}\sigma_k/\sigma_{k+1}$. The retained singular vectors define $\hat F,\hat\Lambda$, with $U_c=\hat F\hat\Lambda^\top+\hat\Xi$. Report

$$
\rho_i=\|\hat\Xi_i\|/\|U_{c,i}\|,
$$

the residual-to-total norm ratio; $\rho_i^2$ is the residual variance fraction. Loadings with row norm exceeding $10^{-8}$ are eligible as systemic readouts. The spectrum dimension describes the panel representation, while Section 3.5 determines the bridge dimension from exposure-specific proxy information. Proposition 4 describes propagation of the latent factor space through the regulatory system.

### 3.3 Support across niche windows

For each width $w\in\{0.2,0.4\}$, construct five niche windows. Width 0.2 uses consecutive non-overlapping partitions; width 0.4 uses windows of $\lceil0.4n\rceil$ patients whose five starting indices are equally spaced and rounded to the nearest integer. Within each, fit every protein on the other proteins by nodewise LASSO with penalty $\alpha$, standardising predictors and response within the window [15, 16]. Retain a directed support entry when its nonzero coefficient frequency is strictly greater than 0.6. With $k=5$, this requires four windows. Symmetrise by taking the union with the transpose to obtain $G$. Frequency aggregation uses the idea of repeated support selection [17]; Appendix A.1 gives its quantile-window consistency argument.

The TCGA setting is $k=5$ for the 60 most variable proteins; $\alpha=0.15$ supplies the reference cohort graph. Both TCGA and simulations cross the two widths with penalties $(0.15,0.20,0.25,0.30)$, producing eight designs. All designs enter the information calculation before the outcome is used. Grid order is width 0.2 followed by width 0.4, with increasing penalty within each width. The one-window comparator applies the same penalty grid and remaining estimator steps to a single global nodewise fit.

### 3.4 Exposure-specific proxy construction

Let $\mathcal E_A$ be a prespecified set of treatment-proxy-eligible proteins. Define $Z_A=N_G(A)\cap\mathcal E_A$. The default includes all neighbours and is used in TCGA and the seven simulation systems. Structural annotations can exclude a known outcome-directed branch or mediator from $\mathcal E_A$ before estimation; the resulting boundary is the one used in Proposition 1. Define $W_A$ as all proteins outside the component containing $A$ and its neighbours, with eligible loading norm. These proteins form the full separated readout pool. The outcome-proxy pool can be larger than the treatment-proxy set: the reduced bridge needs $|W_A|\ge r_b$ and $|Z_A|\ge r_b$. Proposition 1 supplies the structural graph conditions giving these measurements their proximal roles. The eight cohort graphs are reused across exposures, while role sets, information dimension and selected penalty change with $A$.

### 3.5 Joint readout information and reduced bridge

For every design $d=(w,\alpha)$, residualise $W_d$ and $M_d=(A,Z_d^\top)^\top$ on $(1,C)$ to estimate their joint information rank $\hat q_d$.

Whiten each pair by its residual covariance matrices and compute canonical roots $\hat\rho_j$. For candidate rank $q$, the sequential statistic is

$$
B_q=-\left[n-\operatorname{rank}(D_0)-\frac{p_W+p_M+1}{2}\right]\sum_{j>q}\log(1-\hat\rho_j^2),
\qquad d_q=(p_W-q)(p_M-q).
$$

Here $D_0$ is the residualisation design and $p_M$ is the residual predictor dimension. Compare $B_q$ with the chi-square reference of $d_q$ degrees of freedom at level $\alpha_n=\min(0.05,n^{-1/2})$, starting at $q=0$ and stopping at the first accepted rank, with an upper bound of eight. Bartlett's reference gives the Gaussian calibration; Proposition 3 establishes consistency through a root-$n$ covariance expansion. Determine $\hat r_*=\max_d\hat q_d$. For each candidate with $\hat q_d=\hat r_*>0$ and $\min(|W_d|,|Z_d|)\ge\hat r_*$, construct the joint projection below and estimate the conditional information rank $\hat k_d$ of its observed reduced readouts and $Z_d$ after residualising $(1,A,C)$. Define

$$
\widehat{\mathcal E}=\{d:\hat q_d=\hat r_*,\ \hat k_d=\hat r_*,\ \min(|W_d|,|Z_d|)\ge\hat r_*\}.
$$

Among these candidates choose the largest smallest conditional canonical root $\hat\gamma_d=\hat\rho_{\hat r_*,d}^{\mathrm{conditional}}$. Fixed grid order resolves exact ties. If this set is empty, retain the molecular design record with its missing identifying dimension; an effect is fitted only when a candidate supplies the full conditional dimension. The joint rank determines readout dimension, and conditional information determines eligibility and quality. Proposition 3b gives consistency under existence of a conditionally complete candidate.

Fit $\hat W=M\hat\Pi$, with $M=(1,C,A,Z)$. Residualise the fitted readouts on $(1,C)$ and take their leading $\hat r_*$ right singular vectors $P$. Thus exposure contributes to learning the readout coordinates. Define observed readouts $R=WP$ and fitted readouts $\hat R=\hat WP$. Regress the clinical outcome or its censored pseudo-outcome on $(1,C,A,\hat R)$ and report the exposure coefficient. Exposure remains explicit in the outcome equation. Proposition 3a shows that the joint reduction preserves the total-effect bridge.

For either the raw readouts $V=W$ or the reduced readouts $V=R$, record

$$
\nu(V,Z\mid A,C)=\frac{\sigma_{\hat r_*}\{\operatorname{Cov}(\tilde V,\tilde Z)\}}{\|\operatorname{sd}(\tilde V)\|\,\|\operatorname{sd}(\tilde Z)\|},
$$

where tildes denote linear residuals on $(1,A,C)$. Record the conditional canonical roots and sequential rank for $R,Z$ as well. These quantities describe identifying information [18, 19]; the design rule uses dimension and conditional roots without a fixed strength cutoff. Numerical estimability requires the reduced outcome design to have full column rank and the retained joint singular values to exceed a relative tolerance of $10^{-8}$.

### 3.6 Censored clinical outcomes and inference

Estimate $G(t\mid A,C)=P(K>t\mid A,C)$ from a Cox censoring model on $(A,C)$, with censoring event $1-D$ and Breslow baseline [20]. Form $V_\ell=\int_0^\ell I(O>u)/\hat G(u\mid A,C)\,du$ for RMST and $V_\ell^S=I(O>\ell)/\hat G(\ell\mid A,C)$ for survival probability. Each enters the same linear bridge. Proposition 5 identifies the clinical contrasts under conditional independent censoring and the target-specific mean model. Evaluate positivity through 36 months, with a minimum fitted censoring survival of 0.05.

### 3.7 Concentrated bridge confidence sets

For an effect confidence set, remove $(1,C)$ and form $Q=(A,Z)$ using all treatment-proxy coordinates. Pivoted QR of the joint cross-moment selects $r$ nuisance coordinates $T$ with a nonsingular $B=E[TR^\top]$. Retain every remaining coordinate in $U$, with $m=1+|Z|-r$. The bridge gives

$$
L=E[UR^\top]B^{-1},\quad
N=E[UY]-LE[TY],\quad
D=E[UA]-LE[TA],\quad N=\tau_A D.
$$

In the independent workflow, resample estimation patients to estimate the joint covariance of $(\hat N,\hat D)$, refitting covariate projections and censoring while the discovery readout coordinates, roles, dimension and nuisance pivots remain fixed. Invert each component's Fieller inequality at level $\alpha/m$ and intersect the component sets. Theorem 3 gives the moment-inversion construction, and Theorem 4 establishes its independent-estimation law. The point estimator is $(\hat D^\top\hat N)/(\hat D^\top\hat D)$ when defined; bounded, disconnected, unbounded and empty confidence sets describe the available effect information.

### 3.8 Independent discovery and estimation

The inference workflow partitions patients once into 75% discovery and 25% estimation samples, independently of outcomes. Discovery determines molecular filtering, panel selection, median imputations, standardisation, niche curves and eight network designs. The design rule selects roles, dimension and window/penalty; its joint readout coordinates and nuisance moment pivots are then frozen. Estimation patients use these transformations and fixed observed readouts. A clinical exposure contrast is one discovery-sample standard deviation.

Fieller intersection of the estimation moments supplies the effect confidence set. An unavailable discovery design, numerical nuisance failure or fewer than 97.5% complete moment resamples returns the whole real line and no point estimate. Reporting statuses distinguish these failures from real-line sets produced by weak effect information.

The allocation and partition seed are fixed across analyses. Theorem 4 provides inference conditional on the discovery design.

### 3.9 Method outputs

Each exposure record contains its proxy sets, spectrum and bridge dimensions, selected window and penalty, conditional roots and rank, and raw and reduced strength. Estimation adds the effect coefficient, censoring support, resampling completion and confidence-set shape. Reporting rates and independent set coverage use all attempts; point accuracy uses completed fits. Coverage among available fits is also reported.

### 3.10 Quasi-dynamic network interpretation

The weak-form ODE decomposition of idopNetwork describes molecular variation along the niche index: $\dot x_j=Q_{jj}(x_j)+\sum_{k\in\operatorname{pa}(j)}Q_{jk}(x_k)$. On the support graph, the $Q$'s are expanded in an integral Legendre basis of order zero and fitted to the power-curve samples by least squares. The signed weights describe self and cross-protein contributions to the niche curves (Figure 3c; Figures S1–S2). Bridge estimation uses the observed protein levels and reduced readouts defined in Section 3.5.

## 4 Identification and estimation

### 4.1 Structural conditions for network-derived proxies

Condition on measured covariates $C$. Assume consistency, positivity over the exposure contrast of interest, and latent exchangeability $Y^a\perp A\mid(F,C)$. The underlying molecular system is an acyclic structural model with independent errors given $(F,C)$. Let $H$ be the moral graph [21] of its observed-protein DAG after conditioning on $(F,C)$: it joins a parent to its child and joins parents sharing a child. Let $\mathcal O$ be the observed direct parents of the clinical outcome. For survival, these are the observed parents of the uncensored event time. Structural conditions concern this patient-level system; the ODE representation describes niche-level curve contributions.

Let $G_0$ be the population limit of the selected undirected support, and construct $Z_A,W_A$ by the rule in Section 3.4. Two graph conditions connect this construction to the causal model:

- **N1, structural coverage:** $H\subseteq G_0$.
- **N2, exposure boundary:** after removing $A$ from $G_0$, no path joins $Z_A$ to $\mathcal O\setminus\{A\}$.

N1 permits additional edges, which can merge components and reduce the separated proxy pool. N2 specifies the outcome-facing boundary of the eligible exposure neighbourhood. Eligible neighbours can include ancillary children of $A$ that measure the systemic state. Known outcome-directed branches are excluded from Z and remain in the exposure component, outside W; their pathways contribute to the total effect. Under the all-neighbour rule, N2 applies to every observed neighbour.

**Proposition 1 (network-to-proxy validity).** Under N1 and N2, the selected roles satisfy

$$
Y\perp Z_A\mid(A,F,C),\qquad W_A\perp(A,Z_A)\mid(F,C).
$$

The proof is given in Appendix A.4.

Separation establishes the proxy exclusions. Identification additionally requires an outcome bridge and completeness [6, 7, 8]; the linear model below expresses these requirements through loadings and proxy cross-covariance rank.

### 4.2 Explicit bridge and total-effect identification

Write the exposure-specific mean model and proxy equation as

$$
\begin{aligned}
Y&=\mu+\tau_A A+b^\top F+d^\top C+\epsilon_Y,\\
W&=\alpha_W+L_WF+D_WC+\epsilon_W,\\
\operatorname{Proj}(F\mid1,A,Z,C)&=k_0+k_AA+K_ZZ+K_CC.
\end{aligned}
$$

Here $\operatorname{Proj}$ denotes the population linear projection. Assume $E[\epsilon_Y\mid A,F,Z,C]=E[\epsilon_W\mid A,F,Z,C]=0$, and that the first equation also represents the intervention mean. $L_W$ contains propagated outcome-proxy loadings. Assume $\operatorname{rank}(L_W)=\operatorname{rank}(K_Z)=r$, and a nonsingular second-moment matrix of $(1,A,Z,C)$. The conditional distribution of $F$ can be non-Gaussian; its linear projection supplies the moments used by the estimator.

**Proposition 2 (explicit bridge and unique exposure effect).** For any $\eta_W$ solving $L_W^\top\eta_W=b$, the function

$$
h(W,A,C)=\mu-\eta_W^\top\alpha_W+\tau_A A+
\eta_W^\top W+(d-D_W^\top\eta_W)^\top C
$$

satisfies

$$
E[Y\mid A,Z,C]=E[h(W,A,C)\mid A,Z,C],\qquad
E[Y^a]=E[h(W,a,C)].
$$

Its coefficient on $A$ is uniquely $\tau_A$, including when $\eta_W$ is non-unique.

The proof is given in Appendix A.4.

The coefficient $\tau_A$ includes the downstream path contributions defined in Section 2. For any nonsingular $R$, changing coordinates to $RF$ transforms the loadings to $L_WR^{-1}$, the outcome coefficient to $R^{-\top}b$, and the projection coefficient to $RK_Z$. The bridge and exposure effect are invariant to this choice of factor coordinates.

**Table 1.** Components of network-guided proximal identification.

| Component | Role | Mathematical condition |
|---|---|---|
| Exposure neighbourhood $Z_A$ | Connects exposure variation to the systemic state | N2; $\operatorname{rank}(K_Z)=r$ |
| Separated components $W_A$ | Measure the pre-intervention systemic state | N1; $\operatorname{rank}(L_W)=r$ |
| Proxy-information rank | Determines the bridge dimension | $\operatorname{rank}\{\operatorname{Cov}(\tilde W,\tilde Z)\}=r$ |
| Outcome bridge | Transfers proxy information to the clinical mean | $L_W^\top\eta_W=b$ |
| Exposure coefficient | Gives the total intervention contrast | $\eta_A=\tau_A$ |

### 4.3 Rank of identifying proxy information

Let $\tilde W,\tilde Z$ be the residuals of $W,Z$ after population linear projection on $(1,A,C)$.

**Proposition 3 (proxy rank and its estimation).** Under Proposition 2,

$$
\operatorname{Cov}(\tilde W,\tilde Z)=L_WK_Z\operatorname{Cov}(\tilde Z).
$$

If $\operatorname{Cov}(\tilde Z)$ is nonsingular, this matrix has rank $r$. Let $\check W$ and $\check M$ be residuals of $W$ and $M=(A,Z^\top)^\top$ on $(1,C)$. Their joint covariance also has rank $r$:

$$
\operatorname{Cov}(\check W,\check M)=L_W\operatorname{Cov}(\check F,\check M).
$$

Whitening by nonsingular residual covariance matrices preserves these ranks. For fixed proxy dimensions, $r\le8$, positive definite residual covariance matrices, root-$n$ consistent covariance estimates, separated positive canonical roots, and $\alpha_n=\min(0.05,n^{-1/2})$, the sequential rules satisfy $P(\hat r_{\mathrm{joint}}=\hat r_b=r)\to1$.

The proof is given in Appendix A.4.

The joint rank describes systemic directions measured by W and (A,Z); conditional rank describes directions that Z supplies beyond exposure. Under the common systemic mean equations and proxy-error orthogonality, each candidate joint rank is at most the systemic dimension. A candidate spanning all directions determines the grid dimension. Conditional information then selects an identified design through Proposition 3b.

**Proposition 3a (joint readout reduction).** Under Proposition 2, let $\overline W$ be the population linear projection of $W$ on $(1,A,Z,C)$. The covariance of $\overline W$ residualised on $(1,C)$ has rank $r$ and range $\operatorname{col}(L_W)$. For an orthonormal basis $P$ of that range, $P^\top L_W$ is invertible. The reduced readout $R=P^\top W$ has a linear outcome bridge with unique exposure coefficient $\tau_A$.

The proof is given in Appendix A.4.

This reduction uses joint information to estimate readout coordinates and conditional information to identify the effect. A factor direction strongly measured by exposure can have a well-separated joint readout space while its treatment-proxy information is weaker after exposure residualisation. The population reductions span the same space under full rank; their finite-sample signal differs.

For the finite design set $\mathcal D$, let $q_d$ be the joint readout rank and
$r=\max_dq_d$. Let $P_d$ span the joint fitted-readout space at dimension $r$
for candidates with $q_d=r$. Write $R_d=P_d^\top W_d$, and let $k_d$ be the
rank of $\operatorname{Cov}(R_d,Z_d\mid1,A,C)$. Define

$$
\mathcal E=\{d:q_d=r,\ k_d=r,\ \min(|W_d|,|Z_d|)\ge r\},
\qquad
\gamma_d=\sigma_r\{\operatorname{Cor}(\widetilde R_d,\widetilde Z_d)\}.
$$

Correlation here means covariance whitening of each residual view, with
tildes removing $(1,A,C)$. The population design maximises $\gamma_d$ within
$\mathcal E$; the algorithm estimates ranks and roots with fixed grid order
for identical designs and exact ties.

**Proposition 3b (selection of an identified design).** Assume the common
systemic mean model and proxy-error orthogonality of Proposition 2 apply to
every limiting candidate. All joint ranks are at most $r_0$, the systemic
dimension. Suppose a candidate has joint rank $r_0$, full outcome-proxy
loading rank $r_0$, and conditional treatment-proxy rank $r_0$. Assume the
joint spaces are separated, rank decisions are consistent, and $\gamma_d$
has a unique maximum among distinct role designs in $\mathcal E$. Then the
empirical eligible set converges to $\mathcal E$, its dimension converges to
$r_0$, and the selected design identifies the total-effect coefficient. Its
readout bridge is that of Proposition 3a.

The proof is given in Appendix A.4.

### 4.4 Propagated factors and network information

Write $U=U\Theta+F\Lambda^\top+E$ and $B=(I-\Theta)^{-1}$. Assume $\operatorname{Var}(F)=I_r$, $F$ independent of $E$, and positive diagonal $\Psi=\operatorname{Var}(E)$.

**Proposition 4 (factor propagation and precision structure).** If $F$ and $\Lambda$ have column rank $r$, the signal $F\Lambda^\top B$ has column space $\operatorname{span}(F)$. The precision of a molecular row is

$$
\Omega=(I-\Theta)\Psi^{-1}(I-\Theta)^\top
-\tilde\Lambda(I_r+\Lambda^\top\Psi^{-1}\Lambda)^{-1}\tilde\Lambda^\top,
\qquad \tilde\Lambda=(I-\Theta)\Psi^{-1}\Lambda.
$$

The second term is positive semidefinite and has rank at most $r$. The first term has support contained in the moral graph of $\Theta$, with equality in the absence of coefficient cancellations.

The proof is given in Appendix A.4.

The formula explains why the selected support and the latent representation supply complementary information. The systemic state contributes to observed nodewise associations and also creates the cross-proxy signal used by the bridge. N1 specifies structural coverage by the population selection graph; it is an explicit condition on the selected graph in the presence of this latent contribution [22].

### 4.5 Construction and large-sample estimation

For each fixed niche window, define its population nodewise LASSO using the conditional distribution within the prespecified niche-index quantile intervals of each window family. Assume a continuous index distribution, positive window probability, nonsingular within-window predictor covariance, and a positive margin for both active coefficients and inactive KKT inequalities. Assume the preprocessing and loading-eligibility decisions have stable population limits. Appendix A.1 shows that empirical niche-window supports then converge jointly to their population supports. The finite frequency aggregation, prespecified treatment eligibility and fixed protein ordering give stable proxy sets. Assume the limiting candidate designs obey the systemic mean and proxy-error orthogonality conditions in the finite-grid conditions following Proposition 3. N1 and N2 supply sufficient structural conditions for these roles. Assume at least one candidate spans all $r$ joint directions and supplies full conditional information. Proposition 3b selects within that complete candidate set. A positive margin for its smallest conditional-root quality among distinct role designs makes selection stable; its limiting dimension is $r$.

Let $m=(1,A,Z^\top,C^\top)^\top$ and let $H_n$ collect the distinct sample moments of $mm^\top,mW^\top,mY$. After the roles and rank stabilise, least squares, projection onto the retained first-stage space, and second-stage regression define $\hat\tau_A=f(H_n)$. Assume iid patients with finite fourth moments, nonsingular $E[mm^\top]$, a separated rank-$r$ first-stage subspace, a full-rank reduced second-stage design, and positive population conditional information. These are pointwise regularity conditions at a fixed identified model.

**Theorem 1 (construction and effect estimation).** Under the preceding construction and identification conditions, the selected roles and $\hat r_b$ converge to their population values, $\hat\tau_A\to_p\tau_A$, and

$$
\sqrt n(\hat\tau_A-\tau_A)\longrightarrow N(0,J\Sigma_qJ^\top),
\qquad J=\nabla f(H),\quad \Sigma_q=\operatorname{Var}(q_i),\quad H=E[q_i].
$$

Here $q_i$ is the patient moment vector. The gradient includes the first-stage fit and retained projection. Appendix A.2 gives the proof.

**Corollary 1 (patient bootstrap).** Holding the selected roles and rank fixed and refitting both bridge stages and the projection consistently estimates the limiting distribution under Theorem 1's construction conditions.

**Corollary 1b (complete construction bootstrap).** In addition, suppose the within-window empirical moments, fitted curves and preprocessing have bootstrap consistency. Rebuilding supports, roles, ranks and conditional design selection inside each patient bootstrap, then refitting the bridge, consistently estimates the same limiting distribution. Under Theorem 2 the bootstrap also refits censoring. The derivative includes every smooth fitted stage; positive support and selection margins stabilise the discrete stages. Appendix A.2 gives the argument.

### 4.6 Causal survival targets through a censored-outcome bridge

Let $T$ be event time, $K$ censoring time, $O=\min(T,K)$, and $D=I(T\le K)$. Define $G(t\mid A,C)=P(K>t\mid A,C)$. Assume $K\perp(T,F,Z,W)\mid(A,C)$ and $G$ bounded away from zero through the horizon $\ell$. For a complete-data target $L_\ell=\min(T,\ell)$ or $L_t=I(T>t)$, assume the intervention mean and conditional mean have the form in Proposition 2, with target-specific coefficients $\mu_L,\tau_{A,L},b_L,d_L$.

**Proposition 5 (restricted-survival identification).** The observed-data variables

$$
V_\ell=\int_0^\ell\frac{I(O>u)}{G(u\mid A,C)}\,du,
\qquad V_t=\frac{I(O>t)}{G(t\mid A,C)}
$$

have the same conditional means given $(A,F,Z,C)$ as $\min(T,\ell)$ and $I(T>t)$, respectively. Their proximal bridges identify

$$
M_\ell(a)=E[\min(T^a,\ell)],\qquad S_a(t)=P(T^a>t),
$$

and the contrasts satisfy

$$
M_\ell(a)-M_\ell(a')=\tau_{A,\ell}(a-a'),\qquad
S_a(t)-S_{a'}(t)=\tau_{A,t}(a-a').
$$

The proof is given in Appendix A.4.

The horizon-specific mean bridge permits an unrestricted event-time hazard.

**Theorem 2 (survival estimation and joint bootstrap).** In addition to Theorem 1's moment and construction conditions, suppose the censoring estimator has a regular asymptotically linear expansion in the function space used by $V_L$, with corresponding bootstrap validity. Positivity and differentiability of the inverse-censoring map then give

$$
\sqrt n(\hat\tau_{A,L}-\tau_{A,L})
=n^{-1/2}\sum_i\left[J\{q_i(G)-H\}+\mathcal D_G\varphi_{G,i}\right]+o_p(1).
$$

Here $\varphi_{G,i}$ is the censoring-model influence function and $\mathcal D_G$ differentiates the effect through the pseudo-outcome. The patient bootstrap that refits $G$, the first stage, projection and outcome bridge consistently estimates the joint limiting distribution. Appendix A.3 gives the derivative and proof. The application uses a Cox model for censoring on $(A,C)$ and its Breslow baseline; the RMST simulation uses independent censoring and reverse Kaplan–Meier.

### 4.7 Concentrated moment inference

For residualised instruments and readouts, define $T,U,B,L,N,D$ as in Section 3.7. The nuisance bridge moments give $E[TY]=\tau_A E[TA]+B\eta$; substituting into the remaining moments establishes $N=\tau_A D$. If $D^\top D>0$, concentration identifies the same total-effect coefficient. Joint information can keep the nuisance block regular when the final effect-information direction is weak.

**Theorem 3 (concentrated bridge confidence set).** Fix valid molecular roles
and dimension, or suppose their selection stabilises as in Proposition 3b.
Suppose the joint readout space and $B$ are
regular and separated, the nuisance pivot choices have a stable population
limit, and the patient moments have finite fourth moments. For survival also
assume the censoring and positivity conditions of Theorem 2. Let
$\widehat\theta=(\widehat N^\top,\widehat D^\top)^\top$ have a joint central
limit theorem with consistently estimated sampling covariance $\widehat V$.
For each $j$, let $\widehat V_j$ be the covariance submatrix of
$(\widehat N_j,\widehat D_j)$, and assume nonzero variance of its true-effect
null moment. For fixed $m$, set

$$
\mathcal C_{1-\alpha}=\bigcap_{j=1}^m\left\{t:
(\widehat N_j-t\widehat D_j)^2\le
z_{1-\alpha/(2m)}^2
\begin{pmatrix}1&-t\end{pmatrix}
\widehat V_j
\begin{pmatrix}1\\-t\end{pmatrix}\right\}.
$$

Then $\liminf\Pr(\tau_A\in\mathcal C_{1-\alpha})\ge1-\alpha$.
The statement is uniform along sequences with a uniform joint moment CLT,
uniform covariance consistency, a regular separated nuisance block and joint
readout space, stable nuisance pivots, and uniformly nondegenerate null-moment
variances, including sequences with $D\to0$.

The proof is given in Appendix A.4.

Fieller inversion [23] applies to each concentrated moment. The Bonferroni intersection gives coverage through the union bound, allowing dependence among components.

### 4.8 Independent discovery and clinical estimation

Let $\mathcal D$ contain discovery patients and $\mathcal I$ contain independent estimation patients. Molecular preprocessing, the panel, the support grid, proxy roles, dimension, readout matrix $P_{\mathcal D}$ and nuisance moment coordinates are functions of $\mathcal D$. They are fixed when analysing $\mathcal I$. Write $\mathcal G_n$ for the event that an available discovery design has valid proxy roles, dimension $r_0$, $\operatorname{rank}(P_{\mathcal D}^{\top}L_W)=r_0$, and a regular population nuisance block. A discovery design without sufficient information returns $\mathbb R$ and supplies no point estimate.

**Theorem 4 (inference from independent estimation patients).** Conditional on a discovery design in $\mathcal G_n$, suppose Proposition 2 holds for its selected proxies, and the concentrated moment vector in estimation patients has a uniform joint central limit law with a consistently estimated covariance. Assume a fixed upper bound on the number of moment coordinates, uniformly nonsingular covariate projections, a nuisance block with smallest singular value bounded away from zero, and null-moment variances bounded away from zero. For censored targets, impose Proposition 5 and the correctly specified, regular censoring fit of Theorem 2, refitted only in estimation patients and their resamples. Then the intersected Fieller set $\mathcal C_{\mathcal I}(\mathcal D)$ satisfies

$$
\inf_{\mathcal D\in\mathcal G_n}P\{\tau_A\in\mathcal C_{\mathcal I}(\mathcal D)\mid\mathcal D\}\ge1-\alpha-o(1).
$$

The effect-information vector may approach zero. There is no requirement that the winning molecular design converge to one particular candidate. If an available invalid or dimension-incomplete discovery design occurs with probability at most $\varepsilon_n$, returning $\mathbb R$ for unavailable designs and numerical failures gives

$$
P\{\tau_A\in\mathcal C_{\mathcal I}(\mathcal D)\}\ge1-\alpha-\varepsilon_n-o(1).
$$

In particular, consistent discovery of valid complete roles with $\varepsilon_n\to0$ yields unconditional asymptotic coverage at least $1-\alpha$. Covariance consistency includes an increasing number of moment resamples in the asymptotic statement.

The proof is given in Appendix A.5.

## 5 Simulation

### 5.1 Molecular systems and causal targets

The first five fixed acyclic systems contain one exposure, six designated treatment proxies, six designated outcome proxies and 18 additional proteins. Set $X=(\ell_F F\Lambda^\top+E)(I-\Theta)^{-1}$, with independent factor coordinates and structural errors uniform on $(-\sqrt3,\sqrt3)$; proxy-error scales are 0.5 and other scales are 1. Background regulation varies among a branch, chain and three modules, with signed edges of absolute strength 0.3–0.7. Systems I–III have $r=1$, sample sizes 400, 1000 and 1000, and loading scales 0.25, 0.25 and 0.50. Systems IV–V have $r=2$, $n=1000$, and scales 0.25 and 0.35. Their fixed loading seeds generate, respectively, a weak second conditional proxy direction and two stronger directions. Systems VI–VII extend the branch systems to 60 proteins, with 47 background proteins. System VI has $n=411$, $r=1$, scale 0.25 and the loading seed of system I; system VII has $n=352$, $r=2$, scale 0.35 and the loading seed of system V. Their panel and sample sizes correspond to OV and the supplementary LUAD analysis. All matrices and seeds are supplied in the code.

For linear effects, set $Y=0.5A+X\pi+F\lambda_Y+\epsilon_Y$, with independent standard normal outcome error. Outcome-directed background proteins are non-descendants of the exposure, so the total-effect target is 0.5 per raw exposure unit. For censored RMST, generate early event times uniformly between 10 and 14 months and late times between 55 and 65 months. The early-event probability is $0.5-(1.5/24)A-b_F^\top F-\pi_S^\top X$, with coefficient bounds ensuring values strictly between zero and one. Thus

$$
E[\min(T,36)\mid A,F,X]=24+1.5A+24b_F^\top F+24\pi_S^\top X,
$$

and the total RMST effect is exactly 1.5 months per raw exposure unit. Censoring is independent exponential with mean 100 months; reverse Kaplan–Meier supplies the censoring weights. Network construction uses standardised protein levels, while effect regressions retain the raw exposure so that all estimators use the same raw-unit target.

### 5.2 Comparators and evaluation

The representation study isolates network construction and readout learning using a joint-information selector; Section 5.4 evaluates the independent workflow with conditional-information selection. Each of the seven systems uses 200 independent datasets, with replicate seed offset 30000, and 100 patient-bootstrap draws for each outcome. Linear and RMST targets share the molecular observations. Ten estimators analyse identical patients and targets: Joint-quality readout; conditional projection with the same selected roles and rank; the multiscale grid bridge; joint readout using one global nodewise window and the same penalties; correlation-selected proxies; designated joint readout using the true proxy blocks and rank; designated proximal regression using conditional projection with the same known blocks and rank; unadjusted regression; ordinary factor adjustment; and adjustment for true factors. Correlation selection assigns the six largest absolute exposure correlations to Z and the six smallest remaining correlations to W, with joint rank selected by the same sequential rule. The factor estimator chooses dimension by the multiplicative spectral gap and refits factors in each bootstrap draw. The designated estimators isolate readout learning with known valid roles; single-window and correlation comparisons isolate molecular design.

Bootstrap draws hold selected roles and rank fixed and refit the readout projection, both regressions and censoring as appropriate. Evaluation includes fit and interval completion, generating-DAG proxy validity, dimension accuracy, bias, RMSE, 95% coverage and mean interval width. Monte Carlo standard errors accompany effect accuracy and coverage [24]; CSV summaries also give Wilson intervals for reporting and coverage. Bias and RMSE use reported fits; coverage uses completed intervals. Accuracy measures with fewer than five fits are left blank. Rank strata and paired squared-error differences appear in the supplementary results.

### 5.3 Joint representation results

**Table 2.** Joint-quality representation estimator in seven systems. Accuracy entries are estimate (MCSE); rank accuracy uses reported designs.

| Outcome | System | Reported | Bias | RMSE | Coverage | Correct rank (%) |
| --- | --- | --- | --- | --- | --- | --- |
| LINEAR | I | 200/200 | 0.008 (0.006) | 0.087 (0.006) | 0.935 (0.017) | 95.0 |
| LINEAR | II | 200/200 | -0.006 (0.006) | 0.079 (0.004) | 0.950 (0.015) | 92.5 |
| LINEAR | III | 200/200 | -0.001 (0.003) | 0.049 (0.003) | 0.965 (0.013) | 97.5 |
| LINEAR | IV | 200/200 | 0.013 (0.012) | 0.164 (0.010) | 0.965 (0.013) | 95.5 |
| LINEAR | V | 200/200 | 0.012 (0.006) | 0.081 (0.010) | 0.960 (0.014) | 94.0 |
| LINEAR | VI | 200/200 | 0.004 (0.008) | 0.112 (0.008) | 0.940 (0.017) | 96.0 |
| LINEAR | VII | 200/200 | 0.012 (0.009) | 0.121 (0.009) | 0.955 (0.015) | 83.5 |
| RMST | I | 200/200 | -0.027 (0.056) | 0.787 (0.041) | 0.930 (0.018) | 95.0 |
| RMST | II | 200/200 | -0.075 (0.034) | 0.481 (0.023) | 0.945 (0.016) | 92.5 |
| RMST | III | 200/200 | 0.002 (0.036) | 0.512 (0.028) | 0.925 (0.019) | 97.5 |
| RMST | IV | 200/200 | 0.100 (0.096) | 1.356 (0.079) | 0.965 (0.013) | 95.5 |
| RMST | V | 200/200 | 0.009 (0.039) | 0.545 (0.028) | 0.940 (0.017) | 94.0 |
| RMST | VI | 200/200 | 0.004 (0.047) | 0.665 (0.030) | 0.975 (0.011) | 96.0 |
| RMST | VII | 200/200 | -0.015 (0.060) | 0.840 (0.043) | 0.970 (0.012) | 83.5 |

The joint-readout estimator reports 200/200 fits in every system and outcome. Linear-effect coverage ranges from 0.935 to 0.965 and RMST coverage from 0.925 to 0.975. Selected proxy pairs satisfy the generating-DAG exclusions in all seven systems. Rank accuracy ranges from 0.835 to 0.975; Table S4 gives the rank strata.

System IV isolates the contribution of joint readout under weak two-factor information. With the same selected roles and dimension, joint readout gives linear bias 0.013 (MCSE 0.012), compared with 0.203 (0.010) for conditional projection. Coverage increases from 0.625 to 0.965. The paired mean squared error difference, joint minus conditional, is -0.0353 (MCSE 0.0048; 200 pairs). For RMST, coverage increases from 0.900 to 0.965, while RMSE increases from 0.884 to 1.356 and interval width from 2.869 to 5.342 months. Joint readout therefore improves linear bias and coverage in this system with a precision cost for the censored target.

**Table 3.** Paired linear-effect comparisons in the modular, weak-direction and 60-protein two-factor systems. Full comparisons with MCSEs appear in Tables S2–S3.

| System | Estimator | Reported | Bias | RMSE | Coverage |
| --- | --- | --- | --- | --- | --- |
| III | Joint-quality readout | 200/200 | -0.001 | 0.049 | 0.965 |
| III | Same design, conditional projection | 200/200 | 0.002 | 0.049 | 0.965 |
| III | Previous grid bridge | 200/200 | 0.007 | 0.064 | 0.960 |
| III | One global window | 191/200 | 0.107 | 0.195 | 0.869 |
| III | Correlation proxies | 200/200 | 0.118 | 0.184 | 0.845 |
| III | Designated joint readout | 200/200 | 0.000 | 0.052 | 0.955 |
| III | Designated proximal | 200/200 | 0.001 | 0.052 | 0.955 |
| III | Unadjusted | 200/200 | 0.547 | 0.550 | 0.000 |
| III | Factor adjustment | 200/200 | -0.669 | 0.672 | 0.000 |
| III | Oracle factors | 200/200 | -0.003 | 0.050 | 0.945 |
| IV | Joint-quality readout | 200/200 | 0.013 | 0.164 | 0.965 |
| IV | Same design, conditional projection | 200/200 | 0.203 | 0.249 | 0.625 |
| IV | Previous grid bridge | 2/200 | — | — | — |
| IV | One global window | 200/200 | 0.013 | 0.163 | 0.965 |
| IV | Correlation proxies | 200/200 | 0.512 | 0.516 | 0.000 |
| IV | Designated joint readout | 200/200 | 0.014 | 0.182 | 0.950 |
| IV | Designated proximal | 200/200 | 0.078 | 0.170 | 0.915 |
| IV | Unadjusted | 200/200 | 0.511 | 0.515 | 0.000 |
| IV | Factor adjustment | 200/200 | -1.264 | 1.291 | 0.045 |
| IV | Oracle factors | 200/200 | -0.007 | 0.057 | 0.935 |
| VII | Joint-quality readout | 200/200 | 0.012 | 0.121 | 0.955 |
| VII | Same design, conditional projection | 200/200 | 0.038 | 0.116 | 0.940 |
| VII | Previous grid bridge | 197/200 | 0.043 | 0.163 | 0.914 |
| VII | One global window | 173/200 | 0.140 | 0.307 | 0.792 |
| VII | Correlation proxies | 200/200 | 0.509 | 0.519 | 0.025 |
| VII | Designated joint readout | 200/200 | -0.009 | 0.094 | 0.955 |
| VII | Designated proximal | 200/200 | -0.007 | 0.093 | 0.950 |
| VII | Unadjusted | 200/200 | 0.513 | 0.523 | 0.005 |
| VII | Factor adjustment | 200/200 | 0.052 | 0.752 | 0.920 |
| VII | Oracle factors | 200/200 | -0.008 | 0.086 | 0.940 |

The modular system III demonstrates the contribution of network-based proxy construction. Linear RMSE is 0.049 for joint readout, 0.195 for a single global window and 0.184 for correlation-selected proxies. Paired mean squared error differences are -0.0356 (MCSE 0.0038; 191 pairs) and -0.0313 (0.0026; 200 pairs). The corresponding reporting counts are 200/200, 191/200 and 200/200.

In the 60-protein two-factor system VII, linear RMSE is 0.121, compared with 0.307 for the global-window design and 0.519 for correlation proxies. Paired differences are -0.0802 (0.0096; 173 pairs) and -0.2544 (0.0072; 200 pairs), with reporting counts 200/200, 173/200 and 200/200. Global and niche-window designs perform similarly in system IV. The known-role proximal reference has RMSE 0.093 in system VII, quantifying the precision available when valid proxy blocks and dimension are supplied. Tables S2–S4 and the complete paired-comparison records cover all seven systems and both outcomes.

### 5.4 Independent discovery and estimation

With the allocation and moment rules fixed, each of the seven systems generates 200 fresh datasets at offset 140000 and receives one 75:25 partition. Discovery executes molecular construction; estimation computes both targets and 100 patient resamples of concentrated moments, refitting censoring for RMST. Point errors use the raw generating exposure unit. The paired normal reference uses the same estimation patients, point estimator and covariance. Coverage counts all 200 sets, including real-line, disconnected and empty sets (Table 4; Figure 4).

**Table 4.** Independent discovery and estimation. Parentheses give Monte Carlo standard errors. Rank accuracy and effect errors use point fits; set coverage and bounded sets use all 200 attempts.

| Target | System | Point fits | Correct rank | Bias (MCSE) | RMSE (MCSE) | Set coverage | Bounded sets |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LINEAR | I | 200/200 | 0.920 | 0.023 (0.012) | 0.169 (0.008) | 0.985 (0.009) | 135/200 |
| LINEAR | II | 200/200 | 0.960 | -0.012 (0.011) | 0.153 (0.007) | 0.990 (0.007) | 190/200 |
| LINEAR | III | 197/200 | 0.959 | 0.001 (0.008) | 0.112 (0.005) | 0.950 (0.015) | 187/200 |
| LINEAR | IV | 136/200 | 0.949 | 0.074 (0.030) | 0.353 (0.036) | 0.965 (0.013) | 21/200 |
| LINEAR | V | 200/200 | 0.955 | 0.001 (0.010) | 0.142 (0.008) | 0.970 (0.012) | 186/200 |
| LINEAR | VI | 200/200 | 0.905 | 0.022 (0.017) | 0.239 (0.023) | 0.995 (0.005) | 141/200 |
| LINEAR | VII | 194/200 | 0.840 | -0.022 (0.018) | 0.256 (0.024) | 0.995 (0.005) | 142/200 |
| RMST | I | 200/200 | 0.920 | -0.003 (0.112) | 1.580 (0.070) | 0.990 (0.007) | 133/200 |
| RMST | II | 200/200 | 0.960 | 0.082 (0.068) | 0.956 (0.047) | 0.970 (0.012) | 190/200 |
| RMST | III | 197/200 | 0.959 | -0.034 (0.075) | 1.057 (0.049) | 0.955 (0.015) | 188/200 |
| RMST | IV | 136/200 | 0.949 | 0.277 (0.225) | 2.634 (0.291) | 0.995 (0.005) | 21/200 |
| RMST | V | 200/200 | 0.955 | -0.044 (0.070) | 0.994 (0.049) | 0.980 (0.010) | 189/200 |
| RMST | VI | 200/200 | 0.905 | 0.092 (0.121) | 1.704 (0.139) | 0.985 (0.009) | 140/200 |
| RMST | VII | 194/200 | 0.840 | 0.117 (0.131) | 1.829 (0.096) | 0.990 (0.007) | 142/200 |

All available point designs have valid roles under the generating DAG. Correct-dimension frequency ranges from 0.840 to 0.960 and set coverage from 0.950 to 0.995. In systems II, III and V, 186–190 of 200 linear-effect sets and 188–190 RMST sets are bounded, providing finite effect information alongside coverage.

Weak system IV supplies 136/200 point fits, with linear bias 0.074, RMSE 0.353 and overall set coverage 0.965; coverage among point fits is 0.949. It yields 21/200 linear sets that are bounded and 163/200 that are the real line, including 64 unavailable designs. The RMST target also has 21/200 bounded sets. Thus high coverage in this system accompanies limited finite precision. Supporting selection and resampling studies appear in the supplementary methods.

## 6 Application to ovarian-cancer molecular survival data

### 6.1 Ovarian cohort and discovery construction

We use TCGA PanCancer Atlas ovarian-cancer (OV) RPPA and clinical tables from cBioPortal [25, 26, 27]. The cohort includes 411 patients and 245 deaths. The fixed split assigns 308 patients to discovery and 103 to estimation, with 67 deaths in estimation. Discovery selects the 60 most variable proteins after missingness and duplicate-antibody filtering. The bridge includes age. Clinical contrasts correspond to one discovery-cohort protein standard deviation at a 36-month horizon.

Figures 2–3 illustrate LCK and CDH2 molecular representation; Figure 5 gives the selected proxy designs for the three ovarian proteins discussed below. Table S1 and Figures S5–S6 report the supporting analyses in nine other tumour cohorts.

### 6.2 Point estimates of restricted survival contrasts

Clinical estimation uses 300 patient resamples with the molecular design fixed, refitting censoring and concentrated moments. Of the 60 ovarian proteins, 26 supply eligible discovery designs and 24/60 complete point estimation (Figure 6). The panel yields two bounded, three disconnected and 55 real-line RMST sets; survival sets are two bounded, one disconnected and 57 real-line. All endpoint sets include zero; the fitted directions are exploratory signals.

**Table 5.** Three ovarian protein analyses per discovery-cohort standard deviation. Entries give the point estimate followed by its 95% confidence set; survival contrasts are in percentage points. Rank is the discovery-selected readout dimension, and treatment proxies are the selected molecular neighbours.

| Protein | Treatment proxies | Rank | RMST months and set | Survival points and set |
| --- | --- | --- | --- | --- |
| PTEN | CHEK2 | 1 | 1.02; [-2.07, 4.24] | 6.47; [-6.77, 19.31] |
| SERPINE1 | FN1, RBM15 | 2 | -11.55; All real values | -39.48; All real values |
| CCNE1 | CCNB1, BCL2L11, CHEK2 | 2 | -10.22; All real values | -31.62; All real values |

The three examples were chosen after analysis for bounded-set precision (PTEN) and ovarian-cancer biological relevance (SERPINE1 and CCNE1). Their designs share an eight-protein outcome-proxy pool: VHL, GAPDH, IGFBP2, RAB25, GAB2, EEF2K, TP53 and MAP2K1. Each selects window width 0.4 and support penalty 0.2, with rank one for PTEN and two for SERPINE1 and CCNE1. These proxy assignments use the exposure-specific exclusions in Section 4.1 (Table 5; Figure 5).

PTEN has an estimated RMST contrast of 1.02 months and a 95% set of [-2.07, 4.24]; its survival contrast is 6.47 percentage points (Table 5; Figure 7). Clinical interpretation depends on histotype and measurement: a 5400-patient immunohistochemical study associated lower cytoplasmic PTEN with longer survival in high-grade serous cancer [28]. Here the positive coefficient concerns a continuous RPPA abundance contrast through the proximal mean bridge, a different measurement and estimand from the cytoplasmic staining association.

SERPINE1, encoding plasminogen activator inhibitor-1 (PAI-1), has point contrasts of -11.55 RMST months and -39.48 survival percentage points. Higher PAI-1 mRNA expression has been associated with worse ovarian-cancer survival, and PAI-1 knockdown or inhibition reduced growth in PAI-1-expressing ovarian-cancer cell models [29]. This evidence motivates the protein-abundance intervention question and accords with the negative fitted direction.

CCNE1, encoding cyclin E1, has point contrasts of -10.22 RMST months and -31.62 survival percentage points. An Ovarian Tumor Tissue Analysis consortium study of 3029 high-grade serous cancers associated CCNE1 protein overexpression and high-level amplification with shorter survival [30], providing a protein-level context for this negative direction. SERPINE1 and CCNE1 have real-line sets on both endpoints, leaving their effect magnitudes unbounded in this sample.

HSPA1A is the second bounded ovarian RMST contrast: -0.62 months with set [-21.97, 10.19]. ACACA's coefficient of -53.60 months lies outside the RMST target range and is marked at the display boundary in Figure 6.

### 6.3 Pathway-defined intervention boundaries

GAB2 perturbation studies connect ovarian tumour growth and angiogenesis to PI3K and RAS–MAPK signalling [31, 32]. Its ovarian boundary specification assigns these branches to the total intervention effect and excludes their proteins from proxy roles.

In discovery patients, this boundary leaves fewer proxy coordinates than the required joint dimension. Estimating the pathway-defined GAB2 effect therefore requires additional eligible proxies. Supplementary methods and Tables S9–S10 give the full-cohort GAB2 and lung-cancer KDR reference analyses.

## 7 Discussion

Joint readout learning and conditional design selection serve complementary purposes. Exposure information can stabilise the readout space, while conditional information measures the treatment proxies' contribution beyond exposure. Independent estimation then uses the selected measurements to quantify the total-effect contrast and its precision.

In the modular and 60-protein two-factor systems, niche-window construction improves linear-effect estimation over global-window and correlation designs. Joint readout reduces linear bias in the weak two-factor system, where its RMST intervals are wider. Global and niche-window designs perform similarly in some systems. The component study measures these representation differences; the independent study measures coverage and finite precision after discovery selection.

The ovarian application places protein-abundance questions on clinical survival scales. PTEN highlights measurement-specific interpretation; SERPINE1 and CCNE1 relate the fitted directions to ovarian-cancer prognostic and experimental evidence. Future applications can increase identifying information through targeted measurement of eligible treatment proxies and systemic readouts, alongside larger estimation cohorts.

## Software and data

The manuscript, analysis code, study protocols, aggregate results and reproduction instructions are available at https://github.com/ChangeSean/idopNetwork-proximal. TCGA inputs are obtained from cBioPortal using the repository's study identifiers and download script.

## Figure legends

**Figure 1.** Independent discovery and clinical estimation. Discovery determines molecular preprocessing, network candidates, roles, readout dimension and coordinates. Estimation patients supply censored-outcome bridge moments and effect confidence sets.

**Figure 2.** TCGA-OV niche-index power curves for LCK, CDH2, four local treatment proxies and two separated readouts. Points are patient measurements; curve exponents and network roles are annotated.

**Figure 3.** Exposure-specific networks for LCK and CDH2. The upper panels show exposure, proxy and remaining-protein planes with ODE edges at each exposure's grid-selected penalty. The lower panels decompose selected niche curves into self and cross-protein contributions.

**Figure 4.** Independent-sample validation across seven systems, each with 200 fresh datasets. Point error uses available estimates; coverage uses all 200 sets. Bars give 1.96 Monte Carlo standard errors.

**Figure 5.** Selected proxy designs for ovarian PTEN, SERPINE1 and CCNE1. Treatment-proxy neighbourhoods are exposure-specific; the separated eight-protein outcome-proxy pool is shared. Rank and the smallest conditional root describe discovery information. Connectors indicate selected measurement roles.

**Figure 6.** The 24 completed ovarian-cancer point estimates per discovery-cohort standard deviation. Horizontal position gives RMST months or survival percentage points; orange marks the three analyses in Table 5. Open triangles mark off-range coefficients at the display boundary. Figure 7 gives confidence sets for the focused analyses.

**Figure 7.** Ovarian PTEN, SERPINE1 and CCNE1 contrasts on both clinical scales. Points are independent-estimation coefficients and lines are their 95% confidence sets. PTEN has bounded sets on both scales; arrows indicate the real-line sets for SERPINE1 and CCNE1.

**Figure S1.** Full TCGA-OV network on 140 filtered proteins, with 261 signed ODE edges and 18 components. The ten largest hubs are emphasised.

**Figure S2.** TCGA-OV graph by component, with LCK exposure neighbourhood on the left and the complete separated outcome-proxy pool on the right.

**Figure S3.** Fresh conditional-design and joint-quality fit availability across the seven systems, each with 200 attempted datasets. This figure concerns fixed-design reference inference; complete construction appears in Table S7.

**Figure S4.** Full construction completion for eight OV and 17 LUAD selected designs, with 300 attempted reconstructions per exposure. Wilson intervals quantify Monte Carlo uncertainty of the completion proportion.

**Figure S5.** The 58 completed point estimates across ten TCGA cohorts, including 24 ovarian estimates. Cohort labels give completed counts out of 60; open triangles mark off-range coefficients.

**Figure S6.** Full-cohort proxy-information screen across ten cohorts. Labels and marker areas give eligible exposure counts, and strength describes reduced readouts across the eight candidate designs. Orange identifies the original OV and LUAD cohorts. Independent estimates are reported separately in Table S1.

## Appendix A. Construction and inference proofs

### A.1 Niche-window support and proxy-map consistency

Let $V$ denote the population standardised panel and $s(V)=\sum_j V_j$. For the non-overlapping family, population boundaries are the $h/5$ quantiles of $s$. For width $w=0.4$, the five intervals have endpoints at quantiles $u_j=j(1-w)/4$ and $u_j+w$, $j=0,\ldots,4$. With continuous $s$ and positive window mass, empirical quantiles converge. Empirical conditional moments within each window converge to their population counterparts: contributions from observations whose window membership changes are confined to shrinking neighbourhoods of the boundaries. A moment envelope and truncation give the same conclusion with integrable, unbounded observations. Consistent standardisation preserves this argument. Sample-minimum shifts add a common offset to the niche index and leave its ordering and the within-window standardised LASSO unchanged.

For each node and window, positive definite predictor covariance makes the population penalised quadratic criterion strictly convex. Convergence of its empirical moments implies convergence of its minimiser. On the active set, the nonzero coefficient margin preserves signs. On the inactive set, strict KKT slack preserves zeros. There are finitely many nodes and windows, so all support decisions agree jointly with probability tending to one. Applying the implemented frequency threshold, a strict frequency greater than 0.6, and symmetrising therefore yields $P(\hat G=G_0)\to1$.

For loading eligibility, it is sufficient that the centred curve-deviation covariance converges, its retained eigenspace is separated, the largest multiplicative-gap decision is unique, and row norms are separated from the eligibility threshold. These conditions make loading eligibility stable. Convergence of the curve fit follows, for example, from bounded panel support, identifiable power curves with a unique interior least-squares minimiser, and a compact parameter neighbourhood with an integrable objective envelope. Proxy indices follow the fixed protein order. Thus the entire graph-to-role map stabilises. N1 and N2 give structural validity of its limiting roles; Proposition 3 gives consistent bridge rank. For the finite width–penalty grid, its maximum joint rank converges by the conditions following Proposition 3. The eligible set of full conditional dimension stabilises by Proposition 3b, and the maximiser of the smallest conditional-root criterion stabilises when distinct limiting role designs have a positive selection margin; identical roles give the same estimator. This proves the construction part of Theorem 1. The window argument concerns quantile-indexed empirical moments and does not require independently generated windows.

### A.2 Proof of Theorem 1 and Corollary 1

Condition first on the limiting roles and rank. Write $Q=E[mm^\top]$, $R=E[mW^\top]$ and $s=E[mY]$. The first-stage matrix is $\Pi=Q^{-1}R$. The covariance of $\Pi^\top m$ after residualising $(1,C)$ has rank $r$, and its range is the column space of $L_W$. An orthonormal basis $P$ therefore gives $\operatorname{rank}(P^\top L_W)=r$. By Propositions 2 and 3a, the regression on $(1,A,m^\top\Pi P,C)$ has exposure coefficient $\tau_A$.

Inversion of $Q$, the separated spectral projection and inversion of the second-stage moment matrix are differentiable. Repeated eigenvalues within the retained space are allowed because the exposure coefficient is invariant to an orthogonal change of basis in that space. The multivariate central limit theorem gives $\sqrt n(H_n-H)\to N(0,\Sigma_q)$, and the delta method gives Theorem 1. The construction and rank results transfer this conclusion to the selected estimator on an event with probability tending to one.

The empirical moment vector has the usual patient-bootstrap expansion. The same differentiable map, with both stages and projection refitted, gives the bootstrap limit [33, 34], Chapter 23. Holding the realised roles and rank fixed estimates this limit under the construction conditions. This proves Corollary 1. For complete construction resampling, bootstrap consistency of the window moments and curves, positive KKT/loading margins, rank consistency and the finite-grid selection margin make the resampled discrete construction agree with the limiting design with conditional probability tending to one. On that event the same differentiable moment map gives the bootstrap expansion; the complement has vanishing probability. This proves Corollary 1b. The statement is pointwise at a stable identified model.

### A.3 Proof of Theorem 2

Positivity through a fixed horizon makes the inverse-censoring functional differentiable. For a perturbation $g$,

$$
\dot V_\ell[g]= -\int_0^\ell I(O>u)\frac{g(u\mid A,C)}{G(u\mid A,C)^2}\,du,
\qquad
\dot V_t[g]=-I(O>t)\frac{g(t\mid A,C)}{G(t\mid A,C)^2}.
$$

Only the outcome-moment block changes. If $J_s$ denotes the part of $J$ acting on $E[mV_L]$, then $\mathcal D_G g=J_sE[m\dot V_L[g]]$. Expand the empirical moment map jointly with the regular censoring estimator. The empirical-process term is $J\{q_i(G)-H\}$; the nuisance term is $\mathcal D_G\varphi_{G,i}$. Their variance includes their cross-covariance because they are estimated from the same patients. The joint functional delta method gives Theorem 2, and its bootstrap version gives the stated resampling result. Refitting the censoring curve in each draw implements the nuisance contribution.

### A.4 Structural identification and concentrated moment proofs

**Proof of Proposition 1.**

*Proof.* N1 and separation of $W_A$ in $G_0$ imply separation from $(A,Z_A)$ in $H$. The moral-graph criterion therefore gives the second independence. N2 separates $Z_A$ from all observed outcome parents other than $A$ after conditioning on $A$. Adding the independent outcome error gives the first independence. A separation in the full moral graph also separates the relevant ancestral moral graph, so these statements cover all causal paths. $\square$

**Proof of Proposition 2.**

*Proof.* Full column rank of $L_W$ supplies a solution. Conditioning on $(A,F,Z,C)$ makes the two mean functions equal, and averaging gives the bridge equation. $W$ represents the pre-intervention systemic state, so averaging $h(W,a,C)$ gives the intervention mean. For uniqueness, subtract two linear bridge solutions and project their conditional mean difference on $(1,A,Z,C)$. The $Z$ coefficients satisfy $(\eta_W-\eta_W')^\top L_WK_Z=0$. Full row rank of $K_Z$ gives $(\eta_W-\eta_W')^\top L_W=0$. The coefficient of $A$ then gives $\eta_A-\eta_A'=0$. $\square$

**Proof of Proposition 3.**

*Proof.* The proxy mean-zero error is orthogonal to $(1,A,Z,C)$. Residualising the linear projections gives both covariance identities. Full column rank of $L_W$ and full row rank of $K_Z$ give rank $r$ for the conditional matrix. This conditional rank implies full row rank of $\operatorname{Cov}(\check F,\check M)$, so the joint matrix also has rank $r$. For each $k<r$, either canonical-root statistic has an $O(n)$ positive leading term. At $k=r$, residual roots are $O_p(n^{-1/2})$, so the statistic is $O_p(1)$. The chi-square threshold at $1-\alpha_n$ diverges at logarithmic order; it rejects every $k<r$ and accepts $k=r$ with probability tending to one. $\square$

**Proof of Proposition 3a.**

*Proof.* Projecting the proxy equation gives $\overline W=\alpha_W+D_WC+L_W\operatorname{Proj}(F\mid1,A,Z,C)$. Its covariance after residualising $(1,C)$ is $L_WQL_W^\top$, with $Q$ positive definite by the joint-information rank condition implied by conditional completeness. The covariance range is therefore $\operatorname{col}(L_W)$ and $P^\top L_W$ is invertible. The reduced loading matrix $L_R=P^\top L_W$ supplies the solution $L_R^\top\eta_R=b$. Proposition 2 gives the bridge equation and uniqueness of the exposure coefficient. Averaging the reduced bridge after intervening on $A$ gives the same total-effect contrast. $\square$

**Proof of Proposition 3b.**

*Proof.* Proposition 3's covariance factorisations give $q_d\le r_0$ for
every candidate. A full candidate gives $\max_dq_d=r_0$. Finiteness of the
grid makes rank consistency simultaneous. For $q_d=r_0$, $L_{W,d}$ has full
column rank and $P_d^\top L_{W,d}$ is invertible. Thus

$$
\operatorname{Cov}(\widetilde R_d,\widetilde Z_d)
=(P_d^\top L_{W,d})K_{Z,d}\operatorname{Cov}(\widetilde Z_d)
$$

has rank $r_0$ exactly when $K_{Z,d}$ has row rank $r_0$, assuming nonsingular
residual treatment-proxy covariance. Every member of $\mathcal E$ is therefore
conditionally complete. Canonical roots are continuous on separated
covariance spaces and converge uniformly over the finite grid. A positive
quality margin selects the limiting role design with probability tending to
one. Identical roles give the same projection and bridge. Propositions 2 and
3a then identify the unique total-effect coefficient. $\square$

**Proof of Proposition 4.**

*Proof.* Invertibility of $B$ preserves the row rank of $\Lambda^\top B$, giving the column-space statement. The Woodbury identity applied to $(\Lambda\Lambda^\top+\Psi)^{-1}$ gives the precision formula. Expanding its first term produces only parent-child and shared-child entries. $\square$

**Proof of Proposition 5.**

*Proof.* Conditional independent censoring gives
$E[I(O>u)/G(u\mid A,C)\mid T,A,F,Z,C]=I(T>u)$.
Integration and conditional expectation give the restricted-mean identity; evaluation at $t$ gives the survival identity. Proposition 2 applied to either pseudo-outcome gives its intervention mean and contrast. $\square$

**Proof of Theorem 3.**

*Proof.* Nuisance inversion and joint projection are smooth under their
separation conditions. The functional delta method, with the censoring
contribution of Theorem 2 when needed, gives the concentrated moment law.
For each coordinate $N_j-\tau_A D_j=0$, and consistent studentisation yields
an asymptotic standard normal null moment without division by $D_j$.
Each inverted test has asymptotic rejection probability at most $\alpha/m$.
The union bound gives simultaneous retention probability at least
$1-\alpha$, without requiring independent contrasts. The same argument
using the assumed uniform laws proves the uniform statement. $\square$

### A.5 Proof of Theorem 4

Condition on discovery. Selected measurements, their affine preprocessing, $P_{\mathcal D}$ and moment pivots are now constants. Under Proposition 2 and $\operatorname{rank}(P_{\mathcal D}^{\top}L_W)=r_0$, solve $(P_{\mathcal D}^{\top}L_W)^{\top}\eta=b$. The reduced outcome bridge has coefficient $\tau_A$ on $A$; the mean-zero proxy errors are preserved by this fixed projection. The bridge moment equations in the independent estimation distribution are thus $E[Q(Y-\tau_A A-R^{\top}\eta)]=0$, after covariate projection.

Partition $Q$ into the discovery-fixed nuisance coordinates $T$ and remaining coordinates $U$. With $B=E[TR^{\top}]$, $L=E[UR^{\top}]B^{-1}$, cancellation of $\eta$ gives $N=\tau_A D$. Nonsingular nuisance concentration and covariate projections are smooth functions of estimation moments; they do not divide by the effect-information vector $D$. For a censored response, Proposition 5 gives the same equations and the regular censoring expansion adds its patient influence contribution to the joint moment covariance.

At the true $\tau_A$, the uniform joint moment law and consistent covariance imply that each component studentised statistic has limiting standard normal law. There are at most $m_{\max}$ components. Applying the union bound at component size $\alpha/m$ gives conditional coverage at least $1-\alpha-o(1)$ for their intersection. This argument permits $D$ to approach zero and permits arbitrary dependence among the component statistics. Replacing an unavailable confidence calculation by $\mathbb R$ can only increase coverage. Finally average the conditional bound over discovery; the probability of an available design outside the stated identification event contributes at most $\varepsilon_n$ to noncoverage. No assumption about a unique design-quality maximum enters this proof. $\square$

## Supplementary tables

### Supplementary methods and study provenance

Component-study examples were chosen after analysis; Tables S2–S4 give every system, outcome and comparator. The supporting studies below use separate settings and seeds.

#### Allocation development

The 50:50 allocation study uses seed offset 120000. Its weak two-factor system selected rank one in 16/200 attempts, with overall linear coverage 0.905. The 75:25 allocation dedicates more patients to molecular construction and was fixed before the offset-140000 validation batch.

#### Selection and resampling studies

A conditional-information study at offset 50000 uses seven systems with 200 datasets each (Table S6), with linear bias -0.005 to 0.027. Comparisons on common fitted datasets give conditional and joint-quality RMSEs of 0.152 versus 0.151 in system IV and 0.111 versus 0.111 in system VII.

The complete-construction study at offset 70000 rebuilds every molecular stage in three systems, each with 100 datasets and 100 resamples; normal intervals complete in 1/100, 0/100 and 0/100 datasets (Table S7). The concentrated Fieller study at offset 100000 uses seven systems with 200 datasets each, with selected roles and rank fixed; coverage is 0.944–0.985 (Tables S5 and S8).

#### Full-cohort inference procedures

Complete construction inference uses 300 patient resamples. Each rebuilds standardisation, niche ordering, curves, loadings, all eight support graphs, proxy sets, ranks and conditional design selection, then refits censoring, the joint projection and both bridge stages. The initial 60-protein panel and molecular and clinical imputations are held fixed. Before calculating dispersion, convert each coefficient from the resampled standard deviation to the original full-cohort standard deviation.

A finite normal bootstrap interval requires at least 20 successful fits and 97.5% completed attempts; otherwise the interval is unavailable. Corollary 1b supplies the pointwise resampling justification at stable identified models, while the completion threshold measures computational availability. Conditional reference intervals hold roles and rank fixed across 300 draws, refitting projection, both stages and censoring.

The unadjusted comparison fits the same pseudo-outcome on $(1,A,C)$ and refits censoring in the same patient draws. For the concentrated bridge, component null-moment tests at $\tau_A=0$ give $p=\min\{1,m\min_jp_j\}$, matching the Bonferroni intersection. Benjamini–Hochberg $q$-values [35] are calculated separately for the RMST and survival families across returned design-conditioned sets in each cohort. Normal Wald calculations use the separately labelled least-squares reference intervals.

#### Full-cohort clinical reference

Full-cohort molecular analyses supply eight OV and 17 LUAD conditionally complete designs. Table S9 gives their concentrated estimates and sets with roles, rank and pivots fixed. Reconstruction completes 10–25% of OV and 12–35% of LUAD attempts; all complete-construction normal intervals are unavailable under the 97.5% rule (Table S10; Figure S4). These references use full-cohort panels and exposure scales; the independent analyses use discovery-selected panels and scales.

#### Complete clinical panels and additional tumour cohorts

The independent procedure also analyses LUAD, BLCA, BRCA, COADREAD, KIRC, LGG, SKCM, STAD and UCEC. Together with OV, these supply 600 cohort-specific protein contrasts and 1200 endpoint sets; 58/600 exposures complete point estimation. Discovery defines covariates by cohort: age and, where represented, sex and tumour subtype; LUAD uses age and sex. Across the ten cohorts, two RMST sets are bounded, three disconnected and 595 real-line, and every endpoint set includes zero. Table S1 gives sample sizes, design counts and set shapes; Figures S5–S6 show completed coefficients and full-cohort molecular information. Aggregate files contain every exposure's estimate, confidence set and calculation status.

The eight-cohort extension supplies 32 completed estimates and real-line sets for all 480 exposures on both endpoints. COADREAD and STAD designs fail 36-month censoring support; BRCA, LGG and UCEC chiefly encounter insufficient complete moment resamples. KIRC completes all 14 eligible designs with real-line effect sets. Ovarian HSPA1A gives -7.54 survival percentage points with set [-80.32, 36.00], accompanying its bounded RMST set in Section 6. Off-range coefficients include LUAD HSPA1A at 379 percentage points and SKCM BRAF at -53.66 months (Figure S5).

#### Pathway-specific full-cohort reference

The full-cohort boundary analyses use cohort-specific exposure scales. For GAB2, the pathway specification in Section 6.3 removes BRAF from the treatment-proxy pool, leaving no design with the required two-dimensional joint information.

KDR encodes VEGFR2. NSCLC experiments connect amplified KDR to VEGF-induced mTOR signalling and motility [36]. The KDR boundary assigns these pathways to the total effect and removes ITGA2 from the treatment-proxy pool because of its separately documented metastatic branch in LUAD [37]. The selected design has Z=CTNNB1, W=MYC, dimension one, window width 0.4, penalty 0.15 and smallest conditional root 0.214. Its bridge treats MYC as a parallel readout of the systemic proliferative state, outside the KDR intervention path, and requires the outcome exclusion for CTNNB1.

The estimated 36-month RMST contrast is -1.50 months per full-cohort standard deviation, with 95% Fieller set [-4.29, 0.44]; the survival contrast is -8.1 percentage points, with set [-26.33, 2.61]. Reconstruction completes 36/300 attempts (0.120), yielding an unavailable complete-construction interval. The target is a KDR abundance intervention.

**Table S1.** Complete independent analysis across ten cohorts, with 60 attempted proteins per cohort. B/D/R denotes bounded, disconnected and real-line confidence-set counts; real-line counts include unavailable calculations with recorded reasons. Point fits apply to both endpoints.

| Cohort | Discovery / estimation | Estimation deaths | Designs | Point fits | RMST B/D/R | Survival B/D/R |
| --- | --- | --- | --- | --- | --- | --- |
| BLCA | 256/86 | 39 | 19/60 | 8/60 | 0/0/60 | 0/0/60 |
| BRCA | 657/219 | 33 | 18/60 | 1/60 | 0/0/60 | 0/0/60 |
| COADREAD | 345/115 | 17 | 14/60 | 0/60 | 0/0/60 | 0/0/60 |
| KIRC | 341/114 | 42 | 14/60 | 14/60 | 0/0/60 | 0/0/60 |
| LGG | 320/107 | 16 | 18/60 | 0/60 | 0/0/60 | 0/0/60 |
| LUAD | 264/88 | 33 | 10/60 | 2/60 | 0/0/60 | 0/0/60 |
| OV | 308/103 | 67 | 26/60 | 24/60 | 2/3/55 | 2/1/57 |
| SKCM | 237/80 | 37 | 17/60 | 9/60 | 0/0/60 | 0/0/60 |
| STAD | 261/88 | 38 | 23/60 | 0/60 | 0/0/60 | 0/0/60 |
| UCEC | 316/106 | 23 | 11/60 | 0/60 | 0/0/60 | 0/0/60 |

**Table S2.** Complete LINEAR comparison; entries are estimate (MCSE). All 200 attempts enter reporting denominators; generating-DAG role validity concerns the selected proxy pair.

| System | Estimator | Reported | Bias | RMSE | Coverage | Valid roles (%) |
| --- | --- | --- | --- | --- | --- | --- |
| I | Joint-quality readout | 200/200 | 0.008 (0.006) | 0.087 (0.006) | 0.935 (0.017) | 100.0 |
| I | Same design, conditional projection | 200/200 | 0.024 (0.006) | 0.084 (0.004) | 0.940 (0.017) | 100.0 |
| I | Previous grid bridge | 200/200 | 0.026 (0.006) | 0.095 (0.005) | 0.935 (0.017) | 100.0 |
| I | One global window | 200/200 | 0.014 (0.006) | 0.089 (0.007) | 0.940 (0.017) | 100.0 |
| I | Correlation proxies | 183/200 | 0.298 (0.006) | 0.310 (0.006) | 0.257 (0.032) | 100.0 |
| I | Designated joint readout | 200/200 | 0.014 (0.008) | 0.112 (0.005) | 0.920 (0.019) | 100.0 |
| I | Designated proximal | 200/200 | 0.019 (0.008) | 0.112 (0.005) | 0.920 (0.019) | 100.0 |
| I | Unadjusted | 200/200 | 0.411 (0.007) | 0.422 (0.007) | 0.010 (0.007) | — |
| I | Factor adjustment | 200/200 | 0.150 (0.073) | 1.039 (0.085) | 0.940 (0.017) | — |
| I | Oracle factors | 200/200 | 0.003 (0.006) | 0.087 (0.004) | 0.945 (0.016) | — |
| II | Joint-quality readout | 200/200 | -0.006 (0.006) | 0.079 (0.004) | 0.950 (0.015) | 100.0 |
| II | Same design, conditional projection | 200/200 | -0.002 (0.005) | 0.077 (0.004) | 0.950 (0.015) | 100.0 |
| II | Previous grid bridge | 198/200 | 0.005 (0.006) | 0.083 (0.004) | 0.934 (0.018) | 100.0 |
| II | One global window | 200/200 | -0.006 (0.006) | 0.079 (0.004) | 0.950 (0.015) | 100.0 |
| II | Correlation proxies | 193/200 | 0.159 (0.006) | 0.182 (0.006) | 0.694 (0.033) | 100.0 |
| II | Designated joint readout | 200/200 | -0.005 (0.006) | 0.079 (0.004) | 0.945 (0.016) | 100.0 |
| II | Designated proximal | 200/200 | -0.005 (0.006) | 0.079 (0.004) | 0.945 (0.016) | 100.0 |
| II | Unadjusted | 200/200 | 0.199 (0.005) | 0.214 (0.005) | 0.260 (0.031) | — |
| II | Factor adjustment | 200/200 | -1.469 (0.086) | 1.905 (0.059) | 0.665 (0.033) | — |
| II | Oracle factors | 200/200 | -0.006 (0.005) | 0.076 (0.004) | 0.920 (0.019) | — |
| III | Joint-quality readout | 200/200 | -0.001 (0.003) | 0.049 (0.003) | 0.965 (0.013) | 100.0 |
| III | Same design, conditional projection | 200/200 | 0.002 (0.003) | 0.049 (0.003) | 0.965 (0.013) | 100.0 |
| III | Previous grid bridge | 200/200 | 0.007 (0.004) | 0.064 (0.004) | 0.960 (0.014) | 100.0 |
| III | One global window | 191/200 | 0.107 (0.012) | 0.195 (0.010) | 0.869 (0.024) | 100.0 |
| III | Correlation proxies | 200/200 | 0.118 (0.010) | 0.184 (0.007) | 0.845 (0.026) | 100.0 |
| III | Designated joint readout | 200/200 | 0.000 (0.004) | 0.052 (0.003) | 0.955 (0.015) | 100.0 |
| III | Designated proximal | 200/200 | 0.001 (0.004) | 0.052 (0.003) | 0.955 (0.015) | 100.0 |
| III | Unadjusted | 200/200 | 0.547 (0.004) | 0.550 (0.004) | 0.000 (0.000) | — |
| III | Factor adjustment | 200/200 | -0.669 (0.005) | 0.672 (0.005) | 0.000 (0.000) | — |
| III | Oracle factors | 200/200 | -0.003 (0.004) | 0.050 (0.002) | 0.945 (0.016) | — |
| IV | Joint-quality readout | 200/200 | 0.013 (0.012) | 0.164 (0.010) | 0.965 (0.013) | 100.0 |
| IV | Same design, conditional projection | 200/200 | 0.203 (0.010) | 0.249 (0.010) | 0.625 (0.034) | 100.0 |
| IV | Previous grid bridge | 2/200 | — | — | — | 100.0 |
| IV | One global window | 200/200 | 0.013 (0.012) | 0.163 (0.010) | 0.965 (0.013) | 100.0 |
| IV | Correlation proxies | 200/200 | 0.512 (0.005) | 0.516 (0.005) | 0.000 (0.000) | 100.0 |
| IV | Designated joint readout | 200/200 | 0.014 (0.013) | 0.182 (0.014) | 0.950 (0.015) | 100.0 |
| IV | Designated proximal | 200/200 | 0.078 (0.011) | 0.170 (0.009) | 0.915 (0.020) | 100.0 |
| IV | Unadjusted | 200/200 | 0.511 (0.005) | 0.515 (0.005) | 0.000 (0.000) | — |
| IV | Factor adjustment | 200/200 | -1.264 (0.019) | 1.291 (0.016) | 0.045 (0.015) | — |
| IV | Oracle factors | 200/200 | -0.007 (0.004) | 0.057 (0.003) | 0.935 (0.017) | — |
| V | Joint-quality readout | 200/200 | 0.012 (0.006) | 0.081 (0.010) | 0.960 (0.014) | 100.0 |
| V | Same design, conditional projection | 200/200 | 0.021 (0.006) | 0.082 (0.011) | 0.960 (0.014) | 100.0 |
| V | Previous grid bridge | 199/200 | 0.026 (0.007) | 0.098 (0.015) | 0.940 (0.017) | 100.0 |
| V | One global window | 161/200 | 0.059 (0.017) | 0.225 (0.022) | 0.851 (0.028) | 100.0 |
| V | Correlation proxies | 200/200 | 0.400 (0.019) | 0.484 (0.016) | 0.550 (0.035) | 100.0 |
| V | Designated joint readout | 200/200 | 0.001 (0.004) | 0.057 (0.003) | 0.940 (0.017) | 100.0 |
| V | Designated proximal | 200/200 | 0.002 (0.004) | 0.057 (0.003) | 0.940 (0.017) | 100.0 |
| V | Unadjusted | 200/200 | 0.785 (0.004) | 0.787 (0.004) | 0.000 (0.000) | — |
| V | Factor adjustment | 200/200 | -0.818 (0.030) | 0.922 (0.012) | 0.475 (0.035) | — |
| V | Oracle factors | 200/200 | 0.001 (0.004) | 0.049 (0.002) | 0.955 (0.015) | — |
| VI | Joint-quality readout | 200/200 | 0.004 (0.008) | 0.112 (0.008) | 0.940 (0.017) | 100.0 |
| VI | Same design, conditional projection | 200/200 | 0.016 (0.008) | 0.113 (0.008) | 0.930 (0.018) | 100.0 |
| VI | Previous grid bridge | 199/200 | 0.016 (0.008) | 0.108 (0.006) | 0.920 (0.019) | 100.0 |
| VI | One global window | 200/200 | 0.009 (0.008) | 0.109 (0.006) | 0.945 (0.016) | 100.0 |
| VI | Correlation proxies | 135/200 | 0.195 (0.009) | 0.221 (0.010) | 0.667 (0.041) | 100.0 |
| VI | Designated joint readout | 200/200 | 0.003 (0.007) | 0.100 (0.005) | 0.955 (0.015) | 100.0 |
| VI | Designated proximal | 200/200 | 0.005 (0.007) | 0.100 (0.005) | 0.955 (0.015) | 100.0 |
| VI | Unadjusted | 200/200 | 0.191 (0.007) | 0.216 (0.007) | 0.475 (0.035) | — |
| VI | Factor adjustment | 200/200 | -0.194 (0.085) | 1.209 (0.067) | 0.865 (0.024) | — |
| VI | Oracle factors | 200/200 | 0.001 (0.007) | 0.096 (0.005) | 0.920 (0.019) | — |
| VII | Joint-quality readout | 200/200 | 0.012 (0.009) | 0.121 (0.009) | 0.955 (0.015) | 100.0 |
| VII | Same design, conditional projection | 200/200 | 0.038 (0.008) | 0.116 (0.007) | 0.940 (0.017) | 100.0 |
| VII | Previous grid bridge | 197/200 | 0.043 (0.011) | 0.163 (0.010) | 0.914 (0.020) | 100.0 |
| VII | One global window | 173/200 | 0.140 (0.021) | 0.307 (0.015) | 0.792 (0.031) | 100.0 |
| VII | Correlation proxies | 200/200 | 0.509 (0.007) | 0.519 (0.007) | 0.025 (0.011) | 100.0 |
| VII | Designated joint readout | 200/200 | -0.009 (0.007) | 0.094 (0.004) | 0.955 (0.015) | 100.0 |
| VII | Designated proximal | 200/200 | -0.007 (0.007) | 0.093 (0.004) | 0.950 (0.015) | 100.0 |
| VII | Unadjusted | 200/200 | 0.513 (0.007) | 0.523 (0.007) | 0.005 (0.005) | — |
| VII | Factor adjustment | 200/200 | 0.052 (0.053) | 0.752 (0.049) | 0.920 (0.019) | — |
| VII | Oracle factors | 200/200 | -0.008 (0.006) | 0.086 (0.004) | 0.940 (0.017) | — |

**Table S3.** Complete RMST comparison; entries are estimate (MCSE). All 200 attempts enter reporting denominators; generating-DAG role validity concerns the selected proxy pair.

| System | Estimator | Reported | Bias | RMSE | Coverage | Valid roles (%) |
| --- | --- | --- | --- | --- | --- | --- |
| I | Joint-quality readout | 200/200 | -0.027 (0.056) | 0.787 (0.041) | 0.930 (0.018) | 100.0 |
| I | Same design, conditional projection | 200/200 | -0.001 (0.055) | 0.775 (0.040) | 0.930 (0.018) | 100.0 |
| I | Previous grid bridge | 200/200 | 0.006 (0.055) | 0.771 (0.038) | 0.930 (0.018) | 100.0 |
| I | One global window | 200/200 | -0.019 (0.055) | 0.778 (0.040) | 0.930 (0.018) | 100.0 |
| I | Correlation proxies | 183/200 | 0.370 (0.052) | 0.799 (0.039) | 0.923 (0.020) | 100.0 |
| I | Designated joint readout | 200/200 | -0.019 (0.055) | 0.775 (0.040) | 0.940 (0.017) | 100.0 |
| I | Designated proximal | 200/200 | -0.011 (0.055) | 0.772 (0.040) | 0.940 (0.017) | 100.0 |
| I | Unadjusted | 200/200 | 0.498 (0.049) | 0.853 (0.040) | 0.865 (0.024) | — |
| I | Factor adjustment | 200/200 | 0.713 (0.266) | 3.814 (0.200) | 0.960 (0.014) | — |
| I | Oracle factors | 200/200 | -0.041 (0.052) | 0.739 (0.037) | 0.940 (0.017) | — |
| II | Joint-quality readout | 200/200 | -0.075 (0.034) | 0.481 (0.023) | 0.945 (0.016) | 100.0 |
| II | Same design, conditional projection | 200/200 | -0.067 (0.034) | 0.479 (0.023) | 0.940 (0.017) | 100.0 |
| II | Previous grid bridge | 198/200 | -0.054 (0.034) | 0.477 (0.024) | 0.939 (0.017) | 100.0 |
| II | One global window | 200/200 | -0.072 (0.034) | 0.480 (0.023) | 0.945 (0.016) | 100.0 |
| II | Correlation proxies | 193/200 | 0.216 (0.034) | 0.518 (0.024) | 0.927 (0.019) | 100.0 |
| II | Designated joint readout | 200/200 | -0.074 (0.034) | 0.481 (0.023) | 0.940 (0.017) | 100.0 |
| II | Designated proximal | 200/200 | -0.072 (0.034) | 0.481 (0.023) | 0.940 (0.017) | 100.0 |
| II | Unadjusted | 200/200 | 0.292 (0.033) | 0.549 (0.024) | 0.910 (0.020) | — |
| II | Factor adjustment | 200/200 | -1.826 (0.184) | 3.175 (0.139) | 0.865 (0.024) | — |
| II | Oracle factors | 200/200 | -0.066 (0.033) | 0.469 (0.023) | 0.945 (0.016) | — |
| III | Joint-quality readout | 200/200 | 0.002 (0.036) | 0.512 (0.028) | 0.925 (0.019) | 100.0 |
| III | Same design, conditional projection | 200/200 | 0.003 (0.036) | 0.510 (0.028) | 0.920 (0.019) | 100.0 |
| III | Previous grid bridge | 200/200 | 0.013 (0.036) | 0.515 (0.028) | 0.930 (0.018) | 100.0 |
| III | One global window | 191/200 | 0.161 (0.038) | 0.552 (0.028) | 0.927 (0.019) | 100.0 |
| III | Correlation proxies | 200/200 | 0.156 (0.038) | 0.559 (0.029) | 0.905 (0.021) | 100.0 |
| III | Designated joint readout | 200/200 | -0.001 (0.036) | 0.510 (0.028) | 0.925 (0.019) | 100.0 |
| III | Designated proximal | 200/200 | 0.000 (0.036) | 0.510 (0.028) | 0.925 (0.019) | 100.0 |
| III | Unadjusted | 200/200 | 0.800 (0.028) | 0.894 (0.026) | 0.450 (0.035) | — |
| III | Factor adjustment | 200/200 | -0.775 (0.050) | 1.045 (0.045) | 0.760 (0.030) | — |
| III | Oracle factors | 200/200 | -0.004 (0.036) | 0.507 (0.028) | 0.915 (0.020) | — |
| IV | Joint-quality readout | 200/200 | 0.100 (0.096) | 1.356 (0.079) | 0.965 (0.013) | 100.0 |
| IV | Same design, conditional projection | 200/200 | 0.147 (0.062) | 0.884 (0.057) | 0.900 (0.021) | 100.0 |
| IV | Previous grid bridge | 2/200 | — | — | — | 100.0 |
| IV | One global window | 200/200 | 0.105 (0.096) | 1.358 (0.079) | 0.960 (0.014) | 100.0 |
| IV | Correlation proxies | 200/200 | 0.375 (0.031) | 0.579 (0.026) | 0.845 (0.026) | 100.0 |
| IV | Designated joint readout | 200/200 | 0.046 (0.098) | 1.386 (0.088) | 0.950 (0.015) | 100.0 |
| IV | Designated proximal | 200/200 | 0.106 (0.084) | 1.191 (0.070) | 0.910 (0.020) | 100.0 |
| IV | Unadjusted | 200/200 | 0.374 (0.031) | 0.577 (0.026) | 0.845 (0.026) | — |
| IV | Factor adjustment | 200/200 | -0.918 (0.078) | 1.437 (0.069) | 0.890 (0.022) | — |
| IV | Oracle factors | 200/200 | -0.030 (0.033) | 0.469 (0.021) | 0.960 (0.014) | — |
| V | Joint-quality readout | 200/200 | 0.009 (0.039) | 0.545 (0.028) | 0.940 (0.017) | 100.0 |
| V | Same design, conditional projection | 200/200 | 0.018 (0.038) | 0.533 (0.025) | 0.930 (0.018) | 100.0 |
| V | Previous grid bridge | 199/200 | 0.013 (0.038) | 0.530 (0.026) | 0.930 (0.018) | 100.0 |
| V | One global window | 161/200 | 0.049 (0.043) | 0.547 (0.028) | 0.950 (0.017) | 100.0 |
| V | Correlation proxies | 200/200 | 0.292 (0.038) | 0.615 (0.033) | 0.850 (0.025) | 100.0 |
| V | Designated joint readout | 200/200 | 0.006 (0.038) | 0.537 (0.026) | 0.925 (0.019) | 100.0 |
| V | Designated proximal | 200/200 | 0.007 (0.038) | 0.537 (0.026) | 0.925 (0.019) | 100.0 |
| V | Unadjusted | 200/200 | 0.559 (0.031) | 0.709 (0.029) | 0.715 (0.032) | — |
| V | Factor adjustment | 200/200 | -0.579 (0.092) | 1.424 (0.136) | 0.945 (0.016) | — |
| V | Oracle factors | 200/200 | 0.016 (0.037) | 0.522 (0.026) | 0.925 (0.019) | — |
| VI | Joint-quality readout | 200/200 | 0.004 (0.047) | 0.665 (0.030) | 0.975 (0.011) | 100.0 |
| VI | Same design, conditional projection | 200/200 | 0.031 (0.047) | 0.660 (0.029) | 0.975 (0.011) | 100.0 |
| VI | Previous grid bridge | 199/200 | 0.045 (0.047) | 0.661 (0.029) | 0.975 (0.011) | 100.0 |
| VI | One global window | 200/200 | 0.014 (0.048) | 0.671 (0.030) | 0.975 (0.011) | 100.0 |
| VI | Correlation proxies | 135/200 | 0.394 (0.054) | 0.735 (0.038) | 0.978 (0.013) | 100.0 |
| VI | Designated joint readout | 200/200 | 0.011 (0.047) | 0.666 (0.030) | 0.980 (0.010) | 100.0 |
| VI | Designated proximal | 200/200 | 0.015 (0.047) | 0.665 (0.030) | 0.980 (0.010) | 100.0 |
| VI | Unadjusted | 200/200 | 0.407 (0.044) | 0.747 (0.031) | 0.945 (0.016) | — |
| VI | Factor adjustment | 200/200 | -0.834 (0.213) | 3.114 (0.198) | 0.970 (0.012) | — |
| VI | Oracle factors | 200/200 | -0.005 (0.046) | 0.645 (0.030) | 0.975 (0.011) | — |
| VII | Joint-quality readout | 200/200 | -0.015 (0.060) | 0.840 (0.043) | 0.970 (0.012) | 100.0 |
| VII | Same design, conditional projection | 200/200 | 0.030 (0.056) | 0.794 (0.041) | 0.955 (0.015) | 100.0 |
| VII | Previous grid bridge | 197/200 | 0.051 (0.056) | 0.780 (0.043) | 0.959 (0.014) | 100.0 |
| VII | One global window | 173/200 | 0.061 (0.062) | 0.816 (0.051) | 0.965 (0.014) | 100.0 |
| VII | Correlation proxies | 200/200 | 0.412 (0.049) | 0.802 (0.042) | 0.910 (0.020) | 100.0 |
| VII | Designated joint readout | 200/200 | -0.026 (0.056) | 0.796 (0.043) | 0.950 (0.015) | 100.0 |
| VII | Designated proximal | 200/200 | -0.024 (0.056) | 0.795 (0.042) | 0.950 (0.015) | 100.0 |
| VII | Unadjusted | 200/200 | 0.414 (0.049) | 0.804 (0.042) | 0.910 (0.020) | — |
| VII | Factor adjustment | 200/200 | -0.237 (0.179) | 2.543 (0.215) | 0.960 (0.014) | — |
| VII | Oracle factors | 200/200 | -0.034 (0.055) | 0.775 (0.041) | 0.970 (0.012) | — |

**Table S4.** Joint-readout estimates grouped by fitted rank; entries are estimate (MCSE). Strata with fewer than five fits retain their counts and leave accuracy blank.

| Outcome | System | Fitted rank | Reported | Bias | RMSE | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| LINEAR | II | 1 | 185 | -0.006 (0.006) | 0.077 (0.004) | 0.957 (0.015) |
| LINEAR | II | 2 | 15 | 0.003 (0.026) | 0.097 (0.013) | 0.867 (0.088) |
| RMST | II | 1 | 185 | -0.087 (0.035) | 0.485 (0.025) | 0.941 (0.017) |
| RMST | II | 2 | 15 | 0.075 (0.114) | 0.431 (0.056) | 1.000 (0.000) |
| LINEAR | III | 1 | 195 | -0.002 (0.003) | 0.049 (0.003) | 0.964 (0.013) |
| LINEAR | III | 2 | 5 | 0.039 (0.021) | 0.058 (0.010) | 1.000 (0.000) |
| RMST | III | 1 | 195 | -0.007 (0.037) | 0.512 (0.029) | 0.923 (0.019) |
| RMST | III | 2 | 5 | 0.355 (0.190) | 0.520 (0.205) | 1.000 (0.000) |
| LINEAR | I | 1 | 190 | 0.009 (0.006) | 0.082 (0.004) | 0.937 (0.018) |
| LINEAR | I | 2 | 10 | -0.023 (0.052) | 0.158 (0.057) | 0.900 (0.095) |
| RMST | I | 1 | 190 | -0.021 (0.057) | 0.778 (0.042) | 0.926 (0.019) |
| RMST | I | 2 | 10 | -0.146 (0.306) | 0.930 (0.182) | 1.000 (0.000) |
| LINEAR | VI | 1 | 192 | -0.001 (0.008) | 0.105 (0.006) | 0.938 (0.017) |
| LINEAR | VI | 2 | 8 | 0.105 (0.070) | 0.213 (0.081) | 1.000 (0.000) |
| RMST | VI | 1 | 192 | 0.002 (0.048) | 0.667 (0.031) | 0.974 (0.011) |
| RMST | VI | 2 | 8 | 0.065 (0.229) | 0.608 (0.108) | 1.000 (0.000) |
| LINEAR | IV | 2 | 191 | 0.014 (0.012) | 0.163 (0.011) | 0.963 (0.014) |
| LINEAR | IV | 3 | 9 | -0.006 (0.067) | 0.190 (0.031) | 1.000 (0.000) |
| RMST | IV | 2 | 191 | 0.071 (0.096) | 1.330 (0.077) | 0.963 (0.014) |
| RMST | IV | 3 | 9 | 0.717 (0.592) | 1.821 (0.549) | 1.000 (0.000) |
| LINEAR | V | 1 | 1 | — | — | — |
| LINEAR | V | 2 | 188 | 0.009 (0.005) | 0.072 (0.008) | 0.963 (0.014) |
| LINEAR | V | 3 | 11 | 0.011 (0.023) | 0.074 (0.011) | 1.000 (0.000) |
| RMST | V | 1 | 1 | — | — | — |
| RMST | V | 2 | 188 | 0.016 (0.039) | 0.532 (0.026) | 0.941 (0.017) |
| RMST | V | 3 | 11 | -0.203 (0.205) | 0.679 (0.221) | 1.000 (0.000) |
| LINEAR | VII | 2 | 167 | 0.001 (0.008) | 0.102 (0.006) | 0.952 (0.017) |
| LINEAR | VII | 3 | 32 | 0.064 (0.032) | 0.191 (0.030) | 0.969 (0.031) |
| LINEAR | VII | 4 | 1 | — | — | — |
| RMST | VII | 2 | 167 | -0.011 (0.063) | 0.808 (0.046) | 0.964 (0.014) |
| RMST | VII | 3 | 32 | -0.017 (0.179) | 0.995 (0.120) | 1.000 (0.000) |
| RMST | VII | 4 | 1 | — | — | — |

**Table S5.** All fresh concentrated-bridge evaluations. Shape counts are bounded/disconnected/real-line/half-line/empty. Parentheses give Monte Carlo standard errors. Effect errors use all available point fits; coverage includes every returned set; width uses single bounded intervals only.

| Target | System | Sets | Shape counts | Bias (MCSE) | RMSE (MCSE) | Fieller coverage | Normal coverage | Bounded width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LINEAR | I | 200/200 | 181/0/19/0/0 | -0.001 (0.006) | 0.081 (0.004) | 0.985 (0.009) | 0.980 (0.010) | 0.408 |
| LINEAR | II | 200/200 | 185/0/15/0/0 | 0.000 (0.005) | 0.077 (0.004) | 0.975 (0.011) | 0.960 (0.014) | 0.346 |
| LINEAR | III | 198/200 | 189/1/6/0/2 | 0.006 (0.004) | 0.051 (0.003) | 0.944 (0.016) | 0.960 (0.014) | 0.261 |
| LINEAR | IV | 154/200 | 108/20/26/0/0 | 0.018 (0.014) | 0.168 (0.012) | 0.955 (0.017) | 0.942 (0.019) | 1.876 |
| LINEAR | V | 200/200 | 185/1/12/0/2 | 0.006 (0.006) | 0.086 (0.011) | 0.960 (0.014) | 0.940 (0.017) | 0.337 |
| LINEAR | VI | 200/200 | 192/0/8/0/0 | -0.005 (0.008) | 0.109 (0.006) | 0.960 (0.014) | 0.925 (0.019) | 0.521 |
| LINEAR | VII | 197/200 | 171/0/26/0/0 | -0.004 (0.009) | 0.124 (0.010) | 0.964 (0.013) | 0.954 (0.015) | 0.517 |
| RMST | I | 200/200 | 181/2/17/0/0 | 0.122 (0.057) | 0.813 (0.034) | 0.975 (0.011) | 0.960 (0.014) | 3.910 |
| RMST | II | 200/200 | 185/0/15/0/0 | 0.007 (0.035) | 0.497 (0.022) | 0.975 (0.011) | 0.960 (0.014) | 2.246 |
| RMST | III | 198/200 | 191/1/6/0/0 | 0.028 (0.039) | 0.543 (0.030) | 0.955 (0.015) | 0.949 (0.016) | 2.403 |
| RMST | IV | 154/200 | 110/13/31/0/0 | -0.092 (0.105) | 1.305 (0.069) | 0.955 (0.017) | 0.974 (0.013) | 14.770 |
| RMST | V | 200/200 | 187/0/13/0/0 | 0.023 (0.036) | 0.504 (0.027) | 0.975 (0.011) | 0.955 (0.015) | 2.363 |
| RMST | VI | 200/200 | 192/0/8/0/0 | -0.026 (0.057) | 0.800 (0.038) | 0.955 (0.015) | 0.930 (0.018) | 3.799 |
| RMST | VII | 197/200 | 171/0/26/0/0 | -0.010 (0.063) | 0.877 (0.043) | 0.975 (0.011) | 0.959 (0.014) | 3.921 |

**Table S6.** Fresh conditional-design validation. Parentheses give Monte Carlo standard errors. Point fits and intervals have separate denominators.

| Target | System | Point fits | Intervals | Bias (MCSE) | RMSE (MCSE) | Coverage (MCSE) |
| --- | --- | --- | --- | --- | --- | --- |
| LINEAR | I | 198/200 | 198/200 | -0.001 (0.006) | 0.082 (0.004) | 0.949 (0.016) |
| LINEAR | II | 200/200 | 200/200 | -0.005 (0.005) | 0.074 (0.003) | 0.965 (0.013) |
| LINEAR | III | 199/200 | 199/200 | -0.002 (0.004) | 0.054 (0.003) | 0.955 (0.015) |
| LINEAR | IV | 156/200 | 156/200 | 0.027 (0.012) | 0.152 (0.009) | 0.942 (0.019) |
| LINEAR | V | 198/200 | 198/200 | 0.006 (0.004) | 0.063 (0.003) | 0.975 (0.011) |
| LINEAR | VI | 198/200 | 198/200 | 0.010 (0.008) | 0.107 (0.006) | 0.955 (0.015) |
| LINEAR | VII | 197/200 | 197/200 | 0.008 (0.008) | 0.111 (0.006) | 0.954 (0.015) |
| RMST | I | 198/200 | 198/200 | 0.004 (0.057) | 0.794 (0.041) | 0.924 (0.019) |
| RMST | II | 200/200 | 200/200 | -0.018 (0.035) | 0.495 (0.026) | 0.940 (0.017) |
| RMST | III | 199/200 | 199/200 | -0.042 (0.034) | 0.487 (0.023) | 0.950 (0.015) |
| RMST | IV | 156/200 | 156/200 | 0.100 (0.086) | 1.073 (0.063) | 0.974 (0.013) |
| RMST | V | 198/200 | 198/200 | 0.008 (0.034) | 0.473 (0.024) | 0.929 (0.018) |
| RMST | VI | 198/200 | 198/200 | 0.049 (0.051) | 0.722 (0.031) | 0.985 (0.009) |
| RMST | VII | 197/200 | 197/200 | -0.022 (0.066) | 0.925 (0.051) | 0.954 (0.015) |

**Table S7.** Complete construction and fixed-design inference on the same fresh datasets. Coverage is conditional on a completed interval; Monte Carlo standard errors are in parentheses. Coverage and width are not summarised when fewer than five intervals complete.

| Target | System | Inference | Point fits | Intervals | Coverage (MCSE) | Mean width |
| --- | --- | --- | --- | --- | --- | --- |
| LINEAR | I | Complete construction | 100/100 | 1/100 | — | — |
| LINEAR | I | Fixed-design reference | 100/100 | 100/100 | 0.960 (0.020) | 0.312 |
| LINEAR | IV | Complete construction | 80/100 | 0/100 | — | — |
| LINEAR | IV | Fixed-design reference | 80/100 | 80/100 | 0.950 (0.024) | 0.578 |
| LINEAR | VII | Complete construction | 98/100 | 0/100 | — | — |
| LINEAR | VII | Fixed-design reference | 98/100 | 98/100 | 0.980 (0.014) | 0.433 |
| RMST | I | Complete construction | 100/100 | 1/100 | — | — |
| RMST | I | Fixed-design reference | 100/100 | 100/100 | 0.960 (0.020) | 3.049 |
| RMST | IV | Complete construction | 80/100 | 0/100 | — | — |
| RMST | IV | Fixed-design reference | 80/100 | 80/100 | 0.975 (0.017) | 4.926 |
| RMST | VII | Complete construction | 98/100 | 0/100 | — | — |
| RMST | VII | Fixed-design reference | 98/100 | 98/100 | 0.939 (0.024) | 3.386 |

**Table S8.** Fresh concentrated-bridge inference. Parentheses give Monte Carlo standard errors. Unbounded sets have infinite endpoints; other includes finite disconnected or empty sets. All returned shapes enter coverage.

| Target | System | Sets | Single bounded | Unbounded | Other | Fieller coverage | Normal coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LINEAR | I | 200/200 | 181/200 | 19 | 0 | 0.985 (0.009) | 0.980 (0.010) |
| LINEAR | IV | 154/200 | 108/154 | 44 | 2 | 0.955 (0.017) | 0.942 (0.019) |
| LINEAR | VII | 197/200 | 171/197 | 26 | 0 | 0.964 (0.013) | 0.954 (0.015) |
| RMST | I | 200/200 | 181/200 | 19 | 0 | 0.975 (0.011) | 0.960 (0.014) |
| RMST | IV | 154/200 | 110/154 | 44 | 0 | 0.955 (0.017) | 0.974 (0.013) |
| RMST | VII | 197/200 | 171/197 | 26 | 0 | 0.975 (0.011) | 0.959 (0.014) |

**Table S9.** Concentrated-bridge point estimates and 95% Fieller confidence sets, conditional on molecular roles, rank and finite pivots. Full construction completion is reported separately.

| Cohort | Protein | Rank | RMST months and set | Survival points and set | Full completion |
| --- | --- | --- | --- | --- | --- |
| LUAD | BCL2L11 | 2 | -0.01; [-3.87, 4.63] | -1.3; [-24.18, 60.38] | 0.190 |
| LUAD | CASP7 | 2 | -0.51; All real values | -4.1; All real values | 0.353 |
| LUAD | CAV1 | 3 | -1.61; All real values | -9.4; All real values | 0.170 |
| LUAD | CCNB1 | 6 | -1.51; [-81.23, 14.00] | -3.1; [-96.87, 461.56] | 0.343 |
| LUAD | CLDN7 | 5 | -2.09; [-9.80, 4.30] | -8.9; [-50.91, 37.00] | 0.213 |
| LUAD | DIABLO | 2 | -0.15; [-3.17, 2.63] | -1.9; [-17.67, 11.96] | 0.257 |
| LUAD | EEF2 | 2 | -0.20; [-9.89, 2.66] | -1.4; [-41.83, 12.53] | 0.230 |
| LUAD | EIF4G1 | 3 | -3.45; All real values | -19.3; All real values | 0.233 |
| LUAD | FASN | 4 | -4.08; All real values | -18.1; All real values | 0.123 |
| LUAD | G6PD | 2 | 1.43; [-6.40, 76.67] | 10.1; [-29.95, 396.07] | 0.287 |
| LUAD | KDR | 1 | -1.57; [-5.58, 0.76] | -8.4; [-39.58, 5.19] | 0.150 |
| LUAD | PDCD4 | 3 | -0.72; All real values | -4.5; All real values | 0.177 |
| LUAD | PREX1 | 4 | -1.95; All real values | -12.3; All real values | 0.250 |
| LUAD | RBM15 | 3 | 1.89; All real values | -4.7; All real values | 0.203 |
| LUAD | RICTOR | 5 | 0.40; All real values | 2.7; All real values | 0.210 |
| LUAD | SYK | 3 | 1.79; [-1.30, 6.56] | 7.7; [-36.93, 50.37] | 0.190 |
| LUAD | TP53BP1 | 2 | -0.53; [-4.61, 3.22] | 4.2; [-14.52, 24.05] | 0.160 |
| OV | ASNS | 2 | -1.71; [-4.28, 0.13] | 1.5; [-11.53, 11.44] | 0.240 |
| OV | BCL2L11 | 2 | -0.41; All real values | -0.1; All real values | 0.187 |
| OV | BRAF | 3 | 0.93; All real values | 10.5; All real values | 0.183 |
| OV | CDH1 | 3 | -0.64; [-3.66, 3.43] | 5.0; [-2.09, 14.45] | 0.137 |
| OV | CDH2 | 3 | 1.70; All real values | 12.0; All real values | 0.250 |
| OV | CLDN7 | 3 | 0.12; All real values | -2.2; (-∞, 21.80] ∪ [39.93, ∞) | 0.103 |
| OV | GAB2 | 2 | 1.75; [-0.60, 4.74] | 6.8; [-7.13, 18.84] | 0.150 |
| OV | HSPA1A | 2 | 2.71; All real values | 11.7; All real values | 0.223 |

**Table S10.** Complete clinical construction with 300 patient resamples per selected exposure. Completion refers to the whole construction, censoring and bridge pipeline.

| Cohort | Point designs | Attempts | Minimum completion | Median completion | Maximum completion | Full intervals |
| --- | --- | --- | --- | --- | --- | --- |
| OV | 8 | 2400 | 0.103 | 0.185 | 0.250 | 0 |
| LUAD | 17 | 5100 | 0.123 | 0.210 | 0.353 | 0 |

## References

1. Tothill RW, Tinker AV, George J, et al. Novel molecular subtypes of serous and endometrioid ovarian cancer linked to clinical outcome. *Clinical Cancer Research* 2008;14:5198–5208.
2. Cancer Genome Atlas Research Network. Integrated genomic analyses of ovarian carcinoma. *Nature* 2011;474:609–615.
3. Zhang L, Conejo-Garcia JR, Katsaros D, et al. Intratumoral T cells, recurrence, and survival in epithelial ovarian cancer. *New England Journal of Medicine* 2003;348:203–213.
4. Chen C, Jiang L, Fu G, Wang M, Wang Y, Shen B, Liu Z, Wang Z, Hou W, Berceli SA, Wu R. An omnidirectional visualization model of personalized gene regulatory networks. *npj Systems Biology and Applications* 2019;5:38.
5. Dong A, Wu S, Che J, Wang Y, Wu R. idopNetwork: a network tool to dissect spatial community ecology. *Methods in Ecology and Evolution* 2023;14:2272–2283.
6. Miao W, Geng Z, Tchetgen Tchetgen EJ. Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika* 2018;105:987–993.
7. Tchetgen Tchetgen EJ, Ying A, Cui Y, Shi X, Miao W. An introduction to proximal causal inference. *Statistical Science* 2024;39:375–390.
8. Cui Y, Pu H, Shi X, Miao W, Tchetgen Tchetgen EJ. Semiparametric proximal causal inference. *Journal of the American Statistical Association* 2024;119:1348–1359.
9. Ying A, Cui Y, Tchetgen Tchetgen EJ. Proximal causal inference for marginal counterfactual survival curves. arXiv:2204.13144, 2022.
10. Li K, Linderman GC, Shi X, Tchetgen Tchetgen EJ. Regression-based proximal causal inference for right-censored time-to-event data. arXiv:2409.08924, revised 2025.
11. Lipsitch M, Tchetgen Tchetgen E, Cohen T. Negative controls: a tool for detecting confounding and bias in observational studies. *Epidemiology* 2010;21:383–388.
12. Shi X, Miao W, Tchetgen Tchetgen EJ. A selective review of negative control methods in epidemiology. *Current Epidemiology Reports* 2020;7:190–202.
13. Yu M, Shi X, Tchetgen Tchetgen EJ. Fortified proximal causal inference with many invalid proxies. arXiv:2506.13152, 2025.
14. Rakshit P, Shi X, Tchetgen Tchetgen EJ. Adaptive proximal causal inference with some invalid proxies. arXiv:2507.19623, 2025.
15. Meinshausen N, Bühlmann P. High-dimensional graphs and variable selection with the Lasso. *Annals of Statistics* 2006;34:1436–1462.
16. Zhou S. Thresholded Lasso for high dimensional variable selection and statistical estimation. arXiv:1002.1583, 2010.
17. Meinshausen N, Bühlmann P. Stability selection. *Journal of the Royal Statistical Society B* 2010;72:417–473.
18. Cragg JG, Donald SG. Testing identifiability and specification in instrumental variable models. *Econometric Theory* 1993;9:222–240.
19. Stock JH, Yogo M. Testing for weak instruments in linear IV regression. In: Andrews DWK, Stock JH, eds. *Identification and Inference for Econometric Models*. Cambridge University Press; 2005:80–108.
20. Andersen PK, Gill RD. Cox's regression model for counting processes: a large sample study. *Annals of Statistics* 1982;10:1100–1120.
21. Lauritzen SL. *Graphical Models*. Oxford: Clarendon Press; 1996.
22. Chandrasekaran V, Parrilo PA, Willsky AS. Latent variable graphical model selection via convex optimization. *Annals of Statistics* 2012;40:1935–1967.
23. Fieller EC. Some problems in interval estimation. *Journal of the Royal Statistical Society Series B* 1954;16:175–185. doi:10.1111/j.2517-6161.1954.tb00159.x.
24. Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. *Statistics in Medicine* 2019;38:2074–2102. doi:10.1002/sim.8086.
25. Cerami E, Gao J, Dogrusoz U, et al. The cBio Cancer Genomics Portal: an open platform for exploring multidimensional cancer genomics data. *Cancer Discovery* 2012;2:401–404.
26. Li J, Lu Y, Akbani R, et al. TCPA: a resource for cancer functional proteomics data. *Nature Methods* 2013;10:1046–1047.
27. cBioPortal for Cancer Genomics. TCGA PanCancer Atlas Studies [dataset]. Memorial Sloan Kettering Cancer Center; 2018. Accessed 11–12 September 2026. https://www.cbioportal.org/datasets.
28. Martins FC, Couturier DL, Paterson A, et al. Clinical and pathological associations of PTEN expression in ovarian cancer: a multicentre study from the Ovarian Tumour Tissue Analysis Consortium. *British Journal of Cancer* 2020;123:793–802. doi:10.1038/s41416-020-0900-0.
29. Mashiko S, Kitatani K, Toyoshima M, et al. Inhibition of plasminogen activator inhibitor-1 is a potential therapeutic strategy in ovarian cancer. *Cancer Biology & Therapy* 2015;16:253–260. doi:10.1080/15384047.2014.1001271.
30. Kang EY, Weir A, Meagher NS, et al. CCNE1 and survival of patients with tubo-ovarian high-grade serous carcinoma: An Ovarian Tumor Tissue Analysis consortium study. *Cancer* 2023;129:697–713. doi:10.1002/cncr.34582.
31. Duckworth C, Zhang L, Carroll SL, Ethier SP, Cheung HW. Overexpression of GAB2 in ovarian cancer cells promotes tumor growth and angiogenesis by upregulating chemokine expression. *Oncogene* 2016;35:4036–4047. doi:10.1038/onc.2015.472.
32. Sun B, Jensen NR, Chung D, et al. Synergistic effects of SHP2 and PI3K pathway inhibitors in GAB2-overexpressing ovarian cancer. *American Journal of Cancer Research* 2019;9:145–159.
33. Bickel PJ, Freedman DA. Some asymptotic theory for the bootstrap. *Annals of Statistics* 1981;9:1196–1217.
34. van der Vaart AW. *Asymptotic Statistics*. Cambridge University Press; 1998.
35. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B* 1995;57:289–300.
36. Nilsson MB, Giri U, Gudikote J, et al. KDR amplification is associated with VEGF-induced activation of the mTOR and invasion pathways but does not predict clinical benefit to the VEGFR TKI vandetanib. *Clinical Cancer Research* 2016;22:1940–1950. doi:10.1158/1078-0432.CCR-15-1994.
37. Chen Y, Jin L, Ma Y, et al. BACH1 promotes lung adenocarcinoma cell metastasis through transcriptional activation of ITGA2. *Cancer Science* 2023;114:3568–3582. doi:10.1111/cas.15884.
