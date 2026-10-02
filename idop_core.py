# -*- coding: utf-8 -*-
"""
idopNetwork core for static data: transformation, niche ordering, power-law
fitting, patient-level edge selection, and the weak-form ODE decomposition.

@author: Yu Wang (yuwang@bimsa.cn)
"""


import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from sklearn.linear_model import Lasso



# 变换器
def data_transformation(
    data    : pd.DataFrame,
    method  : str,
) -> pd.DataFrame:
    r"""Apply data scaling or transformation to numeric features.

    Note
    ----
    For the original static data
    All transformations are applied column-wise (per feature).

    Parameters
    ----------
    data: (n_samples, n_features) | pd.DataFrame
    method : str
        Supported options:
        - "None"            : Return an unchanged copy of the input.
        - "Minmax_0_1"      : Scale each feature to [0, 1].
        - "Minmax_neg1_1"   : Scale each feature to [-1, 1].
        - "Zscore_shift"    : Z-score then shifted to positive.
        - "Log10_1p"        : log10(1 + x) transformation.
        - "Shift_min"       : Per-column shift so every feature min equals 1 (> 0).

    Returns
    -------
    pd.DataFrame
        Transformed data with the same index and columns as input.

    """

    if method == "None":
        return data

    if method in {"Minmax_0_1", "Minmax_neg1_1"}:
        minimum = data.min(axis=0)
        span = data.max(axis=0) - minimum
        safe_span = span.mask(span == 0.0, 1.0)
        unit = (data - minimum) / safe_span
        return unit if method == "Minmax_0_1" else 2.0 * unit - 1.0

    if method == "Zscore_shift":
        mean = data.mean(axis=0)
        standard_deviation = data.std(axis=0, ddof=0)
        safe_standard_deviation = standard_deviation.mask(
            standard_deviation == 0.0, 1.0
        )
        # 获取 Z-score 值
        zscore = (data - mean) / safe_standard_deviation
        # 将 Z-score 值转换为非负值
        return zscore - zscore.min(axis=0) + 1.0
        
    if method == "Log10_1p":
        if (data <= -1.0).any(axis=None):
            row, column = np.argwhere(data.to_numpy() <= -1.0)[0]
            raise ValueError(
                "Log10_1p requires every value to be greater than -1; invalid value at "
                f"row {data.index[row]!r}, feature {data.columns[column]!r}."
            )
        return np.log10(1.0 + data)


    if method == "Shift_min":
        # Pure per-column translation; guarantees min == 1 (> 0) and preserves
        # row-sum ordering, so the quasi-time index is unchanged.
        return data - data.min(axis=0) + 1.0

    raise ValueError(f"Unsupported method: {method}.")


# 拟动态器
def data_quasi_dynamic(
    data        : pd.DataFrame,
    index_log1p : bool = False,
) -> pd.DataFrame:
    r"""Static dataFrame to quasi dynamic dataFrame.

    For the transformed data.

    Note
    ----
    The original static data $y_j(s_i)$, where
    $i = 1,\cdots,n$ is the sample index,
    $j = 1,\cdots,p$ is the feature index,
    to transformed data $\tilde{y}_j(s_i)$
    to quasi dynamic data $\tilde{y}_j(tau_i)$, where
    $\tau_i = \sigma(T_i), \text{ s.t. } \tau_1 \leq \cdots \leq \tau_n$, where
    $T_i = \sum_{j = 1}^{p} y_j(s_i)$, where
    $\sigma$ is the ascending order function.

    Parameters
    ----------
    data: (n_samples, n_features) | pd.DataFrame

    Returns
    -------
    quasi_dynamic_data: (n_samples, n_features) | pd.DataFrame
        Quasi-dynamic data sorted by row sum in ascending order.

    """

    # sum features for each sample
    row_sum = data.sum(axis=1)
    # sort samples by row sum
    sort_pos = np.argsort(row_sum.to_numpy(), kind="stable")
    # reorder data using the sorted sample labels
    quasi_dynamic_data = data.iloc[sort_pos].copy()
    # obtain the sorted quasi-time values
    quasi_time = row_sum.to_numpy(dtype=float)[sort_pos]

    # apply log1p transformation if required
    if index_log1p:
        if np.any(quasi_time <= 0):
            raise ValueError("Quasi-time values must be positive.")
        quasi_time = np.log1p(quasi_time)

    # set index to quasi-time
    quasi_dynamic_data.index = pd.Index(
        quasi_time,
        name="quasi time"
    )

    return quasi_dynamic_data

# 拟合器

# 幂函数
def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """幂函数：y = a·x^b。"""
    return a * np.power(x, b)

# 幂函数参数
def get_power_function_params(
    data        : pd.DataFrame,
    include_r2  : bool = False,
) -> pd.DataFrame:
    r"""

    Parameters
    ----------
    data: (n_samples, n_features) | pd.DataFrame
    include_r2 : bool, optional
        Whether to include the R² value in the output. Default is False.

    Returns
    -------
    power_function_params: (n_features, 2 or 3) | pd.DataFrame
        The power function parameters for each feature, where 
        rows are features and columns are ["a", "b"] or ["a", "b", "R²"].

    """

    quasi_time = data.index.to_numpy(dtype=float)

    if not np.isfinite(quasi_time).all():
        raise ValueError("quasi-time must contain only finite values.")
    if np.any(quasi_time <= 0):
        raise ValueError(
            "quasi-time must be strictly positive for stable power-law fitting."
        )

    results = []

    for feature, series in data.items():
        y = series.to_numpy(dtype=float)

        (a, b), _ = curve_fit(
            power_law,
            quasi_time,
            y,
            p0=(1.0, 0.5),
            maxfev=10_000,
        )

        result = {
            "feature": feature,
            "a": a,
            "b": b,
        }

        if include_r2:
            predicted = power_law(quasi_time, a, b)
            ss_res = float(np.sum((y - predicted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            result["R²"] = (
                1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
            )

        results.append(result)

    return pd.DataFrame(results).set_index("feature")

# 幂函数样本
def get_power_function_samples(
    params      : pd.DataFrame,
    quasi_time  : pd.Index | np.ndarray,
    n_samples   : int = 100,
) -> pd.DataFrame:
    r"""

    Parameters
    ----------
    params      : (n_features, 2 or 3) | pd.DataFrame
    quasi_time  : (n_samples,) | pd.Index or np.ndarray
    n_samples   : int

    Returns
    -------
    power_function_samples: (n_samples, n_features) | pd.DataFrame

    """

    x = np.asarray(quasi_time, dtype=float)
    sample_x = np.linspace(x.min(), x.max(), n_samples)

    fitted = {
        feature: power_law(
            sample_x,
            params.at[feature, "a"],
            params.at[feature, "b"],
        )
        for feature in params.index
    }

    return pd.DataFrame(
        fitted,
        index=pd.Index(sample_x, name=getattr(quasi_time, "name", None)),
    )


# 选边器
def edge_select(
    data_ecg: pd.DataFrame,
    alpha: float = 0.1, # LASSO regularization parameter
    k: int = 10,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Select edges from fitted idopNetwork data.

    Note
    ----
    从数据中恢复稀疏有向图.
    对每个 target 变量, 其支撑集来源于除 target 变量以外的所有变量.
    每个时间窗内对 x(逐列) 与 y 做 z-score 标准化后再拟合 LASSO,
    系数还原回原始尺度后判断是否为 0.

    Parameters
    ----------
    data_ecg : pd.DataFrame
        Shape = (n_timepoints, n_channels)
        columns = ["time", "feature1", "feature2", ..., "featureN"]
    alpha : float, optional
        LASSO regularization parameter. The default is 0.5.
    k : int, optional
        Number of roughly equal windows to split the idopNetwork into. The default is 10.
    threshold : float, optional
        超过 90 % 以上的 window 中都出现的 source 变量 才会进入支撑集.
        

    Returns
    -------
    edge_supports : pd.DataFrame
        Columns = ["target", "source"]
        Example:
            target | source
            I      | {II,aVR}
            II     | {aVR,aVF}

    """
    lead_columns = [column for column in data_ecg.columns if column != "time"]
    if len(lead_columns) < 2:
        raise ValueError("data_ecg must contain at least two lead columns")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(data_ecg):
        raise ValueError("k cannot exceed the number of time points")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    windows = np.array_split(data_ecg[lead_columns].to_numpy(dtype=float), k, axis=0)
    edge_counts = {
        target: {source: 0 for source in lead_columns if source != target}
        for target in lead_columns
    }

    for window in windows:
        for target_idx, target in enumerate(lead_columns):
            source_indices = [i for i in range(len(lead_columns)) if i != target_idx]
            x = window[:, source_indices]
            y = window[:, target_idx]

            # 标准化 x(逐列) 与 y，让 LASSO 目标尺度 O(1)，改善收敛
            x_mean = x.mean(axis=0)
            x_scale = x.std(axis=0)
            x_scale[x_scale == 0] = 1.0
            x_std = (x - x_mean) / x_scale

            y_mean = float(y.mean())
            y_scale = float(y.std())
            if y_scale == 0.0:
                y_scale = 1.0
            y_std = (y - y_mean) / y_scale

            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=100_000, tol=1e-5)
            model.fit(x_std, y_std)

            # 系数还原到原始尺度: beta_orig = beta_std * y_scale / x_scale
            coefficients = model.coef_ * (y_scale / x_scale)
            for local_idx, coefficient in enumerate(coefficients):
                if coefficient != 0.0:
                    source = lead_columns[source_indices[local_idx]]
                    edge_counts[target][source] += 1

    rows = []
    for target in lead_columns:
        selected = [
            source
            for source in lead_columns
            if source != target and edge_counts[target][source] / k > threshold
        ]
        rows.append({"target": target, "source": "{" + ",".join(selected) + "}"})

    return pd.DataFrame(rows, columns=["target", "source"])


# 求解器内核
def _solve_ode_decomposition(
    quasi_dynamic_data: pd.DataFrame,
    samples           : pd.DataFrame,
    edge_supports     : pd.DataFrame,
    basis_order       : int,
    ridge             : float = 1e-6,
    cross_l1_scale    : float = 5e-4,
    gap_min           : float = 1e-6,
    y_pad             : float = 0.1,
    y_soft_scale      : float = 100.0,
    effect_cap        : float | None = 2.0,
    effect_soft_scale : float = 50.0,
) -> dict:
    r"""Shared ODE weak-form solve: integral Legendre basis + convex optimization.

    Both ``ode_solver`` (edge list) and ``plot_effect_decomposition`` (curves)
    consume this so their results stay consistent.

    Per target a cvxpy problem is solved (mirroring the reference solver):
    - objective  : ||y - intercept - D θ||² + ridge·||θ||²
                   + cross_l1_scale·||θ_cross||₁
    - constraint : intercept + D_self·θ_self ≥ 0            (self dynamics stay valid)
    - constraint : cross cumulative effects all ≥ gap_min or all ≤ -gap_min
                   (one-signed cross contributions, two directions tried)
    - constraint : reconstructed y within [obs_min - y_pad·span,
                   obs_max + y_pad·span]                     (y 不爆表, 软惩罚
                   y_soft_scale·excess², 强系数时近似硬约束且不会不可行)
    - penalty    : each source cumulative effect |effect| > effect_cap·span is
                   penalized by effect_soft_scale·(excess)²  (soft, 防止大正大负
                   抵消的虚高单项; effect_cap=None 关闭)
    - constraint : intercept + D_self·θ_self ≥ 0            (self dynamics stay valid)
    - constraint : cross cumulative effects all ≥ gap_min or all ≤ -gap_min
                   (one-signed cross contributions, two directions tried)

    Returns
    -------
    dict with keys:
    - "features"              : list of feature names
    - "sample_tau"            : (n_t,) time grid (``samples.index``)
    - "intercepts"            : (n_features,) initial fitted sample states
    - "response"              : (n_t, n_features) observed y_j interpolated
    - "predicted_states"      : (n_t, n_features) reconstructed y_j
    - "interaction_functions" : {(target, source): (n_t,) cumulative effect}
    - "support_sets"          : {target: [sources]}
    """

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "ode_solver optimization requires cvxpy."
        ) from exc

    if not isinstance(quasi_dynamic_data, pd.DataFrame):
        raise TypeError("quasi_dynamic_data must be a pandas DataFrame.")
    if not isinstance(samples, pd.DataFrame):
        raise TypeError("samples must be a pandas DataFrame.")
    if not isinstance(edge_supports, pd.DataFrame):
        raise TypeError("edge_supports must be a pandas DataFrame.")
    if basis_order < 0:
        raise ValueError("basis_order must be non-negative.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if y_pad < 0:
        raise ValueError("y_pad must be non-negative.")
    if y_soft_scale < 0:
        raise ValueError("y_soft_scale must be non-negative.")
    if effect_cap is not None and effect_cap <= 0:
        raise ValueError("effect_cap must be positive or None.")
    if effect_soft_scale < 0:
        raise ValueError("effect_soft_scale must be non-negative.")
    if not {"target", "source"}.issubset(edge_supports.columns):
        raise ValueError("edge_supports must contain columns ['target', 'source'].")

    features = list(samples.columns)
    missing = [f for f in quasi_dynamic_data.columns if f not in features]
    if missing:
        raise ValueError(
            "quasi_dynamic_data has features missing from samples: "
            f"{missing}."
        )

    # 解析支撑集: {"a,b"} -> {target: [source, ...]}
    support_sets: dict[str, list[str]] = {}
    for _, row in edge_supports.iterrows():
        target = str(row["target"])
        inner = str(row["source"]).strip().strip("{}")
        sources = [s.strip() for s in inner.split(",") if s.strip()]
        support_sets[target] = [s for s in sources if s in features]

    # 把 quasi_dynamic_data 插值到 samples 的时间网格, 作为弱形式响应
    sample_tau = samples.index.to_numpy(dtype=float)
    observed_tau = quasi_dynamic_data.index.to_numpy(dtype=float)
    observed_values = quasi_dynamic_data[features].to_numpy(dtype=float)
    order = np.argsort(observed_tau, kind="stable")
    observed_tau, observed_values = observed_tau[order], observed_values[order]
    unique_tau, unique_idx = np.unique(observed_tau, return_index=True)
    n_features = len(features)
    response = np.column_stack(
        [
            np.interp(
                sample_tau,
                unique_tau,
                observed_values[unique_idx, j],
            )
            for j in range(n_features)
        ]
    )
    n_t = response.shape[0]
    max_order = basis_order + 1

    sample_states = samples[features].to_numpy(dtype=float)
    if not np.isfinite(sample_states).all():
        raise ValueError("samples must contain only finite values.")

    # 积分 Legendre 基: basis[source, t, r] = ∫_0^t P_r(x_k(s)) dx_k(s)
    from scipy.special import eval_legendre

    basis = np.zeros((n_features, n_t, max_order), dtype=float)
    for source in range(n_features):
        source_values = sample_states[:, source]
        minimum = float(np.min(source_values))
        maximum = float(np.max(source_values))
        if maximum > minimum:
            scaled = -1.0 + 2.0 * (source_values - minimum) / (maximum - minimum)
        else:
            scaled = np.zeros_like(source_values)
        increments = np.diff(scaled)
        for r in range(max_order):
            polynomial = eval_legendre(r, scaled)
            if n_t > 1:
                trapezoids = 0.5 * (polynomial[1:] + polynomial[:-1]) * increments
                basis[source, 1:, r] = np.cumsum(trapezoids)

    intercepts = sample_states[0].copy()
    interaction_functions: dict[tuple[str, str], np.ndarray] = {}
    predicted_states = np.empty_like(response)

    for target_idx, target in enumerate(features):
        block_sources = [target]
        for source in support_sets.get(target, []):
            if source not in block_sources:
                block_sources.append(source)
        source_indices = [features.index(s) for s in block_sources]
        design = np.column_stack([basis[i] for i in source_indices])

        adjusted_response = response[:, target_idx] - intercepts[target_idx]
        self_positions = np.arange(max_order)
        cross_positions = np.arange(max_order, design.shape[1])

        directions = (
            ["self_above_total", "self_below_total"]
            if len(cross_positions)
            else ["self_only"]
        )
        obs_min = float(np.min(response[:, target_idx]))
        obs_max = float(np.max(response[:, target_idx]))
        span = max(obs_max - obs_min, 1e-12)
        y_lower = obs_min - y_pad * span
        y_upper = obs_max + y_pad * span

        best_theta = None
        best_objective = np.inf

        for direction in directions:
            theta_variable = cp.Variable(design.shape[1])
            total_dynamic = design @ theta_variable
            constraints = [
                intercepts[target_idx]
                + design[:, self_positions] @ theta_variable[self_positions]
                >= 0.0,
            ]
            if len(cross_positions):
                cross_dynamic = (
                    design[:, cross_positions] @ theta_variable[cross_positions]
                )
                constrained_cross = cross_dynamic[1:] if n_t > 1 else cross_dynamic
                if direction == "self_above_total":
                    constraints.extend(
                        [
                            constrained_cross <= -gap_min,
                            cp.sum(-constrained_cross)
                            / max(n_t - 1, 1)
                            >= gap_min,
                        ]
                    )
                else:
                    constraints.extend(
                        [
                            constrained_cross >= gap_min,
                            cp.sum(constrained_cross)
                            / max(n_t - 1, 1)
                            >= gap_min,
                        ]
                    )
            regularization = ridge * cp.sum_squares(theta_variable)
            if len(cross_positions):
                regularization += cross_l1_scale * cp.norm1(
                    theta_variable[cross_positions]
                )
            # 软惩罚: 重构 y 超出 [obs_min-y_pad·span, obs_max+y_pad·span] 的部分
            excess_y = cp.pos(
                intercepts[target_idx] + total_dynamic - y_upper
            ) + cp.pos(y_lower - intercepts[target_idx] - total_dynamic)
            regularization += y_soft_scale * cp.sum_squares(excess_y)
            if effect_cap is not None:
                cap = effect_cap * span
                for block_index in range(len(block_sources)):
                    block_slice = slice(
                        block_index * max_order,
                        (block_index + 1) * max_order,
                    )
                    block_effect = (
                        design[:, block_slice] @ theta_variable[block_slice]
                    )
                    excess = cp.pos(block_effect - cap) + cp.pos(
                        -cap - block_effect
                    )
                    regularization += effect_soft_scale * cp.sum_squares(excess)
            problem = cp.Problem(
                cp.Minimize(
                    cp.sum_squares(adjusted_response - total_dynamic)
                    + regularization
                ),
                constraints,
            )
            try:
                problem.solve(solver=cp.CLARABEL, verbose=False)
            except cp.error.SolverError:
                problem.solve(solver=cp.SCS, verbose=False)
            if theta_variable.value is None or problem.status not in {
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            }:
                continue

            theta_value = np.asarray(theta_variable.value, dtype=float).reshape(-1)
            prediction = intercepts[target_idx] + design @ theta_value
            residual = float(
                np.sum(np.square(response[:, target_idx] - prediction))
            )
            if residual < best_objective:
                best_objective = residual
                best_theta = theta_value

        if best_theta is None:
            raise ValueError(
                "No feasible constrained ODE solution for target "
                f"{target!r}. Try a smaller basis_order, a larger ridge, "
                "or a looser edge support."
            )

        total_effect = intercepts[target_idx] * np.ones(n_t)
        for block_index, source in enumerate(block_sources):
            block = best_theta[block_index * max_order:(block_index + 1) * max_order]
            effect = basis[features.index(source)] @ block
            interaction_functions[(target, source)] = effect
            total_effect = total_effect + effect
        predicted_states[:, target_idx] = total_effect

    return {
        "features": features,
        "sample_tau": sample_tau,
        "intercepts": intercepts,
        "response": response,
        "predicted_states": predicted_states,
        "interaction_functions": interaction_functions,
        "support_sets": support_sets,
    }


# 求解器
def ode_solver(
    quasi_dynamic_data: pd.DataFrame, # 用做目标，方程左侧
    samples: pd.DataFrame, # 基展开的输入
    edge_supports: pd.DataFrame, # 选边器的输出
    basis_order: int,
    ridge: float = 1e-6,
    cross_l1_scale: float = 5e-4,
    gap_min: float = 1e-6,
    y_pad: float = 0.1,
    y_soft_scale: float = 100.0,
    effect_cap: float | None = 2.0,
    effect_soft_scale: float = 50.0,
) -> pd.DataFrame:
    r"""Solve the ODE system for the idopNetwork.

    Note
    ----
    方程:
    \[
        dy_{j}(t)/dt = Q_{j \gets j}(x_{j}(t)) + \sum_{k \in S_j} Q_{j \gets k}(x_{k}(t))
    \]
    其中
    - $S_j$ 是变量 $j$ 的支撑集, 由选边器确定.
    - $y_j(t)$ 是拟动态数据.
    - $x_j(t)$ 是拟合曲线的采样数据.
    - $Q_{j \gets j}(x_j(t))$ 是一个基函数展开
        - 选取 Legendre 多项式 P_{r}, 其中 $r = 0,\dots,$basis_order 是基函数的阶数.
        - Q_{j \gets k}(x_k(t)) = \sum_r c_{jkr} P_r(x_k(t))

    两边对 $t$ 积分获取弱形式: 对 $Q$ 做积分 Legendre 基展开,
    响应直接用 $y_j(t)$ 本身(插值到采样网格), 避免对噪声数据求导.
    对每个 target 在支撑集 $\{j\}\cup S_j$ 上用 cvxpy 求解约束凸优化
    (L2 + 跨源 L1 + 自身非负 + 跨源符号一致约束), 见 _solve_ode_decomposition.

    Returns
    -------
    edgelist : pd.DataFrame
        仅含支撑集上的有向边, 列 = ["source", "target", "weight"],
        与 network_data.py 输出的带符号边表格式一致.
        weight 为正负号 \times \sqrt(mean(effect^2)), effect 为该边重构的作用曲线.
    """

    decomposition = _solve_ode_decomposition(
        quasi_dynamic_data, samples, edge_supports, basis_order, ridge,
        cross_l1_scale, gap_min, y_pad, y_soft_scale, effect_cap,
        effect_soft_scale,
    )
    features = decomposition["features"]
    interaction_functions = decomposition["interaction_functions"]
    support_sets = decomposition["support_sets"]

    rows = []
    for target in features:
        for source in support_sets.get(target, []):
            effect = interaction_functions[(target, source)]
            weight = np.sqrt(float(np.mean(np.square(effect))))
            if float(np.mean(effect)) < 0.0:
                weight = -weight
            rows.append({"source": source, "target": target, "weight": weight})

    return pd.DataFrame(rows, columns=["source", "target", "weight"])
