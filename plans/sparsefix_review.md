# Correctness Review: `SPDEMaternReg` in `sparsefix` branch

## Context

The `sparsefix` branch adds `SPDEMaternReg` — a sparse Matern nu=1 regularization class based on Lindgren, Rue & Lindstrom (2011). The branch also fixes sparse matrix handling throughout cofi's solver pipeline. This review checks the implementation against the paper's mathematics.

## Key file: `/home/mk/projects/cofi/src/cofi/utils/_reg_matern.py`

---

## Issue 1 (Bug): `sigma` is NOT the marginal standard deviation

**Severity: High**

The docstring claims sigma is the "Prior marginal standard deviation." This is **incorrect**.

The implementation builds `R = (kappa^2 I - L) / sigma`, giving precision `Q = R^T R = (kappa^2 I - L)^2 / sigma^2`.

For a Matern field with nu=1 in d=2, the paper (p. 427) gives the marginal variance as:

```
sigma^2_marginal = Gamma(nu) / (Gamma(nu + d/2) * (4*pi)^(d/2) * kappa^(2*nu))
                 = 1 / (4*pi*kappa^2)       [for nu=1, d=2]
```

The correct SPDE parameterization (eq 12, p. 436) uses a scaling parameter tau, where:

```
Q = tau^2 * (kappa^2 I - L)^2
```

and `sigma^2_marginal = 1 / (4*pi*kappa^2*tau^2)`.

To achieve a desired marginal std `sigma_target`, you need:

```
tau = 1 / (sigma_target * 2*sqrt(pi) * kappa)
    = L_corr / (sigma_target * 2*sqrt(pi))
```

The current code uses `R = (kappa^2 I - L) / sigma`, i.e. `tau = 1/sigma`. This gives actual marginal std:

```
sigma_actual = 1 / (tau * 2*sqrt(pi) * kappa)
             = sigma * L_corr / (2*sqrt(pi))
             != sigma
```

**Fix**: Compute tau from sigma correctly: `tau = L_corr / (sigma * 2*sqrt(pi))`, then `R = tau * (kappa^2 I - L)`.

---

## Issue 2 (Minor): "Cholesky-like factor" is misleading

**Severity: Low (documentation only)**

The docstring (line 31) calls R "the Cholesky-like factor." R = (kappa^2 I - L)/sigma is **symmetric**, not lower-triangular. It's a symmetric square-root factor: Q = R^T R = R^2. Calling it "Cholesky-like" suggests triangularity.

**Fix**: Replace "Cholesky-like factor" with "symmetric square-root factor" or simply "precision factor."

---

## Issue 3 (Design): Assumes unit grid spacing

**Severity: Medium**

The implementation constructs the discrete Laplacian assuming grid spacing h=1 in both dimensions. On a regular grid with spacing h, the discrete Laplacian should be L/h^2, and the FEM mass matrix (even lumped) scales as h^d.

The paper's Result 2 (eq 10, p. 431) uses:
- C_ij = <psi_i, psi_j> (mass matrix, scales as h^2 for 2D)
- G_ij = <grad psi_i, grad psi_j> (stiffness matrix, independent of h in 2D)
- K_{kappa^2} = kappa^2 * C + G

With lumped mass C_hat (diagonal, entries ~ h^2), the precision becomes:
```
Q = K C_hat^{-1} K = (kappa^2 C_hat + G) C_hat^{-1} (kappa^2 C_hat + G)
```

The current code effectively sets h=1, making C_hat = I. This means L_corr must be specified in **grid cells**, not physical units. The docstring does say "in grid cells" which is consistent, but adding a `grid_spacing` parameter would make the class more useful for real-world problems where the user thinks in physical units.

**Fix (optional)**: Add an optional `grid_spacing` parameter (default 1.0) and scale the Laplacian accordingly. Low priority — the current behavior is internally consistent.

---

## Issue 4 (Correct): Neumann boundary conditions

The 1D Laplacian uses:
```
L[0,0] = -1, L[0,1] = 1      (forward difference at left boundary)
L[-1,-2] = 1, L[-1,-1] = -1   (backward difference at right boundary)
```

This correctly implements Neumann (zero-flux) BCs. The paper discusses Neumann BCs in Section 2.3 and Appendix A.4, noting they inflate boundary variance. The implementation matches the paper.

---

## Issue 5 (Correct): 2D Laplacian via Kronecker product

```python
L_full = kron(I_lon, L_lat) + kron(L_lon, I_lat)
```

This is the standard tensor-product construction for a 2D Laplacian on a regular grid. Correct.

---

## Issue 6 (Correct): Precision factor structure for nu=1

For nu=1 (alpha=2 in d=2), Result 1 in the paper (p. 428-429) shows the GMRF representation on a regular lattice is obtained by convolving the first-order model (eq 6) by itself. On a regular grid with lumped mass, this gives:

```
Q_{2,kappa^2} = K^2 = (kappa^2 I - L)^2
```

(since K = kappa^2 * I + G and G = -L for unit-spacing lumped mass). The implementation's `R = (kappa^2 I - L)/sigma` with `Q = R^2` matches this structure (modulo the sigma scaling issue in Issue 1).

---

## Issue 7 (Minor): L_corr vs range parameter

**Severity: Low (documentation)**

The paper defines the range parameter rho = sqrt(8*nu)/kappa. For nu=1: rho = sqrt(8)/kappa = sqrt(8) * L_corr ~ 2.83 * L_corr.

The docstring correctly describes L_corr as the distance where rho(r) ~ 0.60 (since K_1(1) ~ 0.602). But users familiar with the paper's "range" parameter (where correlation drops to ~0.13) may confuse the two.

**Fix (optional)**: Add a note clarifying the relationship: `range = sqrt(8) * L_corr`.

---

## Issue 8: No dedicated tests for SPDEMaternReg

**Severity: Medium**

No test file exists for `SPDEMaternReg`. Recommended tests:
- Verify R(m0) = 0 when reference_model = m0
- Verify precision matrix Q = R^T R is symmetric positive definite
- Verify gradient and hessian against finite differences
- Compare empirical covariance (from Q^{-1}) against theoretical Matern correlation for a small grid
- Test edge cases: L_corr near boundary, non-square grids

---

## Summary

| Issue | Severity | Type | Action needed |
|-------|----------|------|---------------|
| 1. sigma != marginal std | High | Bug | Fix tau computation |
| 2. "Cholesky-like" | Low | Doc | Rename to "precision factor" |
| 3. Unit grid spacing | Medium | Design | Document or add grid_spacing param |
| 4. Neumann BCs | - | Correct | None |
| 5. Kronecker Laplacian | - | Correct | None |
| 6. Q structure for nu=1 | - | Correct | None |
| 7. L_corr vs range | Low | Doc | Add clarifying note |
| 8. No tests | Medium | Testing | Add test suite |

## Recommended fix order

1. Fix Issue 1 (sigma scaling) — this is a mathematical bug
2. Add tests (Issue 8) — verify the fix and prevent regressions
3. Fix documentation (Issues 2, 7)
4. Optionally add grid_spacing (Issue 3)

## Verification

After fixes:
1. For a large grid (e.g. 100x100), compute Q_inv = inv(Q) and check that diagonal entries approximate `sigma^2` (the user-specified marginal variance)
2. Check that off-diagonal entries of Q_inv match the Matern correlation `r*K_1(r/L_corr) / (L_corr * K_1(1))` scaled by sigma^2
3. Run `python -m pytest tests/` to ensure no regressions
