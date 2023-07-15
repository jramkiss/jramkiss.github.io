# Probability and Statistics

## Section 3: Stochastic Processes

### Chebyshev's Inequality

Assume that $Var(X) < \infty$. Let $c > 0$, then:

$$
\mathcal{P}(|X - E[X]| > c) = \mathcal{E}(\mathcal{I}_{|X - \mathcal{E}(X)| > c}) \le \mathcal{E}(|X - \mathcal{E}(X)|^2 / c^2) = \frac{Var(X)}{c}
$$

The probability that $X$ differs from its mean, $\mathcal{E}(X)$ is bounded above by a quadratic