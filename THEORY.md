# Order inversion under radial reliability scores

## Scope

Scope. This note concerns only $\mathcal{S}_{\mathrm{radial}}(\mu,\Sigma)$. It makes no claim about MSP, energy, kNN, or any non-radial score.

## Scores

Scores are inlier scores: larger means more in-distribution.

$\mu$ and $\Sigma$ are estimated on the ID cloud only. The Mahalanobis radius is

$$
d_\Sigma(z)=\lVert z-\mu\rVert_\Sigma=\sqrt{(z-\mu)^\top\Sigma^{-1}(z-\mu)}.
$$

$\Sigma$ is `sklearn.covariance.LedoitWolf` fit once on the raw ID embeddings. If the fitted covariance is not usable, the run stops. No ridge is added, and Ledoit–Wolf is not refit after the oracle map.

## Class

$$
\mathcal{S}_{\mathrm{radial}}(\mu,\Sigma)=\{\,s=\varphi(-d_\Sigma):\varphi\text{ is Borel and strictly increasing}\,\}.
$$

Borel measurability of $\varphi$ is part of the definition, so the pushforward law of $s(X)$ exists. Constant or non-strict $\varphi$ are outside the class.

## Product measure

Ranking probabilities are population two-sample quantities on $P_{\mathrm{new}}\otimes P_{\mathrm{ID}}$. That is the population AUROC. It is not a claim that patches inside one batch are independent.

## Equivalence

If the law of $d_\Sigma$ has no atoms, then for every $s$ in the class,

$$
\mathbb{P}\bigl(s(X_{\mathrm{new}})>s(X_{\mathrm{ID}})\bigr)=\mathbb{P}\bigl(d_\Sigma(X_{\mathrm{new}})<d_\Sigma(X_{\mathrm{ID}})\bigr).
$$

One strictly increasing radial score inverts if and only if every score in the class inverts if and only if the radius inverts. This is radial order inversion.

## Oracle

For an invertible linear map $A$ with $A\Sigma A^\top=\Sigma$,

$$
T_A(z)=A(z-\mu_{\mathrm{new}})+\mu,
$$

and $d_\Sigma(T_A(z))=\lVert z-\mu_{\mathrm{new}}\rVert_\Sigma$, independent of $A$. The only map implemented is $A=I$:

$$
T(z)=z-\mu_{\mathrm{new}}+\mu.
$$

Rotations are not searched. Rotations are invisible to this class and visible to logits. MSP is outside the class. MSP disagreement is not a falsification.

## Pre-computable radius

Under every such oracle, the post-map radius of a new-site point equals $\lVert x-\mu_{\mathrm{new}}\rVert_\Sigma$ with the ID $\Sigma$, not the new site’s covariance. $\mu_{\mathrm{new}}$ uses site identity. This is an oracle alignment, not a detector. It is not an OOD method.

## Characterization

The class inverts if and only if

$$
\mathbb{P}(d_{\mathrm{new}}<d_{\mathrm{ID}})>\tfrac12,
$$

with $d_{\mathrm{new}}=\lVert X_{\mathrm{new}}-\mu_{\mathrm{new}}\rVert_\Sigma$ and $d_{\mathrm{ID}}=\lVert X_{\mathrm{ID}}-\mu\rVert_\Sigma$.

## Quantile certificate

The certificate is sufficient and not necessary. For $\varepsilon<1-1/\sqrt{2}$, if

$$
F^{-1}_{d_{\mathrm{new}}}(1-\varepsilon)\le F^{-1}_{d_{\mathrm{ID}}}(\varepsilon),
$$

then on continuous laws and the product measure the probability is at least $(1-\varepsilon)^2>1/2$. The Uniform pair $d_{\mathrm{ID}}\sim\mathrm{Unif}[0,1]$, $d_{\mathrm{new}}\sim\mathrm{Unif}[0,0.9]$ inverts and satisfies no such $\varepsilon$.

The only $\varepsilon$ values evaluated, in the toy and on real embeddings, are $0.05$, $0.10$, and $0.20$. A continuous $\varepsilon$ is not searched. The certificate fires if at least one of these three satisfies the inequality. The smallest of the three that fires is recorded. If none fire, the record is none.

## Observability is not a consequence

Small variance of $s$ does not imply failure observability $1/2$. Counterexample: $s\mid\mathrm{correct}=0$, $s\mid\mathrm{incorrect}=\varepsilon$. The separate lemma, not a theorem of this run and not tested on medical data, is

$$
\bigl|\mathcal{O}-\tfrac12\bigr|\le\tfrac12\,\mathrm{TV}\bigl(\mathcal{L}(s\mid\mathrm{correct}),\mathcal{L}(s\mid\mathrm{incorrect})\bigr).
$$

## What else is, and is not, in the class

Euclidean $\lVert z-\mu\rVert_2$ is the class $\Sigma\propto I$. Disagreement with Ledoit–Wolf Mahalanobis is allowed. Disagreement between Mahalanobis and the Gaussian log-likelihood of $\mathcal{N}(\mu,\Sigma)$ is a bug: both are strictly increasing functions of $-d_\Sigma$.

kNN is not in the class.

Geodesic scores, kernel scores, DANN, and information bottleneck are not part of this note.

## Proof sketch

1. Probability is on the product $P_{\mathrm{new}}\otimes P_{\mathrm{ID}}$. For continuous laws of $d_\Sigma$ there are no atoms, so boundary ties have probability 0. A Borel strictly increasing $\varphi$ preserves strict inequalities, and therefore
   $$
   \mathbb{P}\bigl(\varphi(-d_\Sigma(X_{\mathrm{new}}))>\varphi(-d_\Sigma(X_{\mathrm{ID}}))\bigr)=\mathbb{P}(d_{\mathrm{new}}<d_{\mathrm{ID}}).
   $$
   The right-hand side does not depend on $\varphi$. Hence one score in the class inverts if and only if every score in the class inverts if and only if the radius inverts.

2. Let $q=F^{-1}_{d_{\mathrm{ID}}}(\varepsilon)$. On a continuous law this quantile satisfies $\mathbb{P}(d_{\mathrm{ID}}>q)=1-\varepsilon$. The hypothesis $F^{-1}_{d_{\mathrm{new}}}(1-\varepsilon)\le q$ gives $\mathbb{P}(d_{\mathrm{new}}\le q)\ge 1-\varepsilon$.

3. Independence on the product measure yields
   $$
   \mathbb{P}(d_{\mathrm{new}}\le q,\; d_{\mathrm{ID}}>q)=\mathbb{P}(d_{\mathrm{new}}\le q)\,\mathbb{P}(d_{\mathrm{ID}}>q)\ge(1-\varepsilon)^2.
   $$
   On that event $d_{\mathrm{new}}\le q<d_{\mathrm{ID}}$, so $d_{\mathrm{new}}<d_{\mathrm{ID}}$. Therefore $\mathbb{P}(d_{\mathrm{new}}<d_{\mathrm{ID}})\ge(1-\varepsilon)^2$. For $\varepsilon<1-1/\sqrt{2}$ the bound is strictly above $1/2$.

4. The certificate is sufficient. It is not necessary. For $d_{\mathrm{ID}}\sim\mathrm{Unif}[0,1]$ and $d_{\mathrm{new}}\sim\mathrm{Unif}[0,0.9]$,
   $$
   \mathbb{P}(d_{\mathrm{new}}<d_{\mathrm{ID}})=1-0.9/2=0.55>1/2,
   $$
   while $F^{-1}_{d_{\mathrm{new}}}(1-\varepsilon)=0.9(1-\varepsilon)\le\varepsilon=F^{-1}_{d_{\mathrm{ID}}}(\varepsilon)$ forces $\varepsilon\ge 0.9/1.9>0.47$. None of $0.05$, $0.10$, $0.20$ satisfy it.

5. $A=I$ exhausts the class. If $A\Sigma A^\top=\Sigma$ and $\Sigma$ is positive definite, then $A$ is invertible and $A^\top\Sigma^{-1}A=\Sigma^{-1}$. For $v=z-\mu_{\mathrm{new}}$,
   $$
   \lVert Av\rVert_\Sigma^2=v^\top A^\top\Sigma^{-1}Av=v^\top\Sigma^{-1}v=\lVert v\rVert_\Sigma^2.
   $$
   Every $\Sigma$-orthogonal $A$ gives the same $d_\Sigma\circ T_A$, equal to $d_\Sigma\circ T_I$. No other rotation is a new radial score.
