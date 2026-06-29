---

title: "Why we need Reinforcement Learning?"
date: 2026-06-22
draft: false
math: true
tags: ["reinforcement learning", "bellman equation", "monte carlo", "value function"]
categories: ["Research Notes"]
------------------------------

## 1. What is Reinforcement Learning?

Reinforcement Learning (RL), at its core, is a method of learning. [Learning](https://en.wikipedia.org/wiki/Learning) based on wikipedia, is the process of acquiring new understanding and knowledge. To simplify it: when you see something, you know something more than that.

Some classic ways of learning can be shown as examples.

In [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), we learn from supervised examples:

$$
\text{input} \rightarrow \text{target}
$$

![Supervised learning dog example](/dog.png)

The model sees an input and is taught what the correct corresponding target should be. If we later see a different input with the same label, we learn to associate it with the same target.



In [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning), we are not given explicit labels. Instead, the model tries to discover hidden structures, patterns, or relationships within the data by itself.
![Unsupervised learning dog example](/knn.png)


Reinforcement Learning is different. In RL, an agent learns by **interacting with an environment**. It takes actions, observes the consequences, receives rewards, which are not necessarily just numbers, and gradually **reinforces** its behavior toward directions that may lead to higher long-term return.

<div style="display: flex; align-items: center; gap: 24px; margin: 24px 0;">
  <img src="/rl.png" alt="RL example" width="450">

  <p>
    So RL is not only about one-step causal reasoning. It is about training a
    <strong>policy</strong> and developing an understanding of
    <strong>value</strong>.
  </p>
</div>

A policy answers:

$$
\text{What should I do?}
$$

A value function answers:

$$
\text{How good is this situation, if I keep behaving like this?}
$$


---

## 2. Markov Assumption and Policy Forms

When studying RL, we usually begin with an important assumption: the environment satisfies the **Markov property**.

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots)=
P(s_{t+1} \mid s_t, a_t)
$$

This means that, given the current state and action, the next state no longer depends on the entire past history.

In other words, the current state is assumed to contain all the information needed for predicting the future.

Under this assumption, we can define two broad forms of policies.

A **Markov policy** only depends on the current state:

$$
\pi_\theta(a \mid s): \mathcal{S} \rightarrow \Delta(\mathcal{A})
$$

A more general history-dependent policy may depend on the full trajectory so far:

$$
\pi_\theta(a_t \mid s_{1:t}, a_{1:t-1})
$$

In principle, a general policy has more information. It can look at the whole past. But in a Markov environment, the current state already summarizes the useful part of the past.

The goal of both policies is the same: maximize the expected return from the initial state.

$$
\max_\pi V^\pi(s_1)
$$

<p align="center">
  <img src="/bean.png" alt="value maximization" width="300">
</p>

---

## 3. Why is a Markov Policy Enough?

A beautiful result in RL says that, in a Markov environment, we do not lose optimality by restricting ourselves to Markov policies.

More formally, for any history-dependent policy $\pi$, there exists a Markov policy $\mu$ such that

$$
V^{\pi}(s_1) \leq V^\mu(s_1)
$$

The intuition is simple.

If the state really contains all information needed for future prediction, then the agent does not need to remember the entire past. At each state, it only needs to choose an action that leads to the best possible future.

So the optimal decision can be made from the current state alone.

This is one of the reasons why [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process) are so central in RL. They give us a clean mathematical world where memory can be replaced by state [3].

---

## 4. How Do We Know Whether a Policy is Good?

In RL, the objective is special: what we optimize is also how we measure success. The goal itself becomes the metric — accumulated reward.

If a policy achieves a higher return, we call it better.

In a finite-horizon setting, the value function is defined as

$$
V_h^\pi(s)=
E_{\pi}
\left[
\sum_{\tau=h}^{H} r(s_{\tau}, a_{\tau})
\mid s_h = s
\right]
$$

It means: starting from state $s$ at time $h$, if we follow policy $\pi$, what total reward do we expect to collect until the end?

The action-value function, or Q-function, also conditions on the first action:

$$
Q_h^\pi(s,a)=
E_\pi
\left[
\sum_{h'=h}^{H} r(s_{h'}, a_{h'})
\mid s_h = s, a_h = a
\right]
$$

So $V^\pi$ tells us how good a state is, while $Q^\pi$ tells us how good a state-action pair is.

This small difference is extremely important. Once we know $Q^\pi(s,a)$, action selection becomes easier: we can compare actions directly.

---

## 5. Monte Carlo Estimation of Value

One natural way to estimate value is very direct: start from a state, roll out the policy many times, and average the returns.

This is the Monte Carlo idea [1].

$$
V^\pi(s)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\sum_{h=1}^{H} r(s_h, a_h)
\right)_i
$$

Here, $N$ is the number of sampled episodes, and $H$ is the horizon length.

Monte Carlo estimation is conceptually clean. If we sample enough trajectories, the average return becomes a good estimate of the true value.

But the problem is variance.

To see this, consider a simplified infinite-horizon discounted return:

$$
G = \sum_{t=0}^{\infty} \gamma^t r_t
$$

Assume the rewards are independent and identically distributed:

$$
r_t \sim \mathcal{D},
\qquad
E[r_t] = \mu,
\qquad
\mathrm{Var}(r_t) = \sigma^2
$$

Then the variance of the return is

$$
\begin{aligned}
\mathrm{Var}(G)
&=
\mathrm{Var}
\left(
\sum_{t=0}^{\infty} \gamma^t r_t
\right)
\
&=
\sum_{t=0}^{\infty}
\gamma^{2t}
\mathrm{Var}(r_t)
\
&=
\sigma^2
\sum_{t=0}^{\infty}
(\gamma^2)^t
\
&=
\frac{\sigma^2}{1-\gamma^2}
\end{aligned}
$$

When $\gamma$ approaches 1, the horizon becomes effectively long. Since

$$
1-\gamma^2=
(1-\gamma)(1+\gamma)
\approx
2(1-\gamma)
$$

we get

$$
\mathrm{Var}(G)
\approx
\frac{\sigma^2}{2(1-\gamma)}
$$

This shows why Monte Carlo methods can be noisy in long-horizon problems. The longer the effective horizon, the more future randomness is accumulated into the return estimate.

Also, in real scenarios, rewards are rarely i.i.d. The reward at one step is often related to what happened before and what will happen next, so the randomness does not simply average out. This makes estimating return even more unstable.

---

## 6. Bellman Equation: Computing Value Recursively

The Bellman equation gives another way to compute value recursively, and is the core idea behind dynamic programming in sequential decision problems [2].

Instead of waiting until the end of the episode, it breaks the return into two parts:

1. the immediate reward;
2. the value of the next state.

For a fixed policy $\pi$, the expectation Bellman equation is

$$
V_h^\pi(s)=
\sum_a \pi(a \mid s) Q_h^\pi(s,a)
$$

and

$$
Q_h^\pi(s,a)=
r_h(s,a)
+
E_{s' \sim P(\cdot \mid s,a)}
\left[
V_{h+1}^\pi(s')
\right]
$$

This is the central recursive idea in RL.

A value is not estimated only by rolling out to the end. It can also be estimated by looking one step ahead and bootstrapping from another value estimate.

In the infinite-horizon discounted case, a typical one-step TD target is

$$
Q(s_t,a_t)=
r_t
+
\gamma Q(s_{t+1}, a_{t+1})
$$

Compared with Monte Carlo, this target depends only on the immediate reward and the estimated value of the next state-action pair.

Its randomness mainly comes from two sources:

$$
\begin{aligned}
\mathrm{Var}
\left(
r_t + \gamma Q(s_{t+1}, a_{t+1})
\right)
&=
\mathrm{Var}(r_t)
+
\gamma^2
\mathrm{Var}_{s' \sim P}
\left(
Q(s',a')
\right)
\end{aligned}
$$

This is often much lower variance than summing rewards over a long trajectory.

But this lower variance is not free.

---

## 7. What Does Bellman Bootstrapping Sacrifice?

Monte Carlo estimation is unbiased in the sense that, if we sample complete trajectories, the expected return equals the true value:

$$
E[G_t] = V^\pi(s_t)
$$

Bellman-style methods use bootstrapping. The target depends on the current estimate of the value function.

For example, if we use a parameterized Q-function $Q_\theta$, then the target itself contains $Q_\theta$:

$$
r_t + \gamma Q_\theta(s_{t+1}, a_{t+1})
$$

This introduces bias, especially early in training when $Q_\theta$ is inaccurate.

So there is a classic trade-off:

$$
\text{Monte Carlo: low bias, high variance}
$$

$$
\text{Bellman / TD: higher bias during learning, lower variance}
$$

A large part of RL algorithm design can be understood as navigating this trade-off.

---

## 8. Convergence of the Bellman Equation

Since Bellman methods rely on their own current estimates, a natural question appears:

Why should this process converge?

To answer this, we need the idea of a [fixed point](https://en.wikipedia.org/wiki/Fixed_point_(mathematics)).

For optimal control, the Bellman optimality equation is

$$
V_h(s)=
\max_a Q_h(s,a)
$$

with

$$
Q_h(s,a)=
r_h(s,a)
+
\gamma
E_{s' \sim P(\cdot \mid s,a)}
\left[
V*{h+1}(s')
\right]
$$

The expectation Bellman equation evaluates a fixed policy. The optimality Bellman equation searches for the best policy.

This reflects a deep duality in RL:

* policy evaluation asks: how good is this policy?
* policy optimization asks: how good can we possibly be?

To study convergence, define the Bellman optimality operator $\mathcal{T}$:

$$
(\mathcal{T}V)(s)=
\max_a
\left(
r(s,a)
+
\gamma
\sum_{s'} P(s' \mid s,a) V(s')
\right)
$$

Then solving the Bellman equation is equivalent to finding a fixed point:

$$
V = \mathcal{T}V
$$

---

## 8.1 Banach Fixed Point Theorem

The mathematical tool behind this convergence story is the [Banach Fixed Point Theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) [4].

Let $(X,d)$ be a complete metric space. Suppose $\mathcal{T}: X \rightarrow X$ is a contraction mapping, meaning that for some $0 \leq \gamma < 1$,

$$
d(\mathcal{T}u, \mathcal{T}v)
\leq
\gamma d(u,v),
\qquad
\forall u,v \in X
$$

Then three things hold.

First, $\mathcal{T}$ has a unique fixed point $V^*$:

$$
\mathcal{T}V^* = V^*
$$

Second, starting from any initial guess $V_0$, the sequence

$$
V_{k+1} = \mathcal{T}V_k
$$

converges to $V^*$.

Third, the error decreases geometrically. For example, an a priori bound is

$$
d(V_k, V^*)
\leq
\frac{\gamma^k}{1-\gamma}
d(V_1,V_0)
$$

and an a posteriori bound is

$$
d(V_k,V^*)
\leq
\frac{\gamma}{1-\gamma}
d(V_k,V_{k-1})
$$

This theorem gives the formal backbone of value iteration [1, 3].

---

## 8.2 Why is the Bellman Operator a Contraction?

We can prove that the Bellman optimality operator shrinks the distance between any two value functions.

Using the infinity norm,

$$
\lVert V-U \rVert_\infty=
\max_s |V(s)-U(s)|
$$

we have

$$
\begin{aligned}
\left\| {T}V - {T}U \right\|_\infty
&=
\max_s
\left|
({T}V)(s)-
({T}U)(s)
\right|
\end{aligned}
$$


$$
\begin{aligned}
&=
\max_s
\left|
\max_a
\left(
r(s,a)+
\gamma E_{s' \sim P(\cdot \mid s,a)}
\left[
V(s')
\right]
\right)-
\max_a
\left(
r(s,a)+
\gamma E_{s' \sim P(\cdot \mid s,a)}
\left[
U(s')
\right]
\right)
\right|
\end{aligned}
$$


$$
\begin{aligned}
&\leq
\max_s \max_a
\left|
\gamma
\operatorname{E}_{s' \sim P(\cdot \mid s,a)}
\left[
V(s') - U(s')
\right]
\right|
\end{aligned}
$$

$$
\begin{aligned}
&=
\gamma
\max_s \max_a
\left|
\sum_{s'} P(s' \mid s,a)
\left(
V(s') - U(s')
\right)
\right|
\end{aligned}
$$


$$
\begin{aligned}
&\leq
\gamma
\max_s \max_a
\sum_{s'} P(s' \mid s,a)
\left|
V(s') - U(s')
\right|
\\
&\leq
\gamma
\left\| V-U \right\|_\infty .
\end{aligned}
$$
Because $\gamma < 1$, the Bellman operator is a contraction mapping.

Therefore, by the Banach Fixed Point Theorem, value iteration converges to the unique fixed point $V^*$.

This is one of the most elegant results in RL: a simple recursive update can be guaranteed to converge, not because of magic, but because each Bellman update pulls value estimates closer together.

---

## References

[1] Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*. 2nd edition, MIT Press, 2018.  
*Available online: http://incompleteideas.net/book/the-book-2nd.html

[2] Richard Bellman. *Dynamic Programming*. Princeton University Press, 1957.  


[3] Martin L. Puterman. *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley, 1994.  


[4] Stefan Banach. “Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.” *Fundamenta Mathematicae*, 3:133–181, 1922.  



