0.1 Learning PDE Solution Operators via Energy-based Models FrameworkJunyi LiaoLet $\Omega\subset\mathbb{R}^{d}$ be a bounded open domain. We consider a parametric PDE on $\Omega$ given by:$$\begin{cases}L(a,u)=0, & \text{in } \Omega, \\ \alpha u+\beta\frac{\partial u}{\partial n}=g & \text{on } \partial\Omega,\end{cases} \quad (0.1)$$where $a:\Omega\rightarrow\mathbb{R}^{p}$ is the parameter function, $g:\partial\Omega\rightarrow\mathbb{R}$ is given, and $u:\Omega\rightarrow\mathbb{R}$ is the unknown.We assume $a\sim\pi$, where $\pi$ is a probability distribution, called the prior, on a function space $\mathbb{A}$. Then the PDE (0.1) implicitly induces a probability measure $\nu=G_{\#}^{*}\pi$, which is the pushforward measure of $a\sim\pi$ under the solution operator, so the PDE solution $u\sim\nu$.A wide class of PDEs have a variational distribution. In that case, (0.1) is equivalent to:$$u^{*}=G^{*}(a)=\min_{u\in\mathbb{U}}J(a,u),$$where $J:\mathbb{A}\times\mathbb{U}\rightarrow\mathbb{R}$ is the energy functional. Define the Gibbs measure over proposed solutions:$$P(u|a)=\frac{1}{Z_{\beta}(a)}e^{-\beta J(a,u)}$$where $\beta>0$ is an inverse temperature, with the normalizing term:$$Z_{\beta}(a)=\int_{\mathbb{U}}e^{-\beta J(a,u)}d\mu(u), \quad (0.2)$$where $\mu$ is a reference measure on $\mathbb{U}$. For simplicity we write $H(a,u)=\beta J(a,u)$, and the functional $H:\mathbb{A}\times\mathbb{U}\rightarrow\mathbb{R}$ is called the Hamiltonian.As the inverse temperature $\beta\rightarrow\infty$, the probability measure $P(u|a)$ concentrates on the exact solution $u^{*}$ of the PDE (0.1).Deterministic baseline. To conduct a "physics-informed" operator learning, we choose a neural operator (which is often a universal approximator) $G_{\theta}:\mathbb{A}\rightarrow\mathbb{U}$ and minimize the expected energy:$$\min_{\theta} \mathbb{E}_a [H(a, G_{\theta}(a))].$$After training, the neural operator $G_{\theta}$ gives a single solution $u_{\theta}(a)\in\mathbb{U}$ for each input field $a\in\mathbb{A}$.A Probabilistic VariantWe consider a generative neural operator $T_{\theta}:\mathbb{A}\times\mathbb{R}^{q}\rightarrow\mathbb{U}$ with noise input. For an input field $a\in\mathbb{A}$, the model runs as follows:Sample a Gaussian noise $\xi\sim\mathcal{N}(0,I_{q})$,Compute $u=T_{\theta}(a,\xi)$. This induces a conditional distribution$$u\sim Q_{\theta}(\cdot|a)=T_{\theta}(a,\cdot)_{\#}\mathcal{N}(0,I_{q}).$$In this task, we want this pushforward $Q_{\theta}(\cdot|a)$ to approximate the Gibbs measure $P^{*}(u|a)\propto e^{-H(a,u)}$, $u\in\mathbb{U}$.For each $a\in\mathbb{A}$, consider the Kullback-Leibler divergence:$$\text{KL}(Q_{\theta}(\cdot|a)||P^{*}(\cdot|a))=\mathbb{E}_{u\sim Q_{\theta}(\cdot|a)}[\log Q_{\theta}(u|a)+H(a,u)+\log Z(a)],$$where $Z(a)$ is the normalization term. Define the probabilistic operator-learning objective:$$\mathcal{L}_{KL}(\theta) = \mathbb{E}_a [\text{KL} (Q_\theta(\cdot|a) || P^*(\cdot|a))] = \mathbb{E}_{a\sim\pi} [\mathbb{E}_{u\sim Q_\theta(\cdot|a)} [\log Q_\theta(u|a) + H(a, u)]] + \text{Const}.$$This is equivalent to the variational free energy:$$\mathcal{L}_{KL}(\theta)=\mathbb{E}_{a\sim\pi}[\mathbb{E}_{u\sim Q_{\theta}(\cdot|a)}[H(a,u)]-\mathcal{S}(Q_{\theta}(\cdot|a))],$$where $\mathcal{S}=-\mathbb{E}_{u\sim Q_{\theta}(\cdot|a)}[\log Q_{\theta}(u|a)]$ is the entropy of pushforward $Q_{\theta}(\cdot|a)$.1. Training Energy-Based ModelsWhile amortized SVGD provides a principled kernel-based variational framework for approximating the physics-induced Gibbs posterior, it is not the only viable approach for probabilistic operator learning with an implicit generator. In this section, we outline several alternative energy-based methods that avoid explicit density evaluation and offer complementary trade-offs in terms of computational efficiency, scalability, and approximation fidelity. All methods leverage the known Hamiltonian $H(a,u)$ and the ability to compute its functional gradient with respect to the solution field $u$.1.1 Contrastive Divergence and Amortized CD for PDE EnergiesContrastive Divergence (CD) is a classical technique for training energy-based models by approximating likelihood gradients using short Markov chains. In our setting, the energy $H(a,u)$ is fixed by the PDE physics, and the learning objective shifts from estimating an energy function to learning an amortized sampler that produces samples from the Gibbs distribution:$$P^{*}(u|a)\propto \exp(-H(a,u)). \quad (1.1)$$Given an initial sample $u_{0}=T_{\theta}(a,\xi)$ with $\xi\sim\mathcal{N}(0,I)$, we perform $K$ steps of Langevin dynamics targeting $P^{*}(u|a)$:$$u_{k+1}=u_{k}-\eta\nabla_{u}H(a,u_{k})+\sqrt{2\eta}\epsilon_{k}, \quad \epsilon_{k}\sim\mathcal{N}(0,I) \quad (1.2)$$yielding a refined sample $u_{K}$ that is closer to the target distribution. The neural operator is then trained to match the refined sample via an amortized CD objective:$$\mathcal{L}_{CD}(\theta)=\mathbb{E}_{a\sim\pi}\mathbb{E}_{\xi\sim\mathcal{N}}[||T_{\theta}(a,\xi)-\text{stopgrad}(u_{K})||^{2}]. \quad (1.3)$$Interpretation. This procedure can be viewed as amortized MCMC, where the neural operator learns to directly generate samples that would otherwise require iterative physics-based refinement. Unlike SVGD, which enforces diversity through kernel repulsion, CD relies on stochasticity in the Langevin dynamics to explore the energy landscape. This approach is particularly effective when $H(a,u)$ provides a reliable local descent direction but global distributional accuracy is less critical.1.2 Fisher Divergence and Score MatchingAn alternative to KL-based variational objectives is to match distributions via their score functions. The Fisher divergence between a variational distribution $Q_{\theta}(u|a)$ and the target Gibbs distribution $P^{*}(u|a)$ is defined as:$$D_{F}(Q_{\theta}||P^{*})=\frac{1}{2}\mathbb{E}_{u\sim Q_{\theta}(\cdot|a)}[||\nabla_{u}\log Q_{\theta}(u|a)-\nabla_{u}\log P^{*}(u|a)||^{2}]. \quad (1.4)$$Since $\nabla_{u}\log P^{*}(u|a)=-\nabla_{u}H(a,u)$ is known analytically, the primary challenge lies in approximating the score $\nabla_{u}\log Q_{\theta}(u|a)$ of the implicit generator. To this end, we introduce an auxiliary score network $s_{\psi}(a,u)$ and train it via denoising score matching (DSM). Specifically, for perturbed samples $\tilde{u}=u+\sigma\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$, the DSM objective is:$$\mathcal{L}_{DSM}(\psi)=\mathbb{E}_{a,u,\epsilon}[||s_{\psi}(a,\tilde{u})+\frac{1}{\sigma}\epsilon||^{2}] \quad (1.5)$$With a consistent score estimator in place, the Fisher divergence objective becomes:$$\mathcal{L}_{Fisher}(\theta,\psi)=\mathbb{E}_{a\sim\pi}\mathbb{E}_{u\sim Q_{\theta}(\cdot|a)}[||s_{\psi}(a,u)+\nabla_{u}H(a,u)||^{2}]. \quad (1.6)$$Interpretation. Fisher divergence minimization aligns the local geometry of the learned distribution with that of the physics-induced Gibbs measure. Unlike SVGD, which relies on particle interactions, score matching enforces consistency at the level of infinitesimal perturbations, making it particularly attractive for high-dimensional function spaces where kernel methods become expensive.1.3 Sliced Score MatchingFor PDE discretizations with large state dimension, direct score matching may be computationally prohibitive. Sliced score matching offers a scalable alternative by projecting score discrepancies onto random one-dimensional subspaces. Let $v\sim\mathcal{N}(0,I)$ be a random probe direction. The sliced Fisher objective is given by:$$\mathcal{L}_{sliced}(\theta,\psi)=\mathbb{E}_{a,u,v}[(v^{\top}(s_{\psi}(a,u)+\nabla_{u}H(a,u)))^{2}]. \quad (1.7)$$In practice, this objective can be efficiently estimated using Hutchinson-style random projections, making it well-suited for large-scale operator learning problems.Interpretation. Sliced score matching preserves the core principle of Fisher divergence minimization while significantly reducing computational cost. From a functional perspective, it enforces alignment between the projected gradients of the learned and target distributions, which is often sufficient to capture the dominant physics-constrained directions in solution space.Summary. Contrastive divergence, Fisher divergence, and (sliced) score matching provide complementary alternatives to SVGD for probabilistic operator learning. All three methods exploit the availability of a known Hamiltonian $H(a,u)$ and avoid explicit density evaluation for the implicit generator $T_{\theta}$. Among them, amortized CD emphasizes efficient sampling, score matching emphasizes local distributional geometry, and sliced variants prioritize scalability in high-dimensional PDE settings.2. Experiments on ODEs2.1 Poisson's EquationWe consider the boundary value problem in $(0,1)$:$$\begin{cases}\frac{d}{dx}(a(x)\frac{du}{dx}(x))=f(x)\\ u(0)=u(1)=0,\end{cases}$$where $u\in H_{0}^{1}([0,1])$ is the unknown, $a\in L^{\infty}(0,1)\cap C^{1}([0,1])$ is a nonnegative coefficient, and $f\in L^{\infty}(0,1)\cap C([0,1])$ is the forcing.The energy function of this ODE is:$$J(u;a,f)=\frac{1}{2}\int_{0}^{1}a(x)|u^{\prime}(x)|^{2}dx-\int_{0}^{1}f(x)u(x)dx$$Discretization and boundary constraints. We fix the boundary values $u(0)=u(1)=0$, and evaluate the pertinent functions on interior grid points $x_{1},\cdot\cdot\cdot,x_{n}\in(0,1)$. To model a solution $u$, define the full grid solution by fixing endpoints:set $u_{0}=u_{n+1}=0$ always, andstore only interior unknowns $u_{int}=(u_{1},\cdot\cdot\cdot,u_{n})\in\mathbb{R}^{n}.$When evaluating the energy $H$, use the padded vector $u=(u_{0},u_{1},\cdot\cdot\cdot,u_{n},u_{n+1})$.Let $x_{j}=jh$, $j=1,\cdot\cdot\cdot,n$ where $h=\frac{1}{n+1}$. In discretized form, the energy function is:$$J(u;a,f)=\frac{1}{2}hu^{\top}K(a)u-hf^{\top}u,$$where $K(a)$ is a kernel matrix that couples neighboring grid values. Then:$$\nabla_{u}H(u;a,f)=\beta\nabla_{u}J(u;a,f)=\beta(K(a)u-f).$$A finite difference form of $K(a)$ is given by the tridiagonal matrix:$K_{ii}=(a_{i+\frac{1}{2}}+a_{i-\frac{1}{2}})/h^{2}$, $K_{i,i+1}=-a_{i+\frac{1}{2}}/h^{2}$, and $K_{i,i-1}=-a_{i-\frac{1}{2}}/h^{2}$.Each $a_{i+\frac{1}{2}}$ is the value of $a(x)$ evaluated at the midpoint.It is common to use the linear interpolation $a_{i+\frac{1}{2}}=(a_{i}+a_{i+1})/2$ when $a$ is only given on grids, or the harmonic mean $a_{i+\frac{1}{2}}=\frac{2}{a_{i}^{-1}+a_{i+1}^{-1}}$ for discontinuous $a$.In particular, for $a\equiv1$ in the standard Poisson's equation:$$K=\frac{1}{h^{2}}\begin{pmatrix}
2 & -1 & 0 & \cdots & 0 \\
-1 & 2 & -1 & \cdots & 0 \\
0 & -1 & 2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & -1 \\
0 & 0 & 0 & -1 & 2
\end{pmatrix}$$

Nonlinear Poisson's Equation

Consider the non-linear Poisson's equation with Dirichlet BC:$$\begin{cases}
-u''(x) = f(u(x))+ s(x) & \text{in } (0,1), \\
u(0)=u(1)=0.
\end{cases}$$The energy functional of this BVP is$$J(u,s)=\int_0^1\left(\frac{1}{2}| u'(x)|^2+V(u(x))-s(x)u(x)\right) dx,$$where the function $V:[0,1]\to\mathbb{R}$ satisfies $V'(u)=f(u)$. The discretized approximation is$$J(u,s)=\frac{1}{2}hu^\top Ku+h\mathbf{1}^\top V(u)-hs^\top u,$$where $K$ is the kernel matrix (from the standard Laplacian discretization). The energy gradient is$$\nabla_u H(u,s)=\beta\nabla_u J(u,s)=\beta h\left(Ku+\mathbf{1}^\top f(u)-s\right).$$A famous model for phase transitions (e.g., separating oil and water, or magnetic domains) is the Double-Well Potential (Ginzburg-Landau) ODE:$$\begin{cases}
-u''(x) + (u(x)^3 - u(x)) = s(x) & \text{in } (0,1), \\
u(0)=u(1)=0.
\end{cases}$$Here, the reaction term is $f(u) = u-u^3$. The energy functional of this ODE is$$J(u,s) = \int_0^1 \biggl( \frac{1}{2} (u')^2 + \underbrace{\left(\frac{1}{4}u^4 - \frac{1}{2}u^2\right)}_{\text{Double-Well}} - su \biggr) dx.$$

### 2D Darcy Flow
We consider the following boundary problem boundary value problem with Dirichlet boundary condition:
$$
    \nabla\cdot\left(a(x)\nabla(x)\right)=f(x)\ \text{in}\ \Omega=(0,1)^2,$$
$$u=0\ \text{on}\ \partial\Omega.$$
where $u\in H_0^1(\Omega)$ is the unknown, $a\in L^\infty(\Omega)\cap C^1(\Omega)$ is a nonnegative coefficient, and $f\in L^\infty(\Omega)\cap C(\Omega)$ is the forcing. The energy function of this ODE is
$$J(u;a,f)=\frac{1}{2}\int_\Omega a(x)\left\vert\nabla u(x)\right\vert^2\,dx-\int_\Omega f(x)u(x)\,dx.$$
For discretization, we let $u\in\mathbb{R}^{N\times N}$ be values on interior grid points, with $h=1/(N+1)$, and impose $u=0$ on the boundary by padding. We use the approximation for partial derivatives
$$(u_x)_{i+\frac{1}{2},j}=\frac{u_{i+1,j}-u_{i,j}}{h},\quad (u_x)_{i,j+\frac{1}{2}}=\frac{u_{i,j+1}-u_{i,j}}{h},$$
and permeabilities
$$a_{i+\frac{1}{2},j}=\mathrm{avg}(a_{i,j},a_{i+1,j}),\quad a_{i,j+\frac{1}{2}}=\mathrm{avg}(a_{i,j},a_{i,j+1}),$$
where $\mathrm{avg}(\cdot,\cdot)$ can be arithmetic mean (for smooth $a$) or harmonic mean (for strongly varing $a$). Then the discretized energy takes the form
$$J(u;a,f)=\frac{h^2}{2}\sum_{i,j}\left(a_{i+\frac{1}{2},j}(u_x)_{i+\frac{1}{2},j}^2+a_{i,j+\frac{1}{2}}(u_x)_{i,j+\frac{1}{2}}^2\right)-h^2\sum_{i,j}u_{i,j}f_{i,j}.$$
To compute the physical score
$$\nabla_u H(u;a,f)=\beta\nabla_u J(u;a,f),$$
one can depend on either autograd (recommended) or write it explicitly via divergence of fluxes.

### 1D Viscous Burgers Equation
We next consider the one-dimensional viscous Burgers equation as a prototype nonlinear, time-dependent PDE:
$$\nu u_{xx} - u u_x = u_t, \qquad x \in (0,1), \ t \in (0,T),$$
where $\nu > 0$ denotes the viscosity coefficient. Burgers equation serves as a canonical benchmark for nonlinear transport with diffusion and is widely used to evaluate operator-learning methods.
#### Problem setup 
We focus on a time-marching formulation in which the initial condition
$$u^n(x) = u(x, t_n)$$
is treated as the input parameter, and the goal is to predict the solution at the next time step $u^{n+1}(x) = u(x, t_{n+1})$, with $t_{n+1} = t_n + \Delta t$
Periodic boundary conditions are imposed for simplicity and numerical stability.

This formulation allows us to define a probabilistic one-step solution operator
$$
u^{n+1} \sim Q_\theta(\cdot \mid u^n),
\qquad
u^{n+1} = T_\theta(u^n, \xi),
\quad \xi \sim \mathcal{N}(0, I),
$$
where $T_\theta$ is a generative neural operator.

#### Discrete residual energy
Unlike elliptic problems, Burgers equation does not arise as the minimizer of a purely spatial energy functional. Instead, we adopt a physics-informed residual energy corresponding to an implicit time discretization. Using a backward Euler scheme, the discrete PDE residual is
$$\mathcal{R}(u^{n+1}; u^n)
    =
    \frac{u^{n+1} - u^n}{\Delta t}
    +
    u^{n+1} \odot \partial_x u^{n+1}
    -
    \nu \partial_{xx} u^{n+1},
$$
where $\odot$ denotes pointwise multiplication. We define the energy functional
$$
    H(u^n, u^{n+1})
    =
    \frac{1}{2}
    \big\|
        \mathcal{R}(u^{n+1}; u^n)
    \big\|_2^2,
$$
which induces a Gibbs distribution
$$P^*(u^{n+1} \mid u^n) \propto \exp(-H(u^n, u^{n+1}))$$
over the next-step solution.

#### Contrastive Divergence Refinement

Given an initial sample $u_0^{n+1} = T_\theta(u^n, \xi)$, we apply one or more steps of Langevin dynamics to refine the solution:
$$
    u_{k+1}^{n+1}
    =
    u_k^{n+1}
    -
    \eta \nabla_{u^{n+1}} H(u^n, u_k^{n+1})
    +
    \sqrt{2\eta}\,\epsilon_k,
    \qquad \epsilon_k \sim \mathcal{N}(0,I),
$$
and train the neural operator using an amortized contrastive divergence objective:
$$
    \mathcal{L}_{\mathrm{CD}}(\theta)
    =
    \mathbb{E}
    \left[
        \big\|
            T_\theta(u^n,\xi)
            -
            \mathrm{stopgrad}(u_K^{n+1})
        \big\|^2
    \right].
$$