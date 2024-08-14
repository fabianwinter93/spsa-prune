# Algorithm 1: The structured pruning algorithm MINI-LLM
# Input: Dataset D, pre-trained weights W0 in R^d, loss L : R^d -> R, prune ratio p, perturbation scale epsilon.
# Output: The pruned model

# 1: Clear every weight’s sensitivity score S^(W_l^k) = 0;
# 2: Forward via Eq. (7) and estimate each weight’s g_l^k;
# 3: for l in [1, ..., L] do
# 4:    Compute the input activation X_l for l-th layer;
# 5:    Compute every weight’s score S^(W_l^k) via Eq. (9);
# 6: end for
# 7: for l in [1, ..., L] do
# 8:    Keep the important groups S^(G_l) ranked in top 1 - p;
# 9: end for
# 10: return the pruned model.

# Algorithm 1: MeZO

# Require: parameters theta in R^d, loss L : R^d -> R, step budget T, perturbation scale epsilon, batch size B,
#          learning rate schedule {η_t}

# for t = 1, ..., T do
#     Sample batch B ⊂ D and random seed s
#     theta <- PerturbParameters(theta, epsilon, s)
#     l_+ <- L(theta; B)
#     theta <- PerturbParameters(theta, -2 * epsilon, s)
#     l_- <- L(theta; B)
#     theta <- PerturbParameters(theta, epsilon, s)  # Reset parameters before descent
#     
#     projected_grad <- (l_+ - l_-)/(2 * epsilon)
#     Reset random number generator with seed s  # For sampling z
#     for θ_i in θ do
#         z ~ N(0, 1)
#         θ_i <- θ_i - η_t * projected_grad * z
#     end for
# end for
# return θ

# Subroutine PerturbParameters(theta, epsilon, s)
#     Reset random number generator with seed s  # For sampling z
#     for θ_i in θ do
#         z ~ N(0, 1)
#         θ_i <- θ_i + epsilon * z
#     end for
#     return θ  # Modify parameters in place

