function W = solve_w_multiview(Xs, Qs, B, S, eps_val)
% Xs: 1×v cell,   Xs{t} is d_t × n
% Qs: 1×v cell,   Qs{t} is d_t × l
% B:  l × m
% S:  m × n
% W:  v × n, column i stores [w_i^(1),...,w_i^(v)]^T,  sum_t W(t,i)=1
% eps_val: small positive (default 1e-8) to avoid division by zero
% written by tc 2025.8
    if nargin < 5 || isempty(eps_val), eps_val = 1e-8; end
    v = numel(Xs);
    n = size(S,2);

    % Precompute F = B*S (l × n)
    F = B * S;

    % c(t,i) = || X^(t)_:,i - (Q^(t)F)_:,i ||_2^2
    C = zeros(v, n);
    for t = 1:v
        E_t = Xs{t} - Qs{t} * F;     % d_t × n
        C(t, :) = sum(E_t.^2, 1);    % 1 × n
    end

    % Per-sample inverse-error weighting, normalized across views
    Csafe = max(C, eps_val);
    invC  = 1 ./ Csafe;              % v × n, nonnegative
    W     = invC ./ sum(invC, 1);    % normalize per column (sample)

    % Optional: if a sample has zero error in all views, assign uniform weights
    zeroCols = all(C==0,1);
    if any(zeroCols)
        W(:, zeroCols) = 1 / v;
    end

% POWER-NORMALIZE view weights by a concave power (gamma in (0,1])
% W: v×1 (single sample) or v×n (n samples, each column sums to 1)
% Wpn: same size as W, power-normalized and column-wise renormalized
     gamma = 0.25; 
    eps_val = 1e-12; 
%     w_1 = W.^0.25;
%     % ensure nonnegativity and tiny floor for numerical stability
%     W = max(W, 0);
%     W = W ./ max(sum(W,1), eps_val);           % 先确保列和为1（若未归一化）

    % power mapping and column-wise renormalization
    Wpow = (W + eps_val).^gamma;               % 避免 0^gamma 的梯度问题
    Wpn  = Wpow ./ max(sum(Wpow,1), eps_val);  % 每列归一化到 1
    W = Wpn;
end


