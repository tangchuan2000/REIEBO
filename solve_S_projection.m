function S = solve_S_projection(Xs, Qs, B, W, RF, beta)
% Solve:  min_S sum_t ||{X^(t) - Q^(t) B S} diag(w^(t))||_F^2 + beta ||S - RF||_F^2
% s.t.    S' 1 = 1,  S >= 0
% Using B'B = I, (Q^(t))' Q^(t) = I  ==>  S* = proj_simplex( S_hat ) column-wise.
%
% Inputs:
%   Xs   : 1×v cell, Xs{t} is d_t × n
%   Qs   : 1×v cell, Qs{t} is d_t × l   (with Qs{t}' * Qs{t} = I)
%   B    : l × m   (with B' * B = I)
%   W    : v × n   (W(t,i) = w_i^(t))
%   RF   : m × n   (the target matrix in the regularizer)
%   beta : scalar >= 0
%
% Output:
%   S    : m × n, each column on the probability simplex
% written by tc 2025.8
    if nargin < 6, beta = 0; end
    v = numel(Xs);
    n = size(RF, 2);
    m = size(B, 2);
    eps_val = 1e-12;

    % 1) Precompute A^(t) = B' * Q^(t)' * X^(t)  (m × n)
    A = cell(1, v);
    Bt = B';
    for t = 1:v
        A{t} = Bt * (Qs{t}' * Xs{t});   % m × n
    end

    % 2) Column-wise compute \hat{s}_i and Euclidean projection to simplex
    S = zeros(m, n);
    for i = 1:n
        % weights^2 per view for the i-th sample
        wt2 = W(:, i) .^ 2;                         % v×1
        denom = sum(wt2) + beta;                    % scalar
        if denom < eps_val, denom = eps_val; end

        % numerator = sum_t (w_i^(t))^2 a_i^(t) + beta r_i
        num = zeros(m, 1);
        for t = 1:v
            num = num + wt2(t) * A{t}(:, i);
        end
        num = num + beta * RF(:, i);

        s_hat = num / denom;

        % project s_hat to probability simplex: {s>=0, 1^T s = 1}
        S(:, i) = proj_simplex_l2(s_hat);
    end
end

function x = proj_simplex_l2(y)
% Euclidean projection onto the probability simplex Δ = {x>=0, 1^T x = 1}
    y = y(:);
    m = numel(y);
    [ys, ~] = sort(y, 'descend');
    tmpsum = 0; rho = 0;
    for j = 1:m
        tmpsum = tmpsum + ys(j);
        if ys(j) - (tmpsum - 1)/j > 0
            rho = j;
        else
            break;
        end
    end
    theta = (sum(ys(1:rho)) - 1) / rho;
    x = max(y - theta, 0);
end
