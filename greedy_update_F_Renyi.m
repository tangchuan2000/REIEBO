function F = greedy_update_F_Renyi(F, Z, R, alpha, beta)
    
    % Z: m x n 
    % R: m x c    
    % F: c x n 
    % alpha, beta:     
    [m, n] = size(Z);
    c = size(R, 2);    

    RF = R * F;
    RF_old_col_i = zeros(c, 1);
    F_old_col_i = zeros(c, 1);
    col_norms = vecnorm(Z - RF, 2, 1).^2 ;
    reconstruction_sum = sum(col_norms);
    zeros_vector = zeros(c, 1); 

    F_sum_row_div_n_vec = sum(F, 2) ; %Column vector, each element is the sum of F rows, representing the number of samples in each cluster
    for i = 1:n  % for per column           
        min_cost = 1e8;              
        F_old_col_i = F(:, i);
        RF_old_col_i = RF(:,i); %bak
        for j = 1:c  % Traverse each element of F_j                
            vec = zeros_vector;
            vec(j) = 1;
            F(:,i) = vec;   
            F_sum_row_div_n_vec = F_sum_row_div_n_vec - F_old_col_i ; % Only one element is 1, and all others become 0, so subtract all of them first
            F_sum_row_div_n_vec(j) =  F_sum_row_div_n_vec(j) + 1; %Change the sum of the jth line to the sum of the elements in that line after updating to 1

            C = F_sum_row_div_n_vec / n; % p_j                 
            C(C == 0) = 1e4; 
            C = C.^alpha;
            if (alpha == 1)
                log_cost_all = -sum(C .* log(C)); % Degradation to Shannon entropy
            else
                log_cost_all = 1 / (1-alpha) *  log(sum(C)); 
            end

             RF(:,i) = R(:,j);
             new_col_i_norm =  vecnorm(Z(:,i) - RF(:,i))^2; 
             col_norms_old_i = col_norms(i);
             col_norms(i) = new_col_i_norm;
             reconstruction_error =  reconstruction_sum + col_norms(i) - col_norms_old_i;
             cost =    -log_cost_all + beta * reconstruction_error ;

            if cost < min_cost
                min_cost = cost;
                F_old_col_i = F(:, i); 
                RF_old_col_i = RF(:, i);
                reconstruction_sum = reconstruction_error;
            else
                RF(:,i) = RF_old_col_i; %restore to old value
                F(:,i) = F_old_col_i; %restore to old value
                F_sum_row_div_n_vec(j) = F_sum_row_div_n_vec(j) - 1;%restore
                F_sum_row_div_n_vec = F_sum_row_div_n_vec + F_old_col_i ;
                col_norms(i) = col_norms_old_i;
            end                
        end
    end       
end

