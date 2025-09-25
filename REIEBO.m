function [F ,iter ]=REIEBO(X ,Y, d ,numanchor, alpha, beta, lambda)
% X: 1×v cell,   X{t} is d_t × n
    m =numanchor ; 
    numclass =length (unique (Y )); 
    numview =length (X ); 
    numsample =size (Y ,1 );     
    R = eye(m,numclass);

    ww_vector_cell = cell(1, numview);
    ww_vector = ones(1,numsample) / numview;
    for i = 1:numview      
       ww_vector_cell{i} = ww_vector;  
    end 

    P =cell (numview ,1 );  % Q
    A =zeros (d ,m );       % B
    
    
    %A = orth(rand(d, m), 1);
    %tc = A' * A;

    Z =zeros (m ,numsample ); 
%     Q =zeros ( numclass, m); 
%     
%     Q(1,:) = 1;% first row set to 1
    for i =1 :numview 
        di =size (X{i },1 ); 
        P{i}=zeros (di ,d ); 
        %P{i}=orth(rand(di, d), 1);
        X{i}=mapstd (X{i} ,0 ,1 ); 
    end


    rng(5489,'twister');
    F = rand(numclass, numsample);
    F = F ./ sum(F, 1);  

    Z(m,:) = 1;     
    
    flag =1 ; 
    iter =0 ; 
    
    maxIter = 30;
    obj = [];
    while flag 
        iter =iter +1 ; 
        
        %update P /Q
        AZ = A * Z ; 
        for v =1 :numview 
            di =size (X{v},1 );            
            C =X{v} .* repmat(ww_vector_cell{v}.^2,di,1)  * AZ' ; 
            [U ,~,V ]=svd (C ,'econ' );             
            P{v }=U *V' ; 
        end
        clear AZ;
        
        %update A /B        
        part1 =0 ; 
        termpart1 =cell (numview ,1 ); 
        for v =1 :numview         
            termpart1{v} = P{v}' *X{v} .* repmat(ww_vector_cell{v}.^2,d,1)   *Z' ; 
        end
        for v =1 :numview           
            part1 =part1 + termpart1{v}  ;
        end
        [Unew ,~,Vnew ]=svd(part1 ,'econ' ); 
        clear part1;
        
        A = Unew *Vnew' ;       
       
        
        %update Z

        PvA =cell (numview ,1 );     
        for v =1 :numview 
            PvA{v} = P{v}*A ;  
        end
%        options =optimset ('Algorithm' ,'interior-point-convex' ,'Display' ,'off' ); 
%        parfor ji =1 :numsample 
%             ff =0 ; 
%             G = 0;
%             for v =1 :numview 
%                ff = ff - ww_vector_cell{v}(ji).^2  * X{v}(:,ji )' * PvA{v}  ; 
%                %G = G + ww_vector_cell{v}(ji).^2 + beta;
%                G = G + ww_vector_cell{v}(ji).^2 ;
%             end
%             GG = (G+ beta) * eye (m);
%             GG =(GG +GG' )/2 ; 
% 
%             ff = ff - beta * F(:,ji)' * R';
%             Z (:,ji )=quadprog(GG,ff',[],[],ones (1 ,m ),1 ,zeros (m ,1 ),ones (m ,1 ),[],options ); 
%        end
       for i = 1:v
             W(i, :) = ww_vector_cell{i};       % 将第 i 个行向量放到矩阵第 i 行
        end
        Z = solve_S_projection(X, P, A, W, R * F, beta);


         %% optimize R
        J = Z*F';      
        [Ug,~,Vg] = svd(J,'econ');
        R = Ug*Vg';
    
    
         %% optimize F    
         F = greedy_update_F_Renyi(F, Z, R, alpha, beta/lambda);
        
         %% optimize w
        WWWW = solve_w(X, P, A, Z);   %w = solve_w_view(Xt, Qt, B, S, eps_val)    
  
        for v = 1:numview 
          ww_vector_cell{v} = WWWW(v,:); 
        end


        term1 =0 ; 
        termobj =cell (numview ,1 ); 
        for v =1 :numview 
            di =size (X{v},1 ); 
            termobj{v} = norm ((X {v }-PvA{v} *Z)  .* repmat(ww_vector_cell{v},di,1)   ,'fro' );            
        end
        

        for v =1 :numview 
            term1 = term1 +  termobj{v};            
        end
        
        
        F_sum_row_vec = sum(F, 2) ;
        p_j = F_sum_row_vec / numsample; 
        p_j(p_j == 0) = 1e4; 
        p_j = p_j.^alpha;

        if (alpha == 1)
            log_cost = -sum(p_j .* log(p_j)); %  shannon Entropy  
        else
            log_cost = 1 / (1-alpha) *  log(sum(p_j)); 
        end

       
        RF = R * F;
        reconstruction_error = norm(Z - RF, 'fro')^2;
        
        obj(iter)= term1 - log_cost + beta * reconstruction_error ; 

        if ((iter >1 && abs(obj(iter) - obj(iter -1 )) / obj(iter -1 ) < 0.001 )|| iter == maxIter)           
           flag =0 ; 
        end       

    end      
end



