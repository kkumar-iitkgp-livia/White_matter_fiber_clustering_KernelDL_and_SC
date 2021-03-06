function [A,W,final_k,iter_Error] = fGroup_sparse_kernel_dictionary_learning(initial_A,K,n,k,lambda_1,lambda_2,mu_1,Max_iter,tmax_W,TE_thres,minThres_IterError_A,minThres_IterError_W)

%%
% version: updated by Kuldeep

%
% Function summary
%
% Function to perform L1 + L21 group sparsity
% 
% W update using ADMM
% A update using tri matrix factorization approach
%
%%
% Input:  inParam: input struture object with following fields
%         n : number of fibers
%         k : number of desired output clusters
%         Smax : sparsity value (max number of elements; L0 norm)
%         A : initial A matrix
%         K : Kernel matrix
%         Max_iter : maximum number of iterations in main loop
%         TE_thres : Total reconstruction error threshold for termination
%                    criterion
%         minThres_IterError_A : relative/percentage error threhold for 
%                    termination criterion for A update loop
%         rank_tol : relative regularization parameter to avoid/remove
%                    singularity in Weight matrix (W) {can be used to
%                    remove redundant clusters}
%         
%
% Output:  outRes: Output structure object with following fields
%          A: final A matrix
%          W: final W (weight) matrix
%          k: final k (number of cluster) values
%          iter_Error: Reconstruction error in each iteration

%%


% Initialize
%time_W_update = zeros(Max_iter,1);
%time_A_update = zeros(Max_iter,1);
error_term_1 = trace(K) ;
iter_Error = zeros(Max_iter,1)   ; 
Eold = inf                       ;

max_loop_A = 300;

A = initial_A ;
% Precompute matrices
mat_KA = K*A                      ;
mat_AK = mat_KA'                  ;
mat_AKA = A'*(mat_KA)             ;
mat_AKA = (mat_AKA+mat_AKA')/2    ;


% setting algorithm parameters for solving NNLS  (within update for W
% using NNKOMP): 2 options: active set based or IPC based
quad_option = 1 ; 
if(quad_option ==1)
            opts = optimset('Algorithm','interior-point-convex','Display','off')            ;
elseif(quad_option ==2)
            opts = optimset('Algorithm','active-set','Display','off')            ;
end


%%
% Main loop
 for iter = 1:Max_iter  
           
       %disp(strcat('Loop',' ',int2str(iter),' ','running'));
%%
% 1. Update W: Group Sparsity using ADMM solution

       %t_w = cputime                    ;
                       
        Z = zeros([k n])  ;
        U = zeros([k n])  ;  
       
        Eold_W = inf     ; 

        %update W, Z1, Z2, U1, and U2
        term_1_W_update = (mat_AKA + (mu_1)*eye(size(mat_AKA))) ;
        for in_loop_W = 1:tmax_W

            % 1: update W
            temp_mat_UZ = mat_AK - mu_1*(U-Z)  ;
            W = term_1_W_update\temp_mat_UZ;

            % 2: update Z1 (L1 norm shrinkage )
            mat_Z_hat = max( ((W+U) - (lambda_1/mu_1)),0) ;
            
            
            % #: update Z2 (group shrinkage on each row)
            for ind_inner_k = 1:k
                temp_row_norm = norm(mat_Z_hat(ind_inner_k,:),2) ;
                if( temp_row_norm > 0 )
                    Z(ind_inner_k,:) = (max( (temp_row_norm - lambda_2/mu_1),0))*(mat_Z_hat(ind_inner_k,:)/temp_row_norm);   
                else
                    Z(ind_inner_k,:) = (max( (temp_row_norm - lambda_2/mu_1) ,0))*(mat_Z_hat(ind_inner_k,:)/1);
                end
            end


            % 4: update U1
            U = U + (W -Z) ;
            
            
            % 6: compute internal error
            E_W = norm((W-Z),'fro');

            %iter_Error_W(in_loop_W,iter) = E_W;

            % 5: check for convergence
            if (E_W > Eold_W)
                disp('Energy function increased for W ADMM')      ;
                break  ;
            elseif ((Eold_W - E_W)/E_W < minThres_IterError_W)
                disp('No change in energy function for W ADMM')   ;
                break  ;
            end

            Eold_W = E_W   ; 


        end
                  
        W = Z ;
%%
% 3. Update A:   
        %disp('Updating A matrix');
        %t_a = cputime ;          

        Aold = A ;
               
        reg_WWt = 1e-8;        
        mat_WWt = (W*W') + reg_WWt;
        mat_KW = K*W' ;
        
        E_old_A = inf ;
        for loop_A =1:max_loop_A
            
            A = Aold.*( (mat_KW)./(K*Aold*(mat_WWt))); 

            E_A = norm(A-Aold, 'fro')/norm(A,'fro');
                    
            if ( (E_old_A -E_A)/E_old_A < minThres_IterError_A)
                if(E_old_A < E_A)
                   %disp('Increase in E_A');
                   A = Aold ;   % keep last value of A
                end
                break;
            end

            E_old_A = E_A ;
            Aold = A;

        end
        
        % measure time for A update (per iteration)
        %time_A_update(iter) = cputime - t_a ;

%%                    
% update matrices
    
      mat_KA = K*A                      ;
      mat_AK = mat_KA'                  ;
      mat_AKA = A'*(mat_KA)             ;
      mat_AKA = (mat_AKA+mat_AKA')/2    ;
      
%%
 % 4. Compute Reconstruction error 
      
      error_term_2 = trace(2*mat_KA*W) ;
      error_term_3 = trace((W'*mat_AKA)*W) ;
      error_term_4 = lambda_1*sum(W(:)) ;
      error_term_5 = lambda_2*(sum(sqrt(sum(W.*W,2)))) ;
      E_reconst = 0.5*( error_term_1 - error_term_2 + error_term_3 )    ; 
      E = E_reconst + error_term_4 + error_term_5 ;
      iter_Error(iter) = E    ;                  
      

%%
% 5. Check termination criterion
        if ( (Eold -E)/Eold < TE_thres)
            break;
        end
        
        Eold = E   ; 
                                                                        
        
 end

%  outRes.A = A ;
%  outRes.W = W ;
%  outRes.k = k ;  % in case of change in k value
%  outRes.iter_Error = iter_Error ;
 final_k = k ;
 
 end