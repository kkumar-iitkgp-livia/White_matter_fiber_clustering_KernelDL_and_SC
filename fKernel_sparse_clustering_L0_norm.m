function [A,W,final_k,iter_Error] = fKernel_sparse_clustering_L0_norm(A,K,n,k,Smax,Max_iter,TE_thres,minThres_IterError_A,rank_tol)

%%
% version: updated by Kuldeep 

%%
% Function summary
%
% Function to perform Kernel Sparse Clustering (L0-norm)
% 
% W update using L0 NNKOMP
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

disp('start KSC clustering');
%%


% Initialize
%time_W_update = zeros(Max_iter,1);
%time_A_update = zeros(Max_iter,1);
error_term_1 = trace(K) ;
iter_Error = zeros(Max_iter,1)   ; 
Eold = inf                       ;

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
array_time_W_update = zeros(Max_iter,1);
array_time_A_update = zeros(Max_iter,1);
array_time_E_compute = zeros(Max_iter,1);
array_total_time_per_iter = zeros(Max_iter,1);


time_KSC_start = cputime;


 for iter = 1:Max_iter  
          
     start_time_per_iter = cputime; 
       %disp(strcat('Loop',' ',int2str(iter),' ','running'));
%%
% 1. Update W: solve NNLS for each weight vector
       start_time_W =cputime;
                       
       norm_C = diag(mat_AKA)'          ;    
       W = zeros([k n])                 ; 
       
      % Update each column of W: can be run in parallel
      % NNKOMP : Incremental implementation for each column of W
       parfor ind_i = 1:n                         
                                               
            corr_term_1 = mat_KA(ind_i,:) ;  % 1 x k term
            sig_residual = sparse(n,1) ;  % intial 0 residual

            array_AK = mat_AK(:,ind_i);
            tmat_M = mat_KA ; %#ok<*PFBNS>
            tmat_Q = mat_AKA ;

            ind = [];
            Aomp = [] ;

            w = sparse(k,1);

            temp_Mat = []  ;
            temp_Den = []  ;
            term_2 = []     ;

            for t = 1:Smax
                
                % Obtain correaltion 
                Corr = (corr_term_1 - sig_residual'*tmat_M)./norm_C ;

                Corr(:,ind) = -inf;
                [max_corr_val,corr_ind] = max(Corr);

                if(max_corr_val > 0 )
                    ind = [ind; corr_ind;]; 

                    if( t > 1)
                        term_2 = tmat_Q( (ind((1:(t-1))')-1)*k + corr_ind)  ;
                    end
                    temp_Mat = [ temp_Mat term_2; term_2' tmat_Q(corr_ind,corr_ind) ; ];  
                    temp_Den = [ temp_Den; array_AK(corr_ind);] ;                    


                    % NNLS solution (fKNNOMP)
                    x = quadprog((temp_Mat + temp_Mat')/2,-temp_Den,[],[],[],[],zeros(t,1),[],[],opts);

                    % append to Aomp
                    Aomp = [Aomp A(:,corr_ind)];

                    % update residual signal
                    sig_residual = Aomp*x ;

                    w = sparse(ind, 1, x, k, 1) ; 

                end
                 
            end

            % update the weight vector
            if (isempty(ind) == 0)        
                W(:,ind_i) = w ;
            end


       end
       
       array_time_W_update(iter) = cputime - start_time_W ; 

 %% 
 % 2. Remove singularity and insignificant bundles using row sum
        temp_row_sum_W = sum(W,2) ;

        [small_idx,~]=find(temp_row_sum_W < rank_tol*max(temp_row_sum_W)) ;

        % remove those rows which are insignificant compared to
        % largest cluster
        W(small_idx,:) = [] ;
        k = k - length(small_idx);            


        % Modify/Update size of A accordingly
        A(:,small_idx) = [] ;
                    
%%
% 3. Update A:           
        start_time_A = cputime ;
        
%         temp_inv_mat =  W*W'                        ;
%         A = W'/(temp_inv_mat + 1e-8*eye(size(temp_inv_mat)))               ;
        
        Aold = A ;
               
        reg_WWt = 1e-8; 
        max_loop_A = 300 ;
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
        
        array_time_A_update(iter) = cputime - start_time_A ;

%%                    
% update matrices
     start_time_E = cputime ;
      mat_KA = K*A                      ;
      mat_AK = mat_KA'                  ;
      mat_AKA = A'*(mat_KA)             ;
      mat_AKA = (mat_AKA+mat_AKA')/2    ;
      
%%
 % 4. Compute Reconstruction error 
      
      error_term_2 = trace(2*mat_KA*W) ;
      error_term_3 = trace((W'*mat_AKA)*W) ;
      E = 0.5*( error_term_1 - error_term_2 + error_term_3 )    ;  
      iter_Error(iter) = E                              ;                  
      

%%
% 5. Check termination criterion
        if ( (Eold -E)/Eold < TE_thres)
            break;
        end
        
        Eold = E   ; 
        
        array_time_E_compute(iter) = cputime - start_time_E  ;  
        
        
        array_total_time_per_iter(iter) = cputime - start_time_per_iter ;
                                                                        
        
 end

 
 time_KSC_Nys = cputime - time_KSC_start;
 
%  outRes.A = A ;
%  outRes.W = W ;
%  outRes.k = k ;  % in case of change in k value
%  outRes.iter_Error = iter_Error ;
 final_k = k ;
 
end