function [W,C] = fKernelized_k_Means_Clustering(Q,m,gamma,Tmax )

% Function for Kernelized k-mean clustering
% Steps involved: 1) Intialize 2) Learn Dictionary: Obtain W; 3) Obtain
% clusters/labels

%% 
% PArameters
%           Input
%           1) Q: Pairwise fiber distance/similiarity/dissimilarity matrix:
%                 Size:  n x n
%           2) m: the desired number of fiber bundles
%           3) gamma: RBF kernel parameter                                       
%           4) Tmax: Maximum Number of iterations
%
%           Output:
%           1) W: Sparse assignment matix:   Size: n x m
%           2) C: Hard/Label/Cluster assignment matrix:  Size: 1 x n




%%
% initialize

    % Kernel Matrix
    K = exp(-gamma*Q.^2)     ;
    n = size(Q,1)            ;
    I = eye(n)               ;
    % initialize A
    S = randsample(n,m)      ;
    A = I(:,S)                        ;
    iter_Error = zeros(Tmax,1)        ;
    Eold = inf               ;
    
    disp('Starting clustering')       ;

%%
    % dictionary learning
    for iter = 1:Tmax  
                  
            
            
            % kernelized k means_clustering                            
                mat_KA = K*A         ;
                M = A'*mat_KA        ;
                mat_diag_M = diag(M)';
                index = zeros(n,1)   ;
                W = zeros([m n])    ;
                
                % each column of W: can be computed in parallel
                parfor i=1:n                         
                    [~,c_i]= min(mat_diag_M - 2*mat_KA(i,:))       ;
                    index(i) = (i-1)*m + c_i                       ;
                end
                W(index) = 1                                       ;
                       
            % Evaluate Reconstruction error 
            E = trace(K - 2*mat_KA*W + W'*(M*W))        ;
            iter_Error(iter) = E                        ;
            
            if (E > Eold)
                disp('Energy function increased???')    ;
                break                                   ;
            elseif ((Eold - E)/E < 0.001)
                disp('No change in energy function')    ;
                break                                   ;
            end

            Eold = E                                    ; 

               
        % Update Dictionary Matrix A using: MOD
        temp_inv_mat =  W*W'                        ;
        %A = W'/(temp_inv_mat + 1e-10*eye(size(temp_inv_mat)))               ;
        A = W'/temp_inv_mat                         ;
        
    end

%%
    % Obtain hard/Cluster assignment 
    [~, C] = max(W,[],1)                                                    ;

end

