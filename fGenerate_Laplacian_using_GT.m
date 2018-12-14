
sigma = 7 ;
k_ngh = 9 ;  % 1 buffer for diagonal
% Create Laplacian for testing
    % Steps: 1) create Adjacency matrix using EP distance 
    % 2) create Symmetric normalized Laplacian matrix: L = I - D^(-0.5)*Affinity_matrix*D^(-0.5) ;
    
    % load GT data
    
    % GT data 
       
    Base_Data_path = '../../Data/GT/' ;
    inGT_Data = strcat(Base_Data_path,'cell_sampled15_Tracts_10bundlesDataset.mat');
    load(inGT_Data);
    
    % load GT labels
        inGT_labels = strcat(Base_Data_path,'GT_labels_10bundles_apr3.mat');
        load(inGT_labels);    
        G = GT_labels   ;
        
       n = size(G,1)     ;
    
    [Dist_EP1, Dist_EP2] = fDist_Comp_EP(cell_sampled_Tracts,cell_sampled_Tracts);
    
    mean_EP = (Dist_EP1+Dist_EP2)/2 ;
    mean_EP = Dist_EP1 ;
    
    Affinity_matrix = exp(-(mean_EP.*mean_EP)/(sigma*sigma)) ;
    %Affinity_matrix = exp(-(Dist_EP1.*Dist_EP1)/(sigma*sigma)) ;
    
      
    Affinity_matrix(1:(n+1):(n*n))=0;
    D1 = spdiags(1./sqrt(sum(Affinity_matrix,2)),0,n,n) ;
    L = speye(size(Affinity_matrix)) - D1*Affinity_matrix*D1 ; 
    
     L(abs(L)<1e-8)=0;
    %save(strcat(Base_Data_path,'L_EP1_ngh8_Dec12'),'L');
%%

% Option 2: use EP distance to create a binary affinity matrix
%           and then compute Laplacian
    sigma = 7 ;
    k_ngh = 9 ; 
       
    Base_Data_path = '../../Data/GT/' ;
    inGT_Data = strcat(Base_Data_path,'cell_sampled15_Tracts_10bundlesDataset.mat');
    load(inGT_Data);
    
    % load GT labels
        inGT_labels = strcat(Base_Data_path,'GT_labels_10bundles_apr3.mat');
        load(inGT_labels);    
        G = GT_labels   ;
        
       n = size(G,1)     ;
    
    [Dist_EP1, Dist_EP2] = fDist_Comp_EP(cell_sampled_Tracts,cell_sampled_Tracts);
    
    mean_EP = (Dist_EP1+Dist_EP2)/2 ;
    mean_EP = Dist_EP1 ;
    
    Affinity_matrix = exp(-(mean_EP.*mean_EP)/(sigma*sigma)) ;
    
    Affinity_mat(Affinity_mat < 1e-3) = 0 ;
    Affinity_mat(Affinity_mat > 1e-4) = 1 ;
    
    Affinity_matrix(1:(n+1):(n*n))=0;
    D1 = spdiags(1./sqrt(sum(Affinity_matrix,2)),0,n,n) ;
    L = speye(size(Affinity_matrix)) - D1*Affinity_matrix*D1 ; 
    
    
    L= (L+L')/2;
    
 %%   
    
    %% using GT labels
    
    % load GT_labels
    
    Base_Data_path = '../../Data/GT/' ;
    inGT_labels = strcat(Base_Data_path,'GT_labels_10bundles_apr3.mat');
    load(inGT_labels);
    
    G = GT_labels ;
    
    n = size(G,1);
    Affinity_matrix = zeros(n,n);
    
    % normal version
    for i =1:n
       k = G(i);
       ind_set = (G == k);
       Affinity_matrix(i,ind_set) = 1;
       Affinity_matrix(i,i) = 0 ;        
    end
    
%     % sparse version    
%     for i =1:n
%        k = G(i);
%        ind_set = find(G == k);
%        sp_rand_ind_set = randsample(length(ind_set),k_ngh) ;
%        Affinity_matrix(i,ind_set(sp_rand_ind_set)) = 1;
%        Affinity_matrix(i,i) = 0 ;        
%     end
    
    % we need to make Affinity_matrix symmetric; as with random selection its not
    % symmetric
%     a1 = (Affinity_matrix + Affinity_matrix');
%     a1(a1==2) = 1 ;
%     Affinity_matrix = a1 ;
    
%     D = spdiags(sum(Affinity_matrix,2),0,n,n) ;
% 
%             L = D - Affinity_matrix;

            % Normalized L

    D1 = spdiags(1./sqrt(sum(Affinity_matrix,2)),0,n,n) ;
    L = speye(size(Affinity_matrix)) - D1*Affinity_matrix*D1 ; 
    
    
   
    
    %a1 = L( 0<abs(L)<1);