function [ avg_s_k, overall_avg_s ] = fComputeSilhouette( C,D,option_similarity )
% function to compute avg silhouette and overall average silhouette for a
% given clustering/partition (C); the distance matrix (for dissimilarity)
% or Kernel matrix for similarity (D); option_similarity: 1 for similarity
% and 0 for dissimilarity

    C = C(:) ;

    N =length(C);        % num of elements
    k = max(unique(C));  % num of clusters

    % compute a(i): a_i: within cluster mean expcept for that element
%     mat_log_ai = zeros(N,N) ;
%     array_nnz_elem = ones(1,N) ;
    
    % replicate C N times
    template_C = repmat(C, [1 N]);

      % approach 1: loop for a_i
%     for i = 1:N
%        k_val = C(i) ;
%        temp_logical = (C == k_val ) ;
%        temp_logical(i) = 0;   % exlcude that element
%        mat_log_ai(:,i) = temp_logical ;
%        array_nnz_elem(1,i) = nnz(temp_logical) ;
%     end
% 
%     % evaluate within cluster mean: array_ai : 1 x N
%     array_ai = (sum(mat_log_ai .* D))./array_nnz_elem ;

    % approach 2: without loop
    % replicate C' N x 1 times
     template_C_k = repmat(C', [N 1]);
    
    temp_ai = (template_C == template_C_k ) ;
    
    % exclude that particular element
    remove_index = ((1:N)'-1)*N + (1:N)' ;
    temp_ai(remove_index) = 0 ;  
    
    % evaluate within cluster mean: array_ai : 1 x N
    array_ai = (sum(temp_ai .* D))./sum(temp_ai) ;
    

    % Step 2: create b_i    
    temp_mat_bi = zeros(k,N);
    for ind_k = 1:k   

        temp_mat = (template_C == ind_k) ;
        array_nnz_elem = sum(temp_mat) ;
        temp_mat = temp_mat.*D ;

        temp_mat_bi(ind_k,:) = (sum(temp_mat))./array_nnz_elem ; 
    end


    remove_index = ((1:N)' - 1)*k + C ;

    if( option_similarity )
        % similarity measure
        temp_mat_bi(remove_index) = -inf ;
        array_bi = max(temp_mat_bi,[],1) ;
        array_si = (array_ai - array_bi)./(max(array_bi,array_ai)) ;
    else
        % dissimilarity measure
        temp_mat_bi(remove_index) = inf ;
        array_bi = min(temp_mat_bi,[],1) ;
        array_si = (array_bi - array_ai)./(max(array_bi,array_ai)) ;
    end

    % check for division by 0 and set that to 0
    array_si(isnan(array_si)) = 0 ;
    
    
    % overall average silhouette
    overall_avg_s = mean(array_si) ;

    % average silhouette for each cluster
    avg_s_k = zeros(k,1) ;
    for ind_k = 1:k
        temp_mat = array_si'.*(C == ind_k) ;
        avg_s_k(ind_k) = sum(temp_mat)/nnz(temp_mat) ;
    end
    
    % check for division by 0; 
    % it will arise when some cluster is missing in final output
    avg_s_k(isnan(avg_s_k)) = 0 ;

end

