function [ Rand_Measure ] = fCompute_RandIndex_Measures( Gt_labels,C_labels )

%%
% Modified by Kuldeep on April 21, 2015.
%
% Note: no need to pre-process for empty clusters; it will be taken care in
% code. Just pass 2 cluster label arrays
%
%
%%
%Summary: this function evaluates the 1) Rand Index (RI);
% 2) Adjusted Rand Index (ARI) and 3) Normalized Adjusted Rand Index (NARI)
% for given ground truth labels: Gt_labels; and
% the clustering output labels: C_labels
%
%%
% Input:  Gt_labels: ground truth labels; 
%         C_labels: clustering output labels; 
% Output: Rand_Measure: [ RI; ARI; NARI;] (3 x 1 array)




%%
    % Test procedure
%     S = 10 ;
%     R = 10 ;
%     C_labels = randsample(S,500,1) ;
%     Gt_labels = randsample(R,500,1) ;
    
%%
    %rand_measure_option = 1 ;

    Rand_Measure =  zeros(3,1)  ;

    Gt_labels = Gt_labels(:)    ;
    C_labels = C_labels(:)      ;
    
    n =  size(Gt_labels,1)      ;
    
    % compute unique number of clusters in Gt_labels: R
    unq_Gt_labels = unique(Gt_labels)   ;
    R = length(unq_Gt_labels)           ;
    
    % compute unique number of clusters in C_labels: S
    unq_C_labels = unique(C_labels)     ;
    S = length(unq_C_labels)            ;

%%
% step 1: create contingency matrix

      % Test cont matrix
%     n = 10 ;
%     R = 3; S =3;
%     mat_Cont = zeros(R,S);
%     
%     mat_Cont = [ 1 1 0; 1 2 1; 0 0 4; ];

    mat_Cont = zeros(R,S)       ; 
    for ind_s = 1:S
        for ind_r = 1:R
            mat_Cont(ind_r,ind_s) = sum((C_labels == unq_C_labels(ind_s) ).*(Gt_labels == unq_Gt_labels(ind_r) ))           ;        
        end
    end

%%
% step2: obtain array_u and array_v
    array_u =  sum(mat_Cont,2)                                                  ;
    array_v = (sum(mat_Cont,1))'                                                ;

%%
% Step3: obtain parameters for categories of fiber pairs: a,b,c,d,m1,m2,M
    M = n*(n-1)/2                                                               ;

    % before computing combinatorials check elements to be > 1
    a = sum(sum(mat_Cont.*(mat_Cont -1).*(mat_Cont > 1)/2 ))                    ;
    b = sum( array_u.*(array_u -1).*(array_u >1)/2) - a                         ;
    c = sum( array_v.*(array_v -1).*(array_v >1)/2) - a                         ;
    d = M - a - b - c                                                           ;
    
    m1 = a + b                                                                  ;
    m2 = a + c                                                                  ;

%%
% Step4: obtain parameters f and g
    temp_mat_C = zeros(size(mat_Cont))                                          ;
    for ind_u = 1:R
       temp_mat_C(ind_u,:) = mat_Cont(ind_u,:)/array_u(ind_u)                   ; 
    end
    
    f = sum( (sum(temp_mat_C,1)).^2 )                                           ;
    g = sum(sum(temp_mat_C.^2))                                                 ;

%%
% Step5: compute one of the Rand Index measure
    
%     if( rand_measure_option == 1)
%        Rand_Measure(1) = (a+d)/M                                                   ; 
%     elseif(rand_measure_option ==2)
%        Rand_Measure(2) = (a-m1*m2/M )/((m1+m2)/2 - m1*m2/M )                       ;
%     elseif(rand_measure_option ==3)
%        Rand_Measure(3) = (2*f - 2*R*g)/(2*f -R*f -R*R)                             ;
%     end

    
       Rand_Measure(1) = (a+d)/M                                                   ; 
    
       Rand_Measure(2) = (a-m1*m2/M )/((m1+m2)/2 - m1*m2/M )                       ;
    
       Rand_Measure(3) = (2*f - 2*R*g)/(2*f -R*f -R*R)                             ;
    
      
end

