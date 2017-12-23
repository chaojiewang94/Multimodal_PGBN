function [ParaGlobal,PerpCun,AccCun,ParaLocal]  =   Multi_PGBN_Generate_Tags(X_img_all,X_tags_all,prepar,K,T,trial,DataType,dataname,SuPara,Settings,ParaGlobal)
% =======================================================================
% Mutlimodal Poisson Gamma Belief Network
% =======================================================================
% 
% This is the code for the paper "Mutlimodal Poisson Gamma Belief Network" presented at AAAI 2018 
% 
% Created by Chaojie Wang , Bo Chen at Xidian University and Mingyuan Zhou at University of Texas at Austin
% 
% =======================================================================
% LICENSE
% =======================================================================
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% Copyright (c), 2018, Chaojie Wang 
% xd_silly@163.com

%% superparameters
eta     =   SuPara.eta    ;
ac      =   SuPara.ac     ;   bc      =   SuPara.bc   ;
a0pj    =   SuPara.a0pj   ;   b0pj    =   SuPara.b0pj   ;
e0cj    =   SuPara.e0cj   ;   f0cj    =   SuPara.f0cj   ;

IterAll    =   Settings.IterAll   ;

%%  for burnin collection

Phi_img_UsageCun     =   cell(1,T)   ;               
Phi_tags_UsageCun    =   cell(1,T)   ;            
Phi_joint_UsageCun   =   cell(1,T)   ;  
Theta_joint_UsageCun =   cell(T,1)   ;

for t = 1:T
    Phi_img_UsageCun{t}   =   0   ;
    Phi_tags_UsageCun{t}   =   0   ;
    Phi_joint_UsageCun{t}   =   0   ;
    Theta_joint_UsageCun{t} = 0;
end

%% Initialization

X_img     =   full(X_img_all(:,prepar.trindx));           % 在load_GBN中设置
X_tags    =   full(X_tags_all(:,prepar.trindx));
[V_img,N]     =   size(X_img);
[V_tags,N]    =   size(X_tags);

[ii,jj] = find(X_img>eps);
iijj = find(X_img>eps);

if Settings.GibbsInit      
else
    Phi_img = cell(T,1);   Eta = cell(T,1);
    Phi_tags = cell(T,1);
    Phi_joint = cell(T,1);
    for t = 1:T
        Eta{t}  =   eta(t)   ;
        if t == 1
            Phi_img{t}    =   0.2 + 0.8 * rand(V_img,K(1));
            Phi_tags{t}   =   0.2 + 0.8 * rand(V_tags,K(1)); 
            Phi_img{t}    =  bsxfun(@rdivide, Phi_img{t}, max(realmin,sum(Phi_img{t},1)));  %归一化
            Phi_tags{t}   =  bsxfun(@rdivide, Phi_tags{t}, max(realmin,sum(Phi_tags{t},1)));  %归一化   
        else
            Phi_joint{t}  =   0.2 + 0.8 * rand(K(t-1),K(t)); 
            Phi_joint{t}  =  bsxfun(@rdivide, Phi_joint{t}, max(realmin,sum(Phi_joint{t},1)));  %归一化   
        end     
    end
    r_k     =   1/K(T)*ones(K(T),1);    gamma0  =   1   ;    c0  =   1   ;  
end

Xt_to_t1_img= cell(T,1);    Xt_to_t1_tags= cell(T,1);    Xt_to_t1_joint = cell(T,1);
WSZS_img    = cell(T,1);    WSZS_tags    = cell(T,1);    WSZS_joint     = cell(T,1);
Theta_img   = cell(T,1);    Theta_tags   = cell(T,1); 	 Theta_joint    = cell(T,1);
c_j_img     = cell(T+1,1);  c_j_tags     = cell(T+1,1);  c_j_joint      = cell(T+1,1); 
a_j   =   ones(1,N);

for t = 1:T
    Theta_img{t}    =   ones(K(t),N)/K(t);
    Theta_tags{t}   =   ones(K(t),N)/K(t);  
    Theta_joint{t}   =   ones(K(t),N)/K(t);   
    c_j_img{t}  =   ones(1,N);
    c_j_tags{t} =   ones(1,N);
    c_j_joint{t} =     ones(1,N); 
end
c_j_img{T+1}     =   ones(1,N);  p_j_img     =   Calculate_pj(c_j_img,T);
c_j_tags{T+1}    =   ones(1,N);  p_j_tags    =   Calculate_pj(c_j_tags,T);
c_j_joint{T+1}   =   ones(1,N);  p_j_joint   =   Calculate_pj(c_j_joint,T);

TimeAll     =   0   ;
TimEAll = zeros(1, IterAll);
Collection_time = 0;

%% Gibbs Sampling        
for iter   =   1 : IterAll
    tic;

    %% ============= Upward Pass ===================
    for t   =   1:T
        
        if t    ==  1 %&& Tcurrent==1            
            switch DataType
                case 'Count'
                    Xt_img = sparse(X_img);
                    Xt_tags = sparse(X_tags);
                case 'Positive'
                    Rate = Phi_img{1}*Theta_img{1};
                    Rate = 2*sqrt(a_j(jj)'.*X_img(iijj).*Rate(iijj));
                    M  = Truncated_bessel_rnd( Rate );
                    a_j = randg(full(sparse(1,jj,M,1,N))+ac) ./ (bc+sum(X_img,1));
                    Xt_img = sparse(ii,jj,M,V_img,N);  
                    Xt_tags = sparse(X_tags);
            end
            
            if 0
                [Xt_to_t1_img{t},WSZS_img{t}]   =   Multrnd_Matrix_mex_fast(Xt_img,Phi_img{t},Theta_img{t});
                [Xt_to_t1_tags{t},WSZS_tags{t}] =   Multrnd_Matrix_mex_fast(Xt_tags,Phi_tags{t},Theta_tags{t}); 
            else
                [Xt_to_t1_img{t},WSZS_img{t}]   =   Multrnd_Matrix_mex_fast_v2(full(Xt_img),Phi_img{t},Theta_img{t});
                [Xt_to_t1_tags{t},WSZS_tags{t}]   =   Multrnd_Matrix_mex_fast_v2(full(Xt_tags),Phi_tags{t},Theta_tags{t}); 
                Xt_to_t1_joint{t} = Xt_to_t1_img{t} + Xt_to_t1_tags{t}; %% 这边的k咋整？
            end
            
        else       
            [Xt_to_t1_joint{t},WSZS_joint{t}]   =   CRT_Multrnd_Matrix(sparse(Xt_to_t1_joint{t-1}),Phi_joint{t},Theta_joint{t});    
        end
        
        if t==1
            Phi_img{t} = SamplePhi(WSZS_img{t},Eta{t});     
            Phi_tags{t} = SamplePhi(WSZS_tags{t},Eta{t});   
            if nnz(isnan(Phi_img{t})) || nnz(isinf(Phi_img{t}))
                warning(['Phi_img Nan',num2str(nnz(isnan(Phi_img{t}))),'_Inf',num2str(nnz(isinf(Phi_img{t})))]);
                Phi_img{t}(isnan(Phi_img{t}))   =   0   ;
            end      
            if nnz(isnan(Phi_tags{t})) || nnz(isinf(Phi_tags{t}))
                warning(['Phi_tags Nan',num2str(nnz(isnan(Phi_tags{t}))),'_Inf',num2str(nnz(isinf(Phi_tags{t})))]);
                Phi_tags{t}(isnan(Phi_tags{t}))   =   0   ;
            end   
        else
            Phi_joint{t} = SamplePhi(WSZS_joint{t},Eta{t}); 
            if nnz(isnan(Phi_joint{t})) || nnz(isinf(Phi_joint{t}))
                warning(['Phi_joint Nan',num2str(nnz(isnan(Phi_joint{t}))),'_Inf',num2str(nnz(isinf(Phi_joint{t})))]);
                Phi_joint{t}(isnan(Phi_joint{t}))   =   0   ;
            end   
        end  
    end
    
    Xr = CRT_sum_mex_matrix_v2(Xt_to_t1_joint{T}',r_k')';
    [r_k,gamma0,c0]= Sample_rk(full(Xr),r_k,p_j_joint{T+1},gamma0,c0);

    
    
    %% ============= Downward Pass ======================
    if iter>10 
        if T > 1
            p_j_joint{2}  = betarnd(  sum(Xt_to_t1_joint{1},1)+a0pj    ,   sum(Theta_joint{2},1)+b0pj  );
        else
            p_j_joint{2}  = betarnd(  sum(Xt_to_t1_joint{1},1)+a0pj    ,   sum(r_k,1)+b0pj  );         
        end
        
        p_j_joint{2} = min( max(p_j_joint{2},realmin) , 1-realmin);
        c_j_joint{2} = (1-p_j_joint{2})./p_j_joint{2};     
        
        for t   =   3:(T+1)
            if t    ==  T+1
                c_j_joint{t} = randg(sum(r_k)*ones(1,N)+e0cj) ./ (sum(Theta_joint{t-1},1)+f0cj);
            else
                c_j_joint{t} = randg(sum(Theta_joint{t},1)+e0cj) ./ (sum(Theta_joint{t-1},1)+f0cj);
            end
        end
        
        p_j_temp_joint = Calculate_pj(c_j_joint,T);
        p_j_joint(3:end)=p_j_temp_joint(3:end);
        
    end

    for t  =   T:-1:1
        
        if t ==  T
            shape = r_k;   
        else
            shape = Phi_joint{t+1}*Theta_joint{t+1};
        end
       
        if t==1
            Theta_joint{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1_joint{t})),1 );
            tmp_sum  = sum(Theta_joint{t},1);
            tmp_img  = sum(Xt_to_t1_img{t},1);
            tmp_tags = sum(Xt_to_t1_tags{t},1);  
            k_im = (tmp_img./tmp_sum);
            k_tags = (tmp_tags./tmp_sum);
            Theta_img{t}  = bsxfun(@times, Theta_joint{t},   k_im   .* (1./-logmax(1-p_j_img{t})) );
            Theta_tags{t} = bsxfun(@times, Theta_joint{t},   k_tags .* (1./-logmax(1-p_j_tags{t})) ); 
        else
            Theta_joint{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1_joint{t})),     1 ./ (c_j_joint{t+1}-logmax(1-p_j_joint{t})) ); 
        end
        
    end
  
    %% ====================  Figure  ====================
    TimeOneIter     =   toc     ;
    TimEAll(iter)   =   TimeOneIter;
    TimeAll         =   TimeAll + TimeOneIter   ;
    if (mod(iter,10) == 0) | (iter < 5)
        fprintf('--------Train Phase Layer: %d, Iter: %d, Time: %d--------\n',T,iter,TimeOneIter)  ;  
    end
    
    %% ====================  Collection  ====================
    if (iter>(IterAll/2)) && (mod(iter,5)==0)
        Collection_time = Collection_time + 1;
        for t = 1:T
            Phi_img_UsageCun{t} = Phi_img_UsageCun{t} + Phi_img{t};
            Phi_tags_UsageCun{t} = Phi_tags_UsageCun{t} + Phi_tags{t};
            Phi_joint_UsageCun{t} = Phi_joint_UsageCun{t} + Phi_joint{t};
            Theta_joint_UsageCun{t} = Theta_joint_UsageCun{t} + Theta_joint{t};
        end     
    end
end

for t = 1:T
    Phi_img{t} = Phi_img_UsageCun{t}/Collection_time;
    Phi_tags{t} = Phi_tags_UsageCun{t}/Collection_time;
    Phi_joint{t} = Phi_joint_UsageCun{t}/Collection_time;
    Theta_joint{t} = Theta_joint_UsageCun{t}/Collection_time;
end

%%                              fine-tune
for iter   =   1 : IterAll
    tic;

    %%============= Upward Pass ===================
    for t   =   1:T
        if t    ==  1 %&& Tcurrent==1
            switch DataType
                case 'Count'
                    Xt_img = sparse(X_img);
                    Xt_tags = sparse(X_tags);
                case 'Positive'
                    Rate = Phi_img{1}*Theta_img{1};
                    Rate = 2*sqrt(a_j(jj)'.*X_img(iijj).*Rate(iijj));
                    M  = Truncated_bessel_rnd( Rate );
                    a_j = randg(full(sparse(1,jj,M,1,N))+ac) ./ (bc+sum(X_img,1));
                    Xt_img = sparse(ii,jj,M,V_img,N);  
                    Xt_tags = sparse(X_tags);
            end
            if 0
                [Xt_to_t1_img{t},WSZS_img{t}]   =   Multrnd_Matrix_mex_fast(Xt_img,Phi_img{t},Theta_img{t});
                [Xt_to_t1_tags{t},WSZS_tags{t}] =   Multrnd_Matrix_mex_fast(Xt_tags,Phi_tags{t},Theta_tags{t}); 

            else
                tic;
                [Xt_to_t1_img{t},WSZS_img{t}]   =   Multrnd_Matrix_mex_fast_v2(full(Xt_img),Phi_img{t},Theta_img{t});
                [Xt_to_t1_tags{t},WSZS_tags{t}]   =   Multrnd_Matrix_mex_fast_v2(full(Xt_tags),Phi_tags{t},Theta_tags{t}); 
                Xt_to_t1_joint{t} = Xt_to_t1_img{t} + Xt_to_t1_tags{t}; %% 这边的k咋整？
                tttt = toc;
                TimEAll(iter) = tttt;
                fprintf('MultiSample used %f time!\n', tttt);
            end
        else       
            [Xt_to_t1_joint{t},WSZS_joint{t}]   =   CRT_Multrnd_Matrix(sparse(Xt_to_t1_joint{t-1}),Phi_joint{t},Theta_joint{t});    
        end
        
        if t==1
            Phi_img{t} = SamplePhi(WSZS_img{t},Eta{t}); %             figure(25),DispDictionaryImagesc(Phi{1});drawnow;
            Phi_tags{t} = SamplePhi(WSZS_tags{t},Eta{t}); %             figure(25),DispDictionaryImagesc(Phi{1});drawnow
            if nnz(isnan(Phi_img{t})) || nnz(isinf(Phi_img{t}))
                warning(['Phi_img Nan',num2str(nnz(isnan(Phi_img{t}))),'_Inf',num2str(nnz(isinf(Phi_img{t})))]);
                Phi_img{t}(isnan(Phi_img{t}))   =   0   ;
            end      
            if nnz(isnan(Phi_tags{t})) || nnz(isinf(Phi_tags{t}))
                warning(['Phi_tags Nan',num2str(nnz(isnan(Phi_tags{t}))),'_Inf',num2str(nnz(isinf(Phi_tags{t})))]);
                Phi_tags{t}(isnan(Phi_tags{t}))   =   0   ;
            end   
        else
            Phi_joint{t} = SamplePhi(WSZS_joint{t},Eta{t}); 
            if nnz(isnan(Phi_joint{t})) || nnz(isinf(Phi_joint{t}))
                warning(['Phi_joint Nan',num2str(nnz(isnan(Phi_joint{t}))),'_Inf',num2str(nnz(isinf(Phi_joint{t})))]);
                Phi_joint{t}(isnan(Phi_joint{t}))   =   0   ;
            end   
        end
        
    end
    
%%    
    %%====================  Figure  ====================
    TimeOneIter     =   toc     ;
    TimEAll(iter)   = TimeOneIter;
    TimeAll     =   TimeAll + TimeOneIter   ;
    if (mod(iter,10) == 0) | (iter < 5)
        fprintf('--------Layer: %d, Iter: %d, Time: %d, r_k: %d--------\n',T,iter,TimeOneIter,mean(r_k))  ;  
    end
        
end

tmp = num2str(K(1));
for i = 1:size(K,2)-1
    tmp = [tmp,'_',num2str(K(i+1))];
end
eval(['save ',Settings.GibbsSampler,'_',Settings.DataName,'_',Settings.DataType,'_',Settings.Date,'_','layer_',tmp,'_iter_',num2str(IterAll),'.mat']);
