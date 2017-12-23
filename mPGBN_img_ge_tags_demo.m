%% load data

close all
addpath(genpath('./'))
load Multi_PGBN_Generate_Tags_flicker_25_w10_s1_e1_Count_2017_12_18_layer_500_200_iter_200.mat
load labeled_w10.mat

%% 1:1500 train 1501:1995 test
index_in_w10 = 1632;                 %    1642 1632                 
index_in_25000 = w10_labeled_index(index_in_w10); 
figure
imshow(['im',num2str(index_in_25000),'.jpg']);   
title(['\fontsize{10}the topics of',num2str(index_in_25000),'th image'])

%% 读取词表
vocabWords_list = textread('vocab.txt','%s','delimiter',' ');
vocabWords = cell(length(vocabWords_list)/3,1);
for i = 1:length(vocabWords_list)/3
    vocabWords{i,1} = vocabWords_list{3*i-1,1};
end

%%                              img generate tags


%%======== 20news GanZhe   ============

%% initial 

IsTestPerpZhou = 1;
X_img_all_ge   = X_img_all(:,index_in_w10); 
X_img_all_ge  = full(X_img_all_ge);

[ii,jj] = find(X_img_all_ge>eps);
iijj = find(X_img_all_ge>eps);

%%    infer tags from img
if IsTestPerpZhou
    NTest   =   size(X_img_all_ge,2)  ;
    FlagHOTest      =   X_img_all_ge > 0      ;
    
    Xt_to_t1_img_ge    =   cell(T,1);
    Xt_to_t1_joint_ge  =   cell(T,1);
    Theta_img_ge = cell(T,1);
    Theta_joint_ge   =   cell(T,1);
    c_j_img_ge      =   cell(T+1,1);
    c_j_joint_ge    =   cell(T+1,1);

    for t=T:-1:1
        Theta_joint_ge{t}    =   ones(K(t),NTest)/K(t);
        Theta_img_ge{t}      =   ones(K(t),NTest)/K(t);
    end
    
    for t=1:(T+1)
        c_j_img_ge{t}    = ones(1,NTest);
        c_j_joint_ge{t}  = ones(1,NTest);        
    end
    
    p_j_img_ge   = Calculate_pj( c_j_img_ge,T);
    p_j_joint_ge = Calculate_pj( c_j_joint_ge,T); 
end
a_j_ge = ones(1,NTest);

for iter = 1:IterAll
    tic
    for t   =   1:T
        if t    ==  1
            switch DataType
                case 'Count'
                    Xt_img_all_ge = sparse(X_img_all_ge);
                case 'Positive'
                    Rate = Phi_img{1}*Theta_img_ge{1};
                    Rate = 2*sqrt(a_j_ge(jj)'.*X_img_all_ge(iijj).*Rate(iijj));
                    M  = Truncated_bessel_rnd( Rate );
                    a_j_ge = randg(full(sparse(1,jj,M,1,NTest))+ac) ./ (bc+sum(X_img_all_ge,1));
                    Xt_img_all_ge = sparse(ii,jj,M,V_img,NTest);  
            end
            Xt_to_t1_img_ge{t}     =   Multrnd_Matrix_mex_fast_v1(Xt_img_all_ge,Phi_img{t},Theta_img_ge{t}); 
            Xt_to_t1_joint_ge{t}   =   Xt_to_t1_img_ge{t} ;
        else
            Xt_to_t1_joint_ge{t}   =   CRT_Multrnd_Matrix(sparse(Xt_to_t1_joint_ge{t-1}),Phi_joint{t},Theta_joint_ge{t});
        end
    end
        

        if T > 1
            p_j_joint_ge{2}  = betarnd(  sum(Xt_to_t1_joint_ge{1},1)+a0pj    ,   sum(Theta_joint_ge{2},1)+b0pj  );
        else
            p_j_joint_ge{2}  = betarnd(  sum(Xt_to_t1_joint_ge{1},1)+a0pj   ,   sum(r_k,1)+b0pj  );      
        end
        p_j_joint_ge{2} = min( max(p_j_joint_ge{2},realmin) , 1-realmin);
        c_j_joint_ge{2} = (1-p_j_joint_ge{2})./p_j_joint_ge{2};
         
        for t   =   3:(T+1)
            if t    ==  T+1
                c_j_joint_ge{t} = randg(sum(r_k)*ones(1, NTest)+e0cj)    ./ (sum(Theta_joint_ge{t-1},1)+f0cj);
            else
                c_j_joint_ge{t} = randg(sum(Theta_joint_ge{t},1)+e0cj) ./ (sum(Theta_joint_ge{t-1},1)+f0cj);
            end
        end
        p_j_temp_joint = Calculate_pj(c_j_joint_ge,T);
        p_j_joint_ge(3:end)=p_j_temp_joint(3:end);
    
    
    for t  =   T:-1:1
        if t    ==  T
            shape = r_k;         
            Theta_joint_ge{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1_joint_ge{t} )), 1 ./ (c_j_joint_ge{t+1}(1,1:NTest)-logmax(1-p_j_joint_ge{t}(1,1:NTest) ) ) );%尺度参数没有关系，在计算Theta_img Theta_tags时会被消掉

            if t == 1;
                Theta_joint_ge{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1_joint_ge{t} )), 1);%尺度参数没有关系，在计算Theta_img Theta_tags时会被消掉
                tmp_sum  = sum(Theta_joint_ge{t},1);                
                tmp_img  = sum(Xt_to_t1_img_ge{t},1);
                k_img_ge  = tmp_img./tmp_sum;
                Theta_img_ge{t}  = bsxfun(@times, Theta_joint_ge{t}, k_img_ge  .* (1./-logmax(1-p_j_img_ge{t}(1,1:NTest))) ); 
            end
        else
            shape  = Phi_joint{t+1}*Theta_joint_ge{t+1};
            Theta_joint_ge{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1_joint_ge{t} )), 1 ./ (c_j_joint_ge{t+1}(1,1:NTest)-logmax(1-p_j_joint_ge{t}(1,1:NTest) ) ) );%尺度参数没有关系，在计算Theta_img Theta_tags时会被消掉

            if t == 1;
                Theta_joint_ge{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1_joint_ge{t} )), 1);%尺度参数没有关系，在计算Theta_img Theta_tags时会被消掉
                tmp_sum  = sum(Theta_joint_ge{t},1);                
                tmp_img  = sum(Xt_to_t1_img_ge{t},1);
                k_img_ge  = tmp_img./tmp_sum;
                Theta_img_ge{t}  = bsxfun(@times, Theta_joint_ge{t}, k_img_ge    .* (1./-logmax(1-p_j_img_ge{t}(1,1:NTest))) ); 
            end
        end

    end    
    
    TimeOneIter = toc;
    fprintf('-------- Iter: %d, Time: %d, r_k: %d--------\n',iter,TimeOneIter,mean(r_k))  ;  

end

%%  show reconstruct_data
disp('Orginal Words Disp:')
X_tags_full = full(X_tags_all);
Re_tags = Phi_tags{1}*Theta_img_ge{1};

X_index = index_in_w10;
X_tr = X_tags_full(:,X_index);   
Word_index_tr = find(X_tr~=0);   
for i = 1:length(Word_index_tr)       
    fprintf('\t%s \n', vocabWords{Word_index_tr(i)});    
end


disp('Generate Words Disp:')
Pt_n = [Re_tags,[1:2000]'];
Pt_n = fliplr(sortrows(Pt_n, 1)');
for i = 1:20
    fprintf('\t%s %f\n', vocabWords{Pt_n(2, i)}, Pt_n(1, i));
end