function p_j = Calculate_pj2(c_j,T)
p_j=cell(T,T+1);

for t1 = 1:T
    N=length(c_j{t1,2});
    p_j{t1,1}=(1-exp(-1))*ones(1,N);
    p_j{t1,2} = 1./(1+c_j{t1,2});
    for t2 = 3:T+1
        temp = -log(max(1-p_j{t1,t2-1},realmin));
        p_j{t1,t2} = temp./(temp+c_j{t1,t2});
        %p_j{t} = 1./(1+c_j{t}./(-log(max(1-p_j{t-1},realmin))));
        if nnz(isnan(p_j{t1,t2}))
            warning('pj Nan');
            p_j{t1,t2}(isnan(p_j{t1,t2}))= eps ;
        end
    end
end
