%This program is designed to infer Urban Structure from Traffic Flow
%using Cellular Learning Automata
clc
clear
close all
n = 5;  %Number of zones.
t = 10; %Number of time periods.
S = round((4).*rand(n)); %Structural data.
D = round((10).*rand(n,t));

F = ones(n); %Functional data.
for i = 1:n
    for j = 1:n
        temp = 0;
        for t = 1:n
            temp = sqrt(D(i,t)*D(j,t))/sqrt(sum(D(i,:))*sum(D(j,:)));
        end
        F(i,j) = abs(1 - sqrt(1 - temp));
    end
end
F(1:1 + size(F,1):end) = 0;
S(1:1 + size(S,1):end) = 0;
%Making graph symmetric
numberNodes = size(S,1); %Number of CLAs.
NumberLA = numberNodes;
for ii = 1:numberNodes
    S(ii,:)=S(:,ii);
    S(ii,ii) = 0;
end
S = logical(S);
deg = sum(S); %Number of actions for each CLA.
B = S - (deg'*deg./sum(deg));
V = D*transpose(D);
Vbar = V./sum(V,2);
f = @Fitness;
ij = 1:numberNodes;
%%
% Parameters
n1 = 100;   %number of external iterations.
n2 = 100;   %number of internal iterations.
teta = 2;
alpha = 0.5;
beta = 1 - alpha;
ne = 10;    %internal loop termination
a = 0.1;
b = 0.01;


%%
O = zeros(numberNodes);
fitnessFinal = 0;

for it1 = 1:n1
    disp(it1)
    prob = ones(numberNodes,numberNodes)./(numberNodes-1);
    prob(1:numberNodes + 1:numberNodes^2) = 0;
    sumS = zeros(numberNodes,numberNodes);
    for i = 1:numberNodes
        ro = (prob(i,:)./(numberNodes - deg(i)) + Vbar(i,:)).^teta;
        s = ro./sum(ro);
        sumS(i,:) = cumsum(s);
    end
    OEstimated = zeros(numberNodes);
    fitness = 0;
    
    for it2 = 1:n2
        OEstimatedNew = zeros(numberNodes);
        condition = 0;
        tempF = 0;
        while (condition < ne)
            for i = 1:NumberLA
                j = find(rand<sumS(i,:),1,'first');
                beforefitness = f(OEstimatedNew, S, F, alpha, beta);
                OEstimatedNew(i,j) = OEstimatedNew(i,j) + 1;
                if OEstimatedNew(i,j) > S(i,j)
                    afterFitness = f(OEstimatedNew, S, F, alpha, beta);
                    if afterFitness < beforefitness
                        OEstimatedNew(i,j) = OEstimatedNew(i,j) - 1;
                    end
                end
            end
            if tempF < f(OEstimatedNew, S, F, alpha, beta)
                tempF = f(OEstimatedNew, S, F, alpha, beta);
                condition = 0;
            else
                condition = condition + 1;
            end
        end
        tempFitness = f(OEstimatedNew, S, F, alpha, beta);
        OEstimatedNew(1:1 + size(OEstimatedNew,1):end) = 0;
        if tempFitness > fitness
            fitness = tempFitness;
            OEstimated = OEstimatedNew;
            for i = 1:NumberLA
                for j = 1:NumberLA
                    for k = 1:OEstimatedNew(i,j)
                        prob(i,j) = prob(i,j) + a*(1 - prob(i,j));
                        ij1 = ij;
                        ij1(j) = [];
                        prob(i,ij1) = (1 - a).*prob(i,ij1);
                    end
                end
            end
        else
            for i = 1:NumberLA
                for j = 1:NumberLA
                    ij1 = ij;
                    for k = 1:OEstimatedNew(i,j)
                        prob(i,j) = (1 - b).*prob(i,j);
                        ij1 = ij;
                        ij1(j) = [];
                        prob(i,ij1) = b/(numberNodes - 1) + (1 - b).*prob(i,ij1);
                    end
                end
            end
        end
        sumS = zeros(numberNodes,numberNodes);
        for i = 1:numberNodes
            ro = (prob(i,:)./(numberNodes - deg(i)) + Vbar(i,:)).^teta;
            s = ro./sum(ro);
            sumS(i,:) = cumsum(s);
        end
    end
    if fitness > fitnessFinal
        fitnessFinal = fitness;
        O = OEstimated;
    end
end

function y = Fitness(O, S, F, alpha, beta)


    y1 = 1/(sum(sum((O - S).^2)) + 1);
    y2 = sum(sum((O/max(max(O)).*F)/sum(sum(F))));
    y = alpha*y1 + beta*y2;

end