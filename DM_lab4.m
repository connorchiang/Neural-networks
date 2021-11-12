
W1=rand(4,2);
W2=rand(2,4);
W11=W1;
W22=W2;

input_pattern=[1,0,0,0;
               0,1,0,0;
               0,0,1,0;
               0,0,0,1;];

target_pattern = input_pattern;

b1=rand(1,2);
b2=rand(1,4);

N=1000;

%n=1;
for n=1:1:N
    
    Error = 0;
    a_delta_W1=zeros(4,2);
    a_delta_W2=zeros(2,4);
    a_delta_b1=zeros(2,1);
    a_delta_b2=zeros(4,1);
    
    for j= 1:1:4%%%%%%%%%%%%5
        for i=1:1:4
            o_input(i,j) = 1 / ( 1 + exp(-input_pattern(i,j)));
        end   
        
        i_hidden(:,j)= W1.' * o_input(:,j) + b1.';
        %i_hidden(:,j)= W1.' * o_input(:,j) ;
        
        for i=1:1:2
            o_hidden(i,j) = 1 / ( 1 + exp(-i_hidden(i,j)));
        end
        
        i_output(:,j)=W2.' * o_hidden(:,j) + b2.';
        %i_output(:,j)=W2.' * o_hidden(:,j) + b2.';
        
        for i=1:1:4
            o_ouput(i,j) = 1 / ( 1 + exp(-i_output(i,j)));
        end
        
        E_p=0;
        
        for i=1:1:4
            E(i,j) = 1/2 * (o_ouput(i,j)- target_pattern(i,j))^2;
            E_p= E_p + E(i,j);
        end
        Error = Error + E_p;
    
    
    %E_n(n)=Error;
        % Calculate the delta weights of W2
    
        for i = 1:1:4 %which is j
            delta_output(i,j) = (o_ouput(i,j)- target_pattern(i,j)) * o_ouput(i,j) * (1 - o_ouput(i,j));
            for r= 1:1:2 % which is i
                delta_W2(r,i) =   o_hidden(r,j) * delta_output(i,j).' ; % o_i * delta_j
            end
        end
        
        % accumulate delta_W2 before updating W2
        a_delta_W2 = a_delta_W2 + delta_W2;
        a_delta_b2=a_delta_b2 + delta_output(:,j);

        %sum_w_delta =[0;0];
        %for r= 1:1:2
        sum_w_delta = W2 * delta_output(:,j);
                %sum_w_delta(r,1)= sum_w_delta(r,1) + w_delta;
        %end

        for r = 1:1:2
            delta_b1(r,1) = sum_w_delta(r,1)*  o_hidden(r,j) * (1 - o_hidden(r,j));
            for i= 1:1:4
                delta_W1(i,r) =  sum_w_delta(r,1)*  o_hidden(r,j) * (1 - o_hidden(r,j)) * o_input(i,j);
                %delta_b1(r,1) = sum_w_delta(r,1)*  o_hidden(r,j) * (1 - o_hidden(r,j));
            end
        end
        
        a_delta_W1 = a_delta_W1 + delta_W1;
        a_delta_b1 = a_delta_b1 + delta_b1;
    end
    E_n(n)=Error;
    
    rate=3;
        
    W2 = W2 - rate * a_delta_W2;
    W1 = W1 - rate * a_delta_W1;
    b2 = b2 - rate * a_delta_b2.';
    b1 = b1 - rate * a_delta_b1.';

    %Error = Error + E_p;
    
end

subplot(2,1,1);
plot(E_n);
title("Error in 4-2-4 encoder with bias");
xlabel('Iterations');
ylabel('Error');

% % Test part
% for j= 1:1:4%%%%%%%%%%%%5
%     for i=1:1:4
%         o_input(i,j) = 1 / ( 1 + exp(-input_pattern(i,j)));
%     end   
%         
%     i_hidden(:,j)= W1.' * o_input(:,j) + b1.';
%     %i_hidden(:,j)= W1.' * o_input(:,j) ;
%         
%     for i=1:1:2
%         o_hidden(i,j) = 1 / ( 1 + exp(-i_hidden(i,j)));
%     end
%         
%     i_output(:,j)=W2.' * o_hidden(:,j) + b2.';
%         %i_output(:,j)=W2.' * o_hidden(:,j) + b2.';
%         
%     for i=1:1:4
%         o_ouput(i,j) = 1 / ( 1 + exp(-i_output(i,j)));
%     end
% 
% end% Output the test result
% o_ouput


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
for n=1:1:N
    
    Error = 0;
    a_delta_W1=zeros(4,2);
    a_delta_W2=zeros(2,4);
    a_delta_b1=zeros(2,1);
    a_delta_b2=zeros(4,1);
    
    for j= 1:1:4%%%%%%%%%%%%5
        for i=1:1:4
            o_input(i,j) = 1 / ( 1 + exp(-input_pattern(i,j)));
        end   
        
        i_hidden(:,j)= W11.' * o_input(:,j);
        %i_hidden(:,j)= W1.' * o_input(:,j) ;
        
        for i=1:1:2
            o_hidden(i,j) = 1 / ( 1 + exp(-i_hidden(i,j)));
        end
        
        i_output(:,j)=W22.' * o_hidden(:,j);
        %i_output(:,j)=W2.' * o_hidden(:,j) + b2.';
        
        for i=1:1:4
            o_ouput(i,j) = 1 / ( 1 + exp(-i_output(i,j)));
        end
        
        E_p=0;
        
        for i=1:1:4
            E(i,j) = 1/2 * (o_ouput(i,j)- target_pattern(i,j))^2;
            E_p= E_p + E(i,j);
        end
        Error = Error + E_p;
    
    
    %E_n(n)=Error;
        % Calculate the delta weights of W2
    
        for i = 1:1:4 %which is j
            delta_output(i,j) = (o_ouput(i,j)- target_pattern(i,j)) * o_ouput(i,j) * (1 - o_ouput(i,j));
            for r= 1:1:2 % which is i
                delta_W2(r,i) =   o_hidden(r,j) * delta_output(i,j).' ; % o_i * delta_j
            end
        end
        
        % accumulate delta_W2 before updating W2
        a_delta_W2 = a_delta_W2 + delta_W2;
        a_delta_b2=a_delta_b2 + delta_output(:,j);

        %sum_w_delta =[0;0];
        %for r= 1:1:2
        sum_w_delta = W22 * delta_output(:,j);
                %sum_w_delta(r,1)= sum_w_delta(r,1) + w_delta;
        %end

        for r = 1:1:2
            delta_b1(r,1) = sum_w_delta(r,1)*  o_hidden(r,j) * (1 - o_hidden(r,j));
            for i= 1:1:4
                delta_W1(i,r) =  sum_w_delta(r,1)*  o_hidden(r,j) * (1 - o_hidden(r,j)) * o_input(i,j);
                %delta_b1(r,1) = sum_w_delta(r,1)*  o_hidden(r,j) * (1 - o_hidden(r,j));
            end
        end
        
        a_delta_W1 = a_delta_W1 + delta_W1;
        a_delta_b1 = a_delta_b1 + delta_b1;
    end
    E_n(n)=Error;
    
    rate=3;
        
    W22 = W22 - rate * a_delta_W2;
    W11 = W11 - rate * a_delta_W1;
    %b2 = b2 - rate * a_delta_b2.';
    %b1 = b1 - rate * a_delta_b1.';

    %Error = Error + E_p;
    
end

subplot(2,1,2);
plot(E_n);
title("Error in 4-2-4 encoder");
xlabel('Iterations');
ylabel('Error');

% Test part
% for j= 1:1:4%%%%%%%%%%%%5
%     for i=1:1:4
%         o_input(i,j) = 1 / ( 1 + exp(-input_pattern(i,j)));
%     end   
%         
%     i_hidden(:,j)= W11.' * o_input(:,j) ;
%     i_hidden(:,j)= W1.' * o_input(:,j) ;
%         
%     for i=1:1:2
%         o_hidden(i,j) = 1 / ( 1 + exp(-i_hidden(i,j)));
%     end
%         
%     i_output(:,j)=W22.' * o_hidden(:,j) ;
%         i_output(:,j)=W2.' * o_hidden(:,j) + b2.';
%         
%     for i=1:1:4
%         o_ouput(i,j) = 1 / ( 1 + exp(-i_output(i,j)));
%     end
% 
% end% Output the test result
% o_ouput