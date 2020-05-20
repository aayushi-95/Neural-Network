x = []
y = []
for i = 1:50
    x_temp = i + 1
    u = -1 + (1+1)*rand(1,1)
    y_temp = i + 1 + u
    x = [x x_temp]
    y = [y y_temp]
end

% optimal w0 and w1 values by y*psuedo_inv_of_x from summation

a = ones(1, 50);
a1 = [a x];

a2 = transpose(a1);
b = a1*a2;
b2 = inv(b);

pi = a2 * b2;

w = pi*y;

hold on
figure;
p = [w(2) w(1)];
yn = polyval(p,x);
% Normal LLS
scatter(x, y,'r');
% GD LLS
% plot(x, yn,'kx-');
ylabel('Y - values');
xlabel('X - values');
title('Linear Least Squares');
hold off;

