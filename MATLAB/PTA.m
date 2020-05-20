function [omegas, missed] = PTA(model, omega, learning_rate)
    epoch = 1;
    omegas = zeros(100,3);
    missed = [];
 
    mem = 1;
    while (misclassified(model,omega)~=0)
        num = misclassified(model,omega);
       missed = [missed num];
        fprintf('%d\t', missed(epoch));
        epoch = epoch + 1;
       % fprintf('Epoch Number: %d\n ', epoch);
        for i = 1:length(model)
            y = omega(1) + (model(i,1)*omega(2))+(model(i,2)*omega(3));
            y = step_active(y);
            updated_input =[1 model(i,1) model(i,2)];
            desired_output = model(i,3);
            difference = desired_output-y;
            if difference ~= 0
                updated_input(1)= updated_input(1)*learning_rate*difference;
                updated_input(2)= updated_input(2)*learning_rate*difference;
                updated_input(3)= updated_input(3)*learning_rate*difference;
                omega(1) = omega(1)+updated_input(1);
                omega(2) = omega(2)+updated_input(2);
                omega(3) = omega(3)+updated_input(3);
            end
        end
        omegas(mem,:)=omega(1,:);
        mem = mem+1;
    final = misclassified(model,omega);
    end
    omegas = omegas(1:mem-1,:);
    omegas;
    %fprintf('Final weights: %d\n', omegas);
     %fprintf('Missed: %d\n', missed);
    
end