% DEFINE THE WINDOW SIZE M X N

function image = median_filter_to_image(input_image)
M = 3;
N = 3;
% PAD THE MATRIX WITH ZEROS ON ALL SIDES
modify_input_image = padarray(input_image, [floor(M / 2), floor(N / 2)]);

temp_matrix = zeros([size(modify_input_image, 1) size(modify_input_image, 2)]);
med_indx = round((M * N) / 2);
for i = 1:size(modify_input_image, 1) - (M - 1)
    for j = 1:size(modify_input_image, 2) - (N - 1)
        temp = modify_input_image(i:i + (M - 1), j:j + (N - 1), :);
        % RED,GREEN AND BLUE CHANNELS ARE TRAVERSED SEPARATELY
        for k = 1:3
            tmp = temp(:, :, k);
            temp_matrix(i, j, k) = median(tmp(:));
        end
    end
end


%CONVERT THE IMAGE TO UINT8 FORMAT.
image = uint8(temp_matrix);
end