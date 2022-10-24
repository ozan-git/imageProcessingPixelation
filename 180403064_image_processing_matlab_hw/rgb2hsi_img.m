function img = rgb2hsi_img(file)
    I = double(file) / 255;
    HSV = rgb2hsv(file);
    H = HSV(:,:,1);
    S = HSV(:,:,2);
  
% Intensity
    I = sum(I, 3)./3;
  
% Creating the HSL Image
  HSI = zeros(size(file));
  HSI(:,:,1) = H;
  HSI(:,:,2) = S;
  HSI(:,:,3) = I;
  img = HSI;
end