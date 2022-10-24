% Project07

Ia = im2double(imread('Fig0343(a)(skeleton_orig).tiff'));

Ib = imfilter(Ia, -fspecial('laplacian', 0.5) * 3);
Ic = Ia + Ib;
Id = abs(imfilter(Ia, fspecial('sobel'))) + abs(imfilter(Ia, fspecial('sobel')'));
Ie = imfilter(Id, fspecial('average',5));
If = Ib .* Ie;
Ig = Ia + If;
Ih = imadjust(Ig, [], [], 0.5);

figure
imshow(Ia, [])
title('Original image (a)')
figure
imshow(Ib, [])
title('Image in (b)')

figure
imshow(Ic, [])
title('Image in (c)')

figure
imshow(Id, [])
title('Image in (d)')

figure
imshow(Ie, [])
title('Image in (e)')

figure
imshow(If, [])
title('Image in (f)')

figure
imshow(Ig, [])
title('Image in (g)')
figure
imshow(Ih, [])
title('Image in (h)')