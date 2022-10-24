function img = crop_img(file, x1, y1, x2, y2)
[row_image,column_image,channel_image] = size(file);
zeroArray = zeros(row_image,column_image,channel_image);
for i = x1:x2
    for j = y1:y2
        for m=1:channel_image
            zeroArray(i,j,m) = file(i,j,m);
        end
    end
end
img = uint8(zeroArray);
end