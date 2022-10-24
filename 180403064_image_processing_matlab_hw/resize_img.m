
function img = resize_img(file, row, column)
[rowImage, columnImage, channelImage] = size(file);
zeroArray = zeros(row, column, channelImage);

resize_factor_row = rowImage / row;
resize_factor_column = columnImage / column;
for i=1:row
    for j=1:column
        for m=1:channelImage
            zeroArray(i, j, m) = file(int16(i * resize_factor_row), int16(j * resize_factor_column), m);
        end
    end
end
img = uint8(zeroArray);
end
