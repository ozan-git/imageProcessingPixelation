
function B=img_shift(img,n1,n2) 
	[w,h,r]=size(img); 
	B=zeros(w,h,r); 
	for i=n1:w 
		for j=n2:h 
            for m=1:r
                B(i,j,m)=img(i,j,m);
            end
        end
    end
    B= uint8(B);
end
   

  