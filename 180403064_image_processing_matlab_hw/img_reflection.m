function reflection = img_reflection(file,inp)
if(inp == "horizontal")
    reflection = file(:,end:-1:1,:);
elseif(inp == "vertical")
    reflection = file(end:-1:1,:,:);
end   
end
