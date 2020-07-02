function toCopy = createZeroCopy(toCopy)
%creates a copy of a cell array containing matrices, setting all matrix
%elements to 0
for ii = 1:length(toCopy)
    toCopy{ii} = zeros(size(cell2mat( toCopy(ii))));
end
end

