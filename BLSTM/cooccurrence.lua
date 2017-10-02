
get_cooccurences = function(cofilename)
    if cofilename == nil then
        cofilename = "cooccurrence.csv"
    end

    cofile = io.open(cofilename, 'r')
    
    CoTable = {}

    line = cofile:read()
    while line ~= nil do
        line_split = line:split(' ')

        if line_split[3] ~= "start" or line_split[3] ~= 'end' then 
            if CoTable[line_split[1]] == nil then
                CoTable[line_split[1]] = {}
            end
            if CoTable[line_split[1]][tonumber(line_split[2])] == nil then
                CoTable[line_split[1]][tonumber(line_split[2])] = {}
            end

            CoTable[line_split[1]][tonumber(line_split[2])][line_split[3]] = tonumber(line_split[4])
        end

        line = cofile:read()
    end

    torch.save("cooccurrence.t7", CoTable)
end

get_cooccurences()
