taggedCombinedRandomShuffle = function(reviews, tags, summaries)
    local t1 = {}
    local t2 = {}
    local tTags = {}

    local revcnt = 1
    local tmp = {}
    local line = reviews[revcnt]
    while true do
        table.insert(tmp, line)
        if line == "" or line == nil then
            table.insert(t1, localCopy(tmp))
            tmp = {}
            if line == nil then
                break
            end
        end
        revcnt = revcnt + 1
        line = reviews[revcnt]
    end
    
    local sumcnt = 1 
    line = summaries[sumcnt]
    tmp = {}
    local line = summaries[sumcnt]
    while true do
        table.insert(tmp, line)
        if line == "" or line == nil then
            table.insert(t2, localCopy(tmp))
            tmp = {}
            if line == nil then 
                break
            end
        end
        sumcnt = sumcnt + 1
        line = summaries[sumcnt]
    end

    local tagcnt = 1 
    line = tags[tagcnt]
    tmp = {}
    local line = tags[tagcnt]
    while true do
        table.insert(tmp, line)
        if line == "" or line == nil then
            table.insert(tTags, localCopy(tmp))
            tmp = {}
            if line == nil then 
                break
            end
        end
        tagcnt = tagcnt + 1
        line = tags[tagcnt]
    end
 
    print(#tTags, #t1, #t2)
    assert(#t1 == #t2)
    assert(#t2 == #tTags)
    len = #t1
    rand = torch.randperm(len)
    local tmp1 = {}
    local tmp2 = {}
    local tmpTags = {}
    for i = 1, len do
        table.insert(tmp1, localCopy(t1[i]))
        table.insert(tmp2, localCopy(t2[i]))
        table.insert(tmpTags, localCopy(tTags[i]))
    end

    t1 = {}
    t2 = {}
    tTags = {}

    for i = 1, len do
        for j = 1, #tmp1[rand[i]] do
            table.insert(t1, tmp1[rand[i]][j])
            table.insert(t2, tmp2[rand[i]][j])
            table.insert(tTags, tmpTags[rand[i]][j])
            --table.insert(t1, tmp1[i][j])
            --table.insert(t2, tmp2[i][j])
        end
    end
    return t1, tTags, t2 
end


combinedRandomShuffle = function(reviews, summaries)
    local t1 = {}
    local t2 = {}
    local revcnt = 1
    
    local tmp = {}
    local line = reviews[revcnt]
    while true do
        table.insert(tmp, line)
        if line == "" or line == nil then
            table.insert(t1, localCopy(tmp))
            tmp = {}
            if line == nil then
                break
            end
        end
        revcnt = revcnt + 1
        line = reviews[revcnt]
    end
    
    local sumcnt = 1 
    line = summaries[sumcnt]
    tmp = {}
    local line = summaries[sumcnt]
    while true do
        table.insert(tmp, line)
        if line == "" or line == nil then
            table.insert(t2, localCopy(tmp))
            tmp = {}
            if line == nil then 
                break
            end
        end
        sumcnt = sumcnt + 1
        line = summaries[sumcnt]
    end
 
    assert(#t1 == #t2)
    len = #t1
    rand = torch.randperm(len)
    local tmp1 = {}
    local tmp2 = {}
    for i = 1, len do
        table.insert(tmp1, localCopy(t1[i]))
        table.insert(tmp2, localCopy(t2[i]))
    end

    t1 = {}
    t2 = {}

    for i = 1, len do
        for j = 1, #tmp1[rand[i]] do
            table.insert(t1, tmp1[rand[i]][j])
            table.insert(t2, tmp2[rand[i]][j])
            --table.insert(t1, tmp1[i][j])
            --table.insert(t2, tmp2[i][j])
        end
    end
    return t1, t2
end

localCopy = function(tab)
    local tmp = {}
    for i,j in ipairs(tab) do
        tmp[i] = j
    end
    return tmp
end
