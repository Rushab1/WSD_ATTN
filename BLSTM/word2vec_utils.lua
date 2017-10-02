function spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end
    
most_similar = function(word, lookup, Index2Vocab, vocablen, sim_total)
    local similarity = {}
    local t = torch.Tensor(1,1)
    t[1][1] = Vocab[word]
    local v_word = lookup:forward({t})[1][1]:clone()

    for i,j in pairs(Index2Vocab) do
        t = torch.Tensor(1,1)
        t[1][1] = i
        local v = lookup:forward({t})[1][1]:clone()
        similarity[i] = torch.dot(v, v_word)/(torch.norm(v)*torch.norm(v_word))
    end

    local i = 1
    local ms = {}
    for k,v in spairs(similarity, function(t,a,b) return t[b] < t[a] end) do
        ms[Index2Vocab[k]] = v
        print(Index2Vocab[k], v)
        if i == sim_total then
            break
        end
        i = i + 1
    end
    return ms
end

similarity = function(i, j, lookup)
    local t = torch.Tensor(2,1)
    t[1][1] = Vocab[i]
    t[2][1] = Vocab[j]

    local l = lookup:forward({t})[1]

    return torch.dot(l[1], l[2])/ (torch.norm(l[1]) * torch.norm(l[2]))
end

