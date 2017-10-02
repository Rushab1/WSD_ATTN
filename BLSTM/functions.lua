feval2 = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end
    dl_dx:zero()
    Inputs, decoderInputs, Targets = nextBatch()

    Output = model2:forward(Inputs[1])

    dt = torch.IntTensor(batchSize,1)
    
    for _i = 1, batchSize do
        _t= Targets[1][1][_i]
        dt[_i][1] = _t
    end
    loss_x = criterion2:forward(Output, dt)

    dloss_doutput = criterion2:backward(Output, dt)

    b = model2:backward(Inputs[1],dloss_doutput)

    if sgd_params.evalCounter%1 == 0 then
        str_sent = "" 
        enc_batch_size = Inputs[1]:select(2,1)
        for ii = 1,(#enc_batch_size)[1] do
            if enc_batch_size[ii] ~= 0 then
                local word = Index2Vocab[enc_batch_size[ii]]
                str_sent = str_sent .. " " .. word
            end
        end
        
        probabilities = Output:select(1,1)
        probabilities:resize(SecabSize)
        probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)

        str_sent = Index2Secab[wordIds[1]]

        str_sent_target = Index2Secab[Targets[1][1][1]]
        
        print("ACTUAL SUMMARY    : " .. str_sent_target .. "PREDICTED SUMMARY : " .. str_sent )
        model2:forget()
    end
    
    model2:forget()
    return loss_x, dl_dx
end 

function nextBatch()
    local encoderInputs, decoderInputs, decoderTargets = {}, {}, {}

    sentBatch_text = {}
    sentBatch_smy={}
    local max_len_text = 0
    local max_len_smy = 0 
    local j=0, j2

    for j=1,batchSize do 
        local sent_txt = {}
        local sent_smy = {}

        --line = file_txt:read():lower()
        line = reviews[revcnt]; revcnt = revcnt + 1

        while line ~= "" and line~=nil do 
            table.insert(sent_txt,line)
            --line = file_txt:read()
            line = reviews[revcnt]; revcnt = revcnt + 1
            if line ~= nil then
                line = line:lower()
            end
        end
        table.insert(sentBatch_text,sent_txt)
        
        if max_len_text < #sent_txt then
            max_len_text = #sent_txt
        end

        table.insert(sent_smy, '<go>')
        line = summaries[sumcnt]; sumcnt = sumcnt + 1
        while line ~= "" and line~= nil  do     
            table.insert(sent_smy, line)
            line = summaries[sumcnt]; sumcnt = sumcnt + 1
            if line~=nil then
                line = line:lower()
            end
        end 
    
        table.insert(sent_smy, '<eos>')
        table.insert(sentBatch_smy,sent_smy)
        
        if max_len_smy < #sent_smy then
            max_len_smy = #sent_smy
        end 
    end 

    local encoderInputs = torch.IntTensor(max_len_text, batchSize):fill(0)
    local decoderInputs = torch.IntTensor(max_len_smy-1,batchSize):fill(0)
    local decoderTargets= torch.IntTensor(max_len_smy-1,batchSize):fill(0)

    for j=1,batchSize do
        eosOffset = max_len_text - #sentBatch_text[j] --left paddding

        for j2=1,#sentBatch_text[j]do   
            if  Vocab[sentBatch_text[j][j2]] ~= nil then 
                encoderInputs[j2+eosOffset][j] = Vocab[sentBatch_text[j][j2]]
            else
                --encoderInputs[j2+eosOffset][j] = Vocab['<unknown>']
                encoderInputs[j2+eosOffset][j] = 0
            end
        end

        for j2=1,#sentBatch_smy[j]-1 do
            if Secab[sentBatch_smy[j][j2]] ~= nil then 
                decoderInputs[j2][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderInputs[j2][j] = Secab['<unknown>']
            end
        end

        for j2=2,#sentBatch_smy[j] do
            if Secab[sentBatch_smy[j][j2]] ~= nil then
                decoderTargets[j2-1][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderTargets[j2-1][j] = Secab['<unknown>']
            end
        end
    end
    
    if encoderInputs:dim()==0 then
        return -1
    end
    if encoderInputs:size()[1] > 100 then
        encoderInputs=encoderInputs:sub(-100,-1)
    end

    --encoderInputs = encoderInputs:resize(encoderInputs:size()[1])
    --decoderInputs = decoderInputs:resize(decoderInputs:size()[1])
    --decoderTargets = decoderTargets:resize(decoderTargets:size()[1])

    return {encoderInputs}, {decoderInputs}, {decoderTargets}
    --return encoderInputs, decoderInputs, decoderTargets
end 
--*************************************************************

function getVocab(filename)
    numDocumentsTraversed = 0
    local VocabSize = 0
    local Vocab = {}
    local Index2Vocab = {}
    VocabCnt = {}
    file_txt=io.open(filename,'r')

    Vocab['<unknown>'] = 1
    Index2Vocab[1] = '<unknown>'
    VocabSize = 1

    line = file_txt:read():lower()
    while line~=nil do
        if Vocab[line] == nil and line:match('.*[0-9]*.*') ~= nil then 
            VocabSize = VocabSize + 1
            Vocab[line] = VocabSize
            Index2Vocab[VocabSize] = line
            VocabCnt[line] = 1
        end
        
        if Vocab[line] ~= nil and line:match('.*[0-9]*.*') ~= nil then 
            VocabCnt[line] = VocabCnt[line] + 1
        end
        
       line = file_txt:read()
       if (line ~= nil) then
           line=line:lower()
       end

       if line == "" then
           numDocumentsTraversed = numDocumentsTraversed +1
       end
    
       if numDocumentsTraversed > numDocuments then 
           break
       end
    end

    VocabSize2 = 0
    Index2Vocab2 = {}
    Vocab2 = {}
    
    Vocab2['<unknown>'] = 1
    Index2Vocab2[1] = '<unknown>'
    VocabSize2 = 1
    
    for word, index in pairs(Vocab) do 
        if word ~= '<go>' and word ~= '<eos>' and word ~= '<unknown>' then
            if VocabCnt[word] >= minVocabCount then 
                VocabSize2 = VocabSize2 + 1
                Vocab2[word] = VocabSize2
                Index2Vocab2[VocabSize2] = word
            end
        end
    end
    Vocab = nil; Index2Vocab = nil 
    Vocab = Vocab2; Index2Vocab = Index2Vocab2 
    VocabSize = VocabSize2
    return Vocab, Index2Vocab, VocabSize
end


