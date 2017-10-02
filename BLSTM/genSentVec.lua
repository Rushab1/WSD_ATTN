require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'


batchSize = 1000
hiddenSize = 50

cmd = torch.CmdLine()
cmd:option('-model','model.t7')
cmd:option('-set','train')
cmd:option('-csv','true')
opt = cmd:parse(arg)

loaded_model = torch.load(opt.model)
--loaded_model = torch.load('./Models/model12_48')
model = loaded_model.model
Vocab = loaded_model.Vocab
Secab = loaded_model.Secab
Index2Vocab = loaded_model.Index2Vocab
Index2Secab = loaded_model.Index2Secab
SecabSize = #Index2Secab

model_encoder = model.modules[1]
encoder = nn.Sequential()
encoder:add(model_encoder)
encoder:add(nn.Select(1,-1))
encoder:cuda()

------------------------CSV file writing functions-----------------
writeMatrix = function(subtensor, filename)  --write a 2-dimensional tensor to csv file
    local out = assert(io.open(filename, "w")) -- open a file for serialization

    splitter = ","
    for i=1,subtensor:size(1) do
        for j=1,subtensor:size(2) do
            out:write(subtensor[i][j])
            if j == subtensor:size(2) then
                out:write("\n")
            else
                out:write(splitter)
            end
        end
    end

    out:close()
end

CsvWrite = function(t, csvfile)
    Sent = t['Sent']
    Sense = t['Sense']

    len1 = #Sent
    len2 = (#Sent[1])[1]

    SentM = torch.Tensor(len1, len2)

    SentFile = csvfile .. "_sent.csv"
    SenseFile = io.open(csvfile .. "_sense.csv", "w")

    assert(#Sent == #Sense)

    for i  = 1, len1 do
        SenseFile:write(Sense[i] .. "\n")
        for j = 1, len2 do
            SentM[i][j] = Sent[i][j]
        end
    end
    io.close(SenseFile)

    writeMatrix(SentM, SentFile)
end
------------------------------------------------------------

if opt.set == 'train' then
    summary_words = 'summary_words.csv'
    text_words = 'text_words.csv'

elseif opt.set == 'test' then 
    summary_words = 'test_summary_words.csv'
    text_words = 'test_text_words.csv'

elseif opt.set == 'cvtrain' then
    summary_words = 'cv_summary_words.csv'
    text_words = 'cv_text_words.csv'

elseif opt.set == 'cvtest' then
    summary_words = 'cv_summary_words_test.csv'
    text_words = 'cv_text_words_test.csv'

elseif opt.set == 'cvvalid' then
    summary_words = 'cv_summary_words_valid.csv'
    text_words = 'cv_text_words_valid.csv'

end

f = io.open(summary_words,'r')
numDocuments = 0
tmp = f:read()
while tmp ~= nil do
    numDocuments = numDocuments + 1
    tmp = f:read()
    if tmp == "" then
        tmp = f:read()
    end
end

print("Number of Training examples = " .. numDocuments)
file_txt=io.open(text_words, 'r')
file_smy=io.open(summary_words, 'r')

print('Begin')
--reviews = {}
--summaries = {}

--rf = io.open('text_words.csv','r')
--sf = io.open('summary_words.csv','r')

--rev = rf:read():lower()
--while rev ~= nil do
    --table.insert(reviews, rev:lower())
    --rev = rf:read()
--end

--sum = sf:read():lower()
--while sum ~= nil do
    --table.insert(summaries, sum:lower())
    --sum = sf:read()
--end

--revcnt = 1
--sumcnt = 1

function nextBatch()
    local encoderInputs, decoderInputs, decoderTargets = {}, {}, {}

    sentBatch_text = {}
    sentBatch_smy={}
    local max_len_text = 0
    local max_len_smy = 0
    local j=0, j2

    for j=1, batchSize do
        if j > numDocuments then 
            break
        end
        local sent_txt = {}
        local sent_smy = {}

        line = file_txt:read()
        while line ~= "" and line~=nil do 
            line = line:lower()
            table.insert(sent_txt,line)
            line = file_txt:read()
            if line ~= nil then
                line = line:lower()
            end
        end
        table.insert(sentBatch_text,sent_txt)
        
        if max_len_text < #sent_txt then
            max_len_text = #sent_txt
        end

        table.insert(sent_smy, '<go>')
        line = file_smy:read()
        while line ~= "" and line~= nil  do
            line = line:lower()
            table.insert(sent_smy, line)
            line = file_smy:read()
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

    for j=1, batchSize do
        if j > numDocuments then
            break
        end
        eosOffset = max_len_text - #sentBatch_text[j] --left paddding

        for j2=1,#sentBatch_text[j]do   
            if  Vocab[sentBatch_text[j][j2]] ~= nil then 
                encoderInputs[j2+eosOffset][j] = Vocab[sentBatch_text[j][j2]]
            else
                --encoderInputs[j2+eosOffset][j] = Vocab['<unknown>']
                encoderInputs[j2+eosOffset][j] = 0
            end

        end

        --trimmedEosToken = sentBatch_smy[j]:sub(1,-2)
        for j2=1,#sentBatch_smy[j]-1 do
            if Secab[sentBatch_smy[j][j2]] ~= nil then 
                decoderInputs[j2][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderInputs[j2][j] = Secab['<unknown>']
            end
        end
        

        --trimmedGoToken = sentBatch_smy[j]:sub(2,-1)
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

    len1 = encoderInputs:size()[1]
    len2 = encoderInputs:size()[2]
    encoderInputsReverse = torch.IntTensor(encoderInputs:size()):fill(0)

    f = torch.IntTensor(2,len1,len2)
    for _i = 1, len1 do    
        for _j = 1, len2 do 
            --encoderInputsReverse[_i][_j] = encoderInputs[len1-_i+1][_j]
            f[1][_i][_j] = encoderInputs[_i][_j]
            f[2][_i][_j] = encoderInputs[len1-_i+1][_j]
        end
    end

    e = {}
    e[1] = encoderInputs
    e[2] = encoderInputsReverse 

    f = f:cuda()
    return f, {decoderInputs}, {decoderTargets}
    --return {encoderInputs}, {decoderInputs}, {decoderTargets}, sentBatch_smy
end 

function sentBatch(batchCount)
    Inputs, decoderInputs, Targets = nextBatch()
    Output = encoder:forward(Inputs)
    for _i = 1, batchSize do
        if _i + batchCount * batchSize > numDocuments then
            break
        end
        table.insert(Sent_vectors, Output:select(1,_i):clone())
        --table.insert(Senses,Targets[1][1][_i] - 3)
        table.insert(Senses, Targets[1][1][_i])
    end
end

Sent_vectors = {}
Senses = {}

for i = 1, torch.ceil(numDocuments/batchSize) do
    print("Processing Batch" .. i .. " = " .. i*batchSize .. " arctiles")
    sentBatch(i-1)
end

saveVectors = {}
saveVectors['Sent'] = Sent_vectors
saveVectors['Sense'] = Senses
saveVectors['model'] = loaded_model
torch.save("sent_vectors_" .. opt.set .. ".t7", saveVectors)

if opt.csv == 'true' then 
    print("Creating csv files")
    if opt.set == 'cvtrain' then 
        CsvWrite(saveVectors, 'train')
    elseif opt.set == 'cvtest' then 
        CsvWrite(saveVectors, 'test')
    end

end
