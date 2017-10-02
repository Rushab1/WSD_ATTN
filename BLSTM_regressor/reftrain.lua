require 'functions.lua'
require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'


batchSize = 50
numDocuments = 5000
hiddenSize2 = 300
minVocabCount = 0


cmd = torch.CmdLine()
cmd:option('-model','model.t7')
opt = cmd:parse(arg)
model = torch.load(opt.model)
--model = torch.load('Models/model10_540')
Vocab = model['Vocab']
Index2Vocab = model['Index2Vocab']
model = model['model']
VocabSize = #Index2Vocab
print("Vocabulary Size = " .. VocabSize)


print("Getting Secab")
reviews, summaries = {}, {}
rf = io.open('text_words2.csv','r')
sf = io.open('summary_words2.csv','r')
rev = rf:read():lower()
while rev ~= nil do
    table.insert(reviews, rev:lower())
    rev = rf:read()
end

sum = sf:read():lower()
while sum ~= nil do
    table.insert(summaries, sum:lower())
    sum = sf:read()
end

Secab, Index2Secab, SecabSize = getVocab('summary_words2.csv')


model2 = nn.Sequential()
      :add(nn.Sequencer(nn.LookupTableMaskZero(VocabSize, hiddenSize2)))
      :add(nn.Sequencer(nn.FastLSTM(hiddenSize2, hiddenSize2)))
      :add(nn.Sequencer(nn.Linear(hiddenSize2,SecabSize)))
      :add(nn.Sequencer(nn.LogSoftMax()))
      :add(nn.Select(1,-1))

print("loaded model")
model2.modules[1].weight = model.modules[1].weight

criterion2 = nn.SequencerCriterion(nn.ClassNLLCriterion())
model2:cuda()
criterion2:cuda()
print('Done')

print('Getting Parameters')
x, dl_dx = model2:getParameters()
print('Done')

print('Getting sgd_params')
sgd_params = {
    learningRate = 0.1, --changed from 1e-2
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.2
}

for j=1,20 do
    print("------------EPOCH: "..j.."---------------")
    revcnt = 1
    sumcnt = 1
    file_txt = io.open('text_words2.csv','r')
    file_smy = io.open('summary_words2.csv','r')

    for i = 1, numDocuments/batchSize do	
        _, fs = optim.sgd(feval2,x, sgd_params)        
        model2:forget()

        if i%10 == 0 then
            print('error for Batch ' .. sgd_params.evalCounter  .. ' is ' .. " Number Reviews : ".. sgd_params.evalCounter*batchSize .. " : ".. fs[1])
        end

        if i % math.floor(numDocuments/(batchSize)) == 0 then 
            save_model = {}
            save_model['model'] = model2
            save_model['Vocab'] = Vocab
            save_model['Index2Vocab'] = Index2Vocab
            save_model['Secab'] = Secab
            save_model['Index2Secab'] = Index2Secab
            str_save = 'Models/lowermodel_' .. j .."_".. i
            torch.save(str_save, save_model)
        end
    end
end
    
    

