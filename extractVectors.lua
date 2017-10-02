cmd = torch.CmdLine()
cmd:option('-file', '/data/gpuuser2/Data/Glove/glove.6B.300d.txt')
cmd:option('-vector_file', 'Glove.t7')
opt =  cmd:parse(arg)

print("Reading file: " .. opt.file)
f = io.open(opt.file)
print("done")

print("Extracting word vetors. This may take some time.")
line = f:read()

lookup = {}

while line ~= nil do
    sep_line = line:split(' ')
    vec = torch.zeros(#sep_line - 1)
    for i = 1, #sep_line - 1 do
        vec[i] = tonumber(sep_line[i + 1])
    end

    lookup[sep_line[1]] = vec
    
    line = f:read()
end

print("done")

print("Saving word vetors")
torch.save('Glove.t7', lookup)
print("done")
